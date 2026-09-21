#!/usr/bin/env python3
"""Probe: does splitting dense CapRL captions (median 756 tokens, one vector) into
sentences / token chunks / core captions recover cases where full-caption Qwen3
retrieval ranked the target window poorly?

Stages:
    pool     build the 40-case subset (30 worst + 10 controls) and a per-case candidate pool
    lexical  full-corpus TF-IDF sentence-chunk MaxSim vs TF-IDF full caption (all 137 cases)
    dense    Qwen3-Embedding-0.6B on GPU: full vs sentence/chunk128/chunk256/core MaxSim
    report   print and save the comparison tables

The subset only feeds the pool-based dense stage; the lexical stage always covers all 137
cases so the mechanism can be checked at corpus scale without any model.
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.bench_caption_retrieval import (CAPTIONS_PARQUET, CAPTION_DIR, CACHE_DIR,
                                                        MAX_SEQ_LEN, QUERIES, QUERY_PROMPT,
                                                        TOLERANCE_SECONDS, caption_documents,
                                                        first_hit_rank, rank_summary,
                                                        target_windows)
from scripts.benchmarks.bench_visual_embeddings import PE_PROFILE, load_cases, profile_maps

PROBE_DIR = CACHE_DIR / "probe"
POOL_JSON = PROBE_DIR / "pool.json"
LEXICAL_JSON = PROBE_DIR / "lexical_probe.json"
DENSE_JSON = PROBE_DIR / "dense_probe.json"
CORE_JSON = PROBE_DIR / "core_captions_probe.json"
DIAGNOSIS = CACHE_DIR / "diagnosis.json"
QWEN_SMALL = "Qwen/Qwen3-Embedding-0.6B"
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def split_sentences(text, min_words=4):
    parts = [part.strip() for part in SENTENCE.split(text.strip()) if part.strip()]
    merged = []
    for part in parts:
        if merged and len(part.split()) < min_words:
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return merged or [text]

def token_chunks(text, tokenizer, size, overlap):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= size:
        return [text]
    chunks, step = [], size - overlap
    for start in range(0, len(ids), step):
        chunks.append(tokenizer.decode(ids[start:start + size], skip_special_tokens=True))
        if start + size >= len(ids):
            break
    return chunks


class SmallEncoder:
    """Length-sorted, token-budget batches: the shared encode_qwen pads every batch to the
    longest caption, which is the difference between 3 and 30 minutes on long captions."""

    def __init__(self, model_id, dtype, device):
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.model = AutoModel.from_pretrained(model_id, dtype=dtypes[dtype]).to(device).eval()
        self.device = device
        self.dim = self.model.config.hidden_size

    def encode(self, texts, token_budget=16384, max_batch=256):
        torch = self.torch
        lengths = np.asarray([len(ids) for ids in self.tokenizer(
            texts, truncation=True, max_length=MAX_SEQ_LEN,
            add_special_tokens=True)["input_ids"]], dtype=np.int64)
        order = np.argsort(lengths, kind="stable")
        vectors = np.empty((len(texts), self.dim), dtype=np.float32)
        with torch.inference_mode():
            start = 0
            while start < len(order):
                end, tokens = start, 0
                while end < len(order) and (end - start) < max_batch:
                    size = int(lengths[order[end]])
                    if end > start and tokens + size > token_budget:
                        break
                    tokens += size
                    end += 1
                indices = order[start:end]
                encoded = self.tokenizer([texts[index] for index in indices], padding=True,
                                         truncation=True, max_length=MAX_SEQ_LEN,
                                         return_tensors="pt").to(self.device)
                hidden = self.model(**encoded, use_cache=False).last_hidden_state
                vectors[indices] = torch.nn.functional.normalize(
                    hidden[:, -1].float(), dim=-1).cpu().numpy()
                start = end
        return vectors


def window_topk_mean(scores, window_of, n_windows, k):
    order = np.argsort(window_of, kind="stable")
    windows = window_of[order]
    values = scores[order]
    bounds = np.searchsorted(windows, np.arange(n_windows + 1))
    out = np.empty(n_windows, dtype=np.float32)
    for window in range(n_windows):
        segment = values[bounds[window]:bounds[window + 1]]
        top = np.sort(segment)[::-1][:k] if len(segment) else [0.0]
        out[window] = float(np.mean(top))
    return out


def moment_rank_from_scores(scores, window_rows):
    order = np.argsort(-scores)
    return first_hit_rank(order, window_rows)


def video_rank_from_scores(scores, video_of, target_videos):
    for rank, item in enumerate(np.argsort(-scores), 1):
        if video_of[item] in target_videos:
            return rank
    return None


def rank_metrics(ranks):
    summary = rank_summary(ranks)
    summary["mrr"] = round(float(np.mean([1.0 / rank for rank in ranks if rank])), 4) if any(ranks) else 0.0
    summary["top25"] = round(sum(rank is not None and rank <= 25 for rank in ranks) / len(ranks), 4)
    summary["top50"] = round(sum(rank is not None and rank <= 50 for rank in ranks) / len(ranks), 4)
    return summary


def build_pool(args):
    cases = load_cases(QUERIES)
    diagnosis = {row["id"]: row for row in json.loads(DIAGNOSIS.read_text())["cases_detail"]}
    ranked = [(case, diagnosis[case["id"]]["caption_moment_rank"]) for case in cases]
    bad = sorted([item for item in ranked if item[1] is not None and item[1] > 100],
                 key=lambda item: -item[1])[:args.bad_cases]
    control = sorted([item for item in ranked if item[1] is not None and item[1] <= 5],
                     key=lambda item: item[0]["id"])[:args.control_cases]
    subset = bad + control
    log(f"subset: {len(bad)} bad (rank>100) + {len(control)} control (rank<=5)")

    docs, cap_meta = caption_documents(CAPTION_DIR)
    vectors = np.load(CACHE_DIR / "caption_query_vectors.npy").astype(np.float32)
    scores = docs @ vectors.T
    del docs
    gc.collect()
    topk = np.load(CACHE_DIR / "caption_topk.npz")["indices"]
    cap_video = np.asarray([row["video_id"] for row in cap_meta])
    cap_starts = np.asarray([row["window_start_sec"] for row in cap_meta])
    cap_ends = np.asarray([row["window_end_sec"] for row in cap_meta])
    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    _, _, pe_meta = profile_maps(PE_PROFILE)
    fps = {row["video_id"]: row["fps"] for row in pe_meta}

    case_ids = [case["id"] for case in cases]
    pool, windows = {}, {}
    for case, _ in subset:
        row = case_ids.index(case["id"])
        window_rows, target_videos = target_windows(case, cap_video, cap_starts, cap_ends,
                                                    fps, TOLERANCE_SECONDS)
        target_video_rows = np.flatnonzero(np.isin(cap_video, list(target_videos)))
        by_score = target_video_rows[np.argsort(-scores[target_video_rows, row])][:20]
        global_top = [int(index) for index in topk[row][:50]]
        realistic = sorted(set(global_top) | {int(index) for index in by_score})
        oracle = sorted(set(realistic) | set(window_rows))
        pool[case["id"]] = {"realistic": realistic, "oracle": oracle, "targets": window_rows,
                            "target_videos": sorted(target_videos)}
        for index in oracle:
            windows[str(index)] = {"video_id": cap_video[index],
                                   "start": float(cap_starts[index]), "end": float(cap_ends[index]),
                                   "caption": captions[index]}
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    POOL_JSON.write_text(json.dumps({"bad": [c["id"] for c, _ in bad],
                                     "control": [c["id"] for c, _ in control],
                                     "cases": pool, "windows": windows,
                                     "window_count": len(windows)}, indent=1) + "\n")
    log(f"wrote {POOL_JSON}: {len(subset)} cases, {len(windows)} unique windows")


def lexical_probe(args):
    cases = load_cases(QUERIES)
    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    log(f"splitting {len(captions)} captions into sentences")
    chunk_texts, chunk_window = [], []
    for index, caption in enumerate(captions):
        for sentence in split_sentences(caption):
            chunk_texts.append(sentence)
            chunk_window.append(index)
    chunk_window = np.asarray(chunk_window)
    log(f"{len(chunk_texts)} sentences")

    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2,
                                 max_features=500_000, dtype=np.float32)
    documents = vectorizer.fit_transform(chunk_texts)
    queries = normalize(vectorizer.transform([case["text"] for case in cases])).tocsr()
    log(f"tf-idf chunk matrix {documents.shape} nnz={documents.nnz}")

    full_vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2,
                                      max_features=500_000, dtype=np.float32)
    full_documents = full_vectorizer.fit_transform(captions)
    full_queries = normalize(full_vectorizer.transform([case["text"] for case in cases])).tocsr()

    _, cap_meta = caption_documents(CAPTION_DIR)
    cap_video = np.asarray([row["video_id"] for row in cap_meta])
    cap_starts = np.asarray([row["window_start_sec"] for row in cap_meta])
    cap_ends = np.asarray([row["window_end_sec"] for row in cap_meta])
    _, _, pe_meta = profile_maps(PE_PROFILE)
    fps = {row["video_id"]: row["fps"] for row in pe_meta}

    n_windows = len(captions)
    chunk_ranks, chunk_video_ranks, full_ranks, full_video_ranks = [], [], [], []
    for position, case in enumerate(cases):
        window_rows, target_videos = target_windows(case, cap_video, cap_starts, cap_ends,
                                                    fps, TOLERANCE_SECONDS)
        chunk_scores = np.zeros(n_windows, dtype=np.float32)
        values = (documents @ queries[position].T).toarray().ravel()
        np.maximum.at(chunk_scores, chunk_window, values)
        full_scores = (full_documents @ full_queries[position].T).toarray().ravel()
        chunk_ranks.append(moment_rank_from_scores(chunk_scores, window_rows))
        chunk_video_ranks.append(video_rank_from_scores(chunk_scores, cap_video, target_videos))
        full_ranks.append(moment_rank_from_scores(full_scores, window_rows))
        full_video_ranks.append(video_rank_from_scores(full_scores, cap_video, target_videos))

    report = {
        "cases": len(cases), "sentences": len(chunk_texts),
        "tfidf_full": {"moment": rank_metrics(full_ranks), "video": rank_metrics(full_video_ranks)},
        "tfidf_sentence_chunks": {"moment": rank_metrics(chunk_ranks),
                                  "video": rank_metrics(chunk_video_ranks)},
        "moment_ranks_full": full_ranks, "moment_ranks_chunks": chunk_ranks,
    }
    LEXICAL_JSON.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {LEXICAL_JSON}")
    for name in ("tfidf_full", "tfidf_sentence_chunks"):
        log(f"{name}: moment {report[name]['moment']}")
        log(f"{name}: video  {report[name]['video']}")


def dense_probe(args):
    from transformers import AutoTokenizer

    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads(POOL_JSON.read_text())
    windows = pool["windows"]
    subset = pool["bad"] + pool["control"]
    caption_of = {int(index): row["caption"] for index, row in windows.items()}
    tokenizer = AutoTokenizer.from_pretrained(QWEN_SMALL)
    core = json.loads(CORE_JSON.read_text()) if CORE_JSON.exists() else {}
    if not core:
        log(f"warning: {CORE_JSON} missing, skipping the core-caption variant")

    chunks = {"full": {index: [caption] for index, caption in caption_of.items()},
              "sentence": {index: split_sentences(caption) for index, caption in caption_of.items()},
              "chunk128": {index: token_chunks(caption, tokenizer, 128, 32)
                           for index, caption in caption_of.items()},
              "chunk256": {index: token_chunks(caption, tokenizer, 256, 32)
                           for index, caption in caption_of.items()}}
    if core:
        chunks["core"] = {index: core[str(index)]["cores"] for index in caption_of
                          if str(index) in core and core[str(index)].get("cores")}

    encoder = SmallEncoder(QWEN_SMALL, args.dtype, args.device)
    log(f"encoding {len(case_ids)} queries with {QWEN_SMALL} on {args.device}")
    query_cache = PROBE_DIR / "query_vectors_06b.npy"
    if query_cache.exists():
        query_vectors = np.load(query_cache)
    else:
        query_vectors = encoder.encode([QUERY_PROMPT + case["text"] for case in cases],
                                       token_budget=8192, max_batch=64)
        np.save(query_cache, query_vectors)

    variant_ranks = {}
    for name, mapping in chunks.items():
        texts, window_of = [], []
        for index, pieces in mapping.items():
            for piece in pieces:
                texts.append(piece)
                window_of.append(index)
        window_of = np.asarray(window_of)
        vector_cache = PROBE_DIR / f"vectors_{name}.npy"
        window_cache = PROBE_DIR / f"window_of_{name}.npy"
        if vector_cache.exists() and window_cache.exists():
            vectors = np.load(vector_cache)
            window_of = np.load(window_cache)
            log(f"{name}: reusing {vector_cache.name} ({vectors.shape[0]} chunks)")
        else:
            log(f"{name}: encoding {len(texts)} chunks from {len(mapping)} windows")
            vectors = encoder.encode(texts)
            np.save(vector_cache, vectors)
            np.save(window_cache, window_of)
        ordered_windows = np.asarray(sorted(mapping))
        position_of = {int(index): position for position, index in enumerate(ordered_windows)}
        chunk_position = np.asarray([position_of[int(index)] for index in window_of])
        for case in cases:
            entries = pool["cases"].get(case["id"])
            if entries is None:
                continue
            row = case_ids.index(case["id"])
            cosine = vectors @ query_vectors[row]
            per_window_max = np.zeros(len(ordered_windows), dtype=np.float32)
            np.maximum.at(per_window_max, chunk_position, cosine)
            per_window_top2 = window_topk_mean(cosine, chunk_position, len(ordered_windows), 2)
            moments = {}
            for pool_name in ("realistic", "oracle"):
                index_list = entries[pool_name]
                positions = np.asarray([position_of[int(index)] for index in index_list])
                target_set = set(entries["targets"])
                for agg_name, per_window in (("max", per_window_max), ("top2", per_window_top2)):
                    local_order = np.argsort(-per_window[positions])
                    moments[f"{pool_name}_{agg_name}"] = next(
                        (rank for rank, slot in enumerate(local_order, 1)
                         if index_list[slot] in target_set), None)
            variant_ranks.setdefault(name, {})[case["id"]] = moments
        del vectors
        gc.collect()

    report = {"subset": subset, "variants": {}}
    for name, per_case in variant_ranks.items():
        for key in ("realistic_max", "realistic_top2", "oracle_max", "oracle_top2"):
            ranks = [per_case[case_id].get(key) for case_id in subset]
            report["variants"].setdefault(name, {})[key] = rank_metrics(ranks)
    report["per_case"] = {name: {case_id: variant_ranks[name].get(case_id)
                                 for case_id in subset} for name in variant_ranks}
    DENSE_JSON.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {DENSE_JSON}")
    for name, pools in report["variants"].items():
        log(f"{name}: realistic_max {pools['realistic_max']}")


def self_test():
    assert split_sentences("A man stands. He smiles at the camera! Then walks away.") == \
        ["A man stands.", "He smiles at the camera! Then walks away."]
    assert split_sentences("Short. A much longer sentence follows here.") == \
        ["Short.", "A much longer sentence follows here."]
    scores = np.array([0.9, 0.1, 0.5, 0.4], dtype=np.float32)
    window_of = np.array([0, 0, 1, 1])
    assert abs(window_topk_mean(scores, window_of, 2, 1)[1] - 0.5) < 1e-6
    assert abs(window_topk_mean(scores, window_of, 2, 2)[1] - 0.45) < 1e-6
    assert rank_summary([1, 3, None])["found"] == 2
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pool", "lexical", "dense", "all"), default="all")
    parser.add_argument("--bad-cases", type=int, default=30)
    parser.add_argument("--control-cases", type=int, default=10)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--query-batch", type=int, default=16)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.stage in ("pool", "all"):
        build_pool(args)
    if args.stage in ("lexical", "all"):
        lexical_probe(args)
    if args.stage in ("dense", "all"):
        dense_probe(args)


if __name__ == "__main__":
    main()
