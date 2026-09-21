#!/usr/bin/env python3
"""Caption retrieval (Qwen3-Embedding-4B over CapRL shot captions) vs PE-Core keyframe
retrieval on the 137 official English queries (P1 + Round 1/2/3 - 2026), plus fusion
variants that use the caption ranking as a video prior for PE-Core.

Reuses the PE-Core text encoder, hit metrics and RRF helpers from
``bench_visual_embeddings.py`` and the caption matching rule from the Kaggle embedding
kernel, so both branches are scored exactly like their existing benchmarks.

Usage:
    uv run python -m scripts.benchmarks.bench_caption_retrieval            # full run
    uv run python -m scripts.benchmarks.bench_caption_retrieval --reuse    # cached top-k
    uv run python -m scripts.benchmarks.bench_caption_retrieval --diagnose # root-cause stats
    uv run python -m scripts.benchmarks.bench_caption_retrieval --self-test
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

from notebooks.kaggle_qwen3_caption_embed import (EVAL_K, INSTRUCTION, MAX_SEQ_LEN,
                                                  TOLERANCE_SECONDS, first_match)
from scripts.benchmarks.bench_visual_embeddings import (PE_PROFILE, REPORTED_ROUNDS,
                                                        encode_pe, first_hit, load_cases,
                                                        metrics, profile_maps, rrf,
                                                        rrf_equal, round_of)

CACHE_DIR = ROOT / "artifacts/caption_vs_pe_core"
CAPTION_DIR = ROOT / "data/2026/embeddings/qwen3-4b-fp16"
CAPTIONS_PARQUET = ROOT / "data/2026/captions/shot_captions.parquet"
QUERIES = ROOT / "artifacts/official_benchmark_queries_en_v2.json"
PE_BASELINE = ROOT / "benchmarks/visual/visual_embedding_benchmark_2026_en.json"
OUTPUT = ROOT / "benchmarks/visual/caption_vs_pe_core_2026.json"
DIAGNOSIS = CACHE_DIR / "diagnosis.json"
QWEN_MODEL = "Qwen/Qwen3-Embedding-4B"
QUERY_PROMPT = f"Instruct: {INSTRUCTION}\nQuery: "
ALL_ROUNDS = ["Test"] + REPORTED_ROUNDS


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def encode_qwen(texts, model_id=QWEN_MODEL, dtype="bfloat16", device="cpu", batch_size=8):
    """Same contract as the Kaggle kernel's HFEncoder: prompt prepended, left padding,
    last token pooled, L2 normalised; dtype/device only affect numerics, not the recipe."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = AutoModel.from_pretrained(model_id, dtype=dtypes[dtype]).to(device).eval()
    dim = model.config.hidden_size
    vectors = np.empty((len(texts), dim), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=MAX_SEQ_LEN,
                                return_tensors="pt").to(device)
            hidden = model(**encoded, use_cache=False).last_hidden_state
            vectors[start:start + len(batch)] = torch.nn.functional.normalize(
                hidden[:, -1].float(), dim=-1).cpu().numpy()
            log(f"  encoded {min(start + batch_size, len(texts))}/{len(texts)} queries")
    del model
    gc.collect()
    return vectors


def caption_documents(caption_dir):
    paths = sorted(Path(caption_dir).glob("emb_*.npy"))
    if not paths:
        raise FileNotFoundError(f"no caption embeddings in {caption_dir}")
    documents = np.vstack([np.load(path) for path in paths]).astype(np.float32)
    meta = pq.read_table(Path(caption_dir) / "meta.parquet").to_pylist()
    if documents.shape[0] != len(meta):
        raise ValueError(f"{documents.shape[0]} vectors vs {len(meta)} meta rows")
    return documents, meta


def search_topk(documents, queries, k):
    # ponytail: brute-force dot product; 103k x 2560 is ~5 s, add FAISS if the corpus grows.
    indices = np.empty((len(queries), k), dtype=np.int32)
    scores = np.empty((len(queries), k), dtype=np.float32)
    for row, query in enumerate(queries):
        values = documents @ query
        take = min(k, len(values) - 1)
        top = np.argpartition(-values, take)[:take + 1]
        ordered = top[np.argsort(-values[top])][:k]
        indices[row] = ordered
        scores[row] = values[ordered]
    return indices, scores


def pe_topk(vectors, k):
    import faiss

    index = faiss.read_index(str(PE_PROFILE / "flat.ip"))
    distances, indices = index.search(vectors.astype(np.float32), k)
    return indices.astype(np.int32), distances.astype(np.float32)


def video_order(ordered, item_videos):
    seen, out = set(), []
    for item in ordered:
        video = item_videos[item]
        if video not in seen:
            seen.add(video)
            out.append(video)
    return out


def first_video_hit(ordered, target_videos, item_videos):
    for rank, item in enumerate(ordered, 1):
        if item_videos[item] in target_videos:
            return rank
    return None


def filter_by_videos(ordered, keep_videos, item_videos):
    return [item for item in ordered if item_videos[item] in keep_videos]


def boosted_order(ordered, scores, item_videos, video_bonus):
    bonuses = np.array([video_bonus.get(item_videos[item], 0.0) for item in ordered])
    return [ordered[position] for position in np.argsort(-(scores + bonuses))]


def target_video_indices(case, video_index):
    return {video_index[video] for video, _ in case["targets"]}


def by_round_report(found, cases):
    report = {}
    for round_name in ALL_ROUNDS + ["Round 1-3 - 2026 (aggregate)"]:
        if round_name.endswith("(aggregate)"):
            rows = [rank for rank, case in zip(found, cases) if round_of(case["id"]) in REPORTED_ROUNDS]
        else:
            rows = [rank for rank, case in zip(found, cases) if round_of(case["id"]) == round_name]
        report[round_name] = metrics(rows)
    return report


def load_cache(path, case_ids):
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    if list(data["case_ids"]) != case_ids:
        log(f"{path.name}: case ids changed, ignoring cache")
        return None
    return data


def caption_branch(cases, cache_dir, reuse, args):
    texts = [case["text"] for case in cases]
    case_ids = [case["id"] for case in cases]
    topk_path = cache_dir / "caption_topk.npz"
    cached = load_cache(topk_path, case_ids) if reuse else None
    if cached is not None:
        log(f"reusing {topk_path}")
        indices, scores = cached["indices"], cached["scores"]
    else:
        vectors_path = cache_dir / "caption_query_vectors.npy"
        if reuse and vectors_path.exists():
            log(f"reusing {vectors_path}")
            vectors = np.load(vectors_path)
        else:
            log(f"encoding {len(texts)} queries with {args.caption_model} on {args.device}")
            vectors = encode_qwen([QUERY_PROMPT + text for text in texts], args.caption_model,
                                  args.dtype, args.device, args.batch_size)
            np.save(vectors_path, vectors)
        documents, _ = caption_documents(args.caption_dir)
        log(f"searching {documents.shape[0]} captions with {vectors.shape[1]}-d queries")
        indices, scores = search_topk(documents, vectors, args.k)
        np.savez_compressed(topk_path, indices=indices, scores=scores, case_ids=np.array(case_ids))
        del documents
        gc.collect()
    meta = pq.read_table(Path(args.caption_dir) / "meta.parquet").to_pylist()
    return indices, scores, meta


def pe_branch(cases, cache_dir, reuse, args):
    texts = [case["text"] for case in cases]
    case_ids = [case["id"] for case in cases]
    topk_path = cache_dir / "pe_topk.npz"
    cached = load_cache(topk_path, case_ids) if reuse else None
    if cached is not None:
        log(f"reusing {topk_path}")
        indices, scores = cached["indices"], cached["scores"]
    else:
        vectors_path = cache_dir / "pe_query_vectors.npy"
        if reuse and vectors_path.exists():
            log(f"reusing {vectors_path}")
            vectors = np.load(vectors_path)
        else:
            log(f"encoding {len(texts)} queries with PE-Core text tower")
            vectors = encode_pe(texts)
            np.save(vectors_path, vectors)
        log("searching PE-Core l2026 keyframe index")
        indices, scores = pe_topk(vectors, args.k)
        np.savez_compressed(topk_path, indices=indices, scores=scores, case_ids=np.array(case_ids))
    return indices, scores


def validate(name, computed, reference, tolerance_report):
    pairs = [(case_id, computed.get(case_id), reference.get(case_id)) for case_id in reference]
    equal = sum(a == b for _, a, b in pairs)
    tolerance_report[name] = {"cases": len(pairs), "identical": equal,
                              "identical_pct": round(100 * equal / len(pairs), 1),
                              "mismatches": {case_id: [a, b] for case_id, a, b in pairs if a != b}}
    log(f"validation {name}: {equal}/{len(pairs)} identical to {tolerance_report[name]['identical_pct']}%")
    return tolerance_report


STOPWORDS = frozenset(
    "the a an of in on at to and or for with is are was were be been being this that these those "
    "it its as by from into during over under up down out off then when where which who whom whose "
    "what how not no yes there here also has have had do does did but if while".split())


def content_words(text):
    return {word for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in STOPWORDS and len(word) > 2}


def lexical_overlap(query, caption):
    query_words = content_words(query)
    return len(query_words & content_words(caption)) / max(len(query_words), 1)


def target_windows(case, video_of, starts, ends, fps, tolerance):
    rows, videos = [], set()
    for video_id, frame in case["targets"]:
        seconds = frame / fps[video_id]
        mask = ((video_of == video_id) & (starts - tolerance <= seconds)
                & (seconds <= ends + tolerance))
        rows.extend(np.flatnonzero(mask).tolist())
        videos.add(video_id)
    return sorted(set(rows)), videos


def target_frames(case, video_ids, frames, video_index, fps, tolerance):
    rows, videos = [], set()
    for video_id, frame in case["targets"]:
        index = video_index[video_id]
        mask = (video_ids == index) & (np.abs(frames - frame) / fps[video_id] <= tolerance)
        rows.extend(np.flatnonzero(mask).tolist())
        videos.add(index)
    return sorted(set(rows)), videos


def first_hit_rank(order, target_rows):
    targets = set(target_rows)
    for rank, item in enumerate(order, 1):
        if int(item) in targets:
            return rank
    return None


def rank_summary(ranks):
    values = np.asarray([rank for rank in ranks if rank is not None])
    total = len(ranks)
    summary = {"cases": total, "found": len(values)}
    if len(values):
        summary.update({"median": int(np.median(values)),
                        "top1": round(float((values <= 1).sum() / total), 4),
                        "top5": round(float((values <= 5).sum() / total), 4),
                        "top10": round(float((values <= 10).sum() / total), 4),
                        "top100": round(float((values <= 100).sum() / total), 4)})
    return summary


def token_counts(tokenizer, texts):
    counts = []
    for start in range(0, len(texts), 512):
        encoded = tokenizer(texts[start:start + 512], truncation=True, max_length=4096)["input_ids"]
        counts.extend(len(ids) for ids in encoded)
    return np.asarray(counts)


def lexical_baseline(captions, cases, cap_video_of, cap_starts, cap_ends, pe_fps, tolerance):
    """TF-IDF keyword retrieval on the same captions: if a purely lexical ranker matches
    the dense embedding, the ceiling is the caption text itself, not the encoder."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as error:
        log(f"skipping TF-IDF baseline: {error}")
        return None
    vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2,
                                 max_features=400_000, dtype=np.float32)
    scores = (vectorizer.fit_transform(captions)
              @ vectorizer.transform([case["text"] for case in cases]).T).toarray()
    moment, video = [], []
    for position, case in enumerate(cases):
        row = scores[:, position]
        window_rows, target_videos = target_windows(case, cap_video_of, cap_starts, cap_ends,
                                                    pe_fps, tolerance)
        moment.append(first_hit_rank(np.argsort(-row), window_rows))
        mask = np.isin(cap_video_of, list(target_videos))
        best = float(row[mask].max()) if mask.any() else None
        video.append(int((row > best).sum()) + 1 if best is not None else None)
    return {"moment_rank_global": rank_summary(moment),
            "target_video_best_rank_global": rank_summary(video)}


def diagnose(args):
    """Root-cause stats: is the loss in global video identification, in locating the
    moment inside the right video, or in captions that lack the queried detail?"""
    import faiss

    cases = load_cases(args.queries)
    pe_ids, pe_frames, pe_meta = profile_maps(PE_PROFILE)
    pe_fps = {row["video_id"]: row["fps"] for row in pe_meta}
    video_index = {row["video_id"]: row["video_idx"] for row in pe_meta}

    docs, cap_meta = caption_documents(args.caption_dir)
    captions = pq.read_table(args.captions_parquet, columns=["caption"]).column("caption").to_pylist()
    cap_video_of = np.asarray([row["video_id"] for row in cap_meta])
    cap_starts = np.asarray([row["window_start_sec"] for row in cap_meta])
    cap_ends = np.asarray([row["window_end_sec"] for row in cap_meta])

    caption_vectors = np.load(args.cache_dir / "caption_query_vectors.npy").astype(np.float32)
    pe_vectors = np.load(args.cache_dir / "pe_query_vectors.npy").astype(np.float32)
    log(f"scoring {len(cases)} queries against {docs.shape[0]} captions and the PE index")
    caption_scores = docs @ caption_vectors.T
    del docs
    gc.collect()
    index = faiss.read_index(str(PE_PROFILE / "flat.ip"))
    pe_matrix = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    pe_scores = pe_matrix @ pe_vectors.T
    del pe_matrix, index
    gc.collect()

    rows = []
    for position, case in enumerate(cases):
        caption_row = caption_scores[:, position]
        pe_row = pe_scores[:, position]
        caption_order = np.argsort(-caption_row)
        pe_order = np.argsort(-pe_row)
        window_rows, target_videos = target_windows(case, cap_video_of, cap_starts, cap_ends,
                                                    pe_fps, args.tolerance_seconds)
        frame_rows, target_indices = target_frames(case, pe_ids, pe_frames, video_index,
                                                   pe_fps, args.tolerance_seconds)
        caption_video_mask = np.isin(cap_video_of, list(target_videos))
        pe_video_mask = np.isin(pe_ids, list(target_indices))
        caption_video_best = (float(caption_row[caption_video_mask].max())
                              if caption_video_mask.any() else None)
        pe_video_best = float(pe_row[pe_video_mask].max()) if pe_video_mask.any() else None
        target_window = (max(window_rows, key=lambda row: caption_row[row])
                         if window_rows else None)
        target_frame = max(frame_rows, key=lambda row: pe_row[row]) if frame_rows else None
        top1 = int(caption_order[0])
        rows.append({
            "id": case["id"],
            "caption_moment_rank": first_hit_rank(caption_order, window_rows),
            "caption_video_rank": (int((caption_row > caption_video_best).sum()) + 1
                                   if caption_video_best is not None else None),
            "caption_window_rank_in_video": (
                int((caption_row[caption_video_mask] > caption_row[target_window]).sum()) + 1
                if target_window is not None else None),
            "caption_video_top1_ok": bool(cap_video_of[top1] in target_videos),
            "caption_overlap_target": round(lexical_overlap(case["text"], captions[target_window]), 3)
            if target_window is not None else None,
            "caption_overlap_top1": round(lexical_overlap(case["text"], captions[top1]), 3),
            "pe_moment_rank": first_hit_rank(pe_order, frame_rows),
            "pe_video_rank": (int((pe_row > pe_video_best).sum()) + 1
                              if pe_video_best is not None else None),
            "pe_frame_rank_in_video": (
                int((pe_row[pe_video_mask] > pe_row[target_frame]).sum()) + 1
                if target_frame is not None else None),
        })

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.caption_model)
    caption_lengths = token_counts(tokenizer, captions)
    query_lengths = token_counts(tokenizer, [case["text"] for case in cases])
    wrong = [row for row in rows if not row["caption_video_top1_ok"]]
    tfidf = lexical_baseline(captions, cases, cap_video_of, cap_starts, cap_ends,
                             pe_fps, args.tolerance_seconds)

    def moment_rank_given_video_top1(video_key, moment_key):
        return rank_summary([row[moment_key] for row in rows if row[video_key] == 1])

    report = {
        "cases": len(cases), "tolerance_seconds": args.tolerance_seconds,
        "tokens": {"caption_median": int(np.median(caption_lengths)),
                   "caption_p90": int(np.percentile(caption_lengths, 90)),
                   "caption_truncated_at_max": int((caption_lengths >= MAX_SEQ_LEN).sum()),
                   "query_median": int(np.median(query_lengths))},
        "caption": {
            "moment_rank_global": rank_summary([row["caption_moment_rank"] for row in rows]),
            "window_rank_within_target_video": rank_summary(
                [row["caption_window_rank_in_video"] for row in rows]),
            "target_video_best_rank_global": rank_summary([row["caption_video_rank"] for row in rows]),
            "video_top1_accuracy": round(sum(row["caption_video_top1_ok"] for row in rows) / len(rows), 4),
            "moment_rank_given_video_top1": moment_rank_given_video_top1("caption_video_rank",
                                                                         "caption_moment_rank"),
        },
        "pe_core": {
            "moment_rank_global": rank_summary([row["pe_moment_rank"] for row in rows]),
            "frame_rank_within_target_video": rank_summary(
                [row["pe_frame_rank_in_video"] for row in rows]),
            "target_video_best_rank_global": rank_summary([row["pe_video_rank"] for row in rows]),
            "moment_rank_given_video_top1": moment_rank_given_video_top1("pe_video_rank",
                                                                         "pe_moment_rank"),
        },
        "lexical_coverage": {
            "target_caption_mean": round(float(np.mean([row["caption_overlap_target"] for row in rows])), 3),
            "top1_caption_mean": round(float(np.mean([row["caption_overlap_top1"] for row in rows])), 3),
            "target_caption_ge_0.8": round(sum(row["caption_overlap_target"] >= 0.8 for row in rows) / len(rows), 3),
            "target_caption_le_0.3": round(sum(row["caption_overlap_target"] <= 0.3 for row in rows) / len(rows), 3),
        },
        "ambiguity": {
            "caption_video_top1_wrong": len(wrong),
            "wrong_top1_overlap_mean": round(float(np.mean([row["caption_overlap_top1"] for row in wrong])), 3),
            "wrong_target_overlap_mean": round(float(np.mean([row["caption_overlap_target"] for row in wrong])), 3),
            "wrong_top1_overlap_ge_target": sum(row["caption_overlap_top1"] >= row["caption_overlap_target"]
                                                for row in wrong),
        },
        "tfidf_baseline": tfidf,
        "cases_detail": rows,
    }
    args.diagnosis_out.parent.mkdir(parents=True, exist_ok=True)
    args.diagnosis_out.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {args.diagnosis_out}")
    for branch in ("caption", "pe_core"):
        log(f"{branch}:")
        for name, value in report[branch].items():
            log(f"  {name}: {value}")
    log(f"tokens: {report['tokens']}")
    log(f"lexical coverage: {report['lexical_coverage']}")
    log(f"ambiguity: {report['ambiguity']}")
    log(f"tfidf baseline: {report['tfidf_baseline']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--caption-dir", type=Path, default=CAPTION_DIR)
    parser.add_argument("--captions-parquet", type=Path, default=CAPTIONS_PARQUET)
    parser.add_argument("--pe-baseline", type=Path, default=PE_BASELINE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--diagnosis-out", type=Path, default=DIAGNOSIS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--caption-model", default=QWEN_MODEL)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--k", type=int, default=EVAL_K)
    parser.add_argument("--tolerance-seconds", type=float, default=TOLERANCE_SECONDS)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.diagnose:
        diagnose(args)
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(args.queries)
    case_ids = [case["id"] for case in cases]
    log(f"{len(cases)} cases from {args.queries}")

    pe_ids, pe_frames, pe_meta = profile_maps(PE_PROFILE)
    pe_video_idx = pe_ids
    pe_fps = {row["video_id"]: row["fps"] for row in pe_meta}
    video_index = {row["video_id"]: row["video_idx"] for row in pe_meta}

    cap_indices, _, cap_meta = caption_branch(cases, args.cache_dir, args.reuse, args)
    cap_video_of = [row["video_id"] for row in cap_meta]
    cap_video_idx = np.asarray([video_index[video] for video in cap_video_of])
    cap_starts = np.asarray([row["window_start_sec"] for row in cap_meta])
    cap_ends = np.asarray([row["window_end_sec"] for row in cap_meta])

    pe_indices, pe_scores = pe_branch(cases, args.cache_dir, args.reuse, args)

    moment = {}

    def caption_moment(rows):
        return [first_match(row, case, cap_video_of, pe_fps, cap_starts, cap_ends,
                            args.tolerance_seconds, True)
                for row, case in zip(rows, cases)]

    def caption_video(rows):
        return [first_match(row, case, cap_video_of, pe_fps, cap_starts, cap_ends,
                            args.tolerance_seconds, False)
                for row, case in zip(rows, cases)]

    def pe_moment(rows):
        return [first_hit(row, case, pe_ids, pe_frames, pe_meta, args.tolerance_seconds)
                for row, case in zip(rows, cases)]

    def pe_video(rows):
        return [first_video_hit(row, target_video_indices(case, video_index), pe_video_idx)
                for row, case in zip(rows, cases)]

    moment["pe_core"] = pe_moment(pe_indices)
    moment["caption_qwen3_4b"] = caption_moment(cap_indices)
    moment["caption_qwen3_4b_video"] = caption_video(cap_indices)
    moment["pe_core_video"] = pe_video(pe_indices)

    pe_video_ints = [np.asarray(video_order(row, pe_video_idx), dtype=np.int64)
                     for row in pe_indices]
    cap_video_ints = [np.asarray(video_order(row, cap_video_idx), dtype=np.int64)
                      for row in cap_indices]

    def fused_video_hit(fused_rows):
        hits = []
        for row, case in zip(fused_rows, cases):
            target = target_video_indices(case, video_index)
            hits.append(next((rank for rank, item in enumerate(row, 1) if item in target), None))
        return hits

    moment["rrf_video_equal"] = fused_video_hit(rrf_equal([pe_video_ints, cap_video_ints], args.k))
    for weight in (0.25, 0.5, 0.75):
        moment[f"rrf_video_{weight:.2f}"] = fused_video_hit(
            rrf(pe_video_ints, cap_video_ints, weight, args.k))

    cap_video_orders = cap_video_ints
    for depth in (1, 3, 5, 10, 20):
        moment[f"pe_filter_cap_top{depth}"] = [
            first_hit(filter_by_videos(pe_row, set(cap_order[:depth]), pe_video_idx), case,
                      pe_ids, pe_frames, pe_meta, args.tolerance_seconds)
            for pe_row, cap_order, case in zip(pe_indices, cap_video_orders, cases)]

    for weight in (1.0, 4.0):
        boosted = []
        for pe_row, pe_score, cap_order, case in zip(pe_indices, pe_scores, cap_video_orders, cases):
            bonus = {video: weight / (60 + rank) for rank, video in enumerate(cap_order[:10], 1)}
            boosted.append(first_hit(boosted_order(pe_row, pe_score, pe_video_idx, bonus), case,
                                     pe_ids, pe_frames, pe_meta, args.tolerance_seconds))
        moment[f"pe_boost_cap_top10_w{weight:g}"] = boosted

    moment["pe_or_caption"] = [1 if (a is not None or b is not None) else None
                               for a, b in zip(moment["pe_core"], moment["caption_qwen3_4b"])]

    validation = {}
    caption_eval_path = Path(args.caption_dir) / "eval_qwen3-4b-fp16.json"
    if caption_eval_path.exists():
        kaggle = json.loads(caption_eval_path.read_text())
        validate("caption_window_vs_kaggle", dict(zip(case_ids, moment["caption_qwen3_4b"])),
                 kaggle["ranks"], validation)
        validate("caption_video_vs_kaggle", dict(zip(case_ids, moment["caption_qwen3_4b_video"])),
                 kaggle["video_ranks"], validation)
    if args.pe_baseline.exists():
        baseline = json.loads(args.pe_baseline.read_text())
        validate("pe_core_vs_benchmark", dict(zip(case_ids, moment["pe_core"])),
                 dict(zip([case["id"] for case in baseline["cases"]],
                          baseline["methods"]["pe_core"]["ranks"])), validation)

    complementarity = {}
    for round_name in ALL_ROUNDS + ["Round 1-3 - 2026 (aggregate)"]:
        if round_name.endswith("(aggregate)"):
            selected = [case for case in cases if round_of(case["id"]) in REPORTED_ROUNDS]
        else:
            selected = [case for case in cases if round_of(case["id"]) == round_name]
        positions = [case_ids.index(case["id"]) for case in selected]
        pe_hits = [moment["pe_core"][position] is not None for position in positions]
        cap_hits = [moment["caption_qwen3_4b"][position] is not None for position in positions]
        complementarity[round_name] = {
            "cases": len(selected),
            "both": sum(a and b for a, b in zip(pe_hits, cap_hits)),
            "only_pe": sum(a and not b for a, b in zip(pe_hits, cap_hits)),
            "only_caption": sum(b and not a for a, b in zip(pe_hits, cap_hits)),
            "neither": sum(not a and not b for a, b in zip(pe_hits, cap_hits)),
            "union_hits": sum(a or b for a, b in zip(pe_hits, cap_hits)),
            "union_recall": round(sum(a or b for a, b in zip(pe_hits, cap_hits)) / len(selected), 4),
        }

    report = {
        "source": f"{args.queries}: {', '.join(ALL_ROUNDS)}",
        "metric": ("moment: same video and within temporal tolerance of the target moment; "
                   "caption moment uses window +/- tolerance, caption/PE video use first target video"),
        "tolerance_seconds": args.tolerance_seconds, "k": args.k,
        "caption": {"model": args.caption_model, "instruction": INSTRUCTION,
                    "query_prompt": QUERY_PROMPT, "documents": len(cap_meta),
                    "dir": str(args.caption_dir)},
        "cases": cases,
        "validation": validation,
        "complementarity": complementarity,
        "methods": {}, "by_round": {},
    }
    for name, found in moment.items():
        report["methods"][name] = {**metrics(found), "ranks": found}
        report["by_round"][name] = by_round_report(found, cases)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    log(f"wrote {args.output}")

    for name in moment:
        aggregate = report["by_round"][name]["Round 1-3 - 2026 (aggregate)"]
        log(f"{name:26s} R1-3: mrr={aggregate['mrr']:.4f} r@1={aggregate['recall_at_1']:.3f} "
            f"r@5={aggregate['recall_at_5']:.3f} r@100={aggregate['recall_at_100']:.3f} "
            f"hits={aggregate['hits']}/{aggregate['cases']}")


def self_test():
    item_videos = [0, 0, 1, 2, 1]
    assert video_order([0, 1, 2, 3, 4], item_videos) == [0, 1, 2]
    assert first_video_hit([0, 1, 2], {1}, item_videos) == 3
    assert first_video_hit([0, 1], {2}, item_videos) is None
    assert filter_by_videos([0, 2, 4], {1}, item_videos) == [2, 4]
    order = boosted_order(np.array([0, 1, 2]), np.array([0.9, 0.8, 0.1]), item_videos,
                          {1: 1.0})
    assert order == [2, 0, 1]
    assert metrics([1, 10, None])["recall_at_10"] == 2 / 3
    assert rrf_equal([np.array([[1, 2]]), np.array([[2, 3]])], 3)[0].tolist() == [2, 1, 3]
    print("self-test OK")


if __name__ == "__main__":
    main()
