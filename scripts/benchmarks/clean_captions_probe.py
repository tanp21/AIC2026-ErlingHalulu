#!/usr/bin/env python3
"""Probe: can cleaning CapRL captions with DeepSeek make "look-alike" videos distinguishable?

The original captions are never modified — cleaned copies live in a separate artifact and all
scoring happens against the same 40-case / 3,345-window pool as the chunking probe. Three
variants per window:

    stripped     remove boilerplate (logos, timecodes, program titles, camera talk)
    distinctive  short rewrite keeping only specific people/objects/actions/on-screen text
    contrastive  rewrite that emphasises what separates this caption from its 5 nearest
                 neighbours from *other videos* (kNN over the existing 4B embeddings, no
                 query or ground truth involved)

Stages:
    neighbors  kNN (different-video) neighbours for every pool window
    clean      DeepSeek generation of the three variants (resumable)
    score      similarity-before/after + Qwen3-0.6B retrieval comparison on the pool
"""
import argparse
import gc
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.bench_caption_retrieval import (CAPTIONS_PARQUET, CAPTION_DIR, CACHE_DIR,
                                                        QUERIES, QUERY_PROMPT, TOLERANCE_SECONDS,
                                                        caption_documents, first_hit_rank,
                                                        rank_summary, target_windows)
from scripts.benchmarks.bench_visual_embeddings import PE_PROFILE, load_cases, profile_maps
from scripts.benchmarks.probe_caption_chunking import (PROBE_DIR, QWEN_SMALL, SmallEncoder,
                                                       split_sentences, window_topk_mean)

NEIGHBORS_JSON = PROBE_DIR / "caption_neighbors.json"
CLEANED_JSON = PROBE_DIR / "cleaned_captions_probe.json"
CLEANED_LONG_JSON = PROBE_DIR / "cleaned_captions_long.json"
SCORE_JSON = PROBE_DIR / "cleaning_probe.json"
SCORE_4B_JSON = PROBE_DIR / "cleaning_probe_4b.json"
SCORE_PE_JSON = PROBE_DIR / "cleaning_probe_pe.json"
QWEN_4B = "Qwen/Qwen3-Embedding-4B"
POOL_JSON = PROBE_DIR / "pool.json"
POOL = 40  # probe scope note (30 bad + 10 control)
VARIANTS = ("stripped", "distinctive", "contrastive")

CLEAN_SYSTEM = """You rewrite one long, dense video-shot caption into two cleaner versions for text-to-video retrieval.

1) "stripped": same caption with broadcast boilerplate removed. Delete channel logos, timecodes, program titles, "the video opens with", camera/editing commentary, and repeated scene-setting sentences that would appear in most shots of the same show. Keep every remaining fact verbatim in meaning and order. Do not invent.

2) "distinctive": a compact rewrite (at most 60 words) that keeps only details that identify THIS exact moment: specific people, clothing, colors, counts, positions, objects, actions, and on-screen text (quote it). Drop generic studio/kitchen/city/scene descriptions, logos and timecodes. Never invent or guess.

Return JSON only: {"items": [{"id": <id>, "stripped": "...", "distinctive": "..."}]} with one item per input id."""

CONTRAST_SYSTEM = """You rewrite one video-shot caption so it becomes easy to tell apart from similar captions of OTHER videos.

You receive: the caption, plus short excerpts of the 5 most similar captions from other videos.
Rewrite the caption into at most 60 words that keeps the facts but emphasises what is SPECIFIC to this caption: unique people, clothing, colors, counts, positions, objects, actions, and on-screen text (quote it). Remove wording that is shared with the similar captions (generic setting, program names, standard studio descriptions, logos, timecodes).
Never invent facts that are not in the caption.

Return JSON only: {"items": [{"id": <id>, "contrastive": "..."}]} with one item per input id."""

CONTRAST_LONG_SYSTEM = """You rewrite one video-shot caption so it becomes easy to tell apart from similar captions of OTHER videos, WITHOUT losing facts.

You receive: the caption, plus short excerpts of the 5 most similar captions from other videos.
Rewrite the caption in at most {words} words that:
- keeps every fact, especially actions and their order/timing, camera movement that changes what is visible, and all specific visual details (people, clothing, colors, counts, positions, objects, quoted on-screen text);
- prioritises the most distinctive details first; if the caption is too long, compress or drop only the generic wording shared with the similar captions (standard setting, program names, logos, timecodes, repeated commentary), never the actions or specifics;
- never invents facts.

Return JSON only: {"items": [{"id": <id>, "contrastive_long": "..."}]} with one item per input id."""


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def truncate(text, limit=600):
    text = text.strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def valid_text(value):
    return isinstance(value, str) and value.strip()


def parse_items(payload, requested, keys):
    items = (payload or {}).get("items") or []
    found = {}
    for item in items:
        try:
            index = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if index in requested and all(valid_text(item.get(key)) for key in keys):
            found[index] = {key: item[key].strip() for key in keys}
    return found


def pool_windows():
    pool = json.loads(POOL_JSON.read_text())
    return {int(index): row["caption"] for index, row in pool["windows"].items()}, pool


def neighbors_stage(args):
    _, pool = pool_windows()
    indices = np.asarray(sorted(int(index) for index in pool["windows"]))
    docs, meta = caption_documents(CAPTION_DIR)
    video_of = np.asarray([row["video_id"] for row in meta])
    neighbors = {}
    chunk = args.knn_chunk
    for start in range(0, len(indices), chunk):
        rows = indices[start:start + chunk]
        similarity = docs[rows] @ docs.T
        same_video = video_of[rows][:, None] == video_of[None, :]
        similarity[same_video] = -2.0
        top = np.argpartition(-similarity, args.neighbors, axis=1)[:, :args.neighbors]
        for position, row in enumerate(rows):
            order = top[position][np.argsort(-similarity[position, top[position]])]
            neighbors[str(row)] = [int(index) for index in order]
        log(f"kNN {min(start + chunk, len(indices))}/{len(indices)}")
    del docs
    NEIGHBORS_JSON.write_text(json.dumps(neighbors, indent=1) + "\n")
    log(f"wrote {NEIGHBORS_JSON}")


def clean_stage(args, long=False):
    from agent.config import read_dotenv
    from agent.llm import DeepSeekClient, LLMError

    import os

    dotenv = read_dotenv(ROOT / ".env")
    key = (os.environ.get("DEEPSEEK_API_KEY") or dotenv.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    model = args.model or os.environ.get("DEEPSEEK_MODEL") or dotenv.get(
        "DEEPSEEK_MODEL") or "deepseek-flash"
    client = DeepSeekClient(key, model=model, thinking=False)

    captions, _ = pool_windows()
    all_captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    neighbors = json.loads(NEIGHBORS_JSON.read_text())
    output_path = CLEANED_LONG_JSON if long else CLEANED_JSON
    output = json.loads(output_path.read_text()) if output_path.exists() else {}
    if long:
        keys = ("contrastive_long",)
        pending = [index for index in sorted(captions)
                   if not valid_text(output.get(str(index), {}).get(keys[0]))]
    else:
        keys = VARIANTS
        pending = [index for index in sorted(captions)
                   if not all(valid_text(output.get(str(index), {}).get(key)) for key in keys)]
    if args.limit:
        pending = pending[:args.limit]
    log(f"{output_path.name}: {len(captions)} pool windows, {len(output)} done, "
        f"{len(pending)} pending ({model})")

    lock = threading.Lock()
    batches = [pending[start:start + args.batch_size]
               for start in range(0, len(pending), args.batch_size)]

    def run_batch(batch, system, keys, build, max_tokens):
        requested = set(batch)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": build(batch)}]
        parsed = {}
        for attempt in (1, 2):
            try:
                parsed = parse_items(client.chat_json(messages, max_tokens=max_tokens),
                                     requested, keys)
            except LLMError as error:
                log(f"batch {batch[0]}..{batch[-1]} attempt {attempt}: {error}")
                parsed = {}
            missing = requested - set(parsed)
            if not missing:
                break
            log(f"batch {batch[0]}..{batch[-1]} attempt {attempt}: {len(missing)} invalid")
            messages[1]["content"] = build(sorted(missing))
        with lock:
            for index in batch:
                row = output.setdefault(str(index), {"caption": captions[index]})
                row.update(parsed.get(index, {}))
                fallback = set(row.get("fallback", []))
                for key in keys:
                    if valid_text(row.get(key)):
                        fallback.discard(key)
                    else:
                        fallback.add(key)
                if fallback:
                    row["fallback"] = sorted(fallback)
                else:
                    row.pop("fallback", None)
            output_path.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
        log(f"batch {batch[0]}..{batch[-1]} done")

    def build_a(batch):
        return json.dumps([{"id": index, "caption": captions[index]} for index in batch],
                          ensure_ascii=False)

    def build_b(batch):
        return json.dumps([{
            "id": index,
            "caption": captions[index],
            "similar_other_videos": [truncate(all_captions[n]) for n in neighbors[str(index)]],
        } for index in batch], ensure_ascii=False)

    def safe(batch, system, keys, build, max_tokens):
        try:
            run_batch(batch, system, keys, build, max_tokens)
        except Exception as error:  # one bad batch must not kill the run
            log(f"batch {batch[0]}..{batch[-1]} failed: {type(error).__name__}: {error}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        if long:
            system = CONTRAST_LONG_SYSTEM.replace("{words}", str(args.words))
            list(executor.map(lambda batch: safe(batch, system,
                                                 ("contrastive_long",), build_b, 8192), batches))
        else:
            list(executor.map(lambda batch: safe(batch, CLEAN_SYSTEM,
                                                 ("stripped", "distinctive"), build_a, 8192), batches))
            list(executor.map(lambda batch: safe(batch, CONTRAST_SYSTEM,
                                                 ("contrastive",), build_b, 4096), batches))

    counts = {key: sum(1 for row in output.values() if valid_text(row.get(key))) for key in keys}
    log(f"done: {len(output)} windows, valid {counts}, usage {client.usage.as_dict()}")


def stats(ranks):
    summary = rank_summary(ranks)
    summary["mrr"] = round(float(np.mean([1.0 / rank for rank in ranks if rank])), 4) \
        if any(ranks) else 0.0
    summary["top25"] = round(sum(rank is not None and rank <= 25 for rank in ranks) / len(ranks), 4)
    summary["top50"] = round(sum(rank is not None and rank <= 50 for rank in ranks) / len(ranks), 4)
    return summary


def pairwise_similarity(vectors, video_of):
    similarity = vectors @ vectors.T
    mask = video_of[:, None] != video_of[None, :]
    upper = np.triu(mask, k=1)
    values = similarity[upper]
    return {"pairs": int(len(values)), "mean": round(float(values.mean()), 4),
            "median": round(float(np.median(values)), 4),
            "over_0.85": round(float((values > 0.85).mean()), 4),
            "over_0.9": round(float((values > 0.9).mean()), 4)}


def score_stage(args):
    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads(POOL_JSON.read_text())
    bad, control = set(pool["bad"]), set(pool["control"])
    windows, _ = pool_windows()
    indices = np.asarray(sorted(windows))
    position_of = {int(index): position for position, index in enumerate(indices)}
    cap_meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    cap_video = np.asarray([row["video_id"] for row in cap_meta])
    _, _, pe_meta = profile_maps(PE_PROFILE)
    fps = {row["video_id"]: row["fps"] for row in pe_meta}
    query_vectors = np.load(PROBE_DIR / "query_vectors_06b.npy")

    cleaned = json.loads(CLEANED_JSON.read_text())
    texts = {"original": np.asarray([windows[int(i)] for i in indices], dtype=object),
             "first_sentence": np.asarray([split_sentences(windows[int(i)])[0] for i in indices],
                                          dtype=object)}
    for variant in VARIANTS:
        texts[variant] = np.asarray([cleaned[str(i)].get(variant) or windows[int(i)]
                                     for i in indices], dtype=object)
    if CLEANED_LONG_JSON.exists():
        long_cleaned = json.loads(CLEANED_LONG_JSON.read_text())
        texts["contrastive_long"] = np.asarray(
            [long_cleaned[str(i)].get("contrastive_long") or windows[int(i)] for i in indices],
            dtype=object)

    encoder = SmallEncoder(QWEN_SMALL, args.dtype, args.device)
    vectors, similarity, ranks = {}, {}, {}
    for name, items in texts.items():
        cache = PROBE_DIR / f"cleaned_vectors_{name}.npy"
        if cache.exists():
            vectors[name] = np.load(cache)
        else:
            log(f"encoding {len(items)} {name} captions")
            vectors[name] = encoder.encode(list(items))
            np.save(cache, vectors[name])
        video_of = cap_video[indices]
        similarity[name] = pairwise_similarity(vectors[name], video_of)
        log(f"{name}: {similarity[name]}")

    combined = vectors["original"] + vectors["distinctive"]
    combined /= np.linalg.norm(combined, axis=1, keepdims=True) + 1e-9
    vectors["original+distinctive"] = combined
    if "contrastive_long" in vectors:
        mixed = vectors["original"] + vectors["contrastive_long"]
        mixed /= np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-9
        vectors["original+contrastive_long"] = mixed

    ranks = rank_pool(cases, case_ids, pool, position_of, cap_video, query_vectors, vectors)
    report = {"similarity": similarity, "groups": {"bad": sorted(bad), "control": sorted(control)},
              "metrics": metrics_report(ranks, bad, control), "per_case": ranks}
    SCORE_JSON.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {SCORE_JSON}")
    for name in vectors:
        log(f"{name}: bad_oracle {report['metrics'][name]['bad_oracle']}")
        log(f"{name}: control_realistic {report['metrics'][name]['control_realistic']}")


def rank_matrix(cases, case_ids, pool, position_of, cap_video, matrices):
    ranks = {}
    for name, matrix in matrices.items():
        per_case = {}
        for case in cases:
            entries = pool["cases"].get(case["id"])
            if entries is None:
                continue
            case_ranks = {}
            scores = matrix[:, case_ids.index(case["id"])]
            for pool_name in ("realistic", "oracle"):
                index_list = entries[pool_name]
                positions = np.asarray([position_of[int(i)] for i in index_list])
                order = np.argsort(-scores[positions])
                target_set = set(entries["targets"])
                case_ranks[f"{pool_name}"] = next(
                    (rank for rank, slot in enumerate(order, 1) if index_list[slot] in target_set),
                    None)
            # video-level: best window score per video inside the oracle pool
            target_videos = set(entries["target_videos"])
            best_by_video = {}
            for i in entries["oracle"]:
                video = cap_video[int(i)]
                best_by_video[video] = max(best_by_video.get(video, -9.0),
                                           float(scores[position_of[int(i)]]))
            video_order = sorted(best_by_video, key=lambda video: -best_by_video[video])
            case_ranks["video"] = next((rank for rank, video in enumerate(video_order, 1)
                                        if video in target_videos), None)
            per_case[case["id"]] = case_ranks
        ranks[name] = per_case
    return ranks


def rank_pool(cases, case_ids, pool, position_of, cap_video, query_vectors, vectors):
    matrices = {name: (matrix @ query_vectors.T).astype(np.float32)
                for name, matrix in vectors.items()}
    return rank_matrix(cases, case_ids, pool, position_of, cap_video, matrices)


def metrics_report(ranks, bad, control):
    report = {}
    for name in ranks:
        report[name] = {}
        for group, ids in (("bad", bad), ("control", control), ("all", bad | control)):
            for pool_name in ("realistic", "oracle", "video"):
                values = [ranks[name][case_id][pool_name] for case_id in sorted(ids)]
                report[name][f"{group}_{pool_name}"] = stats(values)
    return report


def score_4b_stage(args):
    """Confirmation run: original (cached corpus vectors) vs 60-word and 150-word
    contrastive-clean captions, all embedded with Qwen3-Embedding-4B."""
    from scripts.benchmarks.bench_caption_retrieval import CACHE_DIR as BENCH_CACHE, caption_documents
    from scripts.benchmarks.probe_caption_chunking import SmallEncoder

    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads(POOL_JSON.read_text())
    bad, control = set(pool["bad"]), set(pool["control"])
    windows, _ = pool_windows()
    indices = np.asarray(sorted(windows))
    position_of = {int(index): position for position, index in enumerate(indices)}
    cap_meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    cap_video = np.asarray([row["video_id"] for row in cap_meta])
    _, _, pe_meta = profile_maps(PE_PROFILE)
    fps = {row["video_id"]: row["fps"] for row in pe_meta}
    query_vectors = np.load(BENCH_CACHE / "caption_query_vectors.npy").astype(np.float32)

    docs, _ = caption_documents(CAPTION_DIR)
    vectors = {"original": docs[indices]}
    del docs
    gc.collect()

    cleaned = json.loads(CLEANED_JSON.read_text())
    long_cleaned = json.loads(CLEANED_LONG_JSON.read_text())
    texts = {
        "contrastive": [cleaned[str(i)].get("contrastive") or windows[int(i)] for i in indices],
        "contrastive_long": [long_cleaned[str(i)].get("contrastive_long") or windows[int(i)]
                             for i in indices],
    }
    local_path = PROBE_DIR / "cleaned_local_probe.json"
    if local_path.exists():
        local = json.loads(local_path.read_text())
        texts["contrastive_local"] = [local[str(i)].get("cleaned") or windows[int(i)]
                                      for i in indices]
    diff_path = PROBE_DIR / "cleaned_diff_probe.json"
    if diff_path.exists():
        diff = json.loads(diff_path.read_text())
        texts["contrastive_diff"] = [diff[str(i)].get("cleaned") or windows[int(i)]
                                     for i in indices]
    rewrite_path = PROBE_DIR / "cleaned_local_rewrite_probe.json"
    if rewrite_path.exists():
        rewrite = json.loads(rewrite_path.read_text())
        texts["contrastive_rewrite_local"] = [rewrite[str(i)].get("rewritten") or windows[int(i)]
                                              for i in indices]
    rule_path = PROBE_DIR / "cleaned_rule_probe.json"
    if rule_path.exists():
        rule = json.loads(rule_path.read_text())
        texts["contrastive_rule"] = [rule[str(i)].get("cleaned") or windows[int(i)]
                                     for i in indices]
    # Extra variants from the command line, as "name=path" (or just "path" -> filename stem).
    for spec in args.variants_json or []:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(name).stem, name
        raw = json.loads(Path(path).read_text())
        items, missing = [], 0
        for i in indices:
            row = raw.get(str(i)) or {}
            value = row.get("rewritten") or row.get("cleaned")
            if isinstance(value, str) and value.strip():
                items.append(value)
            else:
                items.append(windows[int(i)])
                missing += 1
        coverage = 1 - missing / len(indices)
        if coverage < args.min_coverage:
            raise SystemExit(
                f"variant {name!r}: only {coverage:.1%} of {len(indices)} pool windows carry a "
                f"rewrite (need {args.min_coverage:.1%}). Refusing to score: the missing windows "
                f"would silently fall back to the ORIGINAL caption and produce a plausible-looking "
                f"but meaningless table.")
        log(f"variant {name}: coverage {coverage:.1%}")
        texts[name] = items
    encoder = SmallEncoder(QWEN_4B, args.dtype, args.device)
    # The original is scored explicitly on THIS encoder. `cleaning_probe_4b.json` never computed
    # it, which is how the 4B original baseline previously got spliced in from the 0.6B report.
    similarity = {"original": pairwise_similarity(vectors["original"], cap_video[indices])}
    log(f"original: {similarity['original']}")
    for name, items in texts.items():
        cache = PROBE_DIR / f"{args.vector_prefix}{name}{args.cache_suffix}.npy"
        if cache.exists():
            vectors[name] = np.load(cache)
        else:
            log(f"encoding {len(items)} {name} captions with {QWEN_4B}")
            vectors[name] = encoder.encode(list(items))
            np.save(cache, vectors[name])
        similarity[name] = pairwise_similarity(vectors[name], cap_video[indices])
        log(f"{name}: {similarity[name]}")
    mixed = vectors["original"] + vectors["contrastive_long"]
    mixed /= np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-9
    vectors["original+contrastive_long"] = mixed

    ranks = rank_pool(cases, case_ids, pool, position_of, cap_video, query_vectors, vectors)
    report = {"model": QWEN_4B, "similarity": similarity,
              "groups": {"bad": sorted(bad), "control": sorted(control)},
              "metrics": metrics_report(ranks, bad, control), "per_case": ranks}
    out_path = Path(args.score_out) if args.score_out else SCORE_4B_JSON
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {out_path}")
    for name in vectors:
        log(f"{name}: bad_oracle {report['metrics'][name]['bad_oracle']}")
        log(f"{name}: control_realistic {report['metrics'][name]['control_realistic']}")


def score_pe_stage(args):
    """PE-Core scored on the same 3,345-window pool: each window takes the best PE
    keyframe that falls inside it, so the comparison with the caption channel is
    apples-to-apples inside one candidate pool."""
    import faiss

    from scripts.benchmarks.bench_caption_retrieval import CACHE_DIR as BENCH_CACHE, PE_PROFILE

    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads(POOL_JSON.read_text())
    bad, control = set(pool["bad"]), set(pool["control"])
    windows, _ = pool_windows()
    indices = np.asarray(sorted(windows))
    position_of = {int(index): position for position, index in enumerate(indices)}
    cap_meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    cap_video = np.asarray([row["video_id"] for row in cap_meta])
    cap_start = np.asarray([row["window_start_sec"] for row in cap_meta])
    cap_end = np.asarray([row["window_end_sec"] for row in cap_meta])
    pe_meta = pq.read_table(PE_PROFILE / "video_meta.parquet").to_pylist()
    pe_video = np.load(PE_PROFILE / "id_video.npy")
    pe_time = np.load(PE_PROFILE / "id_time.npy")
    query_vectors = np.load(BENCH_CACHE / "pe_query_vectors.npy").astype(np.float32)

    frames_by_video = {}
    for row in pe_meta:
        mask = pe_video == row["video_idx"]
        order = np.argsort(pe_time[mask])
        frames_by_video[row["video_id"]] = (pe_time[mask][order], np.flatnonzero(mask)[order])
    window_frames = []
    for index in indices:
        times, frames = frames_by_video[cap_video[int(index)]]
        low, high = np.searchsorted(times, cap_start[index], "left"), \
            np.searchsorted(times, cap_end[index], "right")
        window_frames.append(frames[low:high])
    needed = np.unique(np.concatenate([frames for frames in window_frames if len(frames)]))
    log(f"{len(needed)} PE keyframes fall inside the {len(indices)} pool windows")

    index = faiss.read_index(str(PE_PROFILE / "flat.ip"))
    all_vectors = index.reconstruct_n(0, index.ntotal).astype(np.float32)
    frame_vectors = all_vectors[needed]
    del all_vectors
    gc.collect()
    frame_scores = frame_vectors @ query_vectors.T
    position_of_frame = {int(frame): position for position, frame in enumerate(needed)}
    scores = np.full((len(indices), len(cases)), -9.0, dtype=np.float32)
    for window, frames in enumerate(window_frames):
        if len(frames):
            rows = [position_of_frame[int(frame)] for frame in frames]
            scores[window] = frame_scores[rows].max(axis=0)
    np.save(PROBE_DIR / "pool_pe_scores.npy", scores)

    ranks = rank_matrix(cases, case_ids, pool, position_of, cap_video, {"pe_core": scores})
    report = {"model": "PE-Core-bigG-14-448", "groups": {"bad": sorted(bad), "control": sorted(control)},
              "metrics": metrics_report(ranks, bad, control), "per_case": ranks}
    SCORE_PE_JSON.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {SCORE_PE_JSON}")
    log(f"pe_core: bad_oracle {report['metrics']['pe_core']['bad_oracle']}")
    log(f"pe_core: control_realistic {report['metrics']['pe_core']['control_realistic']}")
    log(f"pe_core: bad_video {report['metrics']['pe_core']['bad_video']}")


def self_test():
    assert truncate("a " * 400, 10).endswith("...")
    assert valid_text("x") and not valid_text("  ")
    assert parse_items({"items": [{"id": "3", "stripped": "a", "distinctive": "b"},
                                  {"id": 4, "stripped": "", "distinctive": "b"}]},
                       {3, 4}, ("stripped", "distinctive")) == {3: {"stripped": "a", "distinctive": "b"}}
    vectors = np.array([[1, 0], [0.9, 0.1], [0.8, 0.2]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    result = pairwise_similarity(vectors, np.array(["a", "a", "b"]))
    assert result["pairs"] == 2 and result["mean"] > 0.9
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",
                        choices=("neighbors", "clean", "clean-long", "score", "score-4b",
                                 "score-pe", "all"),
                        default="all")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--knn-chunk", type=int, default=256)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--words", type=int, default=150)
    parser.add_argument("--variants-json", nargs="+", default=None,
                        help="extra variants to score, as name=path (or path -> filename stem)")
    parser.add_argument("--score-out", default=None,
                        help="report path; defaults to cleaning_probe_4b.json")
    parser.add_argument("--vector-prefix", default="cleaned_vectors_4b_")
    parser.add_argument("--cache-suffix", default="",
                        help="appended to the vector cache name, to avoid stale-cache reuse")
    parser.add_argument("--min-coverage", type=float, default=0.99,
                        help="refuse to score a variant whose rewrite coverage is below this")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.stage in ("neighbors", "all"):
        neighbors_stage(args)
    if args.stage in ("clean", "all"):
        clean_stage(args)
    if args.stage in ("clean-long", "all"):
        clean_stage(args, long=True)
    if args.stage in ("score", "all"):
        score_stage(args)
    if args.stage in ("score-4b", "all"):
        score_4b_stage(args)
    if args.stage in ("score-pe", "all"):
        score_pe_stage(args)


if __name__ == "__main__":
    main()
