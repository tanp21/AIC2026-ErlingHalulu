#!/usr/bin/env python3
"""Staged, anti-overfit tuning + benchmark for PE-Core query expansion (holistic
caption + aspect queries, fused at shot level). See docs/reports/
visual-query-expansion-benchmark-2026.md's plan for the scoring definition,
invariants and staged-tuning protocol this implements.

Dev split = Test + Round 1 + Round 2 (95 cases). Holdout = Round 3 (42 cases),
evaluated once, only with --run-holdout after the config is frozen.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import frame_maps  # noqa: E402
from backend import shots as shots_mod  # noqa: E402
from backend.subquery_fusion import Adjacency, Hit, fuse  # noqa: E402
from scripts.benchmarks.bench_visual_embeddings import (  # noqa: E402
    PE_PROFILE, REPORTED_ROUNDS, encode_pe, metrics, round_of,
)

DEV_ROUNDS = {"Test", "Round 1 - 2026", "Round 2 - 2026"}
HOLDOUT_ROUND = "Round 3 - 2026"
BUCKET_SECONDS = 5.0
TOLERANCE_SECONDS = 10.0

VECTOR_CACHE = ROOT / "artifacts/query_expansion/vector_cache.npz"
RETRIEVAL_CACHE = ROOT / "artifacts/query_expansion/retrieval_cache.npz"
TUNING_LOG = ROOT / "artifacts/query_expansion/tuning_log.jsonl"
CONFIG_PATH = ROOT / "benchmarks/visual/query_expansion_config.json"
BENCHMARK_OUT = ROOT / "benchmarks/visual/query_expansion_benchmark_2026.json"
STORED_EN_BENCHMARK = ROOT / "benchmarks/visual/visual_embedding_benchmark_2026_en.json"

RETRIEVAL_K = 2000


def text_hash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data loading: l2026 profile, shot boundaries, group-id assignment.
# ---------------------------------------------------------------------------

def load_profile():
    id_video = np.load(PE_PROFILE / "id_video.npy")
    id_frame = np.load(PE_PROFILE / "id_frame.npy")
    id_time = np.load(PE_PROFILE / "id_time.npy")
    video_meta = pq.read_table(PE_PROFILE / "video_meta.parquet").to_pylist()
    index = faiss.read_index(str(PE_PROFILE / "flat.ip"))
    meta_by_idx = {row["video_idx"]: row for row in video_meta}
    return id_video, id_frame, id_time, meta_by_idx, index


def load_shot_indices(meta_by_idx):
    """video_idx -> clamped ShotIndex, or None if the video has no shot boundaries."""
    result = {}
    for video_idx, row in meta_by_idx.items():
        path = shots_mod.shots_path(row)
        if not path.exists():
            result[video_idx] = None
            continue
        index = shots_mod.load_path(path)
        fm_path = frame_maps.MAP_DIR / f"{row['video_id']}.npz"
        fm = frame_maps.load_path(fm_path)
        result[video_idx] = index.clamped(len(fm.pts))
    return result


class GroupResolver:
    """row (global l2026 FAISS id) -> group_id, plus an Adjacency for neighbor evidence."""

    def __init__(self, id_video, id_frame, id_time, shot_indices, bucket_seconds=BUCKET_SECONDS):
        self.id_video = id_video
        self.id_frame = id_frame
        self.id_time = id_time
        self.shot_indices = shot_indices
        self.shot_counts = {v: len(idx) for v, idx in shot_indices.items() if idx is not None}
        self.bucket_seconds = bucket_seconds

    def group_of(self, row):
        video_idx = int(self.id_video[row])
        frame_idx = int(self.id_frame[row])
        shot_index = self.shot_indices.get(video_idx)
        if shot_index is not None:
            shot_idx = shot_index.shot_at(frame_idx)
            if shot_idx is not None:
                return (video_idx, shot_idx)
        bucket = int(self.id_time[row] // self.bucket_seconds)
        return (video_idx, "t", bucket)

    def _neighbor(self, group_id, delta):
        if group_id[1] == "t":
            video_idx, _, bucket = group_id
            return (video_idx, "t", bucket + delta)
        video_idx, shot_idx = group_id
        n_shots = self.shot_counts.get(video_idx, 0)
        new_idx = shot_idx + delta
        return (video_idx, new_idx) if 0 <= new_idx < n_shots else None

    def adjacency(self):
        class _View(dict):
            def __init__(self, resolver, delta):
                super().__init__()
                self._resolver, self._delta = resolver, delta

            def get(self, key, default=None):
                return self._resolver._neighbor(key, self._delta)

        return Adjacency(prev=_View(self, -1), next=_View(self, 1))

    @staticmethod
    def video_of(group_id):
        return group_id[0]


# ---------------------------------------------------------------------------
# Encoding + retrieval caches.
# ---------------------------------------------------------------------------

def load_cases_expansion(expansions_path):
    data = json.loads(Path(expansions_path).read_text())
    return data["cases"], data


def collect_texts(cases):
    """All distinct texts needed: each case's caption_en_approved, holistic, and used aspects."""
    texts = set()
    for case in cases:
        if case.get("caption_en_approved"):
            texts.add(case["caption_en_approved"])
        texts.add(case["holistic"]["text"])
        for aspect in case["aspects"]:
            if not aspect.get("dropped"):
                texts.add(aspect["text"])
    return sorted(texts)


def encode_and_cache(texts):
    if VECTOR_CACHE.exists():
        cached = np.load(VECTOR_CACHE, allow_pickle=True)
        vectors = {h: cached[h] for h in cached.files}
    else:
        vectors = {}
    missing = [t for t in texts if text_hash(t) not in vectors]
    if missing:
        print(f"encoding {len(missing)} new texts with PE-Core...", flush=True)
        encoded = encode_pe(missing)
        for text, vec in zip(missing, encoded):
            vectors[text_hash(text)] = vec
        VECTOR_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez(VECTOR_CACHE, **vectors)
    return {t: vectors[text_hash(t)] for t in texts}


def retrieve_cached(texts, vectors_by_text, index, k=RETRIEVAL_K, cache_path=RETRIEVAL_CACHE):
    """Top-k (rows, scores) per text, cached on disk.

    The cache is merged, never rewritten from scratch, so runs over different text sets
    (query expansion v1/v2, paraphrases) keep each other's entries. Entries written by the
    old rows-only format are re-searched once to add scores.
    """
    cache_path = Path(cache_path)
    rows_by_hash, scores_by_hash = {}, {}
    if cache_path.exists():
        with np.load(cache_path) as cached:
            for key in cached.files:
                if key.endswith("_rows"):
                    rows_by_hash[key[:-len("_rows")]] = cached[key]
                elif key.endswith("_scores"):
                    scores_by_hash[key[:-len("_scores")]] = cached[key]

    unique_texts = list(dict.fromkeys(texts))
    missing = [t for t in unique_texts
               if text_hash(t) not in rows_by_hash or text_hash(t) not in scores_by_hash
               or len(rows_by_hash[text_hash(t)]) < k]
    if missing:
        print(f"retrieving top-{k} for {len(missing)} texts...", flush=True)
        vectors = np.stack([vectors_by_text[t] for t in missing]).astype("float32")
        scores, indexes = index.search(vectors, k)
        for text, score_row, index_row in zip(missing, scores, indexes):
            rows_by_hash[text_hash(text)] = index_row
            scores_by_hash[text_hash(text)] = score_row
        payload = {}
        for h, rows in rows_by_hash.items():
            payload[f"{h}_rows"] = rows
            if h in scores_by_hash:
                payload[f"{h}_scores"] = scores_by_hash[h]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(cache_path.name + ".tmp")
        with open(tmp_path, "wb") as handle:
            np.savez(handle, **payload)
        os.replace(tmp_path, cache_path)

    rows = {t: rows_by_hash[text_hash(t)][:k] for t in unique_texts}
    scores = {t: scores_by_hash[text_hash(t)][:k] for t in unique_texts}
    return rows, scores


def retrieve_and_cache(texts, vectors_by_text, index, k=RETRIEVAL_K):
    return retrieve_cached(texts, vectors_by_text, index, k)[0]


def hits_for(text, rows_by_text, id_time, group_resolver, k):
    rows = rows_by_text[text][:k]
    return [Hit(group_id=group_resolver.group_of(int(r)), payload=int(r),
               time=float(id_time[int(r)])) for r in rows]


# ---------------------------------------------------------------------------
# Scoring.
# ---------------------------------------------------------------------------

def first_hit_fused(fused_groups, case, meta_by_idx, id_video, id_frame, tolerance=TOLERANCE_SECONDS):
    for rank, group in enumerate(fused_groups, 1):
        row = group.rep_hit.payload
        video_idx = int(id_video[row])
        frame_idx = int(id_frame[row])
        row_meta = meta_by_idx[video_idx]
        video_id, fps = row_meta["video_id"], row_meta["fps"]
        for target_video, target_frame in case["targets"]:
            if video_id == target_video and abs(frame_idx - target_frame) / fps <= tolerance:
                return rank
    return None


def first_hit_frames(rows, case, meta_by_idx, id_video, id_frame, tolerance=TOLERANCE_SECONDS):
    for rank, row in enumerate(rows, 1):
        row = int(row)
        video_idx = int(id_video[row])
        frame_idx = int(id_frame[row])
        row_meta = meta_by_idx[video_idx]
        video_id, fps = row_meta["video_id"], row_meta["fps"]
        for target_video, target_frame in case["targets"]:
            if video_id == target_video and abs(frame_idx - target_frame) / fps <= tolerance:
                return rank
    return None


def cases_for_split(cases, split):
    if split == "dev":
        return [c for c in cases if round_of(c["id"]) in DEV_ROUNDS]
    if split == "holdout":
        return [c for c in cases if round_of(c["id"]) == HOLDOUT_ROUND]
    if split == "2026":
        return [c for c in cases if round_of(c["id"]) in REPORTED_ROUNDS]
    raise ValueError(split)


def run_config(cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx, group_resolver,
               *, lam, k=60, K=1000, agg="mean", m=2, gamma=0.0, neighbor_scope="aspects",
               T_h=float("inf"), beta=1.0, dedup_seconds=0.0):
    adjacency = group_resolver.adjacency()
    ranks = {}
    for case in cases:
        holistic_hits = hits_for(case["holistic"]["text"], rows_by_text, id_time, group_resolver, K)
        aspect_texts = [a["text"] for a in case["aspects"] if not a.get("dropped")]
        aspect_hits = [hits_for(t, rows_by_text, id_time, group_resolver, K) for t in aspect_texts]
        fused = fuse(holistic_hits, aspect_hits, adjacency, lam=lam, k=k, agg=agg, m=m,
                    gamma=gamma, neighbor_scope=neighbor_scope, T_h=T_h, beta=beta,
                    dedup_seconds=dedup_seconds, video_of=group_resolver.video_of)
        ranks[case["id"]] = first_hit_fused(fused, case, meta_by_idx, id_video, id_frame)
    return ranks


def run_caption_baseline(cases, rows_by_text, meta_by_idx, id_video, id_frame, text_field, K=1000):
    ranks = {}
    for case in cases:
        rows = rows_by_text[case[text_field]][:K]
        ranks[case["id"]] = first_hit_frames(rows, case, meta_by_idx, id_video, id_frame)
    return ranks


def run_caption_shotdedup_baseline(cases, rows_by_text, id_time, meta_by_idx, id_video, id_frame,
                                   group_resolver, text_field, K=1000):
    ranks = {}
    for case in cases:
        hits = hits_for(case[text_field], rows_by_text, id_time, group_resolver, K)
        # Keep the first (best-ranked) hit per group, preserving rank order.
        deduped_rows = []
        seen = set()
        for hit in hits:
            if hit.group_id in seen:
                continue
            seen.add(hit.group_id)
            deduped_rows.append(hit.payload)
        ranks[case["id"]] = first_hit_frames(deduped_rows, case, meta_by_idx, id_video, id_frame)
    return ranks


def round_breakdown(ranks_by_id, cases, rounds):
    by_round = {}
    for round_name in rounds:
        round_ranks = [ranks_by_id[c["id"]] for c in cases if round_of(c["id"]) == round_name]
        by_round[round_name] = metrics(round_ranks)
    pooled = [ranks_by_id[c["id"]] for c in cases if round_of(c["id"]) in rounds]
    by_round["pooled"] = metrics(pooled)
    return by_round


def dev_score(ranks_by_id, cases):
    """Mean rank across the 3 dev rounds (leave-one-round-out style): lower mean(MRR desc,
    R@1 desc) rank across rounds is better. Returns (sort_key, breakdown)."""
    breakdown = round_breakdown(ranks_by_id, cases, DEV_ROUNDS)
    return breakdown


def log_tuning(entry):
    TUNING_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TUNING_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def print_table(name, breakdown, rounds):
    print(f"\n=== {name} ===")
    for round_name in [*rounds, "pooled"]:
        m = breakdown[round_name]
        print(f"  {round_name:28s} MRR={m['mrr']:.3f} R@1={m['recall_at_1']:.3f} "
              f"R@5={m['recall_at_5']:.3f} R@10={m['recall_at_10']:.3f} "
              f"R@100={m['recall_at_100']:.3f} hits={m['hits']}/{m['cases']}")


# ---------------------------------------------------------------------------
# Stages.
# ---------------------------------------------------------------------------

def stage0_sanity(dev_cases, rows_by_text, meta_by_idx, id_video, id_frame):
    if not STORED_EN_BENCHMARK.exists():
        print("WARNING: stored PE-Core benchmark not found, skipping exact-rank sanity check")
        return
    stored = json.loads(STORED_EN_BENCHMARK.read_text())
    stored_cases = stored["cases"]
    stored_ranks = stored["methods"]["pe_core"]["ranks"]
    stored_by_id = dict(zip((c["id"] for c in stored_cases), stored_ranks))

    computed = run_caption_baseline(dev_cases, rows_by_text, meta_by_idx, id_video, id_frame,
                                    "caption_en_approved", K=1000)
    mismatches = []
    for case in dev_cases:
        stored_rank = stored_by_id.get(case["id"])
        computed_rank = computed[case["id"]]
        if stored_rank != computed_rank:
            mismatches.append((case["id"], stored_rank, computed_rank))
    if mismatches:
        print(f"STAGE 0 SANITY FAILED: {len(mismatches)}/{len(dev_cases)} mismatches")
        for case_id, stored_rank, computed_rank in mismatches[:10]:
            print(f"  {case_id}: stored={stored_rank} computed={computed_rank}")
        raise SystemExit(1)
    print(f"STAGE 0 SANITY OK: {len(dev_cases)}/{len(dev_cases)} dev-case ranks match stored PE-Core benchmark exactly")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansions", type=Path,
                        default=ROOT / "artifacts/query_expansion/v1/expansions.json")
    parser.add_argument("--stage", choices=["sanity", "grid", "ablation", "holdout"], required=True)
    parser.add_argument("--run-holdout", action="store_true",
                        help="required in addition to --stage holdout, as a deliberate confirmation")
    args = parser.parse_args()

    cases, header = load_cases_expansion(args.expansions)
    id_video, id_frame, id_time, meta_by_idx, faiss_index = load_profile()
    shot_indices = load_shot_indices(meta_by_idx)
    group_resolver = GroupResolver(id_video, id_frame, id_time, shot_indices)

    texts = collect_texts(cases)
    vectors_by_text = encode_and_cache(texts)
    rows_by_text = retrieve_and_cache(texts, vectors_by_text, faiss_index)

    dev_cases = cases_for_split(cases, "dev")
    holdout_cases = cases_for_split(cases, "holdout")
    all_2026 = cases_for_split(cases, "2026")

    if args.stage == "sanity":
        stage0_sanity(dev_cases, rows_by_text, meta_by_idx, id_video, id_frame)

        approved = run_caption_baseline(dev_cases, rows_by_text, meta_by_idx, id_video, id_frame,
                                        "caption_en_approved", K=1000)
        print_table("caption_approved (dev)", round_breakdown(approved, dev_cases, DEV_ROUNDS), DEV_ROUNDS)

        approved_dedup = run_caption_shotdedup_baseline(
            dev_cases, rows_by_text, id_time, meta_by_idx, id_video, id_frame, group_resolver,
            "caption_en_approved", K=1000)
        print_table("caption_approved_shotdedup (dev)",
                    round_breakdown(approved_dedup, dev_cases, DEV_ROUNDS), DEV_ROUNDS)

        holistic_only = run_config(dev_cases, rows_by_text, id_time, id_video, id_frame,
                                   meta_by_idx, group_resolver, lam=1.0, K=1000)
        print_table("holistic_only / lam=1 (dev)",
                    round_breakdown(holistic_only, dev_cases, DEV_ROUNDS), DEV_ROUNDS)
        return

    if args.stage == "grid":
        run_stage1_grid(dev_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx,
                        group_resolver)
        return

    if args.stage == "ablation":
        # S1* from the Stage-1 grid run: k=10, K=2000, agg=mean, lam=0.75 (max MRR then R@1).
        s1_star = {"lam": 0.75, "k": 10, "K": 2000, "agg": "mean", "m": 2}
        run_stage2_ablations(dev_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx,
                             group_resolver, s1_params=s1_star)
        return

    if args.stage == "holdout":
        if not args.run_holdout:
            raise SystemExit("--stage holdout requires --run-holdout to confirm")
        if not CONFIG_PATH.exists():
            raise SystemExit(f"{CONFIG_PATH} does not exist; freeze a config first")
        run_holdout(all_2026, holdout_cases, rows_by_text, id_time, id_video, id_frame,
                   meta_by_idx, group_resolver)
        return


def run_stage1_grid(dev_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx, group_resolver):
    results = []
    for k in (10, 60):
        for K in (500, 2000):
            for agg in ("mean", "top_m_mean"):
                for lam in (0.25, 0.5, 0.75):
                    ranks = run_config(dev_cases, rows_by_text, id_time, id_video, id_frame,
                                       meta_by_idx, group_resolver, lam=lam, k=k, K=K, agg=agg, m=2)
                    breakdown = round_breakdown(ranks, dev_cases, DEV_ROUNDS)
                    entry = {"stage": 1, "k": k, "K": K, "agg": agg, "lam": lam,
                            "pooled": breakdown["pooled"]}
                    results.append(entry)
                    log_tuning(entry)
                    print(f"k={k:3d} K={K:5d} agg={agg:12s} lam={lam:.2f}  "
                          f"MRR={breakdown['pooled']['mrr']:.3f} "
                          f"R@1={breakdown['pooled']['recall_at_1']:.3f} "
                          f"R@10={breakdown['pooled']['recall_at_10']:.3f}")
    # References
    for k, K in ((60, 2000),):
        for lam, name in ((1.0, "lam=1 (holistic-only ref)"), (0.0, "lam=0 (aspect-only ref)")):
            ranks = run_config(dev_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx,
                              group_resolver, lam=lam, k=k, K=K)
            breakdown = round_breakdown(ranks, dev_cases, DEV_ROUNDS)
            print(f"{name}: MRR={breakdown['pooled']['mrr']:.3f} "
                  f"R@1={breakdown['pooled']['recall_at_1']:.3f}")

    best = max(results, key=lambda r: (r["pooled"]["mrr"], r["pooled"]["recall_at_1"]))
    print(f"\nBEST S1*: {best}")
    best_per_lam = {}
    for lam in (0.25, 0.5, 0.75):
        candidates = [r for r in results if r["lam"] == lam]
        best_per_lam[lam] = max(candidates, key=lambda r: (r["pooled"]["mrr"], r["pooled"]["recall_at_1"]))
        print(f"BEST for lam={lam}: {best_per_lam[lam]}")


def run_stage2_ablations(dev_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx,
                         group_resolver, s1_params=None):
    if s1_params is None:
        raise SystemExit("pass S1* params via code edit once stage 1 has picked them")
    base = dict(s1_params)

    def evaluate(name, **overrides):
        params = {**base, **overrides}
        ranks = run_config(dev_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx,
                          group_resolver, **params)
        breakdown = round_breakdown(ranks, dev_cases, DEV_ROUNDS)
        entry = {"stage": 2, "name": name, "params": params, "pooled": breakdown["pooled"]}
        log_tuning(entry)
        print(f"{name:40s} MRR={breakdown['pooled']['mrr']:.3f} "
              f"R@1={breakdown['pooled']['recall_at_1']:.3f} "
              f"R@10={breakdown['pooled']['recall_at_10']:.3f}")
        return entry

    baseline = evaluate("S1* baseline")
    for gamma in (0.5, 1.0):
        evaluate(f"neighbor gamma={gamma} scope=aspects", gamma=gamma, neighbor_scope="aspects")
    for beta in (0.5, 0.25):
        evaluate(f"soft beta={beta} T_h=200", beta=beta, T_h=200)
    evaluate("hard gate beta=0 T_h=200 (ablation only)", beta=0.0, T_h=200)
    evaluate("dedup_seconds=2", dedup_seconds=2.0)
    return baseline


def run_holdout(all_2026_cases, holdout_cases, rows_by_text, id_time, id_video, id_frame,
                meta_by_idx, group_resolver):
    config = json.loads(CONFIG_PATH.read_text())
    print(f"Frozen config: {config}")
    params = {k: v for k, v in config.items()
             if k in ("lam", "k", "K", "agg", "m", "gamma", "neighbor_scope", "T_h", "beta",
                      "dedup_seconds")}
    ranks = run_config(all_2026_cases, rows_by_text, id_time, id_video, id_frame, meta_by_idx,
                       group_resolver, **params)
    breakdown = round_breakdown(ranks, all_2026_cases, REPORTED_ROUNDS)
    print_table("frozen config (2026, incl. holdout)", breakdown, REPORTED_ROUNDS)
    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(json.dumps({
        "config": config, "ranks": ranks, "breakdown": breakdown,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))
    print(f"wrote {BENCHMARK_OUT}")


if __name__ == "__main__":
    main()
