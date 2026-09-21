#!/usr/bin/env python3
"""What the shot-caption channel finds, and what it adds to PE-Core.

The captions (``docs/caprl-shot-captioning.md``) are a text channel over what
was *shown*, so the question is not only "how good is BM25 over captions" but
"does it find cases PE-Core misses". Both are measured here against the same
cases and the same 10 s tolerance as
``benchmarks/visual/visual_embedding_benchmark_2026_en.json``, whose per-case
PE-Core ranks this script reuses for the comparison -- no FAISS, no GPU, just
Elasticsearch.

    uv run python -m scripts.benchmarks.bench_caption_search
    uv run python -m scripts.benchmarks.bench_caption_search --rounds "Round 2 - 2026"
    uv run python -m scripts.benchmarks.bench_caption_search --method hybrid --vi

A case counts as a hit when a returned caption window is in the gold video and
within the tolerance of a gold frame -- the same rule the agent lives by, since
it searches inside the returned window afterwards.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import caption_search, frame_maps

DEFAULT_BENCHMARK = ROOT / "benchmarks" / "visual" / "visual_embedding_benchmark_2026_en.json"
DEFAULT_OUTPUT = ROOT / "benchmarks" / "captions" / "caption_benchmark_2026.json"
CUTOFFS = (1, 5, 10, 100)


def gold_times(targets) -> dict[str, list[float]]:
    """Gold frames as seconds per video, through the frame map (never fps maths)."""
    times: dict[str, list[float]] = {}
    for video_id, frame_idx in targets:
        try:
            mapping = frame_maps.load(video_id)
            times.setdefault(video_id, []).append(mapping.frame_time(int(frame_idx)))
        except Exception:  # a video without a frame map cannot be scored
            continue
    return times


def caption_ranks(case, method, k, tolerance, use_vi, index):
    """(rank of the first correct caption window, number of hits looked at)."""
    text = case["text_vi"] if use_vi else case["text"]
    search = {
        "keyword": caption_search.search_captions_keyword,
        "semantic": caption_search.search_captions_semantic,
        "hybrid": caption_search.search_captions_hybrid,
    }[method]
    hits = search(text, k, index=index)
    wanted = gold_times(case["targets"])
    for rank, hit in enumerate(hits, 1):
        for gold in wanted.get(hit.video_id, ()):
            near = min(abs(hit.start_sec - gold), abs(hit.end_sec - gold),
                       abs(hit.anchor_sec - gold))
            if hit.start_sec - tolerance <= gold <= hit.end_sec + tolerance or near <= tolerance:
                return rank, len(hits)
    return None, len(hits)


def best_of(rank_a, rank_b):
    """The better of two per-case ranks, or None when neither channel found it.

    The benchmark file stores each visual model's per-case rank, not its full
    ranked list, so a real RRF rank cannot be recomputed here. "Best of both" is
    what a reviewer looking at two lists gets, and an upper bound on what fusing
    them could give -- enough to answer whether the captions add cases at all.
    """
    ranks = [rank for rank in (rank_a, rank_b) if rank]
    return min(ranks) if ranks else None


def summarize(ranks, label):
    scored = [rank for rank in ranks if rank]
    summary = {"label": label, "cases": len(ranks), "hits": len(scored)}
    summary["mrr"] = round(sum(1 / rank for rank in scored) / len(ranks), 4) if ranks else 0.0
    for cutoff in CUTOFFS:
        summary[f"recall_at_{cutoff}"] = round(
            sum(1 for rank in scored if rank <= cutoff) / len(ranks), 4) if ranks else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--index", default=caption_search.CAPTION_INDEX)
    parser.add_argument("--method", choices=["keyword", "semantic", "hybrid"], default="keyword")
    parser.add_argument("--rounds", nargs="*", help="only these case-id prefixes")
    parser.add_argument("--vi", action="store_true", help="query with the Vietnamese prompt")
    parser.add_argument("-k", type=int, default=1000)
    parser.add_argument("--tolerance", type=float, default=10.0, help="seconds")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.benchmark.read_text(encoding="utf-8"))
    cases = data["cases"]
    baseline_ranks = data["methods"]["pe_core"]["ranks"]
    keep = [
        (index, case) for index, case in enumerate(cases)
        if not args.rounds or any(case["id"].startswith(prefix) for prefix in args.rounds)
    ]
    rows, caption_only, combined, baseline_only = [], [], [], []
    for index, case in keep:
        rank, returned = caption_ranks(case, args.method, args.k, args.tolerance, args.vi,
                                       args.index)
        baseline = baseline_ranks[index]
        caption_only.append(rank)
        baseline_only.append(baseline)
        combined.append(best_of(rank, baseline))
        rows.append({"id": case["id"], "caption_rank": rank, "pe_core_rank": baseline,
                     "returned": returned})
        print(f"  {case['id']:34} caption@{rank} pe_core@{baseline}", flush=True)

    report = {
        "benchmark": str(args.benchmark),
        "index": args.index,
        "method": args.method,
        "query_language": "vi" if args.vi else "en",
        "tolerance_seconds": args.tolerance,
        "k": args.k,
        "summaries": [
            summarize(caption_only, f"captions ({args.method})"),
            summarize(baseline_only, "pe_core (from the benchmark file)"),
            summarize(combined, "best of captions or pe_core (fusion upper bound)"),
        ],
        "cases": rows,
    }
    complement = sum(1 for row in rows if row["caption_rank"] and not row["pe_core_rank"])
    report["caption_only_hits"] = complement
    print(json.dumps(report["summaries"], indent=2))
    print(f"cases the captions found and PE-Core missed entirely: {complement}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
