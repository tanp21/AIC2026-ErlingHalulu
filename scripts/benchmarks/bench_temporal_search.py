#!/usr/bin/env python3
"""Choose the production temporal-search configuration from reviewed ground truth."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CANDIDATE_K_VALUES = (500, 1000, 2000, 5000)


def _matches(result, truth, tolerance_seconds):
    if result.get("video_id") != truth.get("video_id"):
        return False
    actual = [event["time"] for event in result.get("events", [])]
    expected = truth.get("event_times", [])
    return len(actual) == len(expected) and all(
        abs(left - right) <= tolerance_seconds for left, right in zip(actual, expected)
    )


def evaluate_results(results, ground_truth, tolerance_seconds=10):
    rank = next((
        index for index, result in enumerate(results, 1)
        if any(_matches(result, truth, tolerance_seconds) for truth in ground_truth)
    ), None)
    return {
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "recall_at_10": float(rank is not None and rank <= 10),
    }


def select_config(rows):
    if not rows:
        raise ValueError("benchmark produced no configuration rows")
    return max(rows, key=lambda row: (
        row["mrr"], row["recall_at_10"], -row["p95_ms"], -row["candidate_k"]
    ))


def _p95(values):
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def _ranked_paths(query, candidate_k, scoring_method):
    from backend.app import TemporalSearchReq, _temporal_audio_hits, temporal_visual_paths
    from backend.temporal_search import fuse_audio, temporal_nms

    request = TemporalSearchReq(
        events=query["events"],
        audio_text=query.get("audio_text"),
        max_gap_seconds=query.get("max_gap_seconds", 120),
    )
    visual = temporal_visual_paths(request, candidate_k, scoring_method)
    paths = visual["paths"]
    if request.audio_text:
        try:
            paths = fuse_audio(paths, _temporal_audio_hits(request.audio_text))
        except Exception:
            pass
    return temporal_nms(paths)[:1000]


def run_benchmark(manifest, warmups=3, runs=10):
    from backend.temporal_search import SCORING_METHODS

    queries = manifest["queries"]
    audio_query = next((item for item in queries if item.get("audio_text")), None)
    if audio_query:
        from backend.app import _temporal_audio_hits
        try:
            _temporal_audio_hits(audio_query["audio_text"])
        except Exception:
            print("warning: audio search unavailable; benchmarking visual fallback", file=sys.stderr)
            queries = [{key: value for key, value in item.items() if key != "audio_text"}
                       for item in queries]
    rows = []
    for candidate_k in CANDIDATE_K_VALUES:
        for scoring_method in SCORING_METHODS:
            reciprocal_ranks = []
            recalls = []
            timings = []
            for query in queries:
                for _ in range(warmups):
                    _ranked_paths(query, candidate_k, scoring_method)
                ranked = None
                for _ in range(runs):
                    started = time.perf_counter()
                    ranked = _ranked_paths(query, candidate_k, scoring_method)
                    timings.append((time.perf_counter() - started) * 1000)
                metrics = evaluate_results(
                    ranked or [], query["ground_truth"],
                    manifest.get("tolerance_seconds", 10),
                )
                reciprocal_ranks.append(metrics["reciprocal_rank"])
                recalls.append(metrics["recall_at_10"])
            rows.append({
                "candidate_k": candidate_k,
                "scoring_method": scoring_method,
                "mrr": statistics.fmean(reciprocal_ranks),
                "recall_at_10": statistics.fmean(recalls),
                "p95_ms": _p95(timings),
            })
    return rows


def _markdown(rows, selected):
    lines = [
        "# Temporal search benchmark", "",
        "| K | Scoring | MRR | Recall@10 | p95 ms |", "|---:|---|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: (item["candidate_k"], item["scoring_method"])):
        lines.append(
            f'| {row["candidate_k"]} | {row["scoring_method"]} | '
            f'{row["mrr"]:.4f} | {row["recall_at_10"]:.4f} | {row["p95_ms"]:.1f} |'
        )
    lines.extend(["", f'Chosen: K={selected["candidate_k"]}, `{selected["scoring_method"]}`.'])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, default=Path("benchmarks/temporal/temporal_ground_truth.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/temporal"))
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    manifest = json.loads(args.ground_truth.read_text())
    if not manifest.get("queries") or any(not item.get("ground_truth") for item in manifest["queries"]):
        raise SystemExit("ground truth must contain at least one reviewed sequence per query")
    rows = run_benchmark(manifest, args.warmups, args.runs)
    selected = select_config(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"warmups": args.warmups, "runs": args.runs, "rows": rows, "selected": selected}
    (args.output_dir / "temporal_benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "temporal_benchmark.md").write_text(_markdown(rows, selected))
    config = {key: selected[key] for key in ("candidate_k", "scoring_method")}
    (args.output_dir / "temporal_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config))


if __name__ == "__main__":
    main()
