#!/usr/bin/env python3
"""Evaluate beam DP and temporal candidate dedup against production baseline."""
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app import TemporalSearchReq, temporal_visual_paths
from scripts.benchmarks.bench_temporal_search import evaluate_results
from backend.temporal_search import temporal_nms


def variants():
    yield {"name": "baseline", "beam_size": 1, "dedup_seconds": 0, "candidate_pool_k": 2000}
    for beam in (2, 4, 8, 16):
        yield {"name": f"beam-{beam}", "beam_size": beam, "dedup_seconds": 0, "candidate_pool_k": 2000}
    for window in (1, 2, 3):
        yield {"name": f"dedup-{window}s", "beam_size": 1, "dedup_seconds": window, "candidate_pool_k": 5000}
    for beam in (2, 4, 8, 16):
        for window in (1, 2, 3):
            yield {"name": f"beam-{beam}+dedup-{window}s", "beam_size": beam,
                   "dedup_seconds": window, "candidate_pool_k": 5000}


def ranked(query, variant):
    request = TemporalSearchReq(events=query["events"], max_gap_seconds=120)
    result = temporal_visual_paths(
        request, 2000, "raw_mean", beam_size=variant["beam_size"],
        candidate_pool_k=variant["candidate_pool_k"],
        dedup_seconds=variant["dedup_seconds"],
    )
    return temporal_nms(result["paths"])[:1000]


def quality(queries, variant, tolerance):
    per_query = []
    for query in queries:
        results = ranked(query, variant)
        metric = evaluate_results(results, query["ground_truth"], tolerance)
        rank = None if metric["reciprocal_rank"] == 0 else round(1 / metric["reciprocal_rank"])
        per_query.append({"id": query["id"], "rank": rank, **metric})
    return {
        "mrr": statistics.fmean(row["reciprocal_rank"] for row in per_query),
        "recall_at_10": statistics.fmean(row["recall_at_10"] for row in per_query),
        "per_query": per_query,
    }


def p95_latency(queries, variant, warmups=3, runs=10):
    timings = []
    for query in queries:
        for _ in range(warmups):
            ranked(query, variant)
        for _ in range(runs):
            started = time.perf_counter()
            ranked(query, variant)
            timings.append((time.perf_counter() - started) * 1000)
    timings.sort()
    return timings[(95 * len(timings) + 99) // 100 - 1]


def markdown(rows, selected):
    lines = [
        "# Temporal beam and candidate-dedup benchmark", "",
        "Audio was unavailable; all variants use visual fallback.", "",
        "| Variant | MRR | Recall@10 | p95 ms | Per-query rank |",
        "|---|---:|---:|---:|---|",
    ]
    for row in rows:
        ranks = ", ".join(f'{item["id"]}:{item["rank"] or "—"}' for item in row["per_query"])
        latency = "—" if row.get("p95_ms") is None else f'{row["p95_ms"]:.1f}'
        lines.append(f'| {row["name"]} | {row["mrr"]:.4f} | {row["recall_at_10"]:.4f} | {latency} | {ranks} |')
    lines.extend(["", f'Chosen: `{selected["name"]}`.'])
    return "\n".join(lines) + "\n"


def main():
    manifest = json.loads((ROOT / "benchmarks/temporal/temporal_ground_truth.json").read_text())
    queries = manifest["queries"]
    rows = []
    for index, variant in enumerate(variants(), 1):
        print(f'quality {index}/20: {variant["name"]}', flush=True)
        rows.append({**variant, **quality(queries, variant, manifest["tolerance_seconds"])})
    best_quality = max((row["mrr"], row["recall_at_10"]) for row in rows)
    baseline = rows[0]
    baseline_quality = (baseline["mrr"], baseline["recall_at_10"])
    if best_quality <= baseline_quality:
        previous = json.loads((ROOT / "benchmarks/temporal/temporal_benchmark.json").read_text())
        baseline["p95_ms"] = previous["selected"]["p95_ms"]
        selected = baseline
    else:
        finalists = [row for row in rows if (row["mrr"], row["recall_at_10"]) == best_quality]
        finalists.append(baseline)
        for row in finalists:
            print(f'timing: {row["name"]}', flush=True)
            row["p95_ms"] = p95_latency(queries, row)
        selected = max(finalists, key=lambda row: (
            row["mrr"], row["recall_at_10"], -row["p95_ms"],
            -row["beam_size"], -row["dedup_seconds"], -row["candidate_pool_k"],
        ))
    output = ROOT / "benchmarks"
    report = {"rows": rows, "selected": selected}
    (output / "temporal_improvements.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "temporal_improvements.md").write_text(markdown(rows, selected))
    print(json.dumps({key: selected[key] for key in (
        "name", "beam_size", "dedup_seconds", "candidate_pool_k", "mrr", "recall_at_10", "p95_ms"
    )}, indent=2))


if __name__ == "__main__":
    main()
