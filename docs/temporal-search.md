# Temporal search

Temporal mode finds an ordered sequence of 2–6 visual events in one video. Frames
must increase strictly and each adjacent pair must be no more than the requested
gap (1–600 seconds, default 120). A video may contribute multiple sequences;
score-ordered temporal NMS keeps every non-overlapping interval.

The visual stage batch-encodes all event descriptions and searches the exact Flat
index. Dynamic programming keeps the best prefix for each endpoint. The production
candidate depth and sequence scorer are loaded from
`benchmarks/temporal/temporal_config.json`; they are deliberately selected by the benchmark
rather than hardcoded.

An optional Vietnamese transcript query searches 1,000 audio hits in parallel.
The best-ranked hit in `[sequence start - 30s, sequence end + 30s]` is fused with
the visual rank using RRF (`k=60`) before NMS. Audio evidence may support multiple
sequences. Audio failure returns visual results with a warning.

## Benchmark and ground truth

`benchmarks/temporal/temporal_ground_truth.json` is reviewed data with this shape:

```json
{
  "tolerance_seconds": 10,
  "queries": [
    {
      "id": "query-id",
      "events": ["first visual event", "second visual event"],
      "audio_text": "optional Vietnamese transcript query",
      "max_gap_seconds": 120,
      "ground_truth": [
        {"video_id": "L01_V001", "event_times": [12.3, 48.7]}
      ]
    }
  ]
}
```

Multiple correct sequences are allowed for each query. A returned sequence is
correct only when the video matches and every event is within ±10 seconds. Run:

```bash
python scripts/benchmarks/bench_temporal_search.py
```

The harness runs 3 warmups and 10 timed executions for every combination of
candidate K (`500, 1000, 2000, 5000`) and scorer (`raw_mean`, `raw_min`,
`shifted_harmonic`, `minmax_mean`, `percentile_mean`). It chooses highest MRR,
then Recall@10, then lower p95 latency, then smaller K. Warm p95 ≤2 seconds is
reported as an operational target, not a correctness gate. Outputs are the raw
JSON report, a Markdown table, and the production config in `benchmarks/temporal/`.

## API

- `POST /search/temporal`: starts a snapshot-backed search.
- `POST /search/temporal/next`: fetches the next 20 video groups.

The ranked pool contains at most 1,000 sequences. Videos are ranked by their best
sequence; sequences inside each video are chronological. Snapshots expire after
600 seconds and are scoped to the authenticated user.
