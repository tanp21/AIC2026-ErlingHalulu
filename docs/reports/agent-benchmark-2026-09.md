# Agent benchmark 2026 — baseline and where the losses are

Run on 2026-09-21, `main` at `0818942`, fast mode, the whole official 2026 set: Test
(23), Round 1 (25), Round 2 (30), Round 3 (36) — 114 query files covering all 137
cases (a TRAKE case per event collapses into one file; the loader rebuilds the
sequence). Ground truth is `targets` from `artifacts/official_benchmark_answers_v2.json`,
scored by the CLI's own harness at the repository's default tolerance of 250 frames.

```bash
# the query files are generated from the answers file, one directory per round
python -m agent run /tmp/bench/round-3-2026 \
  --answers "artifacts/official_benchmark_answers_v2.json::Round 3 - 2026" \
  --out /tmp/bench_out/round-3-2026 --workers 2
```

## Where it stands

| round | n | R@1 | R@5 | R@20 | R@100 | video R@1 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Test | 23 | 0.609 | 0.739 | 0.739 | 0.783 | 0.783 | 0.660 |
| Round 1 | 25 | 0.440 | 0.640 | 0.720 | 0.880 | 0.720 | 0.540 |
| Round 2 | 30 | 0.600 | 0.700 | 0.767 | 0.767 | 0.800 | 0.656 |
| Round 3 | 36 | 0.667 | 0.806 | 0.806 | 0.861 | 0.750 | 0.733 |
| **all** | **114** | **0.588** | 0.728 | 0.763 | 0.861 | **0.763** | 0.669 |

By kind: KIS 0.622 (n=82) · Q&A 0.667 (n=24) · **TRAKE 0.000 (n=8)**.

## The ceiling is 93%, and nothing upstream of verification is the bottleneck

Every gold video is somewhere in the candidate pool (114/114), and the *moment* is too
— within 250 frames — for 106 of 114. So retrieval is not what is losing the queries.

| stage | queries | share |
| --- | ---: | ---: |
| the right moment is in the pool | 106 | 93% |
| ... at fused rank ≤ 1 | 67 | 59% |
| ... at fused rank ≤ 5 | 85 | 75% |
| ... at fused rank ≤ 15 (the verify window in fast mode) | 89 | 78% |
| ... actually shown to the judge | 82 | 72% |
| when shown, the judge scores it highest of everything it saw | **76 of 82** | **93%** |
| the submitted top-1 is the right moment | 67 | 59% |

Read down that column: the agent *finds* the answer for 106 queries and submits it for
67. The judge is the strongest link in the chain — given the right moment it picks it
93% of the time. What leaks is the step in between: **which candidates get judged**.
Seventeen queries have the right moment in the pool but below rank 15, so it is never
shown; seven more have it inside rank 15 and are still skipped, because the selection
spends most of its slots on each query's rank-1 hit rather than on the fused top
(`CandidatePool.to_verify`'s `quota_share=0.6`).

## Failures, by what actually went wrong

| bucket | KIS | Q&A | TRAKE | total |
| --- | ---: | ---: | ---: | ---: |
| **D** right video submitted first, wrong moment | 16 | 4 | 2 | **22** |
| **C** the gold video is never submitted at all | 5 | 0 | 5 | **10** |
| **A1** gold moment verified inside rank 15, another moment submitted | 6 | 2 | 0 | **8** |
| **B** gold moment in the pool, never verified | 2 | 1 | 1 | **4** |
| **A2** gold moment verified but ranked below 15 | 2 | 1 | 0 | **3** |

Buckets A1/A2/B are the leak above: 15 queries the agent already had an answer for. D
is the "right video, wrong moment" the agent's own docs call the commonest miss, and it
is the largest bucket — but note it is *not* the same as being unverifiable: in most of
D the gold moment is in the pool and simply not the one that won.

## What to do, in the order the numbers argue for

### 1. Judge the fused top first, then spend the quota (expected: the A1/B cohorts, ~15 queries)

`to_verify` takes `round(limit * 0.6)` slots for a round-robin over each source query's
best hit, and only then fills from the fused order. That quota exists for a real case
(a caption's rank-1 keyframe sat at fused rank 32 and was missed), but it also skips
fused ranks 2..N whenever the round-robin has candidates to spend them on — seven
queries here had the gold inside rank 15 and unverified.

Change: fill the first `limit - quota` slots from the fused order, then the quota from
the per-source queues. Same call budget, different order. Measure with the same harness
on the same four rounds; the quantity to move is `moment_verified` (72% → ?), not R@1
directly.

### 2. Widen the verify window in fast mode (expected: up to 17 queries, the ≥rank-15 cohort)

`verify_top=15` leaves 17 queries whose answer sits at fused rank 16–100 (93 at ≤50, 99
at ≤100). Chunks run in parallel four at a time, so raising the window costs
`ceil((N-15)/5) - 3` extra Gemini calls per round 1 — roughly +3 calls, ~4 s of wall
time, on a step that already takes 7 s. Given the judge picks the right moment 93% of
the time it sees it, each converted query is close to a direct R@1 gain.

Cheapest version to try first: `verify_top=25` in `FAST_DEFAULTS`, measure, then decide
whether 30 is better. Watch the free-tier rate limit: this is the change most likely to
hit it.

### 3. Do not let the plan name an unknown entity (expected: the C cohort, ~10 queries)

Query 30 is the type case and it was measured end to end. The query describes "4 con X"
and asks what X is; the planner filled X with **"cua"** (crab), so every channel searched
the wrong animal — while the video says **"nghêu"** (clam) fifteen times. Measured on the
live index:

| query | gold video's rank |
| --- | ---: |
| visual: "a girl in a white apron holds a white plate of small shellfish, a vase of purple flowers beside her" — no animal named | **2** |
| audio: "nghêu" | **2** |
| audio: "cua" (what the plan searched) | not found |
| audio: "hôm nay nấu món gì" (what the plan also searched) | not found |

So the retrieval had the answer at rank 2 and the plan's guess threw it away. Change the
planner prompt: when the question asks what an unknown thing *is*, describe it without
naming it, and never put the guess into `transcript_vi`. This is a prompt change, so it
needs the benchmark to confirm it does not cost the queries where the guess is right.

Related, same cohort: the gold video is absent from the submitted list for 5 KIS and 5
TRAKE queries. Classify those individually before touching the index — retrieval finds
93% of moments, so an index rebuild is unlikely to be the answer.

### 4. TRAKE is 0 of 8 and needs its own investigation

Five of the eight never submit the gold video at all, and none of the three that do get
the frames within tolerance. Two possibilities worth separating before changing
anything: the gold rows the harness builds are strict (one video must carry every event,
and only the first accepted frame per event is used), and the agent's `trake_check` /
`trake_refine` path is tuned to rounds 1–2. Check a specific case by hand — `query-p2-8-trake`
and `query-p1-18-trake` are in the C bucket — before assuming which side is wrong.

### 5. Fix the two structural wastes (cheap, no measurement needed)

- **Web lookups ran for 18 of 114 queries (16%).** The cooking lookup is started in the
  background before planning but is collected with `_collect_lookups(min(3.0, remaining - 8))`,
  which is zero once a run is over budget — so the signal that named the right dish for
  query 24 is usually dropped. Either collect it deterministically before the answer
  step or start it earlier.
- **Two of every twelve round-1 slots go to caption queries that 404** (measured across
  14 saved runs: 28/197 planned queries, 14%). The client latches and drops them, but
  only after they have taken slot budget and a round trip. Either build
  `build_caption_index` or tell the planner the channel is absent before it writes
  `caption_en`.

## What the numbers say *not* to do

- **Rebuild the visual index for recall.** The moment is already in the pool for 93% of
  queries; a better encoder would have to beat a ceiling that is not the binding one.
- **Change the judge model.** Given the right moment it scores it top 93% of the time.
- **Add more search channels.** The caption channel is planned for every query and does
  not exist here; more channels that are not shown the answer do not help.

## Reproducing

```bash
# query files, one directory per round, from the answers file
python - <<'EOF'
import json, re
from collections import defaultdict
from pathlib import Path
data = json.load(open("artifacts/official_benchmark_answers_v2.json"))
rounds = defaultdict(list)
for case in data["cases"]:
    prefix, _, rest = case["id"].partition("/")
    rounds[prefix].append((rest.split(":")[0], case.get("text") or ""))
for prefix, items in rounds.items():
    out = Path("/tmp/bench") / re.sub(r"[^a-z0-9]+", "-", prefix.lower()).strip("-")
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    for qid, text in items:
        if text.strip() and qid not in seen:
            seen.add(qid)
            (out / f"{qid}.txt").write_text(text.strip(), encoding="utf-8")
EOF
```

Then the four `python -m agent run ... --answers "...json::<round>"` invocations above.
`/tmp/classified.json` (per-query pool/verification/evidence facts) and
`/tmp/moment_ceiling.json` (the frame-level ceiling table) are produced by the throwaway
scripts written during this session; both are small enough to regenerate from the saved
run JSONs and the answers file.

Known caveat: two runs of this benchmark produced 0.652 and 0.609 R@1 on the same Test
round with identical code, so treat differences under ~0.05 on a single round as noise.
The 114-query aggregate is the number to move.
