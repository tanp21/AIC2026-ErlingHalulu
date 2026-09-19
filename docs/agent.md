# Retrieval agent

`python -m agent` takes a Vietnamese competition query (KIS, Q&A or TRAKE), plans
searches across every channel the server offers, checks candidates by looking at
them, and writes a ranked, submission-ready answer for a teammate to confirm.

The agent is a client of the search server: it runs anywhere (a laptop is fine),
calls the server over HTTP and never ships in the production image.

## How one query is answered

There are three modes. They share every step except what happens after round 1.

**`--mode fast`** (the default) takes about 15-20 s per query (median; TRAKE ~30 s):

```text
t=0   cooking query? ──► Gemini + Google Search: which "Món ngon mỗi ngày" dish/episode (background)
      DeepSeek plan, no thinking (~2 s) ──► plan lookups start in the background
      round 1: one /agent/search batch (~0.5-1 s)
      Gemini verifies 15 candidates as 3 parallel calls of 5 (~5 s): each query's best
        hits round-robin, then the rest by fusion
      best ≥ 9 and the falsifier (~5 s) accepts it? ── yes ──► done
      otherwise ──► one DeepSeek rewrite (no thinking, sees verdicts + web notes)
                    + the target scene searched inside the top 5 videos
                    + neighbours of a 5-8 match ──► batch ──► parallel verification
      Q&A: reading at the top 2 moments  ∥  a web answer (recipe for cooking shows,
           a news/fact search otherwise, none for counts) ──► reconciled
      TRAKE: no point verification; each event searched inside the 3-4 leading videos
             and joined in order ≤ 45 s apart; those paths and the fused server paths
             checked event by event in parallel; the best one's events re-picked in
             parallel (the unrefined path is kept as the next row)
```

**`--mode auto`** is fast mode, then the deep tool loop (up to 120 s in total) only for
a KIS/Q&A query whose best verdict is still under 7.

Where the time went before (deep mode, round 1 of 2026): the multi-turn DeepSeek loop
with thinking, sequential Gemini calls, and a few requests that stalled. Measured on the
live server: one Gemini judge call takes ~4 s whether it looks at 2 or 8 frames, a
DeepSeek call without thinking 1-3 s, a batch 0.3-4 s. Two tail fixes matter as much as
the parallelism:

- A Gemini call still pending after 8 s is **hedged**: a duplicate request races it and
  the first answer wins (`GeminiClient(hedge_after=...)`).
- About one thumbnail request in 50 stalls for ~30 s over the tailnet while an
  immediate retry takes 0.3 s, so images time out after 5 s and are retried
  (`agent/search_client.py`).

**`--mode deep`** is the original multi-step tool loop, for when there is time:

```text
query ──► DeepSeek plan ──► round 1: one /agent/search batch ──► Gemini verifies top 12
            (JSON)            visual (EN caption + paraphrases/scene)       │
                              temporal (multi-scene / TRAKE)              │ best ≥ 9 (and, for Q&A,
                              transcript (VI)                            │ an answer read)? ── yes ──► finalize
                              OCR (distinctive on-screen strings)         │ no
            Gemini + Google Search for names the query only describes     ▼
                                                              DeepSeek tool loop (≤ 6 steps, ≤ 300 s)
                                                              search · inspect · neighbours ·
                                                              read_transcript · web_search · finish
                                                                          │
finalize: ranking (verified matches first) · Q&A answer read from the frame/speech ·
          TRAKE: top 6 paths checked event by event, best 2 moved to each event's first keyframe
```

- **Round 1 is deterministic** after the plan: `agent/plan.py` turns the plan into one
  batch, target scene first, so a query with a clear match needs no agent turns at all.
- **Candidates are moments, not hits** (`agent/pool.py`). Keyframe hits within 4 s of
  each other merge; transcript passages and OCR shots back every moment inside their
  span. Ranking is reciprocal-rank fusion plus a video-level term, then Gemini verdicts
  override it: verified matches (≥ 6) first, contradicted ones (≤ 3) last.
- **What gets verified is not just the fused top.** Fusion favours moments that many
  weak hits agree on, and on round 2 it left a caption's rank-1 keyframe (the gold frame
  of p2-3) unverified at fused rank 32. `CandidatePool.to_verify` gives 60% of the
  verification slots round-robin to each query's best unverified hits, then fills the
  rest by fusion.
- **Gemini looks, DeepSeek decides.** Gemini scores labelled keyframes 0-10 against the
  original Vietnamese query, reads Q&A answers, and picks exact TRAKE frames. DeepSeek
  plans, rewrites queries and chooses tools. Neither model is the source of a Q&A answer
  on its own: answers are read from the video, and web search only resolves names or
  cross-checks. A loop that spends its budget without calling `finish` gets one last
  finish-only turn, so what it learned (a name found on the web, its own ranking) is not
  lost. For Q&A the final answer then comes from `finish`, or else from a dedicated
  reading of the top moment (full frame, neighbours, transcript, and web notes as hints).
- **An early stop must survive a falsifier.** Before round 1 is accepted, Gemini sees
  the top candidate with its neighbouring keyframes and tries to contradict each detail
  of the query. A rejection keeps the score (its first live run rejected two correct
  matches over details a few keyframes cannot show) but hands the run to the loop, with
  the objection attached for the human. On the first live run, 2 of 13 early stops were
  confident wrong matches. Deep mode only.
- **Q&A answers are read at up to 3 moments** of the top video and decided by majority
  (ties to confidence): one keyframe can hide what is being counted or read.
- **Web search is automatic where it pays** (`agent/web.py`). Cooking queries (15 of 55
  in 2026 rounds 1-2, mostly "Món ngon mỗi ngày") get a Google-Search lookup of the dish
  from the raw description before planning, so the rewrite round can search the
  transcript for the dish name. For cooking Q&A, a second lookup starts from the top
  video's opening narration (where the host names the dish) and asks for the published
  recipe. Q&A whose answer is spoken or world knowledge gets a news lookup, because ASR
  misspells names ("Đèo Bứa" for "Đèo Tà Pứa"). Web answers end in an `ĐÁP ÁN:` line;
  `reconcile_answer` keeps a confident video reading, lets the web decide an unsure
  one, and for spoken answers takes the web's spelling of the same name. `--no-web`
  turns these off.
- **TRAKE paths are fused, re-found inside the video, then judged per event.** Round 1
  runs a tight temporal query (45 s maximum gap between events: every TRAKE key so far
  spans under ~20 s) and a loose one (300 s). Paths are ranked by both, plus how
  strongly the other channels support their video, since the query's context sentence
  (an intro shot) is not an event. The server's DP picks videos well but can take one
  event from minutes away, so each event is also searched inside the leading videos and
  the hits are joined in order by a client-side DP (`pool.local_path`). On round 2 this
  turned p2-8 from a miss (right video, last event 5 min late) into a hit. Verification scores a frame against the whole
  query, where an intro shot the query mentions for context can outscore the events
  themselves. Each of the server's top paths is therefore sent to Gemini as E1..En and
  scored against its own events before the best ones are refined.

### Why this shape

Drawn from agents built for the same kind of competition:

- *Can Agents Win the Video Browser Showdown?* (arXiv 2609.07311). An autonomous
  agent matched experts on textual KIS. The target was outside the initial top-100 in
  46% of solved tasks and was recovered by rewriting (78% of the time after an empty
  look) and by searching inside candidate videos. Most errors came from committing too
  early, and a separate check rejected a third of wrong submissions. Hence the
  verify-before-commit loop, video-scoped search, and a human doing the final check.
- *Deep Video Discovery* (arXiv 2505.18079). A small set of search-centric tools at
  several granularities beat fixed workflows. It also named two failure traps: endless
  inspecting and repeating queries. Hence the step and time budgets and the
  duplicate-query guard.
- RAPID (AIC 2024), QUEST and MADTempo (AIC 2025). Parallel query variants, and
  resolving out-of-knowledge entities before searching. Hence the lookup step.
- The team's own benchmarks (`docs/reports/visual-holistic-paraphrase-benchmark-2026.md`,
  `visual-query-expansion-benchmark-2026.md`). Faithful paraphrases fused by *embedding
  mean* (caption weight 0.5) beat the single caption on dev and holdout. Rank fusion of
  the same paraphrases got worse with every one added, and shortened "aspect" captions
  lost outright. Hence one visual query per scene: the caption plus up to 4 faithful
  paraphrases, fused server-side.
- The deployed transcript index is built from cleaned transcripts that write numbers as
  digits. On the live server "130.000" found its video and "một trăm ba mươi ngàn" did
  not, so transcript queries keep digits (`AIC_AGENT_SPOKEN_NUMBERS=1` spells them out
  for a raw-ASR index).

## Server side: `POST /agent/search`

One request runs up to 12 sub-queries in parallel. A sub-query that fails reports its
own `error` instead of failing the batch.

```json
{"queries": [
  {"kind": "visual",   "id": "v0", "text": "red boat on a river", "paraphrases": ["a river with a red boat on it"],
                       "limit": 200, "video_ids": ["L22_V001"]},
  {"kind": "similar",  "id": "s0", "video_id": "L22_V001", "frame_idx": 1234},
  {"kind": "audio",    "id": "a0", "text": "cá nhám", "method": "keyword"},
  {"kind": "ocr",      "id": "o0", "text": "Nà Ní"},
  {"kind": "temporal", "id": "t0", "events": ["...", "..."], "audio_text": "...", "max_gap_seconds": 180}
]}
```

- `paraphrases` (up to 4) are encoded with `text` and fused into one query vector by
  `backend.subquery_fusion.combine_vectors` (the frozen `B*` weights), then searched once.
- `video_ids` scores *every* keyframe of those videos: it reconstructs their Flat index
  rows instead of filtering a corpus-wide top-k.
- Hits are compact: `video_id`, `frame_idx`, `time`, `score`, `rank`, plus `text`,
  `start` and `end` for transcript and OCR hits. Temporal hits are videos with their
  best 3 paths.
- It runs under the coordinator identity `<login>#agent`. The agent can therefore share
  a teammate's login without the two cancelling each other's searches (latest-wins
  answers 409 when one login overlaps itself). The agent serializes its own batches.

### Servers without `/agent/search`

With `AIC_AGENT_SEARCH_API=auto` (the default), the first 404 from `/agent/search`
switches the client to `agent/compat.py`. That module maps every sub-query onto the
public `/search`, `/search/image` and `/search/temporal` routes, paging `/search/next`
for video-scoped queries. The agent runs unchanged, with three costs:

- Sub-queries run one at a time. Each audio query pools its full 10k-result snapshot,
  about 5-10 s on the team server.
- Paraphrases are dropped: the public routes take one text, and rank-fusing paraphrases
  measured worse than the caption alone.
- Searches run under the caller's own login, so a teammate searching in the UI with
  that login can supersede them. The client retries 409s.

## Setup

```bash
uv sync --only-group server --only-group agent   # or add --group agent to your usual sync
```

Configuration comes from the environment, falling back to the repository `.env`
(gitignored):

| Variable | Default | |
| --- | --- | --- |
| `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` | required | |
| `AIC_AGENT_URL` | `http://127.0.0.1:8000` | The team server, e.g. its `https://<host>.<tailnet>.ts.net` name |
| `AIC_AGENT_USER`, `AIC_AGENT_PASSWORD` | unset | Basic Auth, only for a deployment that still uses it. Over Tailscale, `tailscale serve` identifies you and no credentials are needed |
| `AIC_AGENT_TRUSTED_USER` | unset | Direct-to-uvicorn development only: sent as `X-AIC-User` |
| `DEEPSEEK_MODEL` / `GEMINI_MODEL` | `deepseek-flash` / `gemini-3.8-flash` | |
| `DEEPSEEK_THINKING` | `enabled` | Thinking mode for deep-mode loop turns (fast mode never thinks) |
| `DEEPSEEK_PLAN_THINKING` | `disabled` | Planning takes 2-3 s without thinking and 12-28 s with it, and plan quality was comparable |
| `GEMINI_THINKING_LEVEL` | `low` | |
| `AIC_AGENT_SEARCH_API` | `auto` | `batch` requires `/agent/search`; `compat` always uses the public routes |
| `AIC_AGENT_SPOKEN_NUMBERS` | `0` | `1` spells digits out in transcript queries (raw-ASR index only) |

## Commands

```bash
python -m agent check                                    # server, DeepSeek and Gemini reachable?
python -m agent plan queries/SOTUYEN2-bo-de-thi.zip      # plans + round-1 queries only; no server needed
python -m agent ask "Nhiều con rùa nhỏ đang ăn rau cải" --kind kis
python -m agent run queries/SOTUYEN1-bo-de-thi.zip --answers queries/Round12026_answer.zip
python -m agent run queries/SOTUYEN1-bo-de-thi.zip --mode deep --only query-p1-9-qa   # hard case, minutes
python -m agent eval artifacts/agent/<run> --answers queries/Round12026_answer.zip --tolerance 250
```

`run` and `ask` write `artifacts/agent/<timestamp>/`:

- `<query>.json`: plan, ranking with verdicts and evidence, answer, and the full tool trace.
- `submission/<query>.csv` and `submission.zip`, in the organisers' format (at most 100
  rows, answers quoted when needed).
- `report.html`: the top 20 candidates per query with thumbnails loaded from the
  server, for the final human check.
- `usage.json`: token counts per model. With `--answers`, also `scores.json`.

Labelled data for evaluation: `queries/round1-2025.csv` (answers to
`queries/query-p1-groupA`), `queries/Round12026_answer.zip` (answers to
`SOTUYEN1-bo-de-thi.zip`), and the team's visual benchmark for later rounds, which
carries gold frames but no Q&A answer strings:

```bash
python -m agent run queries/SOTUYEN2-bo-de-thi.zip   --answers "benchmarks/visual/visual_embedding_benchmark_2026_en.json::Round 2 - 2026"
```
 The organisers do not publish a frame tolerance, so
`eval` counts a frame within `--tolerance` frames (default 250, about the 10 s the team's
visual benchmarks use) of any gold frame as a hit, or anywhere between the two frames
of an organiser key that marks a clip by its first and last frame. Use its
numbers to compare agent versions, not to predict the leaderboard.

## Results so far

Same 10 s tolerance as `benchmarks/visual/visual_embedding_benchmark_2026_en.json`. The
PE-Core row is that benchmark's single human-approved caption per case. Its case sets
differ slightly (it scores TRAKE events separately), so read the comparison as
direction, not exact margin.

| Round (queries) | Method | R@1 | R@5 | MRR | Video@1 | Time per query (median / p90) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Round 1 (25) | PE-Core caption | 22.2% | 37.0% | 0.31 | | |
| Round 1 (25) | deep v2 / v3 (2026-09-18, compat) | 56% / 52% | 76% / 72% | 0.63 / 0.61 | 84% / 72% | 29 s / ~170 s |
| Round 1 (25) | fast v6 / v7 (2026-09-19, batch) | 48% / 48% | 60% / 68% | 0.54 / 0.56 | 60% / 72% | 18-20 s / 28-40 s |
| Round 2 (30) | PE-Core caption | 38.9% | 63.9% | 0.50 | | |
| Round 2 (30) | deep v3 (compat) | 66.7% | 73.3% | 0.70 | 86.7% | 99 s / ~250 s |
| Round 2 (30) | fast v6 | 56.7% | 73.3% | 0.64 | 76.7% | 19 s / 35 s |

- Fast mode costs roughly a tenth of R@1 for a 5x shorter run; `--mode auto` sits
  between them. R@5 is level, which matters most when a human checks the list.
- Round 2 was never used while tuning deep mode; fast mode's fixes (verification
  quota, local TRAKE paths, scene coherence) were found on both rounds.
- One query is 3-4 points, and LLM plans and verdicts vary between runs: the same
  fast code scored 40-56% R@1 on round 1 across runs. Treat a few points as noise.
- Round-1 retrieval is the same in both modes (the gold video is in a round-1 visual
  top 20 for 25 of 30 round-2 queries); what differs is how many candidates are looked
  at and re-searched.
- TRAKE: 2/2 on round 2 with the in-video local paths (previously 1/2). Round 1's p1-16
  (lion dance) still misses: the gold video reaches the top 4 only in some runs.
- Q&A answers: 2/4 on round 1 in the latest run. p1-17's pass name is fixed by the web
  lookup ("Đèo Bứa" in the ASR, "Đèo Tà Pứa" in the news) when the right video is on
  top; counting (p1-15: 10-11 read, 12 in the key) stays wrong even at Gemini's `high`
  thinking level, which also took 20-28 s. The team benchmark has no answer strings
  for round 2.
- Remaining failure types: right video but wrong frame in long lecture shots; confident
  look-alikes among same-genre videos (lion dances, cycling); VLM counting.

## Tuning

- Prompts: all in `agent/prompts.py`. Iterate with `python -m agent plan`, which is
  cheap and fast.
- Budgets: `--mode`, `--time-budget` (default 20 s fast, 300 s deep), `--max-steps`
  (deep), and `AgentOptions` / `FAST_DEFAULTS` in `agent/orchestrator.py`
  (`verify_top`, `verify_chunk`, `accept_score`, `early_stop_score`, `trake_refine`).
  The budget is soft: a step already started finishes, and a Q&A answer is always read.
- Fusion weights: `KIND_WEIGHTS`, `POINT_WINDOW_SECONDS` and `VIDEO_WEIGHT` in
  `agent/pool.py`.

## Not done yet

- **Shot captions** as a search channel, once they are indexed server-side. Add a
  `caption` kind to `/agent/search` and a plan field.
- **Image exemplars** for out-of-knowledge entities (QUEST/MADTempo style): fetch a web
  image and search with `/search/image`.
- **Group priors**: past answer keys show L26 is cooking, L25 lectures, L24 lion dance
  and L23 cycling. A team-maintained group description could bias the plan.
- **UI integration**: pushing the ranking into the collaboration workspace instead of a
  static report.
