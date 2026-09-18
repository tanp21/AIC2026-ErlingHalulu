# Retrieval agent

`python -m agent` takes a Vietnamese competition query (KIS, Q&A or TRAKE), plans
searches across every channel the server offers, checks candidates by looking at
them, and writes a ranked, submission-ready answer for a teammate to confirm.

The agent is a client of the search server: it runs anywhere (a laptop is fine),
calls the server over HTTP and never ships in the production image.

## How one query is answered

```text
query ──► DeepSeek plan ──► round 1: one /agent/search batch ──► Gemini verifies top 12
            (JSON)            visual (EN captions, 2-3 per scene)         │
                              temporal (multi-scene / TRAKE)              │ best ≥ 9 (and, for Q&A,
                              transcript (VI, numbers spelled out)        │ an answer read)? ── yes ──► finalize
                              OCR (distinctive on-screen strings)         │ no
            Gemini + Google Search for names the query only describes     ▼
                                                              DeepSeek tool loop (≤ 6 steps, ≤ 300 s)
                                                              search · inspect · neighbours ·
                                                              read_transcript · web_search · finish
                                                                          │
finalize: ranking (verified matches first) · Q&A answer read from the frame/speech ·
          TRAKE: each event moved to the first neighbouring keyframe that shows it
```

- **Round 1 is deterministic** after the plan: `agent/plan.py` turns the plan into one
  batch, target scene first, so a query with a clear match needs no agent turns at all.
- **Candidates are moments, not hits** (`agent/pool.py`). Keyframe hits within 4 s of
  each other merge; transcript passages and OCR shots back every moment inside their
  span. Ranking is reciprocal-rank fusion plus a video-level term, then Gemini verdicts
  override it: verified matches (≥ 6) first, contradicted ones (≤ 3) last.
- **Gemini looks, DeepSeek decides.** Gemini scores labelled keyframes 0-10 against the
  original Vietnamese query, reads Q&A answers, and picks exact TRAKE frames. DeepSeek
  plans, rewrites queries and chooses tools. Neither model is the source of a Q&A answer
  on its own: answers are read from the video, and web search only resolves names or
  cross-checks.

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
  resolving out-of-knowledge entities before searching. Hence 2-3 caption variants per
  scene and the lookup step.

## Server side: `POST /agent/search`

One request runs up to 12 sub-queries in parallel. A sub-query that fails reports its
own `error` instead of failing the batch.

```json
{"queries": [
  {"kind": "visual",   "id": "v0", "text": "red boat on a river", "limit": 200, "video_ids": ["L22_V001"]},
  {"kind": "similar",  "id": "s0", "video_id": "L22_V001", "frame_idx": 1234},
  {"kind": "audio",    "id": "a0", "text": "cá nhám", "method": "keyword"},
  {"kind": "ocr",      "id": "o0", "text": "Nà Ní"},
  {"kind": "temporal", "id": "t0", "events": ["...", "..."], "audio_text": "...", "max_gap_seconds": 180}
]}
```

- `video_ids` scores *every* keyframe of those videos: it reconstructs their Flat index
  rows instead of filtering a corpus-wide top-k.
- Hits are compact: `video_id`, `frame_idx`, `time`, `score`, `rank`, plus `text`,
  `start` and `end` for transcript and OCR hits. Temporal hits are videos with their
  best 3 paths.
- It runs under the coordinator identity `<login>#agent`. The agent can therefore share
  a teammate's login without the two cancelling each other's searches (latest-wins
  answers 409 when one login overlaps itself). The agent serializes its own batches.

## Setup

```bash
uv sync --only-group server --only-group agent   # or add --group agent to your usual sync
```

Configuration comes from the environment, falling back to the repository `.env`
(gitignored):

| Variable | Default | |
| --- | --- | --- |
| `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` | required | |
| `AIC_AGENT_URL` | `http://127.0.0.1:8000` | The team server (HAProxy) URL |
| `AIC_AGENT_USER`, `AIC_AGENT_PASSWORD` | unset | HAProxy Basic Auth login |
| `AIC_AGENT_TRUSTED_USER` | unset | Direct-to-uvicorn development only: sent as `X-AIC-User` |
| `DEEPSEEK_MODEL` / `GEMINI_MODEL` | `deepseek-flash` / `gemini-3.8-flash` | |
| `DEEPSEEK_THINKING` | `enabled` | Thinking mode for tool-loop turns |
| `DEEPSEEK_PLAN_THINKING` | `disabled` | Planning takes 2-3 s without thinking and 12-28 s with it, and plan quality was comparable |
| `GEMINI_THINKING_LEVEL` | `low` | |

## Commands

```bash
python -m agent check                                    # server, DeepSeek and Gemini reachable?
python -m agent plan queries/SOTUYEN2-bo-de-thi.zip      # plans + round-1 queries only; no server needed
python -m agent ask "Nhiều con rùa nhỏ đang ăn rau cải" --kind kis
python -m agent run queries/SOTUYEN1-bo-de-thi.zip --answers queries/Round12026_answer.zip
python -m agent eval artifacts/agent/<run> --answers queries/Round12026_answer.zip --tolerance 50
```

`run` and `ask` write `artifacts/agent/<timestamp>/`:

- `<query>.json`: plan, ranking with verdicts and evidence, answer, and the full tool trace.
- `submission/<query>.csv` and `submission.zip`, in the organisers' format (at most 100
  rows, answers quoted when needed).
- `report.html`: the top 20 candidates per query with thumbnails loaded from the
  server, for the final human check.
- `usage.json`: token counts per model. With `--answers`, also `scores.json`.

Labelled data for evaluation: `queries/round1-2025.csv` (answers to
`queries/query-p1-groupA`) and `queries/Round12026_answer.zip` (answers to
`SOTUYEN1-bo-de-thi.zip`). The organisers do not publish a frame tolerance, so
`eval` counts a frame within `--tolerance` frames of the gold clip as a hit. Use its
numbers to compare agent versions, not to predict the leaderboard.

## Tuning

- Prompts: all in `agent/prompts.py`. Iterate with `python -m agent plan`, which is
  cheap and fast.
- Budgets: `--max-steps`, `--time-budget`, and `AgentOptions` in
  `agent/orchestrator.py` (`verify_top`, `early_stop_score`, `trake_refine`).
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
