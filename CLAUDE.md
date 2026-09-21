# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A multimodal video retrieval system for VBS-style competitions (AIC HCMC). A FastAPI backend serves a React/Vite SPA and answers visual, audio (Vietnamese transcripts), hybrid, OCR (on-screen text), and temporal queries against pre-built FAISS and Elasticsearch indexes. Deployed to a five-person team through direct IPv4 port forwarding and HAProxy Basic Auth.

Detailed setup, deployment, and ops procedures live in `docs/`; this file covers what you need to work in the code.

## Commands

Run everything from the repository root.

```bash
# Backend tests (stdlib unittest — there is no pytest config)
# -t . is required: it makes `tests` a proper package, so its domain subfolders
# (tests/audio, tests/ocr, ...) don't collide with the same-named scripts/* subfolders.
.venv/bin/python -m unittest discover -s tests -t . -v

# A single module / class / test
.venv/bin/python -m unittest tests.temporal.test_temporal_search
.venv/bin/python -m unittest tests.temporal.test_temporal_api.TemporalRequestTests
.venv/bin/python -m unittest tests.test_access_control.AccessMiddlewareTests.test_rejects_user_outside_allowlist

# Frontend
cd frontend && npm test -- --run      # vitest, non-watch
cd frontend && npm run typecheck      # tsc -b
cd frontend && npm run build          # emits ../static/dist, which the backend serves
cd frontend && npm run dev            # Vite dev server, proxies /search /thumb /frame /video /indexes to :8000
```

`uv run python -m unittest ...` also works; the repo convention is to prefix `uv` invocations with
`UV_CACHE_DIR=/tmp/aic_uv_cache` (see `docs/getting-started.md`).

Running the backend locally requires the access config, or startup fails with a `RuntimeError`:

```bash
export AIC_ALLOWED_USERS=owner@example.com,u2@example.com,u3@example.com,u4@example.com,u5@example.com
export AIC_ADMIN_USER=owner@example.com
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Full stack (production path): `export AIC_UID=$(id -u) AIC_GID=$(id -g) AIC_HF_HUB_DIR="$HOME/.cache/huggingface/hub" AIC_VLM_DIR="$HOME/.cache/aic/vlm"` then `docker compose up -d --build --wait`.

Offline tooling runs as modules, never as file paths:

```bash
uv run python -m scripts.indexing.build_flat_profile --help   # creates indexes/profiles/*
uv run python -m scripts.indexing.build_audio_index --help    # populates Elasticsearch
uv run python -m scripts.indexing.build_ocr_index --help      # populates the OCR Elasticsearch index
uv run python -m scripts.indexing.build_caption_index --help  # populates the shot-caption Elasticsearch index
uv run python -m scripts.indexing.build_frame_maps --help     # creates indexes/frame_maps/*.npz
uv run python -m scripts.extraction.detect_shots --help       # creates data/<year>/shots/*.json
uv run python -m scripts.embedding.embed_keyframes --help

# Local VLM snapshots for AIC_AGENT_VISION=local (needs the network; the container
# is offline and read-only, so this runs here and the result is mounted in)
uv run python -m scripts.vlm.fetch_models --help              # creates ~/.cache/aic/vlm/*
uv run python -m scripts.benchmarks.bench_agent_vision --help # measures candidates on the agent's own job
```

The retrieval agent is an HTTP client of the server. The team runs it from the **Retrieval agent** UI tab (server-side, keys pasted per run); as a CLI it runs on a laptop (`uv sync --group agent`, keys in that machine's `.env`; see `docs/agent.md`):

```bash
uv run python -m agent plan queries/SOTUYEN1-bo-de-thi.zip        # planning only, no server needed
uv run python -m agent run queries/SOTUYEN1-bo-de-thi.zip --answers queries/Round12026_answer.zip   # fast, ~20 s/query; --mode auto|deep spend more
```

Once per clone: `git config core.hooksPath .githooks` — the pre-commit hook is a fail-closed Gitleaks scan.

## Architecture

### Retrieval modes

| Mode | Pipeline | Entry point |
| --- | --- | --- |
| Visual (English) | PE-Core text/image embedding → FAISS inner-product index | `visual_search` / `visual_vector_search` in `backend/app.py` |
| Audio (Vietnamese) | Qwen3-Embedding-0.6B kNN + BM25 in Elasticsearch, merged by RRF | `backend/audio_search.py` |
| Hybrid | Both branches in parallel `ThreadPoolExecutor`, fused around matching video timestamps | `fuse_search_results` in `backend/app.py` |
| OCR (on-screen text) | ASCII-folded BM25 keyword match against one OCR'd keyframe per shot | `ocr_search_response` / `backend/ocr_search.py` |
| Caption (shot descriptions) | BM25 over one CapRL caption per shot window (optional Qwen3 kNN + RRF); agent-only so far | `backend/caption_search.py`, `_agent_caption_hits` in `backend/app.py` |
| Temporal | Batch-encode 2–6 ordered events → per-event candidates → DP over the best prefix per endpoint → optional audio RRF → temporal NMS | `backend/temporal_search.py` |

`SearchReq.mode` is `Literal["visual", "audio", "hybrid", "ocr"]`, but `ocr_text` is also a cross-cutting filter: any non-empty `ocr_text` on a `visual`/`audio`/`hybrid` request runs OCR search in parallel and restricts results to hits whose shot overlaps an OCR match (`_with_ocr_filter` / `filter_search_results` in `backend/app.py`) — it doesn't just switch the mode. The frontend simplifies this further: `App.tsx`'s `MODE_OPTIONS` only exposes `hybrid` and `temporal` as top-level toggles, then derives the actual `visual`/`audio`/`ocr`/`hybrid` request mode from which of the visual/audio/OCR fields are filled in.

Hybrid degrades gracefully: if the audio branch raises, the visual results are returned with a `warning` field rather than an error. OCR degrades the opposite way — `_with_ocr_filter` raises **503** if the OCR branch fails, since a filter that silently no-ops would misrepresent the results as OCR-matched.

### Module map

- `backend/app.py` — all routes, request models, orchestration, and the snapshot stores. The largest file; start here.
- `backend/core.py` — PE-Core model loading (lazy, lock-guarded), text/image encoding, and `ProfileStore`. **Builds a module-level `PROFILE_STORE` at import time**, so `from backend import core, app` touches `indexes/` on disk. This is why tests mock `backend.core.*` and `backend.app.*` symbols rather than constructing real indexes.
- `backend/access_control.py` — the trusted-ingress ASGI middleware and `LatestWinsCoordinator` (per-user rate limiting + latest-request-wins scheduling).
- `backend/audio_search.py` — Elasticsearch index definition, query builders, passage construction, transcript cues. `transcript_path()` reads the raw ASR parquet unless `AIC_TRANSCRIPT_PATH` names another file; cleaned transcripts are opt-in, never adopted just by existing on disk.
- `backend/caption_search.py` — the caption half of Elasticsearch: BM25 over the offline
  CapRL shot captions (`docs/caprl-shot-captioning.md`), one document per caption window,
  built by `scripts/indexing/build_caption_index.py`. Dense vectors are optional and off by
  default (`--dense`), so the channel ships without an embedding pass over 103k documents.
  It reaches what CLIP cannot express — counts, the order of actions inside one shot, and
  sentences written on a slide — and is optional at runtime: with no index, the agent's
  caption sub-queries fail alone and the channel is dropped for that run.
- `backend/ocr_search.py` — the OCR half of Elasticsearch: BM25-only (no dense vector — on-screen text is short overlays meant for literal keyword match), ASCII-folded analyzer, one document per shot built from its representative keyframe. Mirrors `backend/audio_search.py`'s query-building style. See `docs/reports/ocr-shot-audit.md` for why one keyframe per shot and what accuracy bar was accepted.
- `scripts/audio/` — offline transcript cleaning (not in the image). `postprocess.py` holds pure transforms over a timed word stream, `pipeline.py` fixes their order, `capu.py` is the only part that loads a model. Every transform is off unless a benchmark justified it; most did not — see `docs/reports/audio-transcript-cleaning.md`.
- `scripts/ocr/` — offline OCR extraction (not in the image): `engines.py`/`alt_engines.py` wrap the candidate OCR backends, `run_corpus.py`/`run_corpus_remote.py` drive the full-corpus pass, `postprocess.py` suppresses burned-in furniture (logos, tickers) before `scripts/indexing/build_ocr_index.py` consumes the output.
- `backend/temporal_search.py` — pure functions: candidate dedup, sequence DP, scoring methods, audio fusion, NMS. No I/O; easy to test directly.
- `backend/collaboration.py` — SQLite-backed competition workspace (`/api/collab/*`): round import from ZIP, per-query assignment/answers with optimistic-concurrency `version` fields, and CSV/ZIP submission export.
- `backend/agent_runner.py` — server-hosted retrieval-agent runs for the UI (`/api/agent/*`): an in-memory owner-scoped run registry (one active run per user, 1 h TTL), a bounded executor, and per-run DeepSeek/Gemini keys that are never persisted. Each run drives the `agent/` orchestrator (`--mode fast` by default, selectable per run) over loopback (`AIC_AGENT_INTERNAL_URL`, identity `<login>` → the server's `<login>#agent` coordinator slot) and writes its JSON, `report.html` and `submission.zip` under `AIC_AGENT_OUT_DIR`. While a run is live, each poll carries a `progress` snapshot of what Gemini is judging (with the same context snippets), and `POST /api/agent/runs/{id}/review` lets the owner accept/reject a candidate, type a Q&A answer or stop; the orchestrator applies these as verdicts no model can overwrite and stops waiting on Gemini (`agent/review.py`).
- `backend/search_core.py` — compute-plane access. With `AIC_CORE_URL` unset every primitive calls `core`/`frame_maps` in-process; with it set (a teammate's machine) encode/FAISS/annotate work is forwarded to the owner's `/core/*` endpoints. Local branches go through module attributes so existing `backend.app.core.*` test patches keep applying. See `docs/split-development.md`.
- `backend/core_service.py` — the other half of the split: `/core/*` endpoints (owner side) and `CoreProxyMiddleware`, which forwards media/data routes (`/thumb`, `/frame`, `/video`, `/clip`, `/shots`, `/frame-map`, `/transcript`, `/context`, `/videos`, `/indexes`, `/profiles`) to `AIC_CORE_URL` (teammate side). Installed before `install_access_control` so the proxy sits inside the identity/limits middleware; registered with `add_api_route` so `app.routes` stays iterable.
- `backend/shots.py` — TransNetV2 shot boundaries (`data/<year>/shots/<group>/<id>.json`) plus exact-PTS
  clip extraction for `/videos`, `/shots/{id}`, `/clip/{id}/{shot}`. Boundaries are generated
  offline by `scripts.extraction.detect_shots`; the 2026 dataset has none until that runs. Also
  owns `media_catalogue()`, the on-disk video scan that `backend.app.resolve()` falls back to — this is
  why every media route serves the 2025 `K*` videos that no embedding profile covers. See
  `docs/clip-export.md` for the shot-dashboard/clip-export UI this backs. Also serves the CapRL shot
  captions (`caption_for_frame`, `GET /captions/{video_id}`) that the retrieval agent reads as text
  context. Read one video's `data/2026/captions/parts/<group>/<id>.parquet` at a time behind an LRU:
  the corpus-wide `shot_captions.parquet` is a single row group, so filtering it does not help and it
  costs ~5 s and ~900 MB resident. Only 2026 was captioned; `K*` videos answer `available: false`.
- `backend/shot_labels.py` — reviewer-facing shot exclusions (`intro`/`preview`/`presenter`/`outro`/`other`), stored one JSON-per-video under `artifacts/shot-labels/` and served through `PATCH /shots/{video_id}/labels`. Optimistic concurrency via a `revision` counter; a `shots_digest` hash ties a label document to the exact shot boundaries it was made against, so re-detecting shots invalidates stale labels (`LabelConflict`) instead of silently mislabeling. `scripts/indexing/suggest_shot_exclusions.py` pre-populates suggestions for a reviewer to confirm — see `docs/shot-exclusions.md`. This excludes shots from the visual index without touching their embeddings or the transcript/OCR indexes.
- `backend/frame_maps.py` — PTS arrays per video; the source of truth for frame ↔ time.
- `agent/` — the LLM retrieval agent (`python -m agent` and the UI's server-hosted runs). DeepSeek plans and drives tools, Gemini verifies keyframes and does Google-Search lookups; it talks to the server only through `POST /agent/search` (a parallel multi-channel batch in `backend/app.py`, run under the coordinator identity `<login>#agent` so it never supersedes the human's own searches) plus the `/thumb`, `/frame`, `/context`, `/captions` and `/transcript` routes. `agent/pool.py` (moment fusion) and `agent/plan.py` (plan → round-1 batch) are pure and carry most of the logic. See `docs/agent.md`.
- `agent/vlm.py` — the vision backends: `GeminiClient`'s judgement role, or a VLM on this machine (`AIC_AGENT_VISION=local`), in-process or over HTTP (`AIC_VLM_URL`, the same local/remote split as `backend/search_core.py`). All five jobs in `agent/vision.py` reach a model through one `judge()` call, so the backend is swappable without touching them. It renders the response schema back into words (the prompts leave the output shape to Gemini's `response_json_schema`), fans the per-image jobs out one frame per call with a short Vietnamese prompt, and stamps ids from the label rather than trusting the reply. `agent/vlm_models.py` loads the three candidates. Weights are materialised by `scripts/vlm/fetch_models.py` and mounted read-only — the container cannot download them.
- `scripts/benchmarks/bench_agent_vision.py` — measures candidate local VLMs on the agent's own verification job: one shared candidate pool built from real searches, scored by how well each model's 0-10 verdicts separate gold from non-gold. End-to-end R@1 is deliberately not the metric; the agent's own docs record it varying 40-56% across runs of identical code.
- `backend/thumb.py` — exact-PTS ffmpeg thumbnail extraction with a bounded semaphore and a size-capped LRU disk cache.

### Snapshot pagination (not offset pagination)

`/search` and `/search/temporal` compute the full ranked pool once, store it in a `SearchSnapshotStore`, and return the first page plus an opaque `next_cursor`. `/search/next` and `/search/temporal/next` walk that snapshot.

- One live snapshot per user; issuing a new search drops the previous one.
- 600 s TTL, refreshed on each page fetch; an expired or foreign cursor returns **410**.
- Caps: 1000 results pooled; 100 results per visual page, 20 video groups per temporal page.
- `TemporalSnapshotStore` subclasses the store and renames `results`/`total_results` to `videos`/`total_videos` on the way out.

### Access model

`TrustedIngressAccessMiddleware` wraps every request. `/healthz` is the only anonymous route; everything else requires the internal `X-AIC-User` header set by HAProxy, whose value must be in `AIC_ALLOWED_USERS`. `AccessConfig.from_env()` rejects anything other than **exactly five unique logins** with `AIC_ADMIN_USER` among them.

For the Tailscale-only deployment (no public port forwarding, no Basic Auth), `tailscale serve` on the host terminates the tailnet HTTPS connection and proxies to `127.0.0.1:8000`, adding a `Tailscale-User-Login` header naming the connecting tailnet member. HAProxy copies that header into `X-AIC-User` — but only when the request arrived over loopback/the Docker bridge (`acl local_proxy` in `haproxy.cfg`); anything else is rejected with 403, and `compose.yaml` publishes the ingress port as `127.0.0.1:8000` so nothing external can reach it directly anyway. Each `AIC_ALLOWED_USERS` entry must be a teammate's real Tailscale account login. See `docs/team-deployment.md`.

The middleware also enforces per-route body limits before the handler runs (16 KiB default, 8 MiB for `POST /search/image`, 256 KiB for `POST /agent/search`, 1 MiB for an owner's `POST /api/collab/rounds`), buffers the body with a 10 s timeout, holds global + per-user semaphores for `/video/*` and `/thumb|/frame/*`, and emits an `audit user=… path=… status=…` log line per request.

Owner-only (`request.state.is_admin`): non-`flat` indexes, round upload, round activation.

### Index profiles

`indexes/profiles/manifest.json` defines named profiles (currently `l2025`, `l2026`) with a default. When the manifest is absent, `ProfileStore` falls back to the legacy full-dataset layout directly in `indexes/`. Profile directories are resolved and checked for escapes from the profile root. FAISS indexes are lazily loaded into a bounded LRU keyed by `(profile, index_name)`.

### Frame identity

Frame ↔ time conversion goes through `backend.frame_maps.FrameMap` (PTS + time base), never through fps arithmetic. The same map is shipped to the browser via `GET /frame-map/{video_id}` and reimplemented in `frontend/src/frameMap.ts`; changes to one must mirror the other. `backend.thumb.ffmpeg_command` seeks by exact PTS with `-copyts`.

### Competition workflow

`RULES_GUIDELINES.md` (Vietnamese, from the organizers) defines the three query kinds — KIS, Q&A, TRAKE — and their CSV submission formats. `backend/collaboration.py` implements this end to end: it parses a ZIP of `<name>-{kis,qa,trake}.txt` query files into an active round, tracks per-query assignees and answers, and exports a ZIP of `submission/*.csv` in the required format. Frontend counterpart: `frontend/src/collab.tsx` and `collabApi.ts`.

## Conventions and gotchas

- Backend tests are stdlib `unittest` with `unittest.mock.patch` targeting `backend.app.*` / `backend.core.*` symbols. Frontend tests are Vitest + Testing Library (`frontend/src/test/setup.ts`).
- Some tests assert repository invariants rather than behavior — `tests/test_frontend_serving.py` checks the Dockerfile mentions `backend/temporal_search.py` and `benchmarks/temporal/temporal_config.json`; `tests/test_runtime_mounts.py` checks mount-derived paths (including `backend/shot_labels.py`); `tests/test_repository_security.py` checks the Gitleaks setup. **A new top-level runtime module must be added to the Dockerfile `COPY` line.**
- `benchmarks/temporal/temporal_config.json` (candidate K + scorer) is generated by `scripts/benchmarks/bench_temporal_search.py` against `benchmarks/temporal/temporal_ground_truth.json`. Re-run the benchmark instead of hand-editing it. `benchmarks/audio/audio_config.json` and `benchmarks/ocr/ocr_config.json` work the same way, from `bench_audio_search --sweep`/`bench_audio_capu` and `bench_ocr.py` respectively.
- Offline, GPU-heavy, or dependency-conflicting tooling gets its own venv outside the main `.venv` and outside `pyproject.toml`'s tracked groups — e.g. `.venv-captioning` (CapRL shot captioning, `requirements-captioning.txt`, see `docs/caprl-shot-captioning.md`) and `.venv-ocr` (the production `OcrPipeline` in `scripts/ocr/engines.py`, used by `scripts/indexing/build_ocr_index.py`). Neither ships in the production image. The PaddleOCR/EasyOCR/VLM-OCR bake-off venvs from the engine comparison (`docs/reports/ocr-shot-audit.md`) were deleted once the comparison concluded — reinstall from that report if reproducing it.
- `benchmarks/audio/audio_ground_truth.json` is 48 real competition queries with their gold video, recovered from the round-1 answer keys in `queries/`. One query is 2.08 points, so treat any single-query difference as noise. Note that BM25 lowercases and strips punctuation in both the bench and Elasticsearch, so capitalisation and punctuation changes are only visible under `--dense`.
- The agent's dependencies are pinned in both the `agent` group (CLI-only installs) and the `server` group, because `backend/agent_runner.py` imports `agent/` in-process. Server-hosted runs need outbound HTTPS to DeepSeek and Gemini, so the `app` service joins the `aic-egress` network; every other service stays on the internal network.
- Split-host development: `AIC_CORE_URL` switches `backend/search_core.py` to remote mode and activates `CoreProxyMiddleware`, so a teammate's app runs with no `data/`, `indexes/`, models or Elasticsearch while the owner's host stays the compute plane. Production keeps `AIC_CORE_URL` unset. See `docs/split-development.md`.
- A new runtime dependency must go in **both** `[project].dependencies` and `[dependency-groups].server` in `pyproject.toml` — the production image installs only the `server` group (no TensorFlow, no notebook toolchain).
- The container runs read-only and offline for models (`HF_HUB_OFFLINE=1`); only the `aic-egress` network gives the app outbound HTTPS (agent runs). Models must already be in the host Hugging Face cache. Writable paths are `/cache` (thumbnails), `/workspace` (collab SQLite + agent run output), and `tmpfs /tmp`.
- `data/` (a symlink) and `indexes/` are gitignored local mounts. Never commit them, `.env.secure`, or generated artifacts.
- Startup calls `warmup_search()`, which loads metadata, PE-Core, the flat FAISS index, and Qwen — expect 45–60 s before `Application startup complete`.

## Further reading

`docs/getting-started.md` (setup, indexing, local dev) · `docs/audio-search.md` · `docs/temporal-search.md` · `docs/clip-export.md` (per-shot clips for captioning) · `docs/shot-exclusions.md` (reviewer shot labelling) · `docs/caprl-shot-captioning.md` · `docs/agent.md` (retrieval agent) · `docs/team-deployment.md` (direct-IP deployment) · `docs/operations.md` (failure playbook). Methodology write-ups for offline decisions (audio cleaning, OCR engine choice, hybrid fusion tuning, **which vision backend the agent should judge with**) live under `docs/reports/` — the last is `docs/reports/agent-vision-backends-2026.md`.
