# Audio transcript search

The app indexes the 873 transcripts for L21-L30 as **duration-bounded semantic
passages** (`--strategy semantic_duration`, `build_audio_index.py`'s default)
and searches them with Elasticsearch BM25 plus dense kNN. Python combines the
two rankings with reciprocal rank fusion, so Elasticsearch Basic is
sufficient. Query and passage embeddings use `Qwen/Qwen3-Embedding-0.6B`;
visual search remains on the existing CLIP/FAISS indexes.

## Chunking strategy

Passages are grouped along sentence boundaries wherever a topic shift is
detected (a windowed change-score over sentence embeddings, local-max
candidates within the top 15th percentile per video — see
`backend.audio_search.make_bounded_semantic_passages`), constrained to
`min_duration_seconds=10` / `target_duration_seconds=40` /
`max_duration_seconds=75`, with 1-sentence overlap between adjacent chunks.
This replaced the earlier fixed 60-second/15-second-stride sliding window.

The chosen bounds come from a **hybrid** benchmark (44 KIS/QA queries from
`benchmarks/audio/audio_ground_truth.json`; real PE-Core+FAISS visual search;
`backend.app.fuse_search_results` after its adaptive per-hit `max_time_delta` fix,
`backend/app.py:964`) — not an audio-only benchmark. An earlier audio-only-only sweep
favoured a shorter config (min=8/target=25/max=40) because longer chunks'
coarse anchor timestamp fell outside `fuse_search_results`' old *fixed* 30s
fusion window; that was an artifact of the fusion window, not evidence
against longer chunks. Once fixed, the current bounds tie or beat every
shorter config tried on hybrid recall while indexing the fewest passages
(8,361 vs 30,547 for the old sliding baseline). A follow-up benchmark against
the 4 TRAKE queries, run through the real `backend/temporal_search.py` per-event DP
(not just gold-video recall), found no meaningful regression: the DP's
per-event localization is driven entirely by visual candidates and is
identical across chunking strategies — audio only nudges the final rank via
RRF, and it does so *better* under the current bounds for the one TRAKE video
with substantive narration (rank 2→1 vs. the old sliding default). Full
numbers in `benchmarks/audio/semantic_chunk_config.json`.

## Pause-aware boost (Phase 2, experimental — off by default)

`SemanticChunkConfig.pause_boost_enabled` (`--pause-boost` / `--no-pause-boost`,
default **off**) lets an unusually long pause between sentences — relative to
that video's own pause distribution, `pause_relative_percentile` — influence
which boundary is chosen once a chunk is already in `[target_duration,
max_duration]`. Two mechanisms were benchmarked (fixed config C throughout,
44 KIS/QA queries, full numbers and manual-review notes in
`benchmarks/audio/semantic_chunk_config.json`'s `pause_boost_check`):

- **Ungated** (`strong_semantic OR pause_boosted` — any long pause is eligible
  on its own, no semantic support required): retrieval doesn't clearly
  regress, but pause ends up driving 70–86% of *all* chunk boundaries in the
  corpus, and manual reading of a 12-boundary sample across news, cooking,
  travel/documentary, and sports content found only about half were real
  topic transitions — the rest were just a breath or rhythm pause inside one
  continuing statement, especially in interview-style news and live sports
  commentary. **Rejected as a production candidate** — the mechanism overrides
  the semantic signal far more than it augments it. The code path still
  exists (`backend.audio_search._assemble_bounded_chunks(..., borderline_candidates=None)`)
  but only as a directly-callable function for historical ablation/testing —
  it is **not reachable from any CLI flag or from
  `make_bounded_semantic_passages`**, so it can never run in production.
- **Gated** (`strong_semantic OR (borderline_semantic AND pause_boosted)`,
  the only pause behavior `pause_boost_enabled=True` actually runs — a second,
  wider `_breakpoint_candidates` percentile, `pause_borderline_semantic_percentile`
  (default 30.0), gates the pause signal so it can only *promote* a boundary
  the semantic score was already borderline on, never manufacture one alone):
  every promoted boundary is confirmed (100% in benchmark) to independently
  clear the borderline-semantic bar, the pause-influenced share of all
  boundaries drops to 37–46%, and manual review quality rises to 7/10 (70%)
  real topic transitions — a clear improvement over the ungated mechanism.
  Retrieval, however, does not improve over pause-off by a clear enough margin
  on this 44-query benchmark to justify shipping it: the headline recall dip
  traces mostly to one boundary-position-sensitive query (root-caused in the
  benchmark notes), but 44 queries (~2.3 points each) can't rule out a real,
  small effect either way.

**Decision**: `pause_boost_enabled` stays `False` in production (both
`SemanticChunkConfig`'s default and both CLIs' argparse default). The gated
implementation is fully tested and kept in the codebase as an optional
feature — pass `--pause-boost` to exercise it — for re-benchmarking later
(e.g. against a larger ground truth) rather than enabling it now.

## One-time online preparation

Docker Desktop must be running with WSL integration enabled. The Elasticsearch
container is limited to 4 GB with a 2 GB JVM heap. The PE-Core checkpoint needs
more memory than WSL2's default allocation on a 32 GB host. Configure
`C:\Users\<user>\.wslconfig` before loading Visual or Hybrid search:

```ini
[wsl2]
memory=24GB
swap=8GB
networkingMode=mirrored
```

Run `wsl --shutdown` from PowerShell after changing the file, reopen WSL, and
mount the data disk again. Then, from WSL:

```bash
docker compose pull
UV_CACHE_DIR=/tmp/aic_uv_cache uv sync
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -c \
  'from sentence_transformers import SentenceTransformer; SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")'
```

Pinning the Docker image in `compose.yaml` and keeping the Hugging Face cache
makes the competition runtime independent of network access.

## Transcript cleaning (optional)

The ASR output is unpunctuated, uncapitalised, and spells every number out.
`scripts/audio/` can repair that, writing `step_data_clean.parquet` and
`transcription_clean.parquet` into `data/2026/audio/generated/`.

**Generating a cleaned transcript does not deploy it.** The runtime reads the
raw parquet unless `AIC_TRANSCRIPT_PATH` names another file — writing an
artifact must never silently change what a running server serves:

```bash
export AIC_TRANSCRIPT_PATH=$PWD/data/2026/audio/generated/step_data_clean.parquet
```

Set it before `uvicorn` and before `build_audio_index`, or leave it unset to
stay on the untouched ASR output. `compose.yaml` sets this by default for the
`docker compose` deployment — restoration recovers proper nouns the raw ASR
garbles entirely (e.g. a cyclist's name coming out as `martinlas`/`martin lát`
instead of "Martin Laas"), which matters for keyword search even though it's
otherwise a subtitle-quality change. Local `uv run uvicorn` still needs the
export above to match.

Only **capitalisation and punctuation restoration** is enabled, via the
Gemini API (`scripts/audio/gemini_clean.py`) — it does not improve recall
either, but it takes subtitle cues from 0% to 50–74% ending at a real
sentence boundary. An earlier local-model restorer (`dragonSwing/xlm-roberta-capu`)
and a set of deterministic transforms (boilerplate removal, confidence
filtering, number normalisation) were retired: each moved gold-video recall
by at most one query out of 48, in both directions, and are no longer in the
codebase. Full numbers in `docs/reports/audio-transcript-cleaning.md`
(historical — describes the retired local-model pipeline).

**Cleaning repairs; it never removes.** The corpus's definition is narrow —
fix what the recogniser got wrong (spelling, tone marks, obvious mishearings),
add punctuation and capitalisation, change nothing else. Gemini does not
reliably obey that on its own: told only not to remove content, it went on
deleting university ad reads, song lyrics and repeated formulas, and folding
spoken formulas into notation. So the instruction now lists each of those
explicitly as forbidden, and two guards back it up in
`scripts/audio/gemini_clean.py`: a response is rejected (the raw words are kept
and the chunk retried on the next run) if it comes back cut short, or if it
drops more than `MAX_DELETED_PER_1000_WORDS` words per 1000. A chunk-level
length check alone was not enough — a long video can lose a 668-word stretch
and still come back at 98.6% overall. The prompt is part of the cache key, so
editing it invalidates every cached response rather than silently replaying
answers produced under the old wording.

```bash
export GEMINI_API_KEY=...

# Apply benchmarks/audio/audio_config.json's gemini_model (or pass --gemini-model
# directly). Resumable: a killed run only re-pays for chunks without a cached
# response (data/2026/audio/.gemini_cache/).
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -m scripts.audio.build_clean_transcripts

# Capitalisation and punctuation are invisible to BM25 — both this bench and
# Elasticsearch's standard analyser lowercase and drop punctuation — so they can
# only be measured against the embeddings. Compare the two parquets in a fresh
# process so the corpus embedder is not resident alongside anything else.
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -m scripts.benchmarks.bench_audio_search --dense \
  --transcripts data/2026/audio/step_data_clean.parquet \
  --baseline-transcripts data/2026/audio/step_data.parquet
```

Restoration needs network access and a `GEMINI_API_KEY`; every other step is
pure Python. Run this on the dev machine: the container is read-only and
offline (`HF_HUB_OFFLINE=1`), and `GEMINI_API_KEY` is a dev-machine-only
secret that is never part of `.env.secure` or read by anything under the
Docker image.

## Build the transcript index

```bash
docker compose up -d --wait
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -m scripts.indexing.build_audio_index --recreate
```

With no `--strategy` flag this now builds `semantic_duration` passages using
the bounds described above. To rebuild against the old fixed sliding window
instead (e.g. to reproduce a pre-migration comparison), pass
`--strategy sliding`; `scripts.benchmarks.bench_audio_search --duration-sweep`
is the tool for re-justifying the bounds themselves if the transcript source
or embedding model ever changes.

The build validates all 873 transcript sources and verifies the generated
passage count against Elasticsearch. `L24_V022` contains only ASR placeholder
tokens (`<`), so the current batch has 872 searchable videos and one explicitly
reported unsearchable video.

The indexer reads whichever transcript `backend.audio_search.transcript_path()` resolves
to — the raw ASR parquet unless `AIC_TRANSCRIPT_PATH` names another file — so
the index and the subtitle cues always come from the same source. Because
Elasticsearch bakes the resolved text and embeddings in at build time, changing
`AIC_TRANSCRIPT_PATH` only takes effect after rerunning this command with
`--recreate`.

## Run offline

```bash
export HF_HUB_OFFLINE=1
docker compose up -d --wait
UV_CACHE_DIR=/tmp/aic_uv_cache uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Uvicorn blocks during startup while it warms PE-Core, Qwen, CUDA, and the
default Flat FAISS index. On the tested machine this takes roughly 45–60
seconds; wait for `Application startup complete` before opening the UI. Run
only one Uvicorn process (`--workers 1` is the default), because every process
loads its own models and multi-gigabyte FAISS index. IVF and HNSW remain lazy
and incur a one-time load if selected.

Hybrid mode accepts separate queries: `visual_text` in English for PE-Core and
`audio_text` in Vietnamese for transcript search. The older `text` request
field remains supported. If one Hybrid query is empty, only the other search
branch runs.

An optional `ocr_text` is a strict video filter for those visual/audio results:
the video must contain an OCR hit for the full folded phrase. The matching text
may appear in a different shot, such as a title card naming the event location.
With `ocr_text` alone, `/search` performs the existing OCR-only search. OCR
filter failures return `503` instead of silently returning unfiltered results.

`backend.app.fuse_search_results` only credits an audio hit toward a visual hit's
score if their timestamps land within a window of each other
(`backend/app.py:964`). That window now **scales with the matching passage's own
duration** (`_audio_time_window`, floor 30s, ceiling 120s) instead of a fixed
30 seconds — a longer semantic passage's anchor can legitimately sit further
from the exact visual moment it's relevant to, so it gets more slack. This
was benchmarked, not guessed: see `benchmarks/audio/semantic_chunk_config.json`.

Check Elasticsearch before the event:

```bash
curl -fsS http://127.0.0.1:9200/_cluster/health
curl -fsS http://127.0.0.1:9200/aic_audio_v1/_count
```

`AUDIO_EMBED_DEVICE=cuda` is the default. Set it to `cpu` if simultaneous
CLIP and Qwen inference exceeds available VRAM. If Elasticsearch is down,
Audio mode returns HTTP 503 and Hybrid mode falls back to Visual with an
`audio_unavailable` warning.

On the tested RTX 3060, warm Qwen query embedding plus Elasticsearch takes
about 120 ms on CUDA and 1.5 seconds on CPU. After building the index, restart
WSL once before the event to release CUDA memory cached by short-lived build
and benchmark processes, then mount the data disk and start Docker again.
