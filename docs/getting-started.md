# Getting Started

This guide covers the supported Docker workflow, one-time transcript indexing, and local development.

## Prerequisites

- Windows 11 with WSL2
- Docker Desktop with WSL integration enabled
- NVIDIA drivers and GPU access from Docker
- Python 3.12 and [`uv`](https://docs.astral.sh/uv/) for local tooling
- Node.js 22 for frontend development
- Prepared `data/` and `indexes/` directories
- The PE-Core and `Qwen/Qwen3-Embedding-0.6B` models in the local Hugging Face cache

For a 32 GB host, PE-Core benefits from a larger WSL allocation. Create or update `C:\Users\<user>\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB
networkingMode=mirrored
```

Run `wsl --shutdown` in PowerShell after changing it, reopen WSL, and remount the data disk.

## 1. Verify the host

Run these commands from the repository root in WSL:

```bash
test -f data/2026/audio/transcription.parquet && echo "transcripts found"
nvidia-smi
docker info --format '{{json .Runtimes}}'
```

The repository expects runtime videos below `data/<year>/videos/`, visual indexes in `indexes/`, and transcript data in `data/2026/audio/`. These directories are never committed.

### Build the Round 1 L-only Flat profiles

Create the two switchable profiles before starting the application. Both sources are the consolidated shard layout (`emb_*.npy` + `meta.parquet` + `info.json`) produced by `scripts.indexing.shard_precomputed`.

```bash
uv run python -m scripts.indexing.build_flat_profile \
  --source data/2025/embeddings \
  --name l2025 --label 'L · embeddings 2025' --default
uv run python -m scripts.indexing.build_flat_profile \
  --source data/2026/embeddings \
  --name l2026 --label 'L · embeddings 2026'
```

The generated `indexes/profiles/manifest.json` activates Round 1 mode. Only L videos and the exact Flat index are exposed; choose the 2025 or 2026 embedding profile in the search sidebar. Removing or moving the local manifest restores the legacy full-dataset index on the next application restart.

### Adding a group to a built profile

`l2026-v2`/`naflex-l2026-v2` are the L+M+S profiles; `l2026-v3`/`naflex-l2026-v3` are the same
profiles with the N traffic-camera groups appended. A group that arrives later goes in with
`extend_flat_profile` rather than a rebuild — the base profile's vectors and ids are carried
over unchanged, so every hit the old profile returned comes back at the same index, and the
switch is one manifest edit (`--base` is left on disk, so going back is another).

```bash
uv run python -m scripts.indexing.extend_flat_profile \
  --base l2026-v2 --name l2026-v3 --label 'L+M+S+N · embeddings 2026' \
  --source data/2026/embeddings_n2026/pe-core --frame-from pts-ms --default
```

`--frame-from pts-ms` is what the N groups need. Their videos are VFR and the organisers'
map-keyframes say their `frame_idx` is only a relative label there, scoring KIS/QA on
`pts_time`; so each keyframe is placed at the frame of *our* copy whose PTS carries its
`pts_ms` (the column `scripts.embedding.attach_pts_ms` joins in from the organisers' own
`map-keyframes` CSV). Indexing the label instead would put 78 of the 298 videos seconds away
from the moment the query asked for — up to 69 s on `N029_V002` — while the video and the
keyframe image would still look plausible. The profile's `info.json` records the worst
placement error it saw, which is bounded by the frame gaps in our copy (0.05 s for 98.9% of
the 88,453 N keyframes, 1.0 s at worst).

If the videos need frame maps first, `scripts.indexing.build_batch2_frame_maps --groups N031`
builds them from `data/<year>/videos/<group>/` (it reads both `.mp4` and the N groups' `.mov`).

### The N keyframe thumbnails

`/thumb` cannot cut a frame out of an N video at all: those files are VFR with a container whose
sample table disagrees with its packets, so a seek decodes nothing and a `pts` match finds
nothing. Their thumbnails come from the organisers' own keyframe archives instead — which are
also what the N embeddings were taken from — resized offline into a store keyed by the
`(video_id, frame_idx)` a hit reports:

```bash
uv run python -m scripts.indexing.build_keyframe_thumbnails \
  --zips /path/to/Keyframes_N*.zip --workers 8
```

It writes `data/2026/keyframe_thumbs/{160,320,640,1280}/<group>/<video_id>/<frame>.jpg` (~17 GB
for the 88k N keyframes; 1280 is what `/frame` and the agent's judging use). `backend.thumb`
consults it before touching a video, so both hosts that serve media to the team need the
directory — and a frame with no stored image falls back to an index-based extraction, which
works but decodes the video from the start (seconds, not milliseconds).

## 2. Prepare dependencies and models

```bash
docker compose pull
UV_CACHE_DIR=/tmp/aic_uv_cache uv sync
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -c \
  'from sentence_transformers import SentenceTransformer; SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")'
```

The production container runs offline, so download required model files before the event.

## 3. Build the transcript index

Start Elasticsearch and create the audio index once:

```bash
docker compose up -d elasticsearch --wait
export AIC_TRANSCRIPT_PATH=$PWD/data/2026/audio/generated/step_data_clean.parquet
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -m scripts.indexing.build_audio_index --recreate
```

The builder validates the transcript inputs and confirms the generated passage count in Elasticsearch. See [Audio Transcript Search](audio-search.md) for the current batch details.

## 4. Configure application access

```bash
cp .env.secure.example .env.secure
chmod 600 .env.secure
```

Edit `.env.secure` and replace all placeholders:

```dotenv
AIC_ALLOWED_USERS=owner@example.com,teammate1@example.com,teammate2@example.com,teammate3@example.com,teammate4@example.com
AIC_ADMIN_USER=owner@example.com
```

Exactly five unique Tailscale account logins are required, and the administrator must be included in the allowlist. Never commit `.env.secure`.

## 5. Start the stack

```bash
export AIC_UID=$(id -u) AIC_GID=$(id -g)
export AIC_HF_HUB_DIR="${HOME}/.cache/huggingface/hub"
docker compose up -d --build --wait
docker compose logs -f app
```

If the model cache lives elsewhere, set `AIC_HF_HUB_DIR` to its actual `hub` directory. Startup warms PE-Core, Qwen, CUDA, and the Flat FAISS index and may take 45–60 seconds. Wait for `Application startup complete`.

Verify the local ingress:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -i http://127.0.0.1:8000/session
```

The health endpoint returns `200`; everything else requires an identity, which HAProxy takes from the `Tailscale-User-Login` header `tailscale serve` adds. Continue with [Venue Deployment](venue-deployment.md) to run this alongside a rented GPU instance.

## Search with an example image

Visual and Hybrid modes accept one query image alongside the optional English visual query. Choose a JPEG, PNG, or WebP file, drag it into the query-image area, or focus that area and paste a screenshot. Images are limited to 6 MiB and 20 megapixels and are processed only for the current request.

When both text and an image are present, **Image influence** controls their mix and defaults to 50%. An image-only search automatically uses 100%. To refine existing results, click the `≈` button on a keyframe; the application immediately searches the full visual index using that frame and excludes the frame itself from the new results. Image queries are intentionally unavailable in Audio and Temporal modes.

## Local development

Install dependencies and build the UI:

```bash
UV_CACHE_DIR=/tmp/aic_uv_cache uv sync
cd frontend
npm ci
npm run build
cd ..
```

Set the five-user configuration, ensure Elasticsearch is reachable, and start one Uvicorn worker:

```bash
export AIC_ALLOWED_USERS=owner@example.com,user2@example.com,user3@example.com,user4@example.com,user5@example.com
export AIC_ADMIN_USER=owner@example.com
export AIC_TRANSCRIPT_PATH=$PWD/data/2026/audio/generated/step_data_clean.parquet
UV_CACHE_DIR=/tmp/aic_uv_cache uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Protected routes still require the internal `X-AIC-User` header. Use the Docker/HAProxy path for realistic end-to-end access testing; HAProxy removes client-supplied copies and sets it after authentication.

To develop against the owner's GPU/index host instead of a full local stack (no GPU, `data/`, `indexes/` or Elasticsearch needed), see [Split Development](split-development.md).

## Utility commands

Run Python tools as modules from the repository root:

```bash
uv run python -m scripts.indexing.build_audio_index --help
uv run python -m scripts.indexing.build_frame_maps --help
uv run python -m scripts.embedding.embed_keyframes --help
```

Start notebooks from the same directory so relative paths remain valid:

```bash
uv run jupyter lab --notebook-dir=.
```
