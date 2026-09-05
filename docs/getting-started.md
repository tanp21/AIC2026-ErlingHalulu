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

Create the two switchable profiles before starting the application. The first command reads the per-frame 2025 embeddings directly; use `--source data/2025/embeddings` instead when the verified shards have already been prepared from that tree.

```bash
uv run python -m scripts.indexing.build_flat_profile \
  --source data/2025/embeddings/PE-Core-bigG-14-448 \
  --name l2025 --label 'L · embeddings 2025' --default
uv run python -m scripts.indexing.build_flat_profile \
  --source data/2026/embeddings \
  --name l2026 --label 'L · embeddings 2026'
```

The generated `indexes/profiles/manifest.json` activates Round 1 mode. Only L videos and the exact Flat index are exposed; choose the 2025 or 2026 embedding profile in the search sidebar. Removing or moving the local manifest restores the legacy full-dataset index on the next application restart.

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

Exactly five unique HAProxy usernames are required, and the administrator must be included in the allowlist. Never commit `.env.secure` or `haproxy-users.cfg`.

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

The health endpoint returns `200`; `/session` requires HAProxy Basic Auth. Continue with [Direct-IP Team Deployment](team-deployment.md) to create the password file and expose the application to teammates.

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
