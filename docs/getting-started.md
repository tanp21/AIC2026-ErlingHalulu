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

Exactly five unique Tailscale identities are required, and the administrator must be included in the allowlist. Never commit `.env.secure`.

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

The health endpoint returns `200`; `/session` returns `401` because only Tailscale Serve injects a trusted identity. Continue with [Secure Team Deployment](team-deployment.md) to expose the application to teammates.

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
UV_CACHE_DIR=/tmp/aic_uv_cache uv run uvicorn app:app --host 127.0.0.1 --port 8000 --workers 1
```

Protected routes still require the `Tailscale-User-Login` header. Use the Docker/Tailscale path for realistic end-to-end access testing.

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
