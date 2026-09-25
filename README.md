# AIC Search

<div align="center">

**Fast multimodal video retrieval for VBS-style competitions.**

Search keyframes by visual semantics, Vietnamese speech transcripts, or both—then jump directly to the matching moment.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![React 19](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Secret scanning](https://github.com/tanp21/AIC2026-ErlingHalulu/actions/workflows/gitleaks.yml/badge.svg)](https://github.com/tanp21/AIC2026-ErlingHalulu/actions/workflows/gitleaks.yml)

[Getting started](docs/getting-started.md) · [Audio search](docs/audio-search.md) · [Venue deployment](docs/venue-deployment.md) · [Operations](docs/operations.md)

</div>

## Overview

AIC Search is a GPU-accelerated retrieval system built for time-critical video search. It combines PE-Core visual embeddings and FAISS with Qwen transcript embeddings and Elasticsearch, exposing all three search modes through one responsive web interface.

| Mode | Best query language | Retrieval pipeline |
| --- | --- | --- |
| **Visual** | English | PE-Core → FAISS |
| **Audio** | Vietnamese | Qwen embeddings + BM25 → Elasticsearch |
| **Hybrid** | English visual query + Vietnamese transcript query | Parallel retrieval → reciprocal rank fusion |

## Highlights

- **Purpose-built multimodal search** with separate visual and transcript query fields.
- **Low-latency warm runtime** for competition use, with models and the default FAISS index loaded at startup.
- **Precise navigation** from a result to its thumbnail, source frame, timestamp, or original video.
- **Five-person deployment** over Tailscale: `tailscale serve` supplies the identity, HAProxy enforces an explicit user allowlist, and administrative operations are owner-only.
- **Resource-aware concurrency** with per-user rate limits, latest-request-wins scheduling, and bounded FFmpeg work.
- **Offline-ready containers** with read-only data, indexes, and model mounts.

## Architecture

Single host — everything on one machine:

```text
Browser ──► tailscale serve ──► HAProxy ──► FastAPI + React
                                               ├──► PE-Core ──► FAISS visual indexes
                                               ├──► Qwen ─────► Elasticsearch indexes
                                               └──► FFmpeg ───► thumbnails, frames, and video
```

Two hosts — the venue split in [Venue Deployment](docs/venue-deployment.md), where the heavy
compute moves to a rented GPU instance and the media stays on the machine at the venue:

```text
Browser ──► tailscale serve ──► HAProxy ──► FastAPI   (media host)
                                               ├──► FFmpeg ───► media, from local disk
                                               └──► SSH tunnel ──► FastAPI   (compute host)
                                                                     ├──► PE-Core ──► FAISS
                                                                     ├──► Qwen ─────► Elasticsearch
                                                                     └──► the retrieval agent
```

Hybrid retrieval runs the visual and transcript branches in parallel and merges their rankings around matching video timestamps.

## Quick start

The production path uses Docker Desktop with WSL2 integration, an NVIDIA GPU, prepared `data/` and `indexes/` directories, and a local Hugging Face model cache.

```bash
cp .env.secure.example .env.secure
# Replace every placeholder with the five teammates' real Tailscale account logins.

export AIC_UID=$(id -u) AIC_GID=$(id -g)
export AIC_HF_HUB_DIR="${HOME}/.cache/huggingface/hub"
docker compose up -d --build --wait
```

Open `http://127.0.0.1:8000/healthz` to verify the service. Identity comes from `tailscale serve`, which adds a `Tailscale-User-Login` header that HAProxy turns into `X-AIC-User`; see [Venue Deployment](docs/venue-deployment.md) for the two-host setup and [Getting Started](docs/getting-started.md) for the single-host one.

### Local development server (no Docker)

Requires `data/` and `indexes/` already prepared and the Hugging Face model cache populated locally.

```bash
export AIC_ALLOWED_USERS=owner@example.com,u2@example.com,u3@example.com,u4@example.com,u5@example.com
export AIC_ADMIN_USER=owner@example.com
uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Open `http://127.0.0.1:8000/healthz`; the first search request loads PE-Core on demand, so it is slower than the rest.

See [Getting Started](docs/getting-started.md) for prerequisites, data preparation, index creation, local development, and complete startup instructions.

## Repository layout

```text
.
├── backend/               FastAPI runtime and retrieval logic (app.py, core.py, ...)
├── frontend/              React and Vite web interface
├── scripts/               Indexing, embedding, extraction, and benchmarks
├── notebooks/             Experimental and data-preparation workflows
├── tests/                 Backend and security regression tests
├── docs/                  User, deployment, and operations guides
├── compose.yaml           Hardened production stack
└── Dockerfile             Reproducible multi-stage application image
```

`data/`, `indexes/`, model caches, generated results, and real environment files are local-only and excluded from Git.

## Documentation

- [Getting Started](docs/getting-started.md) — prerequisites, setup, indexing, and development
- [Audio Transcript Search](docs/audio-search.md) — transcript preparation, retrieval behavior, and performance
- [Shot Clip Export](docs/clip-export.md) — cutting per-shot clips for video-captioning experiments
- [Venue Deployment](docs/venue-deployment.md) — compute on a rented GPU instance, media on the local machine, joined by an SSH tunnel
- [Operations](docs/operations.md) — monitoring, fair-use controls, troubleshooting, and incident response
- [Keyframe Downscale Report](docs/reports/keyframe_hd_downscale_report.md) — retrieval and image-quality evaluation
- [Shot Exclusions](docs/shot-exclusions.md) — review intro/preview/MC shots and rebuild filtered profiles

## Development

```bash
# Backend
.venv/bin/python -m unittest discover -s tests -t . -v

# Frontend
cd frontend
npm test -- --run
npm run typecheck
npm run build
```

Activate the repository's fail-closed secret-scanning hook once after cloning:

```bash
git config core.hooksPath .githooks
```

Every commit is scanned with Gitleaks locally, and GitHub Actions scans the complete reachable history on pushes and pull requests. Never commit credentials; if one is exposed, revoke it before cleaning the Git history.

## Project status

This repository is competition-focused and under active development. Its datasets, generated indexes, model weights, and credentials are intentionally not distributed with the source tree.
