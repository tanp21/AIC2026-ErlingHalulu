# Split Development (owner compute host + teammate app host)

A teammate can run the whole application on their own machine while the heavy,
hardware-bound pieces — PE-Core, FAISS, Qwen3-Embedding, Elasticsearch and the
`data/` disk — stay on the owner's host. Code changes are then tested locally
without pushing a branch, waiting for a merge, and redeploying the team server.

`AIC_CORE_URL` also runs in the other direction. When the media is what has to stay
local — a venue whose network is too congested to stream video and frames over — the
laptop sets `AIC_ROLE=media` and forwards the *API* instead, while `/thumb`, `/frame`,
`/video` and `/clip` are answered from its own disk. See `docs/venue-deployment.md`.

## How it splits

```text
Teammate laptop (no GPU, no data/, no indexes/, no ES)          Owner host (GPU + data)
┌──────────────────────────────────────────────────┐           ┌──────────────────────────────┐
│ FastAPI app :8000          AIC_CORE_URL = ...    │ Tailscale │ FastAPI app :8000            │
│  ├ /search, /search/temporal, /agent/search      │  serve    │  ├ production routes         │
│  │   orchestration, fusion, DP, ranking          │ ────────► │  ├ /core/* compute endpoints │
│  ├ /api/collab/*   (local SQLite)                │           │  │   PE-Core · FAISS · Qwen3  │
│  ├ /api/agent/*    (local run registry)          │           │  │   Elasticsearch            │
│  ├ / (static dist) or Vite dev :5173             │           │  └ /thumb /frame /video ...  │
│  └ proxy /thumb /frame /video /clip /shots       │           │      ffmpeg + data/ (238G)   │
│      /frame-map /transcript /context /videos     │           └──────────────────────────────┘
│      /indexes /profiles                          │
└──────────────────────────────────────────────────┘
```

One codebase, one switch: with `AIC_CORE_URL` unset the app behaves exactly as
before (the production container and the owner's host). With it set, the
primitives in `backend/search_core.py` are forwarded to the owner's `/core/*`
endpoints and the media/data routes above are reverse-proxied.

| Runs on the teammate | Runs on the owner |
| --- | --- |
| All routes and request validation | PE-Core text/image encoding |
| Hybrid fusion, temporal DP/NMS, OCR filter, RRF, ranking | FAISS visual search (incl. scoped `video_ids` and batch temporal queries) |
| Snapshot pagination (`/search/next`) | Qwen3 encoding, audio FlatIP, Elasticsearch execution |
| Collab SQLite, agent runner + agent CLI, frontend | Media extraction (ffmpeg) and `data/`, `indexes/` files |

## Owner host

Nothing new to run: `/core/*` ships inside `backend/app.py`'s app, behind the same
Tailscale + HAProxy identity as everything else. Keep `AIC_CORE_URL` unset so the
owner keeps executing locally.

```bash
docker compose up -d --build --wait     # as usual
tailscale serve status                  # https://<host>.<tailnet>.ts.net → 127.0.0.1:8000
curl -fsS http://127.0.0.1:8000/healthz
```

The teammate's requests arrive through `tailscale serve`, so HAProxy stamps
`X-AIC-User` with their real tailnet login — it must be one of the five
`AIC_ALLOWED_USERS`. No token or extra port is involved. `/core/*` exposes the
same data the search UI already does, to the same five people.

## Teammate host

Prerequisites: Python 3.12 + `uv`, Node 22 for the frontend. No Docker, no GPU,
no `data/`, `indexes/`, Hugging Face cache or Elasticsearch.

```bash
uv sync --only-group server --only-group agent
npm --prefix frontend ci

export AIC_ALLOWED_USERS=<five team logins>       # same list as the owner's
export AIC_ADMIN_USER=<owner login>
export AIC_CORE_URL=https://tanp21-desktop.taild28bdb.ts.net   # the team server
export AIC_DEV_USER=<your own tailnet login>      # stands in for HAProxy locally

uv run uvicorn backend.app:app --host 127.0.0.1 --port 8000 --workers 1
```

Startup logs `warm-up remote core <url> (api 1, ...)` and skips local model
loading. Then, in another shell:

```bash
cd frontend
AIC_DEV_BACKEND=http://127.0.0.1:8000 AIC_DEV_USER=$AIC_DEV_USER npm run dev
```

The agent CLI and the UI's **Retrieval agent** tab both work against the local
app, which forwards their searches to the owner:

```bash
export AIC_AGENT_URL=http://127.0.0.1:8000
export AIC_AGENT_TRUSTED_USER=$AIC_DEV_USER
export DEEPSEEK_API_KEY=... GEMINI_API_KEY=...
uv run python -m agent check
uv run python -m agent run queries/SOTUYEN1-bo-de-thi.zip
```

### Optional: direct-to-uvicorn core

If the owner is running uvicorn without HAProxy (no `tailscale serve`), set
`AIC_CORE_USER=<a login in AIC_ALLOWED_USERS>` so outbound calls carry
`X-AIC-User`; the teammate's own login works. Over `tailscale serve` this is
unnecessary — HAProxy deletes client-supplied `X-AIC-User` and uses the tailnet
identity instead.

## `/core/*` contract

Versioned by `CORE_API_VERSION` in `backend/core_service.py`; `/core/healthz`
reports it. A client/server mismatch is logged at startup but not fatal.

| Endpoint | Purpose |
| --- | --- |
| `GET /core/healthz` | liveness + `api_version` |
| `GET /core/catalog?profile=` | default profile, indexed video ids, index info |
| `POST /core/encode/text` | PE-Core text vectors |
| `POST /core/encode/image` | PE-Core image vector (raw image bytes) |
| `POST /core/encode/qwen` | Qwen3 query vectors |
| `POST /core/visual/search` | FAISS search, annotated hits (video/frame/pts/time) |
| `POST /core/visual/frame-embedding` | embedding of an indexed (or extractable) frame |
| `POST /core/frame/annotate` | anchor times / frame ids → frame index, PTS, time |
| `POST /core/audio/flat-search` | owner-side audio FlatIP (null when absent) |
| `POST /core/es/search` | Elasticsearch execution for the audio/OCR query builders |

Everything else the teammate needs is a plain proxy of the owner's routes:
`/thumb /frame /video /clip /shots /frame-map /transcript /context /videos
/indexes /profiles`.

## Notes and limitations

- Search latency grows by one or two tailnet round trips per primitive; a warm
  visual search measured ~0.7 s on the team tailnet versus ~0.3 s local.
- The owner's host must be reachable and awake; media streaming and search both
  depend on it. `AIC_CORE_URL` down → searches answer 503, media proxies 502.
- `/compare` and shot-label writes stay owner-only; the teammate's
  non-admin identity is enforced by the owner's access middleware.
- Changing a `/core/*` payload is a contract change: the owner must redeploy
  before teammates on the new client code can use it. Orchestration and agent
  changes live entirely in the teammate's app now, so they need no redeploy.

## Troubleshooting

```bash
# Is the owner's compute plane reachable from the teammate?
curl -fsS https://tanp21-desktop.taild28bdb.ts.net/core/healthz

# Which mode is a running app in?
grep 'warm-up remote core' <log>   # remote mode logs this line at startup
```

`503 core compute host unreachable` means the teammate app could not reach
`AIC_CORE_URL`; `403 user is not allowed` means the tailnet login is missing from
the owner's `AIC_ALLOWED_USERS`.
