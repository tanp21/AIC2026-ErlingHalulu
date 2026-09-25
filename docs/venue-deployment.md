# Venue deployment (compute on a rented instance, media on the WSL desktop)

The desktop cannot hold Qwen3-Embedding-4B next to PE-Core and NaFlex, and the exam room's
network is congested enough that shipping frames over it is the wrong shape. So the heavy half
moves to a rented GPU instance and the heavy *bytes* stay on the desktop.

| | Compute host (vast.ai instance) | Media host (WSL2 desktop at the venue) |
| --- | --- | --- |
| Runs | PE-Core, NaFlex, FAISS, Qwen3-Embedding, Elasticsearch, the retrieval agent, collab | the SPA, ffmpeg |
| Answers | `/core/*`, `/search*`, `/agent/search`, `/api/agent/*`, `/api/collab/*`, `/api/check`, `/ocr/*` | `/`, `/static/*`, `/thumb/*`, `/frame/*`, `/video/*`, `/clip/*`, `/api/dres/*` |
| Has on disk | its own corpus, `indexes/`, models | the corpus, `frame_maps/`, the built frontend |
| Config | `AIC_MEDIA_HOST_LOGIN=<media host login>` | `AIC_ROLE=media`, `AIC_CORE_URL=http://core-tunnel:9000` |

Both hosts run the same code; the difference is one environment variable. A teammate's browser
talks only to the desktop, so media never crosses the venue WAN and the frontend needs no
cross-origin support. What crosses is the JSON the compute host returns.

The point of the whole arrangement: **the agent's frames come from the instance's own disk**. It
looks at candidates and judges them there, so a query never spends the congested link on an image.

For the reverse split (a teammate's app with no local data), see `docs/split-development.md`.

## What the instance cannot do

This is the part that surprises, and it shapes everything below. A vast.ai instance is an
**unprivileged Docker container**, not a VM. Read `/etc/vast-agents-guide.md` on it, but the
three consequences that matter here:

- **No Docker-in-Docker.** The app runs bare under **supervisor**; Elasticsearch runs from its
  tarball. `compose.yaml` does not apply to this host.
- **No `/dev/net/tun` and no `CAP_NET_ADMIN`.** Tailscale cannot run. The link to the media host
  is an **SSH tunnel** instead (`compose.yaml`'s `core-tunnel` service).
- **No sysctl changes.** `vm.max_map_count` stays at the host's 65530, so Elasticsearch must bind
  `network.host: 127.0.0.1` — that puts it in development mode, where the bootstrap check is
  skipped. Binding it any wider makes it refuse to start.

Two more facts to plan around:

- **`workspace_is_volume` is `false`** on the standard instance: stop/start preserves the
  container filesystem, but **recycle or destroy wipes it**. Keep the instance running through
  the competition; a recycle costs the corpus download and the Elasticsearch rebuild.
- **The NVIDIA driver is older than the newest PyTorch wheels.** A driver reporting CUDA 12.7
  cannot load `torch==2.13.0+cu130`, which is what `uv sync` installs by default. It fails
  silently — `torch.cuda.is_available()` is just `False`. After `uv sync`:

  ```bash
  export UV_CACHE_DIR=/tmp/aic_uv_cache
  uv pip install --reinstall --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match "torch<2.12" "torchvision<0.27"
  ```

  A later `uv sync` will undo this. Check `torch.version.cuda` after any dependency change.

## What to rent

| Resource | Sizing | Why |
| --- | --- | --- |
| GPU | RTX 3090 24 GB or better | PE-Core fp16 ~9 GB plus Qwen3-Embedding-4B fp16 ~8 GB; a third model resident at once is what OOMs, not either alone |
| Disk | 500 GB | see the corpus table — it fits, but only just, and the archives are deleted as they are extracted |
| RAM | 32 GB+ | the flat FAISS index for 590k keyframes is ~3 GB resident before metadata |
| CPUs | 8+ | ffmpeg thumbnails and FAISS both want cores |

Prefer a host with high bandwidth: the corpus arrives over that link, not yours. A 3090-class
instance measured ~380 MB/s down with `aria2c -x16`.

## Creating the instance

1. Rent it (RTX 3090, disk ≥ 500 GB, high bandwidth), take the public-IP SSH option.
2. `ssh -p <port> root@<host>`.
3. Clone or `rsync` the repo to `/workspace/aic`. It needs no GPU, no `data/`, no `indexes/`.
4. Install dependencies and fix torch, as above.
5. Write `.env.secure` (five logins, admin, OpenRouter/DeepSeek/Gemini keys) and
   `.env.instance` — see `docs/getting-started.md` and the file the repo already ships for the
   container deployment, adapted for host paths (`AIC_DATA_DIR=/workspace/aic/data`, and so on).
   Set `AIC_MEDIA_HOST_LOGIN=<the desktop's login>`; leaving it unset makes the instance ignore
   every delegated identity, so the whole team would share one coordinator slot and one agent run.
6. Download models into `HF_HOME`, then rebuild Elasticsearch rather than uploading its volume
   — but only `build_audio_index` works from the checked-in data. See *Elasticsearch* below.
7. Install the supervisor units, start them, and check `curl http://127.0.0.1:8000/healthz`.

## Getting the corpus there

**Enumerate first; do not assume a year.** The profile manifests decide what is needed, and the
profile labels are misleading about where their videos live:

```bash
.venv/bin/python -c "
import pyarrow.parquet as pq, collections
for p in ['l2026-v2','naflex-l2026-v2']:
    r = pq.read_table(f'indexes/profiles/{p}/video_meta.parquet').to_pylist()
    print(p, len(r), collections.Counter(x['year'] for x in r), sorted({x['group'] for x in r}))
"
```

As of the 2026 set this answers: **1189 videos, all `year=2026`**, groups `L21..L30`, `M01..M10`,
`S01` — about 330 GB, plus the 298 `N001..N100` traffic-camera videos (~96 GB) that
`l2026-v3`/`naflex-l2026-v3` carry. Nothing from `data/2025/videos/` is referenced, despite
`l2025` being one of the profiles.

The N videos are `Video_N001-N010.zip` … `Video_N091-N100.zip` (also in `corpus-urls.txt`).
They belong on the compute host as much as on the desktop: the agent reads candidate frames
from the host it runs on, and `/shots` and `/frame-map` are forwarded there. The desktop's own
copies came from the same mirror, so every frame map and embedding agrees with the bytes the
instance holds.

The videos come from the organiser's archives, not from YouTube. Re-downloading a different
rendition is the wrong move: `frame_maps` and every embedding were built from these exact files,
and a re-encode or an fps difference shifts every frame index silently — the search returns the
right video at the wrong moment, with nothing on screen to show it.

| What | Size | How |
| --- | --- | --- |
| `Videos_*.zip` / `Video_*.zip` | ~320 GB | `aria2c -x16 -s16` on the instance, one at a time; extract, then delete the archive |
| `data/2026/**` minus `videos/` and `embeddings*/` | ~5.6 GB | `rsync` from the desktop |
| `data/2026/keyframe_thumbs/{160,320,640,1280}` | ~17 GB | `rsync` from the desktop — the only thumbnail source for the N groups (their videos carry no usable timestamps, so ffmpeg cannot extract one), built with `scripts.indexing.build_keyframe_thumbnails` |
| `indexes/frame_maps/`, profile metadata, `*.flat.ip` | ~9 GB | `rsync` from the desktop |
| Hugging Face models | ~36 GB | download on the instance |
| Elasticsearch data | ~2 GB | rebuild on the instance |

Two naming traps when extracting:

- Batch 1 names archives `Videos_L21_a.zip`, and splits L26 across `_a` through `_e`, so the
  archive name is not a group name.
- Batch 2 names some archives after a *range* (`Video_N001-N010.zip`).

So take each **member's own** prefix as the group, and normalise the separator (`S01-V001.mp4`
and `L21_V001.mp4` both become `<group>_V<nnn>`); `scripts/extraction/ingest_zip_videos.py`
documents the same rule.

## Elasticsearch

`build_audio_index` builds `aic_audio_v1` from `data/2026/audio/generated/step_data_clean.parquet`
with the defaults; that is the whole recipe.

The other two builders do not currently run from the data in the tree, and the same is true on
the desktop — both are pre-existing breakages, not something the split introduced:

- **OCR.** `build_ocr_index` reads `entry["lines"]` to pick each shot's anchor frame, but
  `data/2026/ocr/ocr_2026.json` and every `raw/out_*.json` carry only `{frame_idx, text}`. Every
  shot is therefore judged empty and the run ends with `0 shots -> aic_ocr_2026_v1` and no error.
- **Captions.** `build_caption_index` calls `int(mapping.frame_at_media_time(...))` and dies with
  `TypeError: int() argument must be ... not 'NoneType'` on the first caption whose video has no
  frame map — there is no skip for that case. `aic_captions_2026_v1` does not exist on the
  desktop either, so leaving it out reproduces production; the agent drops the caption channel
  for the run, as `docs/agent.md` describes.

Do not try to *rebuild* `aic_audio_v1` to match a cluster that already has it. The desktop's
index was built with `--strategy topic` (DeepSeek chunking, cached under
`data/2026/audio/.deepseek_topic_cache/`), and re-running the builder reproduces neither the
passage count nor the coverage — it also fails its own validation on the `S01` videos, which the
desktop's index does not contain. Port it.

### Porting an index between clusters

The only thing that reaches a cluster is the app container, and its rootfs is read-only, so the
script
(`artifacts/venue/export-index.py` on the instance) goes in on stdin:

```bash
docker exec -i aic_hcmc-app-1 python - definition aic_audio_v1 > audio.def.json
docker exec -i aic_hcmc-app-1 python - definition aic_ocr_2026_v1 > ocr.def.json
docker exec -i aic_hcmc-app-1 python - bulk       aic_audio_v1 > audio.ndjson
docker exec -i aic_hcmc-app-1 python - bulk       aic_ocr_2026_v1 > ocr.ndjson
```

Then, on the receiving host: `DELETE` the index, `PUT` it with the exported **definition**, and
`POST /_bulk` the NDJSON. Three things bite:

- **The definition, not the mapping.** Both indexes declare custom analyzers in `settings`
  (`vi_folded` and friends). Copy the mapping alone and creation is refused with
  `analyzer [vi_folded] has not been configured in mappings` — or worse, if the index already
  exists, the documents load into an index that scores every query differently while the
  mapping still looks right.
- **`dense_vector` values are not in `_source`.** A `_source` dump drops every embedding
  silently: the documents load, `_count` looks right, and every search returns nothing. Export
  with `"fields": ["embedding"]` as well and merge it in.
- **`http.max_content_length` is 100 MB.** A 17k-passage index is ~265 MB of NDJSON; `split -l`
  it into chunks and bulk them in sequence. Splitting on an even line count keeps each
  action/source pair together.

## The media host

Same repo, same image, with `AIC_ROLE=media` and `AIC_CORE_URL=http://core-tunnel:9000` in
`.env.secure`. Its `.env` needs the four `AIC_CORE_SSH_*` values (host, port, user, key path)
that the `core-tunnel` service substitutes.

The tunnel runs as a **container**, not on the host, and that is deliberate: a container's
`127.0.0.1` is its own, so a host-side tunnel would have to bind `0.0.0.0` for `app` to reach it
— and the compute host's only authentication is the `X-AIC-User` header, so anything on the venue
LAN could then impersonate any teammate. Inside `aic-internal` nothing else can reach it.

Two things make the container slightly unusual, both in `Dockerfile.tunnel`:

- It runs as `AIC_UID`, not root. The file-sharing layer behind Docker Desktop refuses to let
  container root read a `0600` key it does not own, and `ssh` will not start when its uid has no
  passwd entry — hence the `adduser` in the image with `AIC_UID` passed as a build arg.
- `StrictHostKeyChecking=accept-new`, because a rented host has no stable key to pin ahead of time.

## Checking it is actually split

```bash
# media answered locally: a real JPEG, and no /thumb in the compute host's audit log
docker compose exec app python -c "
import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/thumb/K01_V001/0?w=320')
print(r.status, r.headers['content-type'])"

# the tunnel reaches the compute host
docker compose exec app python -c "
import urllib.request; print(urllib.request.urlopen('http://core-tunnel:9000/healthz').read())"

# the compute host sees the TEAMMATE, not the desktop, on forwarded routes
grep 'audit user=' <compute host log> | tail -1
```

While a retrieval agent run is going, watch the desktop's log: a `/thumb` or `/frame` line
appearing there during verification means the split has broken and images are crossing the venue
link again.

## When the link to the compute host drops

Media keeps working — the corpus and the SPA are local — but search, agent runs, collaboration
and "Check by AI" all fail. In order of preference:

1. `docker compose ps core-tunnel` and `docker compose logs core-tunnel`. The service reconnects
   on its own every 5 s; a long run of "tunnel dropped" means the instance is down or its sshd
   is not accepting forwards.
2. Restart the instance and re-run the supervisor units.
3. Last resort: unset `AIC_CORE_URL` and `AIC_ROLE` on the desktop and give it the FAISS indexes
   and models it needs. It will be slow and short on VRAM, but it searches.

## DRES stays on the media host

`/api/dres/*` is in `LOCAL_PREFIXES`, so it is answered where it is used rather than forwarded.
Three reasons, any one of which would be enough:

- It talks to an **external** server (`AIC_DRES_BASE_URL`), so it wants a host with real
  outbound access and the team's own credentials — nothing the rented instance should hold.
  `.env.dres` belongs on the media host; the instance does not need it and should not have it.
- It builds its submission payload from the **media**: `/api/dres/frame-id` needs the profile
  metadata, `/api/dres/frame` the frame map, and the payload the videos. The media host has all
  three, including for the 2025 `K` videos the compute host was never given.
- A submission is a stateful, rate-limited conversation with one competition server. Keeping it
  on one machine is the point.

## The audio channel only works on the compute host

`AIC_ROLE=media` does not cause this — it is true of the container image wherever it runs. The
image installs `ffmpeg` and nothing else, so it has no C compiler, and the Qwen embedder reaches
a Triton kernel that wants one:

```
audio search unavailable: Failed to find C compiler.
Please specify via CC environment variable or set triton.knobs.build.impl.
```

Audio search therefore answers **503** on the desktop and any other host running the container
directly, while the instance — an ordinary Ubuntu box with a toolchain — answers normally. Adding
a compiler to the image (`apt-get install ... gcc` alongside `ffmpeg`) is the one-line fix, and
it costs image size that only the fallback path needs.

## What the split costs

- **Videos the compute host was never given are invisible to its JSON routes.** The corpus
  there is whatever the served profiles reference — the 2026 `L`/`M`/`S` groups here. A 2025
  `K` video still plays and still thumbnails, because those routes are local, but `/shots` and
  `/frame-map` for it 404: those are forwarded, and the compute host has no such video. Clip
  Studio on those videos is therefore unavailable. Add their groups to the download list if
  that matters.
- **`/shots` stays forwarded on purpose.** Its `PATCH` sibling writes reviewer labels to the
  compute host, which is where they belong — the exclusion is baked into the visual index at
  build time, so a label written anywhere else would silently disagree with the index.

## Tuning knobs

- **`/frame-map/{id}` is forwarded**, not local: the frontend fetches it to seek, and a long
  video's PTS array is a few hundred KB. If opening a video feels slow over the venue link, add
  `/frame-map/` to `LOCAL_PREFIXES` in `backend/core_service.py` — the handler reads only
  `frame_maps` off local disk, and the media host has it. `/transcript/` is the same shape.
- **`AIC_THUMB_FFMPEG_SLOTS`** (default 12) bounds concurrent ffmpeg on the media host. It is the
  desktop's CPU, not the instance's, that thumbnails cost.
- **`AUDIO_EMBED_DEVICE=cpu`** if PE-Core and Qwen3-Embedding together exceed the instance's
  VRAM. It costs query latency, not correctness.
