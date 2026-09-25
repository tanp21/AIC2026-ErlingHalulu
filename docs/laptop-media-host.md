# The laptop media host (Windows, Docker Desktop, no WSL2)

The competition venue has a switch, not a tailnet. Five laptops are wired to it; one of
them holds the corpus and serves everyone, and the same laptop keeps its WiFi up for the
two things that need the internet: DRES submission and the SSH tunnel to the compute host.

```
   teammate laptops ──switch──►  THIS LAPTOP  ──WiFi──►  vast.ai instance
   (browser, Basic Auth)         SPA + media            search, agent, collab
                                 ffmpeg, DRES           Elasticsearch, GPU
```

This is the deployment in `docs/venue-deployment.md` with two substitutions, because a
switch provides neither of the things that document relies on:

| | Venue deployment (desktop) | Laptop media host |
| --- | --- | --- |
| Identity | `tailscale serve` adds `Tailscale-User-Login` | HTTP Basic Auth in `haproxy.lan.cfg` |
| Ingress config | `haproxy.cfg` | `haproxy.lan.cfg` |
| Compose | `compose.yaml` | `compose.yaml` + `compose.media.yaml` |
| Everything else | identical — same image, same `AIC_ROLE=media`, same `core-tunnel` | |

The app itself does not change. `backend/core_service.py` still forwards every non-media
route to `AIC_CORE_URL`; the diff is entirely about how a request gets an identity and
where the process is allowed to listen.

## What the laptop has to hold

A media host serves `/`, `/static`, `/thumb`, `/frame`, `/video`, `/clip`, `/frame-map` and
`/api/dres/*` from local disk. Everything else leaves over the tunnel. So it needs the
corpus and the metadata those routes read — and **not** the FAISS indexes, the models, or
Elasticsearch.

| Path | Size | Why it is needed | If it is missing |
| --- | --- | --- | --- |
| `data/<year>/videos/**` | ~330 GB + 96 GB N | the footage itself | nothing plays |
| `indexes/frame_maps/*.npz` | 80 MB | exact frame↔time; `/frame-map`, `/clip`, DRES | `/thumb` and `/frame` 500 on every hit |
| `indexes/frame_maps/manifest.json` | 4 KB | the size/mtime signature DRES verifies before a submission | every DRES submission is refused |
| `data/<year>/shots/**` | ~18 MB | TransNetV2 boundaries for `/clip/{id}/{shot}` | Clip Studio is unavailable |
| `indexes/profiles/**` (no `*.ip`) | ~50 MB | `video_meta.parquet` + id maps; `resolve()` and DRES both read them | every route 404s on a video id |
| `indexes/video_meta.parquet`, `id_*.npy` | ~14 MB | only so `build_frame_maps` can enumerate the corpus | the re-sign step below cannot run |
| `data/2026/keyframe_thumbs/{160,320,640,1280}` | 22 GB | the **only** thumbnail source for the N traffic-camera groups | `/thumb` 500s for every N frame |
| `data/<year>/videos_faststart/**` | optional | playback starts without a round-trip to the file tail | videos still play, just slower to open |

`data/2025/` is not referenced by any profile, but `/video`, `/thumb` and `/clip` still
serve the `K*` videos from it through `shots.media_catalogue()`. Copy it if the team wants
those to play; `/shots` and `/frame-map` for them are forwarded and will 404, because the
compute host was never given them either.

## One-time setup on Windows

**1. Docker Desktop.** Linux containers, which means either the WSL2 backend or the Hyper-V
one — Windows has no third option. "No WSL2" as in *not developing inside a WSL distro* is
fine: nothing here is run from a Linux shell on the laptop, and both backends behave the
same for this stack. Give it the drive holding the repo under *Settings → Resources → File
sharing*.

**2. Turn on the GPU passthrough** if you want it (the media path never uses CUDA, so this
is only for later experiments): `nvidia-smi` on the Windows side, then check
`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`.

**3. Allow inbound 8000 on the switch only.** Windows Firewall blocks it by default, and
the teammates' browsers will simply time out. Scope the rule to the LAN subnet so the WiFi
interface stays closed:

```powershell
New-NetFirewallRule -DisplayName "AIC ingress (switch)" -Direction Inbound `
  -Protocol TCP -LocalPort 8000 -Action Allow -Profile Private `
  -RemoteAddress 192.168.0.0/16
```

**4. Clone and configure.**

```powershell
git clone <repo> C:\aic ; cd C:\aic
copy .env.laptop.example .env
copy .env.secure.example .env.secure
copy .env.dres.example .env.dres
git config core.hooksPath .githooks
```

Fill in three files that are deliberately not in git:

- `.env` — from the template; `AIC_INGRESS_BIND` is the laptop's Ethernet address
  (`ipconfig`), and the four `AIC_CORE_SSH_*` values point at the instance.
- `.env.secure` — the five logins, plus `AIC_ROLE=media`, `AIC_CORE_URL=http://core-tunnel:9000`
  and `AIC_CORE_USER=<the login this laptop acts as>`. That last one is not cosmetic: it is
  the identity the compute host sees for every forwarded request, and it must equal
  `AIC_MEDIA_HOST_LOGIN` **on the instance** or the whole team collapses into one
  coordinator slot. Compare the two by hand once.
- `.env.dres` — the team's DRES account. It stays on this laptop; the instance has no
  business holding competition credentials.

**5. Generate the password file.** Only the passwords are asked for; the usernames come
from `AIC_ALLOWED_USERS`.

```powershell
uv run python -m scripts.media.make_lan_users
```

## The corpus has to be *copied*, not re-downloaded

This is the trap. `backend/dres.py::_source` refuses a submission whose video differs from
`indexes/frame_maps/manifest.json` in **size or mtime**, because a frame map that no longer
describes its file would submit a frame_id for the wrong moment. A freshly extracted
archive carries the extraction time, so a re-downloaded corpus fails every DRES submission
— while search, playback and thumbnails all work perfectly, which is what makes it
confusing at 2am.

Two ways out, best first:

```bash
# On the laptop, with the repo's venv. Verifies each existing map against the file's own
# PTS timeline (a packet probe, not a decode) and re-records the signature.
uv run python -m scripts.indexing.build_frame_maps

# No ffmpeg on the laptop? This re-signs on a matching file size alone, and refuses any
# video whose length changed.
uv run python -m scripts.indexing.build_frame_maps --resign-only --data-dir D:/aic/data
```

`python -m scripts.media.check_media_host` runs both checks (and the rest of the table
above) and prints which videos need it. It runs without host Python at all:

```powershell
docker compose -f compose.yaml -f compose.media.yaml build app
docker compose -f compose.yaml -f compose.media.yaml run --rm `
  -v ./scripts:/app/scripts:ro app python -m scripts.media.check_media_host
```

Copying with `robocopy /COPY:DAT /DCOPY:DAT` or `rsync -a` preserves timestamps and skips
the problem entirely — but run the check anyway, because a partial copy looks identical
from the outside.

## Running it

```powershell
docker compose -f compose.yaml -f compose.media.yaml up -d --build --wait
```

Three services come up — `app`, `ingress`, `core-tunnel`. There is no Elasticsearch and no
GPU reservation; `compose.media.yaml` explains each removal. First build takes a few
minutes (npm build + uv install in the image).

Startup still takes 45–60 s. The app logs `warm-up media host metadata` and
`warm-up remote core …` — if the second warns that the core host is unreachable, the
tunnel is down and search will 502 while media keeps working. That is the designed
behaviour, not a crash.

### Checking the split

```powershell
# media answered locally: a real JPEG
docker compose -f compose.yaml -f compose.media.yaml exec app python -c `
  "import urllib.request; r=urllib.request.urlopen('http://127.0.0.1:8000/thumb/L21_V001/0?w=320'); print(r.status, r.headers['content-type'])"

# the tunnel reaches the compute host
docker compose -f compose.yaml -f compose.media.yaml exec app python -c `
  "import urllib.request; print(urllib.request.urlopen('http://core-tunnel:9000/healthz').read())"
```

From a teammate's laptop, `http://<AIC_INGRESS_BIND>:8000/` should prompt for a password
and then load the SPA. `http://<AIC_INGRESS_BIND>:8000/healthz` answers without one — that
is the only anonymous route, matching what `TrustedIngressAccessMiddleware` allows.

While a retrieval agent run is going, watch this laptop's log: a `/thumb` or `/frame` line
appearing *here* during verification is correct (the images are local), while one appearing
on the instance means the split has broken and frames are crossing the WiFi link.

## Competition day

```powershell
ipconfig                                   # confirm AIC_INGRESS_BIND still matches
docker compose -f compose.yaml -f compose.media.yaml up -d
python -m scripts.media.check_media_host   # or the docker one above
```

Before the round: have each teammate log in once (`/healthz` first, then the SPA) so a
wrong password is found early rather than mid-query. Keep the WiFi up — DRES submission and
the tunnel both need it, and neither needs inbound access, which is why the ingress is
pinned to the switch address.

## When something is wrong

| Symptom | Cause |
| --- | --- |
| Teammates time out; `curl` from the laptop works | Windows Firewall step 3, or `AIC_INGRESS_BIND` is the WiFi address |
| 403 on every request | `haproxy.lan.cfg` is not mounted — the Tailscale `haproxy.cfg` refuses LAN sources by design |
| Browser re-prompts for a password on every asset | `haproxy-users.cfg` is missing or was mounted as a directory (Docker creates a *directory* when a bind source does not exist) |
| 401 from the app | HAProxy authenticated, but the username is not in `AIC_ALLOWED_USERS`; usernames must match exactly |
| 502 on search, media fine | the tunnel is down: `docker compose … logs core-tunnel`, then check the instance |
| `/thumb` 500s for N videos only | the keyframe store is missing — ffmpeg cannot decode those containers at all |
| DRES refuses every submission | the mtime/size re-sign step above |
| Everyone shares one agent run and one search slot | `AIC_CORE_USER` here does not match `AIC_MEDIA_HOST_LOGIN` on the instance |
| Search works but `/clip/` 422s for N videos | those groups have no shot boundaries (`data/2026/shots/` lacks them); run `scripts.extraction.detect_shots`, or accept it |

`docs/operations.md` has the instance-side playbook; `docs/venue-deployment.md` explains
what the compute host does with each forwarded route.
