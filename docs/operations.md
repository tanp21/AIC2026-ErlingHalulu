# Operations

Use this guide to monitor the competition server, diagnose common failures, and respond to access incidents.

## Runtime behavior

- One Uvicorn worker avoids loading duplicate GPU models and multi-gigabyte FAISS indexes.
- PE-Core, Qwen, CUDA, and the Flat index are warmed during startup.
- Up to five searches may run globally, with one active search per user.
- A newer queued query supersedes an older query from the same user.
- Each user may submit ten new searches per ten seconds; cached result-page requests do not count toward this limit.
- Normal request bodies are limited to 16 KiB and queries to 2,000 characters. Image-search multipart bodies are limited to 8 MiB, with at most 6 MiB of image data.
- A successful search caches up to 1,000 ranked results for the user and returns them in pages of 100 as the result view scrolls. Each user keeps only their newest snapshot, and an idle snapshot expires after ten minutes.
- Thumbnail and frame extraction allows four global FFmpeg tasks and two per user.
- IVF, HNSW, and index comparison are owner-only and execute exclusively.
- Audit logs contain identity, method, path, status, and latency—not query text.

## Routine checks

```bash
docker compose ps
docker compose logs --tail=100 app
docker stats
nvidia-smi
```

Check service health and the transcript index:

```bash
curl -fsS http://127.0.0.1:8000/healthz
docker compose exec elasticsearch curl -fsS http://localhost:9200/_cluster/health
docker compose exec elasticsearch curl -fsS http://localhost:9200/aic_audio_v1/_count
```

During a five-user load test, watch GPU memory, host memory, CPU, and container logs:

```bash
nvidia-smi -l 1
docker stats
docker compose logs -f app
```

The current operational targets are a warm Hybrid query within 1.5 seconds, five concurrent queries within 4 seconds each without OOM, and at least 3 Mbps per user for five simultaneous original-video streams when host upload is at least 20 Mbps.

## Start, restart, and stop

```bash
# Start or apply configuration changes
docker compose up -d --build --wait

# Restart only the application
docker compose restart app

# Stop services while preserving Elasticsearch and thumbnail volumes
docker compose stop
```

After restarting WSL, remount `data/` before starting the stack.

## Common failures

### `data/... not found`

The data disk is not mounted at the repository's `data/` directory. Remount it and restart the app.

### Docker socket permission denied

Re-enable Docker Desktop's WSL integration, restart Docker Desktop if needed, and reopen the WSL shell.

### Model unavailable in offline mode

Set `AIC_HF_HUB_DIR` to the host Hugging Face `hub` directory containing both required models. The application container intentionally cannot download from the Internet.

### Port 8000 is already in use

Stop the old Uvicorn or container process. Compose publishes HAProxy on `0.0.0.0:8000`.

### Direct-IP URL returns `401`

Confirm the username exists in `haproxy-users.cfg`, then validate both HAProxy configuration files and recreate the ingress container. A stale or incorrect password also returns `401`.

### Direct-IP URL returns `403`

Compare the lowercase username in `haproxy-users.cfg` with `AIC_ALLOWED_USERS` in `.env.secure`, then recreate the app and ingress containers after any change.

### Audio mode returns `503`

Check Elasticsearch health and the `aic_audio_v1` document count. Hybrid search continues with visual results and reports an `audio_unavailable` warning when Elasticsearch fails.

### Image search returns `413` or `422`

`413` means the file or multipart request exceeds its limit. Resize or recompress the image below 6 MiB. `422` means the upload is empty, malformed, animated, over 20 megapixels, or not a JPEG, PNG, or WebP. The server decodes uploads in memory and never stores them.

### More results returns `410`

The cached search snapshot expired or was replaced by a newer successful query. Run the original search again to create a fresh snapshot.

### CUDA memory pressure

Set `AUDIO_EMBED_DEVICE=cpu` if PE-Core and Qwen cannot fit simultaneously. This lowers VRAM use but increases warm audio-query latency.

## Emergency access shutdown

Run this in elevated Windows PowerShell to immediately block ingress while leaving the local containers running:

```powershell
Disable-NetFirewallRule -DisplayName "AIC HTTP ingress"
```

Then delete the `AIC_HTTP` port-forward rule from the F6601P and inspect the audit log:

```bash
docker compose logs app
```

If credentials may have been exposed, replace their hashes before restoring access. Do not restore access until the allowlist, password file, port-forward rule, and logs have been reviewed.

## Secret hygiene

Activate the repository hook after every clone:

```bash
git config core.hooksPath .githooks
```

The hook runs Gitleaks against staged changes and fails closed when neither a local Gitleaks binary nor Docker is available. GitHub Actions scans the complete reachable history on every push and pull request. Real secrets must never be added to an allowlist or example file.
