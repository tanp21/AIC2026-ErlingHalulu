# Audio transcript search

The app indexes the 873 transcripts for L21-L30 as overlapping 60-second
passages and searches them with Elasticsearch BM25 plus dense kNN. Python
combines the two rankings with reciprocal rank fusion, so Elasticsearch Basic
is sufficient. Query and passage embeddings use
`Qwen/Qwen3-Embedding-0.6B`; visual search remains on the existing CLIP/FAISS
indexes.

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

## Build the transcript index

```bash
docker compose up -d --wait
UV_CACHE_DIR=/tmp/aic_uv_cache uv run python -m scripts.indexing.build_audio_index --recreate
```

The build validates all 873 transcript sources and verifies the generated
passage count against Elasticsearch. `L24_V022` contains only ASR placeholder
tokens (`<`), so the current batch has 872 searchable videos and one explicitly
reported unsearchable video.

## Run offline

```bash
export HF_HUB_OFFLINE=1
docker compose up -d --wait
UV_CACHE_DIR=/tmp/aic_uv_cache uv run uvicorn app:app --host 127.0.0.1 --port 8000
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
