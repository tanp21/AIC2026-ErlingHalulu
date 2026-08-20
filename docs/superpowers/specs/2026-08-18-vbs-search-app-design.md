# VBS-style search web app (FAISS) — design

## Goal
Local web app for VBS-like text search over precomputed CLIP embeddings, with
multiple FAISS index types, real-time thumbnail extraction, and range-streamed
video playback.

## Data
- Embeddings: `data/2025/embeddings/` — 855,632 vectors, dim 1280, L2-normalized
  float32, model `hf-hub:timm/PE-Core-bigG-14-448`. Sharded `emb_%06d.npy`
  (50000/shard) + `meta.parquet` (`shard_idx,row_idx,video_id,frame_idx`).
- Videos: K groups → `data/2025/videos`, L groups → `data/2026/videos`.
  605 K + 873 L = 1478 mp4, all present. `frame_idx` = real video frame number;
  L mixes 30/25 fps so fps is probed per video (ffprobe, cached).

## Decisions
- Index all 855,632 (K + L). Metric = Inner Product (vectors normalized → cosine).
- Index types: FlatIP (ground truth), IVFFlat (nlist 4096), HNSW (M=32).
- Thumbnails: ffmpeg seek `frame_idx/fps` (keyframes mismatch, so never use them).
- Backend FastAPI, deps faiss-cpu/fastapi/uvicorn. Text encoder on GPU (fp16).

## Architecture
```
`python -m scripts.indexing.build_index` → indexes/{flat,ivf,hnsw}.ip + id_video.npy + id_frame.npy
                 + video_meta.parquet + info.json
core.py        → CLIP text encoder + meta/index loading (shared app+bench)
thumb.py       → ffmpeg extract + disk cache
app.py         → FastAPI: /search /thumb /frame /video(Range) /indexes /compare
`python -m scripts.benchmarks.bench_index` → latency + recall@k vs Flat
static/        → frontend (grid, index dropdown, compare)
```

## Memory
855k×1280×4B ≈ 4.4 GB/index. Build streams shards (one index resident at a
time); runtime lazy-loads a single index, evicting the previous (compare mode
runs sequentially).

## Endpoints
- `POST /search {text,index,k,nprobe?,ef?}` → ranked results + latency + index info
- `POST /compare {text,k}` → latency + recall@k for flat/ivf/hnsw
- `GET /thumb/{video}/{frame}?w=` → JPEG (scaled)
- `GET /frame/{video}/{frame}` → full-res JPEG
- `GET /video/{video_id}` → mp4 with Range (play at timestamp via `#t=`)
- `GET /indexes` → index list + config

## Verification
- `python -m scripts.indexing.build_index --self-test`: rank-0 self-match + meta count check.
- Post-build spot-check: 20 random ids verify flat returns the same id
  (guards id_map ordering).
- Smoke: `/search` JSON, `/thumb` JPEG, `/video` 206 on Range, compare recall>0.9@k=50.
