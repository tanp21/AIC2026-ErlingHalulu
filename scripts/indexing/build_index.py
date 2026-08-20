#!/usr/bin/env python3
"""Build FAISS indexes (flat/ivf/hnsw) + id_map over data/2025/embeddings.

Videos: K groups -> data/2025/videos, L groups -> data/2026/videos.
fps is probed per video (L mixes 30/25 fps), cached to video_meta.parquet.

Outputs indexes/: flat.ip, ivf.ip, hnsw.ip, id_video.npy, id_frame.npy,
video_meta.parquet, info.json.
"""
import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
EMB_DIR = DATA / "2025" / "embeddings"
INDEX_DIR = ROOT / "indexes"
DIM = 1280
SHARD_SIZE = 50000


def log(msg):
    print(f"[build] {msg}", flush=True)


def probe_fps(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate", "-of",
         "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    num, den = r.stdout.strip().split("/")
    return round(float(num) / float(den), 3)


def build_video_meta(vocab, idx_of):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    cache = INDEX_DIR / "video_meta.parquet"
    fps_cache = {}
    if cache.exists():
        for r in pq.read_table(cache).to_pylist():
            fps_cache[r["video_id"]] = r["fps"]

    to_probe = [v for v in vocab if v not in fps_cache]
    if to_probe:
        log(f"probing fps for {len(to_probe)} videos ...")
        paths = [
            str(DATA / ("2025" if v.startswith("K") else "2026") / "videos" / v[:3] / f"{v}.mp4")
            for v in to_probe
        ]
        with ThreadPoolExecutor(max_workers=16) as ex:
            fps = list(tqdm(ex.map(probe_fps, paths), total=len(paths), desc="ffprobe"))
        fps_cache.update(dict(zip(to_probe, fps)))

    rows = []
    for v in vocab:
        year = "2025" if v.startswith("K") else "2026"
        path = DATA / year / "videos" / v[:3] / f"{v}.mp4"
        assert path.exists(), f"missing video {path}"
        rows.append({
            "video_idx": idx_of[v], "video_id": v, "group": v[:3],
            "year": year, "path": str(path), "fps": fps_cache[v],
        })
    pq.write_table(pa.Table.from_pylist(rows), cache)
    log(f"video_meta written ({len(rows)} videos)")
    return {r["video_idx"]: r for r in rows}


def load_meta_order():
    t = pq.read_table(EMB_DIR / "meta.parquet")
    c = t.to_pydict()
    n = len(c["video_id"])
    info = json.loads((EMB_DIR / "info.json").read_text())
    assert n == info["count"], f"meta count {n} != info {info['count']}"
    vocab = list(dict.fromkeys(c["video_id"]))
    idx_of = {v: i for i, v in enumerate(vocab)}
    return c, vocab, idx_of, n


def iter_shards(n_shards):
    for i in range(n_shards):
        yield np.load(EMB_DIR / f"emb_{i:06d}.npy")


def build(args):
    c, vocab, idx_of, n = load_meta_order()
    log(f"{n} embeddings, {len(vocab)} videos")

    build_video_meta(vocab, idx_of)

    id_video = np.array([idx_of[v] for v in c["video_id"]], dtype=np.int32)
    id_frame = np.asarray(c["frame_idx"], dtype=np.int32)
    np.save(INDEX_DIR / "id_video.npy", id_video)
    np.save(INDEX_DIR / "id_frame.npy", id_frame)
    log("id_map saved")

    n_shards = (n + SHARD_SIZE - 1) // SHARD_SIZE

    flat = faiss.IndexFlatIP(DIM)
    for x in tqdm(iter_shards(n_shards), total=n_shards, desc="flat"):
        flat.add(x)

    rng = np.random.default_rng(0)
    for gi in rng.choice(n, 20, replace=False):
        sh, ro = divmod(int(gi), SHARD_SIZE)
        vec = np.load(EMB_DIR / f"emb_{sh:06d}.npy")[ro:ro + 1]
        _, I = flat.search(vec, 1)
        assert I[0][0] == gi, f"id_map mismatch at global {gi}: got {I[0][0]}"
    log("id_map ordering verified (20 spot checks)")

    faiss.write_index(flat, str(INDEX_DIR / "flat.ip"))
    del flat

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(DIM), DIM, args.nlist)
    train = np.vstack([np.load(EMB_DIR / f"emb_{i:06d}.npy") for i in range(4)])
    log(f"training IVF (nlist={args.nlist}) on {len(train)} vectors")
    ivf.train(train)
    del train
    ivf.nprobe = 256
    for x in tqdm(iter_shards(n_shards), total=n_shards, desc="ivf"):
        ivf.add(x)
    faiss.write_index(ivf, str(INDEX_DIR / "ivf.ip"))
    del ivf

    hnsw = faiss.IndexHNSWFlat(DIM, args.hnsw_m)
    hnsw.hnsw.efConstruction = 400
    for x in tqdm(iter_shards(n_shards), total=n_shards, desc="hnsw"):
        hnsw.add(x)
    faiss.write_index(hnsw, str(INDEX_DIR / "hnsw.ip"))
    del hnsw

    (INDEX_DIR / "info.json").write_text(json.dumps({
        "dim": DIM, "count": n,
        "indexes": {
            "flat": {"type": "FlatIP", "metric": "ip", "exact": True},
            "ivf": {"type": "IVFFlat", "metric": "l2", "nlist": args.nlist, "nprobe_default": 256},
            "hnsw": {"type": "HNSW", "metric": "l2", "M": args.hnsw_m,
                     "efConstruction": 400, "efSearch_default": 256},
        },
    }, indent=2))
    log(f"DONE: indexes -> {INDEX_DIR}")


def self_test():
    x = np.load(EMB_DIR / "emb_000000.npy")[:5000]
    idx = faiss.IndexFlatIP(DIM)
    idx.add(x)
    for i in (0, 100, 4999):
        _, I = idx.search(x[i:i + 1], 5)
        assert I[0][0] == i, f"self-match failed at {i}: {I[0]}"
    load_meta_order()
    print("self-test OK: rank-0 self-match + meta/info count consistent")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--nlist", type=int, default=4096)
    p.add_argument("--hnsw-m", type=int, default=64)
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        build(args)


if __name__ == "__main__":
    main()
