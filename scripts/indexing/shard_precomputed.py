#!/usr/bin/env python3
"""Concatenate precomputed per-frame embeddings into 50000-row shards.

Reads data/2025/embeddings/PE-Core-bigG-14-448/{GROUP}/{VIDEO}/{frame:06d}.npy
(one L2-normalized (1280,) float32 vector per file) and writes, flat into
data/2025/embeddings/, the same layout as data/2026/embeddings/:
  emb_%06d.npy   -> (50000, 1280) float32
  meta.parquet   -> shard_idx, row_idx, video_id, frame_idx
  info.json

Resume-safe: committed state is reconstructed from meta.parquet + shards.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

MODEL = "hf-hub:timm/PE-Core-bigG-14-448"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class ShardWriter:
    def __init__(self, out_dir, dim, shard_size=50000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.out_dir / "meta.parquet"
        self.dim = dim
        self.shard_size = shard_size
        self.buf = np.empty((shard_size, dim), dtype=np.float32)
        self.fill = 0
        self.shard_idx = 0
        self.total = 0
        self.done = set()
        self.meta_rows = []
        self._pending = []
        self._resume()

    def _resume(self):
        if not self.meta_path.exists():
            return
        rows = pq.read_table(self.meta_path).to_pylist()
        self.meta_rows = [
            (r["shard_idx"], r["row_idx"], r["video_id"], r["frame_idx"]) for r in rows
        ]
        self.done = {(r[2], r[3]) for r in self.meta_rows}
        self.total = len(self.meta_rows)
        self.shard_idx = self.total // self.shard_size
        self.fill = self.total % self.shard_size
        if self.fill > 0:
            part = self.out_dir / f"emb_{self.shard_idx:06d}.npy"
            assert part.exists(), f"inconsistent state: {part} missing"
            self.buf[: self.fill] = np.load(part)

    def add(self, feat, video_id, frame_idx):
        self.buf[self.fill] = feat
        self._pending.append((self.shard_idx, self.fill, video_id, frame_idx))
        self.fill += 1
        self.total += 1
        if self.fill == self.shard_size:
            self.checkpoint(finalize=True)

    def checkpoint(self, finalize=False):
        if self.fill == 0:
            return
        np.save(self.out_dir / f"emb_{self.shard_idx:06d}.npy", self.buf[: self.fill])
        self.meta_rows.extend(self._pending)
        self._pending.clear()
        self._write_meta()
        if finalize:
            self.shard_idx += 1
            self.fill = 0

    def _write_meta(self):
        table = pa.table(
            {
                "shard_idx": pa.array([r[0] for r in self.meta_rows], type=pa.int32()),
                "row_idx": pa.array([r[1] for r in self.meta_rows], type=pa.int32()),
                "video_id": pa.array([r[2] for r in self.meta_rows], type=pa.string()),
                "frame_idx": pa.array([r[3] for r in self.meta_rows], type=pa.int32()),
            }
        )
        pq.write_table(table, self.meta_path)

    def close(self):
        self.checkpoint(finalize=True)

    def write_info(self):
        (self.out_dir / "info.json").write_text(
            json.dumps(
                {
                    "model": MODEL,
                    "dim": self.dim,
                    "count": self.total,
                    "shard_size": self.shard_size,
                    "normalized": True,
                    "dtype": "float32",
                },
                indent=2,
            )
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="data/2025/embeddings/PE-Core-bigG-14-448")
    p.add_argument("--out-dir", default="data/2025/embeddings")
    p.add_argument("--shard-size", type=int, default=50000)
    args = p.parse_args()

    src = Path(args.src)
    files = sorted(src.rglob("*.npy"))
    log(f"{len(files)} embedding files under {src}")

    dim = int(np.load(files[0]).shape[0])
    log(f"dim = {dim}")

    writer = ShardWriter(args.out_dir, dim, args.shard_size)
    log(f"resume: {writer.total} committed, {len(writer.done)} done")

    n = 0
    skipped = []
    for fp in tqdm(files, desc="shard", unit="vec"):
        video_id = fp.parent.name
        frame_idx = int(fp.stem)
        if (video_id, frame_idx) in writer.done:
            continue
        try:
            writer.add(np.load(fp), video_id, frame_idx)
        except (EOFError, ValueError):
            skipped.append(str(fp))
            continue
        n += 1

    writer.checkpoint()
    writer.write_info()
    writer.close()
    log(f"DONE: {n} new vectors, {writer.total} total -> {args.out_dir}")
    if skipped:
        log(f"skipped {len(skipped)} unreadable files:")
        for s in skipped:
            log(f"  {s}")


if __name__ == "__main__":
    main()
