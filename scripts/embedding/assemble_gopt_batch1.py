#!/usr/bin/env python3
"""Concatenate the 4 disjoint SigLIP2 GOPT Kaggle-notebook output parts into
one sharded embedding source, ready for scripts/indexing/build_flat_profile.py.

Each part (embeddings/ViT-gopt-16-SigLIP2-384-part{1..4}-of4/ downloaded from
its own Kaggle notebook) covers a disjoint set of videos from the "Batch 1
2026" manifest (L21-L30) -- see docs on scripts/embedding/embed_all.py's
sibling Kaggle notebooks. This script does not deduplicate; it verifies the
four parts really are disjoint (no repeated (video_id, frame_idx) key) and
re-shards the concatenation at the standard 50000-row shard size, using the
same meta.parquet / info.json schema as scripts/embedding/embed_all.py's
ShardWriter.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SHARD_SIZE = 50_000


def iter_part(part_dir: Path):
    meta_path = part_dir / "meta.parquet"
    rows = pq.read_table(meta_path).to_pylist()
    rows.sort(key=lambda r: (r["shard_idx"], r["row_idx"]))
    shard = shard_idx = None
    for row in rows:
        if row["shard_idx"] != shard_idx:
            shard_idx = row["shard_idx"]
            shard = np.load(part_dir / f"emb_{shard_idx:06d}.npy", mmap_mode="r")
        yield row["video_id"], int(row["frame_idx"]), np.asarray(shard[row["row_idx"]])


def assemble(part_dirs, output, model, dim):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    # Collect everything first, then globally sort by (video_id, frame_idx) --
    # this must match the row order scripts/embedding/embed_all.py's per-group,
    # alphabetically-collected keyframe walk produces for l2026/naflex-l2026, since
    # scripts/indexing/build_flat_profile.py assigns video row-indices in first-
    # encounter order and the visual benchmark assumes identical row mappings
    # across profiles. The 4 Kaggle parts are otherwise ordered by a load-
    # balancing video partition, not by video_id.
    all_rows = []  # (video_id, frame_idx, vector)
    seen = set()
    for part_dir in part_dirs:
        part_dir = Path(part_dir)
        info = json.loads((part_dir / "info.json").read_text())
        if info["dim"] != dim:
            raise ValueError(f"{part_dir}: dim {info['dim']} != expected {dim}")
        part_count = 0
        for video_id, frame_idx, vector in iter_part(part_dir):
            key = (video_id, frame_idx)
            if key in seen:
                raise ValueError(f"duplicate (video_id, frame_idx) across parts: {key}")
            seen.add(key)
            all_rows.append((video_id, frame_idx, np.array(vector, copy=True)))
            part_count += 1
        if part_count != info["count"]:
            raise ValueError(
                f"{part_dir}: read {part_count} rows, info.json says {info['count']}"
            )
        print(f"[assemble] {part_dir.name}: {part_count} rows (running total {len(all_rows)})")

    all_rows.sort(key=lambda row: (row[0], row[1]))

    output.mkdir(parents=True)
    buf = np.empty((SHARD_SIZE, dim), dtype=np.float32)
    fill = shard_idx = total = 0
    meta_rows = []

    def flush():
        nonlocal fill, shard_idx
        if fill == 0:
            return
        np.save(output / f"emb_{shard_idx:06d}.npy", buf[:fill])
        shard_idx += 1
        fill = 0

    for video_id, frame_idx, vector in all_rows:
        buf[fill] = vector
        meta_rows.append((shard_idx, fill, video_id, frame_idx))
        fill += 1
        total += 1
        if fill == SHARD_SIZE:
            flush()
    flush()

    pq.write_table(pa.table({
        "shard_idx": pa.array([r[0] for r in meta_rows], type=pa.int32()),
        "row_idx": pa.array([r[1] for r in meta_rows], type=pa.int32()),
        "video_id": pa.array([r[2] for r in meta_rows], type=pa.string()),
        "frame_idx": pa.array([r[3] for r in meta_rows], type=pa.int32()),
    }), output / "meta.parquet")
    (output / "info.json").write_text(json.dumps({
        "model": model, "dim": dim, "count": total,
        "shard_size": SHARD_SIZE, "normalized": True, "dtype": "float32",
    }, indent=2) + "\n")
    print(f"[assemble] DONE: {total} embeddings -> {output}")
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parts", nargs="+", required=True, help="the 4 part directories, in any order")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="timm/ViT-gopt-16-SigLIP2-384")
    p.add_argument("--dim", type=int, default=1536)
    args = p.parse_args()
    assemble(args.parts, args.output, args.model, args.dim)


if __name__ == "__main__":
    main()
