#!/usr/bin/env python3
"""Merge the per-account shards of a sharded caption-embedding run.

Each Kaggle account in a sharded run writes a complete output directory
(``emb_*.npy`` + ``meta.parquet`` + ``info.json``) covering its own contiguous
slice of the caption rows. This concatenates those directories back into one
with vectors in global row order and writes fresh ``info.json``/``checksums.json``.

Usage:
    uv run python -m scripts.embedding.merge_caption_embedding_shards \
        --inputs data/2026/embeddings/qwen3-8b-fp16-shard0 \
                 data/2026/embeddings/qwen3-8b-fp16-shard1 \
                 data/2026/embeddings/qwen3-8b-fp16-shard2 \
        --output data/2026/embeddings/qwen3-8b-fp16
    uv run python -m scripts.embedding.merge_caption_embedding_shards --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


OUTPUT_SHARD_SIZE = 50_000
META_COLUMNS = ("group", "video_id", "shot_index", "window_index", "window_count",
                "window_start_sec", "window_end_sec")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_rows(input_dir: Path):
    meta = pq.read_table(input_dir / "meta.parquet").to_pylist()
    position = 0
    for shard_path in sorted(input_dir.glob("emb_*.npy")):
        shard = np.load(shard_path, mmap_mode="r")
        for row in range(shard.shape[0]):
            if position >= len(meta):
                raise RuntimeError(f"{input_dir}: {shard_path} has more rows than meta")
            yield meta[position], shard[row]
            position += 1
    if position != len(meta):
        raise RuntimeError(f"{input_dir}: meta has {len(meta)} rows but shards have {position}")


def merge(inputs, output):
    inputs = [Path(item) for item in inputs]
    if len(inputs) < 1:
        raise ValueError("need at least one input directory")
    collected = []
    for input_dir in inputs:
        for row, vector in iter_rows(input_dir):
            collected.append((int(row["index"]), row, input_dir, vector))
    collected.sort(key=lambda item: item[0])
    indices = [item[0] for item in collected]
    if indices != list(range(len(indices))):
        missing = sorted(set(range(len(indices))) - set(indices))[:5]
        raise RuntimeError(f"shards do not cover 0..{len(indices) - 1} "
                           f"(duplicates or missing rows, e.g. {missing})")
    dim = int(collected[0][3].shape[0])
    if any(item[3].shape[0] != dim for item in collected):
        raise RuntimeError("inconsistent vector dimensions across shards")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("emb_*.npy"):
        stale.unlink()
    buffer = np.empty((OUTPUT_SHARD_SIZE, dim), dtype=np.float32)
    fill = shard_index = 0
    for _, _, _, vector in collected:
        buffer[fill] = vector
        fill += 1
        if fill == OUTPUT_SHARD_SIZE:
            np.save(output / f"emb_{shard_index:06d}.npy", buffer)
            shard_index += 1
            fill = 0
    if fill:
        np.save(output / f"emb_{shard_index:06d}.npy", buffer[:fill])

    meta = {"index": pa.array(indices, type=pa.int32())}
    for column in META_COLUMNS:
        meta[column] = pa.array([item[1][column] for item in collected])
    pq.write_table(pa.table(meta), output / "meta.parquet")

    info = json.loads((inputs[0] / "info.json").read_text())
    info["count"] = len(collected)
    info["videos"] = len({item[1]["video_id"] for item in collected})
    info["shard_index"] = 0
    info["shard_count"] = 1
    info["merged_from"] = [str(input_dir) for input_dir in inputs]
    info["merged_sources"] = [
        {key: json.loads((input_dir / "info.json").read_text()).get(key)
         for key in ("shard_index", "shard_count", "count", "units_per_second", "strategies")}
        for input_dir in inputs
    ]
    (output / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    checksums = {path.name: sha256_file(path) for path in sorted(output.iterdir())
                 if path.is_file() and path.name != "checksums.json"}
    (output / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    return info


def self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)

        def write_part(name, rows, dim=3, shard_size=3):
            directory = root / name
            directory.mkdir()
            source = np.empty((len(rows), dim), dtype=np.float32)
            for position, (index, _) in enumerate(rows):
                source[position] = index * 100 + np.arange(dim)
            shard = 0
            for start in range(0, len(rows), shard_size):
                np.save(directory / f"emb_{shard:06d}.npy", source[start:start + shard_size])
                shard += 1
            table = {"index": pa.array([index for index, _ in rows], type=pa.int32())}
            for column in META_COLUMNS:
                table[column] = pa.array([row[column] for _, row in rows])
            pq.write_table(pa.table(table), directory / "meta.parquet")
            (directory / "info.json").write_text(json.dumps({
                "run": "test", "dim": dim, "count": len(rows), "model": "test",
                "shard_index": 0, "shard_count": 3,
            }) + "\n")

        def row(index):
            return {"index": index, "group": "L99", "video_id": f"L99_V{index:03d}",
                    "shot_index": 0, "window_index": 0, "window_count": 1,
                    "window_start_sec": float(index), "window_end_sec": float(index) + 1.0}

        write_part("a", [(index, row(index)) for index in range(0, 4)])
        write_part("b", [(index, row(index)) for index in range(4, 7)])
        info = merge([root / "a", root / "b"], root / "out")
        assert info["count"] == 7 and info["shard_count"] == 1
        merged = np.vstack([np.load(path) for path in sorted((root / "out").glob("emb_*.npy"))])
        assert merged.shape == (7, 3)
        meta = pq.read_table(root / "out/meta.parquet").to_pylist()
        assert [item["index"] for item in meta] == list(range(7))
        for index in range(7):
            assert merged[index, 0] == index * 100, (index, merged[index])

        write_part("c", [(0, row(0)), (0, row(1))])
        try:
            merge([root / "a", root / "c"], root / "out2")
        except RuntimeError as error:
            assert "do not cover" in str(error)
        else:
            raise AssertionError("merge accepted duplicate rows")
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.inputs or not args.output:
        parser.error("--inputs and --output are required")
    info = merge(args.inputs, args.output)
    print(f"merged {info['count']} vectors from {len(args.inputs)} shards -> {args.output}")


if __name__ == "__main__":
    main()
