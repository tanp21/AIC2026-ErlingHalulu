#!/usr/bin/env python3
"""Build an exact FAISS Flat profile from raw or sharded embeddings."""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend import frame_maps
from backend import shot_labels
from backend import shots


def iter_embeddings(source, prefix):
    meta_path = source / "meta.parquet"
    if meta_path.exists():
        rows = pq.read_table(meta_path).to_pylist()
        shard = shard_idx = None
        for row in rows:
            if not row["video_id"].startswith(prefix):
                continue
            if row["shard_idx"] != shard_idx:
                shard_idx = row["shard_idx"]
                shard = np.load(source / f"emb_{shard_idx:06d}.npy", mmap_mode="r")
            yield row["video_id"], int(row["frame_idx"]), np.asarray(shard[row["row_idx"]])
        return
    for path in sorted(source.glob(f"{prefix}*/*/*.npy")):
        yield path.parent.name, int(path.stem), np.load(path)


def build_profile(source, output, base_index, prefix="L", batch_size=2048,
                  shot_label_dir=None, data_dir=shots.DATA):
    source, output, base_index = map(Path, (source, output, base_index))
    if output.exists():
        raise FileExistsError(f"profile already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        base_rows = pq.read_table(base_index / "video_meta.parquet").to_pylist()
        base_meta = {row["video_id"]: row for row in base_rows}
        exclusion_cache = {}
        excluded_by_reason = {}

        def excluded_reason(video_id, frame_idx):
            if shot_label_dir is None:
                return None
            if video_id not in exclusion_cache:
                row = base_meta[video_id]
                label_path = shot_labels.label_path(video_id, shot_label_dir)
                if not label_path.exists():
                    exclusion_cache[video_id] = None
                else:
                    index = shots.load_path(shots.shots_path(row, data_dir))
                    document = shot_labels.load(video_id, index.bounds, shot_label_dir)
                    exclusion_cache[video_id] = (index, {
                        int(position): label["reason"]
                        for position, label in document["labels"].items()
                        if label["state"] == "excluded"
                    })
            cached = exclusion_cache[video_id]
            if cached is None:
                return None
            index, labels = cached
            position = index.shot_at(frame_idx)
            return labels.get(position)
        index = None
        vectors, video_ids, frame_ids = [], [], []
        video_order = {}

        def flush():
            nonlocal vectors
            if vectors:
                index.add(np.stack(vectors).astype("float32", copy=False))
                vectors = []

        for video_id, frame_idx, vector in iter_embeddings(source, prefix):
            vector = np.asarray(vector)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"invalid embedding: {video_id}/{frame_idx}")
            if index is None:
                index = faiss.IndexFlatIP(len(vector))
            elif len(vector) != index.d:
                raise ValueError(f"embedding dimension mismatch: {video_id}/{frame_idx}")
            if video_id not in base_meta:
                raise ValueError(f"video metadata unavailable: {video_id}")
            reason = excluded_reason(video_id, frame_idx)
            if reason is not None:
                excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1
                continue
            video_order.setdefault(video_id, len(video_order))
            vectors.append(vector)
            video_ids.append(video_order[video_id])
            frame_ids.append(frame_idx)
            if len(vectors) == batch_size:
                flush()
        flush()
        if index is None:
            raise ValueError(f"no embeddings found for prefix {prefix!r}")

        id_video = np.asarray(video_ids, dtype=np.int32)
        id_frame = np.asarray(frame_ids, dtype=np.int32)
        rows = []
        for video_id, video_idx in video_order.items():
            rows.append({**base_meta[video_id], "video_idx": video_idx})
        times = np.empty(len(id_frame), dtype=np.float64)
        for video_idx, video_id in enumerate(video_order):
            positions = np.flatnonzero(id_video == video_idx)
            mapping = frame_maps.load_path(base_index / "frame_maps" / f"{video_id}.npz")
            frames = id_frame[positions]
            if np.any(frames < 0) or np.any(frames >= len(mapping.pts)):
                raise ValueError(f"indexed frame outside frame map: {video_id}")
            times[positions] = (mapping.pts[frames] - mapping.pts[0]) * (
                mapping.time_base_num / mapping.time_base_den
            )

        faiss.write_index(index, str(staging / "flat.ip"))
        np.save(staging / "id_video.npy", id_video)
        np.save(staging / "id_frame.npy", id_frame)
        np.save(staging / "id_time.npy", times)
        pq.write_table(pa.Table.from_pylist(rows), staging / "video_meta.parquet")
        result = {
            "count": index.ntotal, "video_count": len(rows), "dim": index.d,
            "excluded_count": sum(excluded_by_reason.values()),
            "excluded_by_reason": excluded_by_reason,
        }
        (staging / "info.json").write_text(json.dumps({
            **result,
            "indexes": {"flat": {"type": "FlatIP", "metric": "ip", "exact": True}},
        }, indent=2))
        os.replace(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def register_profile(profile_root, name, label, result, default=False):
    profile_root = Path(profile_root)
    manifest_path = profile_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "default": name, "cache_size": 2, "profiles": {}
    }
    if default:
        manifest["default"] = name
    manifest["profiles"][name] = {
        "label": label, "directory": name,
        "count": result["count"], "video_count": result["video_count"],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, default=Path("indexes/profiles"))
    parser.add_argument("--base-index", type=Path, default=Path("indexes"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", default="L")
    parser.add_argument("--shot-label-dir", type=Path,
                        help="filter only reviewed exclusions from this directory")
    parser.add_argument("--data-dir", type=Path, default=shots.DATA)
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    result = build_profile(
        args.source, args.profile_root / args.name, args.base_index, args.prefix,
        shot_label_dir=args.shot_label_dir, data_dir=args.data_dir,
    )
    register_profile(args.profile_root, args.name, args.label, result, args.default)
    print(f"[flat-profile] {args.name}: {result['count']} vectors, "
          f"{result['video_count']} videos, {result['excluded_count']} excluded "
          f"{result['excluded_by_reason']}")


if __name__ == "__main__":
    main()
