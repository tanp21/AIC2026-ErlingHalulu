#!/usr/bin/env python3
"""Append a new embedding source to an existing Flat profile, as a new profile.

    uv run python -m scripts.indexing.extend_flat_profile \
        --base l2026-v2 --name l2026-v3 --label 'PE-Core · L+M+S+N 2026' \
        --source data/2026/embeddings_n2026/pe-core --frame-from pts-ms --default

``--base`` is never touched: the extended profile is a copy of its vectors, ids and video
metadata with the new source's keyframes appended, and the switch happens in
``indexes/profiles/manifest.json``, so going back is one manifest edit.

Why this rather than another ``build_flat_profile`` run: that builder reads one source
directory, and the L/M/S embeddings behind ``l2026-v2`` are already consolidated in it --
rebuilding means shipping ~6 GB of embeddings to the host to re-add 590k vectors that are
already there. Appending keeps every existing index position meaning what it meant, so a hit
from the new profile is the same hit the old one returned, at the same index.

``--frame-from pts-ms`` is what the N (traffic-camera) groups need. Their videos are VFR and
the organisers' map-keyframes call ``frame_idx`` a relative label there, scoring KIS/QA on
``pts_time`` instead; so each keyframe is appended at the frame of *our* copy whose PTS carries
that time (``KeyframeResolver``), not at the label. Pass ``frame-idx`` for a CFR source.

The new source is appended as it comes: shot exclusions are a property of the videos already
in the base profile, and a group arriving later has no reviewed labels yet. If it needs them,
run ``build_flat_profile --shot-label-dir`` over a merged source instead.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.indexing.build_flat_profile import (
    KeyframeResolver, iter_embeddings, register_profile,
)


def extend_profile(base, output, source, base_index, prefix="", frame_from="frame-idx",
                   batch_size=2048):
    base, output, source, base_index = map(Path, (base, output, source, base_index))
    if not base.exists():
        raise FileNotFoundError(f"base profile does not exist: {base}")
    if output.exists():
        raise FileExistsError(f"profile already exists: {output}")
    if frame_from not in ("frame-idx", "pts-ms"):
        raise ValueError(f"unknown frame_from: {frame_from!r}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        index = faiss.read_index(str(base / "flat.ip"))
        id_video = np.load(base / "id_video.npy")
        id_frame = np.load(base / "id_frame.npy")
        id_time = np.load(base / "id_time.npy")
        if not len(id_video) == len(id_frame) == len(id_time) == index.ntotal:
            raise ValueError(f"base profile is inconsistent: {base}")
        rows = pq.read_table(base / "video_meta.parquet").to_pylist()
        video_idx = {row["video_id"]: int(row["video_idx"]) for row in rows}
        if len(video_idx) != len(rows):
            raise ValueError(f"base profile repeats a video id: {base}")
        next_video_idx = max((int(row["video_idx"]) for row in rows), default=-1) + 1
        catalogue = {row["video_id"]: row for row in
                     pq.read_table(base_index / "video_meta.parquet").to_pylist()}
        resolver = KeyframeResolver(base_index)

        new_video, new_frame, new_time = [], [], []
        added, worst_frame_off, new_videos = [], 0.0, 0
        for video_id, frame_idx, pts_ms, vector in iter_embeddings(source, prefix):
            vector = np.asarray(vector)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"invalid embedding: {video_id}/{frame_idx}")
            if len(vector) != index.d:
                raise ValueError(f"embedding dimension mismatch: {video_id}/{frame_idx}")
            if frame_from == "pts-ms":
                if pts_ms is None:
                    raise ValueError(f"source carries no pts_ms for {video_id}/{frame_idx}")
                frame_idx, off = resolver.frame_for(video_id, pts_ms)
                worst_frame_off = max(worst_frame_off, off)
            if video_id not in video_idx:
                row = catalogue.get(video_id)
                if row is None:
                    raise ValueError(f"video metadata unavailable: {video_id}")
                rows.append({**row, "video_idx": next_video_idx})
                video_idx[video_id] = next_video_idx
                next_video_idx += 1
                new_videos += 1
            added.append(vector)
            new_video.append(video_idx[video_id])
            new_frame.append(frame_idx)
            new_time.append(resolver.time_of(video_id, frame_idx))
            if len(added) == batch_size:
                index.add(np.stack(added).astype("float32", copy=False))
                added.clear()
        if added:
            index.add(np.stack(added).astype("float32", copy=False))
        if not new_frame:
            raise ValueError(f"no embeddings found in {source} for prefix {prefix!r}")

        faiss.write_index(index, str(staging / "flat.ip"))
        np.save(staging / "id_video.npy",
                np.concatenate([id_video, np.asarray(new_video, dtype=np.int32)]))
        np.save(staging / "id_frame.npy",
                np.concatenate([id_frame, np.asarray(new_frame, dtype=np.int32)]))
        np.save(staging / "id_time.npy", np.concatenate([id_time, np.asarray(new_time)]))
        pq.write_table(pa.Table.from_pylist(rows), staging / "video_meta.parquet")
        info = json.loads((base / "info.json").read_text())
        result = {
            "count": index.ntotal, "video_count": len(rows), "dim": index.d,
            "appended": len(new_frame), "appended_videos": new_videos,
            "excluded_count": info.get("excluded_count", 0),
            "excluded_by_reason": info.get("excluded_by_reason", {}),
            "extended_from": base.name, "frame_from": frame_from,
            "frame_time_off_seconds": worst_frame_off,
        }
        (staging / "info.json").write_text(json.dumps({
            **info, **result,
            "indexes": info.get("indexes", {"flat": {"type": "FlatIP", "metric": "ip"}}),
        }, indent=2))
        os.replace(staging, output)
        return result
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True,
                        help="existing profile name under --profile-root")
    parser.add_argument("--source", type=Path, required=True,
                        help="sharded embeddings (meta.parquet + emb_*.npy) to append")
    parser.add_argument("--profile-root", type=Path, default=Path("indexes/profiles"))
    parser.add_argument("--base-index", type=Path, default=Path("indexes"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--frame-from", choices=("frame-idx", "pts-ms"), default="frame-idx")
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()

    base = args.profile_root / args.base
    result = extend_profile(base, args.profile_root / args.name, args.source,
                            args.base_index, args.prefix, args.frame_from)
    register_profile(args.profile_root, args.name, args.label, result, args.default)
    print(f"[extend-profile] {args.name} <- {args.base}: +{result['appended']} vectors "
          f"from {result['appended_videos']} new videos, now {result['count']} vectors over "
          f"{result['video_count']} videos, frames from {result['frame_from']} "
          f"(worst {result['frame_time_off_seconds']:.3f}s off)")


if __name__ == "__main__":
    main()
