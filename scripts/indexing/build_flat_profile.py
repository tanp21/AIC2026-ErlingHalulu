#!/usr/bin/env python3
"""Build an exact FAISS Flat profile from raw or sharded embeddings.

    uv run python -m scripts.indexing.build_flat_profile \
        --source data/2026/embeddings_merged/pe-core --prefix '' \
        --name l2026-v2 --label 'L+M+S · embeddings 2026'

``--prefix ''`` takes every group the source holds. For the N (traffic-camera) groups, whose
organisers' map-keyframes call ``frame_idx`` a relative label and time KIS/QA answers by
``pts_time``, pass ``--frame-from pts-ms``: each keyframe is then placed at the frame of our
own copy whose PTS carries its ``pts_ms`` (see ``KeyframeResolver``)."""
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
    """``(video_id, frame_idx, pts_ms, vector)`` per keyframe, in the source's own order.

    ``pts_ms`` is the organisers' real playback time for the sources that carry it
    (``scripts.embedding.attach_pts_ms`` adds the column for the N groups) and ``None``
    otherwise -- a frame index is a frame's identity for every CFR group.
    """
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
            yield (row["video_id"], int(row["frame_idx"]), row.get("pts_ms"),
                   np.asarray(shard[row["row_idx"]]))
        return
    for path in sorted(source.glob(f"{prefix}*/*/*.npy")):
        yield path.parent.name, int(path.stem), None, np.load(path)


class KeyframeResolver:
    """Places a keyframe that is known only by its time at a frame of the file we hold.

    The N (traffic-camera) videos are VFR and the organisers' own map-keyframes say their
    ``frame_idx`` is a relative label there, while ``pts_time`` is the real playback time and
    what a KIS/QA answer is scored on. So for those two groups the source's frame index is not
    an identity: the keyframe belongs at the frame of *our* copy whose PTS carries that time --
    which is also what makes a hit's thumbnail, its playback position and its submitted
    millisecond agree with each other. Holds one frame-time array per video: resolving is a
    ``searchsorted`` per keyframe, but building that array is not.
    """

    def __init__(self, base_index):
        self.base_index = Path(base_index)
        self._maps = {}

    def _times(self, video_id):
        times = self._maps.get(video_id)
        if times is None:
            times = frame_maps.load_path(
                self.base_index / "frame_maps" / f"{video_id}.npz").times()
            self._maps[video_id] = times
        return times

    def time_of(self, video_id, frame_idx):
        """The playback time the frame map gives a frame of the file we hold."""
        return float(self._times(video_id)[frame_idx])

    def frame_for(self, video_id, pts_ms):
        """``(frame_idx, seconds_off)``: the nearest frame, and how far its time is off."""
        target = float(pts_ms) / 1000.0
        times = self._times(video_id)
        position = int(np.searchsorted(times, target))
        if position <= 0:
            frame = 0
        elif position >= len(times):
            frame = len(times) - 1
        else:
            before, after = position - 1, position
            frame = before if abs(times[before] - target) <= abs(times[after] - target) else after
        return frame, abs(float(times[frame]) - target)


def build_profile(source, output, base_index, prefix="L", batch_size=2048,
                  shot_label_dir=None, data_dir=shots.DATA, frame_from="frame-idx"):
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
        if frame_from not in ("frame-idx", "pts-ms"):
            raise ValueError(f"unknown frame_from: {frame_from!r}")
        resolver = KeyframeResolver(base_index) if frame_from == "pts-ms" else None
        worst_frame_off = 0.0
        index = None
        vectors, video_ids, frame_ids = [], [], []
        video_order = {}

        def flush():
            nonlocal vectors
            if vectors:
                index.add(np.stack(vectors).astype("float32", copy=False))
                vectors = []

        for video_id, frame_idx, pts_ms, vector in iter_embeddings(source, prefix):
            vector = np.asarray(vector)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError(f"invalid embedding: {video_id}/{frame_idx}")
            if index is None:
                index = faiss.IndexFlatIP(len(vector))
            elif len(vector) != index.d:
                raise ValueError(f"embedding dimension mismatch: {video_id}/{frame_idx}")
            if video_id not in base_meta:
                raise ValueError(f"video metadata unavailable: {video_id}")
            if resolver is not None:
                if pts_ms is None:
                    raise ValueError(f"source carries no pts_ms for {video_id}/{frame_idx}")
                frame_idx, off = resolver.frame_for(video_id, pts_ms)
                worst_frame_off = max(worst_frame_off, off)
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
        if resolver is not None:
            # How far a keyframe had to move to land on a frame of our copy. Bounded by one
            # frame interval of the video, so a value in the seconds means the source's
            # pts_ms and this frame map are not of the same rendition.
            result["frame_from"] = "pts-ms"
            result["frame_time_off_seconds"] = worst_frame_off
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
    parser.add_argument("--frame-from", choices=("frame-idx", "pts-ms"), default="frame-idx",
                        help="how a source keyframe is identified: its frame index (every CFR "
                             "group), or the pts_ms the VFR N groups are scored on")
    parser.add_argument("--default", action="store_true")
    args = parser.parse_args()
    result = build_profile(
        args.source, args.profile_root / args.name, args.base_index, args.prefix,
        shot_label_dir=args.shot_label_dir, data_dir=args.data_dir, frame_from=args.frame_from,
    )
    register_profile(args.profile_root, args.name, args.label, result, args.default)
    resolved = (f", frames from pts_ms (worst {result['frame_time_off_seconds']:.3f}s off)"
                if "frame_from" in result else "")
    print(f"[flat-profile] {args.name}: {result['count']} vectors, "
          f"{result['video_count']} videos, {result['excluded_count']} excluded "
          f"{result['excluded_by_reason']}{resolved}")


if __name__ == "__main__":
    main()
