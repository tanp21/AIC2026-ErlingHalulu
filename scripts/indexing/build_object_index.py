"""Build the Polars/Parquet object store from detector samples and FAISS keyframes.

Only keyframes of a profile's videos can be matched to detector samples, and only the groups
in ``SEARCH_PREFIXES`` can be matched at all -- the 2025 ``K`` *catalogue* rows are in the
base index without any detector source. Every sample still lands in the playback partition of
its video, whether or not it is one of them, which is what lets a video be looked at before it
is searchable.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
from uuid import uuid4

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from backend import frame_maps


ROOT = Path(__file__).resolve().parents[2]
MAX_DELTA = 0.25
#: Groups whose keyframes can be matched to a detector source. N was playback-only while no
#: FAISS profile carried its videos; the extended profiles (`extend_flat_profile`) do now, and
#: the builder follows the profile, not a list: a group in a profile but not here would index
#: its frames for playback and silently never match them.
SEARCH_PREFIXES = ("L", "M", "S", "N")
SOURCE_COLUMNS = ["video_id", "frame_idx", "time", "labels", "scores", "boxes"]
PLAYBACK_SCHEMA = pa.schema([
    ("video_id", pa.string()), ("frame_idx", pa.int64()), ("time", pa.float64()),
    ("labels", pa.list_(pa.string())), ("scores", pa.list_(pa.float64())),
    ("boxes", pa.list_(pa.list_(pa.float64()))),
])
FRAME_SCHEMA = pa.schema([
    ("id", pa.uint32()), ("video_id", pa.string()), ("frame_idx", pa.int64()),
    ("time", pa.float64()), ("pts", pa.int64()), ("source_frame_idx", pa.int64()),
    ("source_time", pa.float64()), ("delta", pa.float64()), ("profile_mask", pa.uint64()),
])
DETECTION_SCHEMA = pa.schema([
    ("frame_id", pa.uint32()), ("profile_mask", pa.uint64()), ("score", pa.float64()),
    ("x0", pa.float64()), ("y0", pa.float64()), ("x1", pa.float64()),
    ("y1", pa.float64()), ("cx", pa.float64()), ("cy", pa.float64()),
    ("det_idx", pa.uint16()),
])


def _profile_frames(profiles_dir, maps_dir, prefixes=SEARCH_PREFIXES):
    manifest = json.loads((profiles_dir / "manifest.json").read_text())
    names = sorted(manifest["profiles"])
    if len(names) > 64:
        raise ValueError("object store supports at most 64 FAISS profiles")
    masks = {name: 1 << index for index, name in enumerate(names)}
    frames = {}
    targets = defaultdict(list)
    maps = {}
    for name in names:
        directory = profiles_dir / manifest["profiles"][name]["directory"]
        rows = pq.read_table(directory / "video_meta.parquet", columns=["video_idx", "video_id"]).to_pylist()
        videos = {row["video_idx"]: row["video_id"] for row in rows}
        video_indexes = np.load(directory / "id_video.npy", mmap_mode="r")
        frame_indexes = np.load(directory / "id_frame.npy", mmap_mode="r")
        for video_idx, frame_idx in zip(video_indexes, frame_indexes):
            video_id, frame_idx = videos[int(video_idx)], int(frame_idx)
            if not video_id.startswith(prefixes):
                continue
            key = (video_id, frame_idx)
            frame = frames.get(key)
            if frame is None:
                mapping = maps.get(video_id)
                if mapping is None:
                    mapping = maps[video_id] = frame_maps.load_path(maps_dir / f"{video_id}.npz")
                if frame_idx < 0 or frame_idx >= len(mapping.pts):
                    raise ValueError(f"FAISS frame is outside its frame map: {key}")
                frame = {"id": len(frames) + 1, "video_id": video_id, "frame_idx": frame_idx,
                         "time": mapping.frame_time(frame_idx), "pts": int(mapping.pts[frame_idx]),
                         "profile_mask": 0}
                frames[key] = frame
                targets[video_id].append((frame["time"], frame["id"]))
            frame["profile_mask"] |= masks[name]
        print(f"profile {name}: {len(video_indexes)} indexed keyframes", flush=True)
    ordered = {video: ([row[0] for row in sorted(values)], [row[1] for row in sorted(values)])
               for video, values in targets.items()}
    return frames, ordered, maps, masks


def _read_sources(sources, stage, targets, maps, maps_dir):
    best = {}
    labels = set()
    stats = {"rows": 0, "invalid_time": 0, "invalid_box": 0,
             "parquet_frames": 0, "parquet_detections": 0}
    partitioning = ds.partitioning(pa.schema([("video_id", pa.string())]), flavor="hive")
    for prefix, source in sources:
        parquet = pq.ParquetFile(source)

        def batches():
            for batch in parquet.iter_batches(batch_size=4096, columns=SOURCE_COLUMNS):
                playback = []
                for row in batch.to_pylist():
                    stats["rows"] += 1
                    video_id = row["video_id"].replace("-", "_")
                    if not video_id.startswith(prefix):
                        continue
                    frame_idx, second = int(row["frame_idx"]), float(row["time"])
                    names, scores, boxes = row["labels"], row["scores"], row["boxes"]
                    if not (len(names) == len(scores) == len(boxes)):
                        raise ValueError(f"misaligned detection arrays: {video_id}/{frame_idx}")
                    for label, score, box in zip(names, scores, boxes):
                        if (not label or len(box) != 4 or not math.isfinite(score)
                                or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)):
                            stats["invalid_box"] += 1
                            raise ValueError(f"invalid detection: {video_id}/{frame_idx}")
                        labels.add(label)
                    if frame_idx < 0 or not math.isfinite(second):
                        stats["invalid_time"] += 1
                        continue
                    playback.append({"video_id": video_id, "frame_idx": frame_idx, "time": second,
                                     "labels": names, "scores": scores, "boxes": boxes})
                    stats["parquet_frames"] += 1
                    stats["parquet_detections"] += len(names)
                    target = targets.get(video_id)
                    if target is None:
                        continue
                    mapping = maps.get(video_id)
                    if mapping is None:
                        mapping = maps[video_id] = frame_maps.load_path(maps_dir / f"{video_id}.npz")
                    if (frame_idx >= len(mapping.pts)
                            or abs(mapping.frame_time(frame_idx) - second) > 0.01):
                        stats["invalid_time"] += 1
                        continue
                    seconds, frame_ids = target
                    start = bisect.bisect_left(seconds, second - MAX_DELTA)
                    stop = bisect.bisect_right(seconds, second + MAX_DELTA)
                    for position in range(start, stop):
                        frame_id = frame_ids[position]
                        delta = abs(seconds[position] - second)
                        old = best.get(frame_id)
                        if old is None or (delta, second) < old[:2]:
                            best[frame_id] = (delta, second, frame_idx, names, scores, boxes)
                if playback:
                    yield pa.RecordBatch.from_pylist(playback, schema=PLAYBACK_SCHEMA)

        ds.write_dataset(
            batches(), stage / "playback", schema=PLAYBACK_SCHEMA, format="parquet",
            partitioning=partitioning, basename_template=f"{prefix}-part-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore", max_open_files=64,
            max_rows_per_group=4096,
        )
        print(f"source {prefix}: {parquet.metadata.num_rows} detection frames read", flush=True)
    stats["aligned"] = len(best)
    return best, sorted(labels), stats


def _write_search(stage, frames, best, labels, masks, stats):
    by_id = {frame["id"]: frame for frame in frames.values()}
    label_ids = {label: index for index, label in enumerate(labels)}
    writers = {}
    buffers = defaultdict(list)
    frame_writer = pq.ParquetWriter(stage / "keyframes.parquet", FRAME_SCHEMA, compression="zstd")
    profile_counts = {name: 0 for name in masks}
    search_detections = 0
    try:
        frame_buffer = []
        for frame_id in sorted(best):
            frame = by_id[frame_id]
            delta, second, source_idx, names, scores, boxes = best[frame_id]
            frame_buffer.append({**frame, "source_frame_idx": source_idx,
                                 "source_time": second, "delta": delta})
            for name, mask in masks.items():
                profile_counts[name] += bool(frame["profile_mask"] & mask)
            if len(frame_buffer) == 50_000:
                frame_writer.write_table(pa.Table.from_pylist(frame_buffer, schema=FRAME_SCHEMA))
                frame_buffer.clear()
            for det_idx, (label, score, box) in enumerate(zip(names, scores, boxes)):
                if det_idx > 65535:
                    raise ValueError(f"too many detections on frame {frame_id}")
                y0, x0, y1, x1 = box
                index = label_ids[label]
                buffers[index].append({"frame_id": frame_id, "profile_mask": frame["profile_mask"],
                                       "score": score, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                                       "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2, "det_idx": det_idx})
                search_detections += 1
                if len(buffers[index]) >= 50_000:
                    writer = writers.get(index)
                    if writer is None:
                        writer = writers[index] = pq.ParquetWriter(
                            stage / "labels" / f"{index:03d}.parquet", DETECTION_SCHEMA, compression="zstd"
                        )
                    writer.write_table(pa.Table.from_pylist(buffers[index], schema=DETECTION_SCHEMA))
                    buffers[index].clear()
        if frame_buffer:
            frame_writer.write_table(pa.Table.from_pylist(frame_buffer, schema=FRAME_SCHEMA))
        for index, rows in buffers.items():
            if not rows:
                continue
            writer = writers.get(index)
            if writer is None:
                writer = writers[index] = pq.ParquetWriter(
                    stage / "labels" / f"{index:03d}.parquet", DETECTION_SCHEMA, compression="zstd"
                )
            writer.write_table(pa.Table.from_pylist(rows, schema=DETECTION_SCHEMA))
    finally:
        frame_writer.close()
        for writer in writers.values():
            writer.close()
    stats.update(keyframes=len(best), detections=search_detections)
    return {name: {"mask": mask, "keyframes": profile_counts[name]} for name, mask in masks.items()}


def build(sources, profiles_dir, maps_dir, output_dir, prefixes=SEARCH_PREFIXES):
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".objects-", dir=output_dir))
    try:
        (stage / "labels").mkdir()
        frames, targets, maps, masks = _profile_frames(profiles_dir, maps_dir, prefixes)
        best, labels, stats = _read_sources(sources, stage, targets, maps, maps_dir)
        profiles = _write_search(stage, frames, best, labels, masks, stats)
        (stage / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "labels": labels, "profiles": profiles,
            "alignment_max_seconds": MAX_DELTA, "stats": stats,
        }, indent=2))
        stage.chmod(0o755)
        generation = output_dir / f"polars-{uuid4().hex[:12]}"
        os.replace(stage, generation)
        pointer = output_dir / f".current-{uuid4().hex[:8]}"
        try:
            pointer.symlink_to(generation.name, target_is_directory=True)
            os.replace(pointer, output_dir / "current")
        finally:
            pointer.unlink(missing_ok=True)
        return {**stats, "output": str(generation)}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="PREFIX=PARQUET")
    parser.add_argument("--profiles-dir", type=Path, default=ROOT / "indexes" / "profiles")
    parser.add_argument("--maps-dir", type=Path, default=ROOT / "indexes" / "frame_maps")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "indexes" / "objects")
    parser.add_argument("--search-prefixes", default=",".join(SEARCH_PREFIXES),
                        help="comma-separated video groups whose keyframes may be matched; "
                             "every other group stays playback-only")
    args = parser.parse_args(argv)
    sources = []
    for entry in args.source:
        prefix, separator, value = entry.partition("=")
        if not separator or prefix not in {"L", "M", "S", "N"} or not Path(value).is_file():
            parser.error(f"invalid source {entry!r}; expected L|M|S|N=/path/file.parquet")
        sources.append((prefix, Path(value)))
    if not {"L", "M", "S"}.issubset({prefix for prefix, _ in sources}):
        parser.error("L, M and S sources are required for keyframe search")
    prefixes = tuple(p.strip() for p in args.search_prefixes.split(",") if p.strip())
    if not prefixes:
        parser.error("--search-prefixes must name at least one group")
    started = time.monotonic()
    print(json.dumps({**build(sources, args.profiles_dir, args.maps_dir, args.output_dir, prefixes),
                      "seconds": round(time.monotonic() - started, 1)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
