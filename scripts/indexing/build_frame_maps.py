#!/usr/bin/env python3
"""Precompute exact decoded-frame PTS maps for every indexed video."""
import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from backend import frame_maps


ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "indexes"
MAP_DIR = INDEX_DIR / "frame_maps"
MANIFEST = MAP_DIR / "manifest.json"


def parse_probe(data):
    streams = data.get("streams", [])
    if len(streams) != 1 or "time_base" not in streams[0]:
        raise ValueError("ffprobe did not return one video time base")
    time_base = Fraction(streams[0]["time_base"])
    raw_pts = [frame.get("best_effort_timestamp") for frame in data.get("frames", [])]
    if not raw_pts or any(value is None for value in raw_pts):
        raise ValueError("video has frames without best-effort PTS")
    return frame_maps.FrameMap(
        np.asarray(raw_pts, dtype=np.int64),
        time_base.numerator,
        time_base.denominator,
    )


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=time_base:frame=best_effort_timestamp",
        "-threads", "1", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return parse_probe(json.loads(result.stdout))


def probe_packets(path):
    """Fast timeline check only; never replaces a decoded-frame probe on mismatch."""
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=time_base:packet=pts", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    return parse_packet_probe(json.loads(result.stdout))


def parse_packet_probe(data):
    streams = data.get("streams", [])
    packets = data.get("packets", [])
    if len(streams) != 1 or "time_base" not in streams[0] or not packets or any("pts" not in p for p in packets):
        raise ValueError("packet PTS unavailable")
    time_base = Fraction(streams[0]["time_base"])
    return frame_maps.FrameMap(
        np.sort(np.asarray([p["pts"] for p in packets], dtype=np.int64)),
        time_base.numerator,
        time_base.denominator,
    )


def verify_or_probe(path, existing_map):
    """Reuse a decoded map only when the new file's entire PTS timeline is identical."""
    if existing_map.exists():
        try:
            packets = probe_packets(path)
            previous = frame_maps.load_path(existing_map)
            if (packets.time_base_num == previous.time_base_num
                    and packets.time_base_den == previous.time_base_den
                    and np.array_equal(packets.pts, previous.pts)):
                return previous, True
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass
    return probe(path), False


def signature(path):
    stat = Path(path).stat()
    return {"path": str(path), "size": stat.st_size, "mtimeNs": stat.st_mtime_ns}


def save_manifest(videos):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=MAP_DIR, mode="w", suffix=".json", delete=False) as tmp:
        temp_path = Path(tmp.name)
        json.dump({"version": 2, "videos": videos}, tmp, indent=2, sort_keys=True)
    try:
        os.replace(temp_path, MANIFEST)
    finally:
        temp_path.unlink(missing_ok=True)


def save_map(video_id, mapping):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    target = MAP_DIR / f"{video_id}.npz"
    with tempfile.NamedTemporaryFile(dir=MAP_DIR, suffix=".npz", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.savez_compressed(
            temp_path,
            pts=mapping.pts,
            time_base_num=np.int64(mapping.time_base_num),
            time_base_den=np.int64(mapping.time_base_den),
        )
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)


def aligned_times(id_video, id_frame, video_ids, load_map):
    times = np.empty(len(id_frame), dtype=np.float64)
    for video_idx in np.unique(id_video):
        rows = np.flatnonzero(id_video == video_idx)
        mapping = load_map(video_ids[int(video_idx)])
        frames = id_frame[rows]
        if np.any(frames < 0) or np.any(frames >= len(mapping.pts)):
            raise ValueError(f"indexed frame outside PTS map: {video_ids[int(video_idx)]}")
        times[rows] = (mapping.pts[frames] - mapping.pts[0]) * (mapping.time_base_num / mapping.time_base_den)
    return times


def refresh_times(directory, load_map=frame_maps.load):
    """Refresh search-result seconds from the same PTS maps used by DRES."""
    directory = Path(directory)
    rows = pq.read_table(directory / "video_meta.parquet").to_pylist()
    video_ids = {int(row["video_idx"]): row["video_id"] for row in rows}
    times = aligned_times(
        np.load(directory / "id_video.npy"), np.load(directory / "id_frame.npy"),
        video_ids, load_map,
    )
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".npy", delete=False) as tmp:
        temp_path = Path(tmp.name)
    try:
        np.save(temp_path, times)
        os.replace(temp_path, directory / "id_time.npy")
    finally:
        temp_path.unlink(missing_ok=True)
    return len(times)


def build(workers):
    MAP_DIR.mkdir(parents=True, exist_ok=True)
    rows = pq.read_table(INDEX_DIR / "video_meta.parquet").to_pylist()
    document = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    old = document.get("videos", {}) if document.get("version") == 2 else {}
    current = dict(old)
    pending = []
    for row in rows:
        video_id, path = row["video_id"], Path(row["path"])
        sig = signature(path)
        if old.get(video_id) == sig and (MAP_DIR / f"{video_id}.npz").exists():
            continue
        pending.append((video_id, path, sig))

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {executor.submit(verify_or_probe, path, MAP_DIR / f"{video_id}.npz"): (video_id, sig) for video_id, path, sig in pending}
        for job in as_completed(jobs):
            video_id, sig = jobs[job]
            try:
                mapping, verified = job.result()
                if not verified:
                    save_map(video_id, mapping)
                current[video_id] = sig
                save_manifest(current)
                print(f"[frame-map] {video_id}: {len(mapping.pts)} frames{' (PTS verified)' if verified else ' (decoded)'}", flush=True)
            except Exception as exc:
                failures.append(f"{video_id}: {exc}")

    save_manifest(current)
    if failures:
        raise RuntimeError("frame-map failures:\n" + "\n".join(failures))

    frame_maps.load.cache_clear()
    count = refresh_times(INDEX_DIR)
    print(f"[frame-map] base: {len(rows)} videos, {count} indexed timestamps")
    for directory in sorted((INDEX_DIR / "profiles").iterdir()) if (INDEX_DIR / "profiles").exists() else []:
        if directory.name.startswith(".") or not (directory / "id_frame.npy").exists():
            continue
        count = refresh_times(directory)
        print(f"[frame-map] {directory.name}: {count} indexed timestamps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    build(max(1, args.workers))


if __name__ == "__main__":
    main()
