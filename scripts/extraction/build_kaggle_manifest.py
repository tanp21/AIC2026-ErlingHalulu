#!/usr/bin/env python3
"""Build the Kaggle keyframe manifests.

The keyframe PNGs are too large to upload to Kaggle, so only the video zips went up.
These manifests give a Kaggle notebook everything it needs to re-extract exactly the
frames the existing embeddings were built from: the video's path inside the Kaggle
mount, its fps, and its frame indices.

One file per batch/year:

  batch1_2025  L21..L30, frames from data/2025/embeddings/meta.parquet
  batch2_2025  K01..K20, frames from data/2025/embeddings/meta.parquet
  batch1_2026  L21..L30, frames from data/2026/embeddings/meta.parquet

Frame indices come from meta.parquet (not from data/*/keyframes on disk) because that
is what id_frame.npy in the served indexes references.

  uv run python -m scripts.extraction.build_kaggle_manifest --out data/kaggle_manifests
  uv run python -m scripts.extraction.build_kaggle_manifest --refresh-layout
"""
import argparse
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LAYOUT_PATH = Path(__file__).with_name("kaggle_zip_layout.json")

BASE_URL = "https://aic-data.ledo.io.vn"

# Zips whose contents must never enter the layout. Videos_L25_b duplicates
# L25_V050..L25_V088 from Videos_L25_a and was not uploaded to the Kaggle dataset.
SKIP_ZIPS = frozenset({"Videos_L25_b"})

# Every L zip on the host, including the ones SKIP_ZIPS drops, so --refresh-layout
# re-reads the full picture and the filtering stays visible in one place.
L_ZIPS = (
    "Videos_L21_a", "Videos_L22_a", "Videos_L23_a", "Videos_L24_a",
    "Videos_L25_a", "Videos_L25_b",
    "Videos_L26_a", "Videos_L26_b", "Videos_L26_c", "Videos_L26_d", "Videos_L26_e",
    "Videos_L27_a", "Videos_L28_a", "Videos_L29_a", "Videos_L30_a",
)

BATCH1_ROOT = "/kaggle/input/datasets/tommr211"
BATCH1_DATASET = "aic-batch1-videos"
BATCH2_ROOT = "/kaggle/input/datasets/taiphamnguyenanh"

# K01..K07 -> part1, K08..K14 -> part2, K15..K20 -> part3.
BATCH2_PARTS = ((7, "part1"), (14, "part2"), (20, "part3"))

MANIFESTS = (
    {
        "name": "batch1_2025",
        "batch": "batch1",
        "year": "2025",
        "prefix": "L",
        "source": "data/2025/embeddings/meta.parquet",
    },
    {
        "name": "batch2_2025",
        "batch": "batch2",
        "year": "2025",
        "prefix": "K",
        "source": "data/2025/embeddings/meta.parquet",
    },
    {
        "name": "batch1_2026",
        "batch": "batch1",
        "year": "2026",
        "prefix": "L",
        "source": "data/2026/embeddings/meta.parquet",
    },
)


def dataset_slug(batch, group):
    """Kaggle dataset directory a group lives in, relative to the manifest root."""
    if batch == "batch1":
        return BATCH1_DATASET
    number = int(group[1:])
    for last, part in BATCH2_PARTS:
        if number <= last:
            return f"aic-batch2-videos-{part}"
    raise KeyError(f"no batch2 part covers {group}")


def folder_index(layout):
    """video_id -> zip folder, rejecting a video listed under two folders."""
    index = {}
    for folder, members in layout.items():
        for video_id in members:
            if video_id in index:
                raise ValueError(
                    f"{video_id} appears in both {index[video_id]} and {folder}")
            index[video_id] = folder
    return index


def zip_folder(video_id, group, index):
    """Extracted-zip directory holding a video, e.g. Videos_L26_c or Videos_K01."""
    if video_id in index:
        return index[video_id]
    if group.startswith("K"):
        return f"Videos_{group}"
    raise KeyError(f"no zip folder for {video_id}")


def relative_path(video_id, group, batch, index):
    folder = zip_folder(video_id, group, index)
    return f"{dataset_slug(batch, group)}/{folder}/video/{video_id}.mp4"


def load_frames(meta_path, prefix):
    """video_id -> sorted unique frame indices, for videos whose id starts with prefix."""
    table = pq.read_table(meta_path, columns=["video_id", "frame_idx"])
    video_ids = np.asarray(table["video_id"].to_pylist())
    frame_idx = table["frame_idx"].to_numpy()
    keep = np.char.startswith(video_ids, prefix)
    video_ids, frame_idx = video_ids[keep], frame_idx[keep]

    order = np.argsort(video_ids, kind="stable")
    video_ids, frame_idx = video_ids[order], frame_idx[order]
    unique, starts = np.unique(video_ids, return_index=True)
    return {
        str(vid): sorted({int(f) for f in chunk})
        for vid, chunk in zip(unique, np.split(frame_idx, starts[1:]))
    }


def load_video_meta(path):
    """video_id -> (group, fps)."""
    table = pq.read_table(path, columns=["video_id", "group", "fps"]).to_pydict()
    out = {}
    for video_id, group, fps in zip(table["video_id"], table["group"], table["fps"]):
        if fps is None:
            raise ValueError(f"{video_id} has no fps in {path}")
        out[video_id] = (group, float(fps))
    return out


def build_manifest(spec, frames_by_video, video_meta, index):
    root = BATCH1_ROOT if spec["batch"] == "batch1" else BATCH2_ROOT
    videos = {}
    for video_id in sorted(frames_by_video):
        if video_id not in video_meta:
            raise KeyError(f"{video_id} is in {spec['source']} but not in video_meta")
        group, fps = video_meta[video_id]
        videos[video_id] = {
            "rel": relative_path(video_id, group, spec["batch"], index),
            "fps": fps,
            "frames": frames_by_video[video_id],
        }
    return {
        "batch": spec["batch"],
        "year": spec["year"],
        "root": root,
        "source": spec["source"],
        "video_count": len(videos),
        "keyframe_count": sum(len(v["frames"]) for v in videos.values()),
        "videos": videos,
    }


def load_layout(path=LAYOUT_PATH):
    return json.loads(Path(path).read_text())


# --- remote zip layout ------------------------------------------------------
#
# The host serves the source zips but honours Range only intermittently: roughly a
# third of requests stream the whole multi-GB object instead. So check the status and
# the advertised length before reading anything, and retry when it looks wrong.

# The host 403s the default Python-urllib User-Agent.
_HEADERS = {"User-Agent": "curl/8.5.0"}


def _remote_size(url, timeout=60):
    request = urllib.request.Request(url, headers=_HEADERS, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.headers["Content-Length"])


def _fetch_tail(url, size, span, attempts=6, timeout=180):
    start = max(0, size - span)
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(
            url, headers={**_HEADERS, "Range": f"bytes={start}-{size - 1}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            length = int(response.headers.get("Content-Length", size))
            if response.status == 206 and length <= span * 2:
                return response.read()
            # Range ignored; drop the connection rather than pull the whole object.
            print(f"  range ignored (attempt {attempt}, {length} bytes offered), retrying",
                  file=sys.stderr)
    raise RuntimeError(f"{url}: server never honoured a range request")


def parse_central_directory(buffer):
    """Entry names from every PK\\x01\\x02 record in a zip central-directory slice."""
    names, pos = [], 0
    while True:
        pos = buffer.find(b"PK\x01\x02", pos)
        if pos < 0 or pos + 46 > len(buffer):
            return names
        name_len, = struct.unpack("<H", buffer[pos + 28:pos + 30])
        end = pos + 46 + name_len
        if name_len and end <= len(buffer):
            try:
                names.append(buffer[pos + 46:end].decode("utf-8"))
            except UnicodeDecodeError:
                pass
        pos += 4


def fetch_zip_members(folder, span=1_500_000):
    url = f"{BASE_URL}/{folder}.zip"
    size = _remote_size(url)
    names = parse_central_directory(_fetch_tail(url, size, span))
    return sorted({Path(n).stem for n in names if n.endswith(".mp4")})


def refresh_layout(path=LAYOUT_PATH, span=1_500_000):
    layout = {}
    for folder in L_ZIPS:
        members = fetch_zip_members(folder, span=span)
        marker = " (skipped)" if folder in SKIP_ZIPS else ""
        print(f"{folder}: {len(members)} videos{marker}", flush=True)
        if folder not in SKIP_ZIPS:
            layout[folder] = members
    Path(path).write_text(json.dumps(layout, indent=1, sort_keys=True) + "\n")
    total = sum(len(v) for v in layout.values())
    print(f"wrote {path} ({len(layout)} folders, {total} videos)")
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/kaggle_manifests",
                        help="directory the manifest JSONs are written to")
    parser.add_argument("--repo-root", default=str(ROOT))
    parser.add_argument("--refresh-layout", action="store_true",
                        help="re-read the zip listings from the host and rewrite "
                             "kaggle_zip_layout.json, then exit")
    args = parser.parse_args()

    if args.refresh_layout:
        refresh_layout()
        return

    repo = Path(args.repo_root)
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = repo / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    index = folder_index(load_layout())
    video_meta = load_video_meta(repo / "indexes" / "video_meta.parquet")

    for spec in MANIFESTS:
        frames = load_frames(repo / spec["source"], spec["prefix"])
        manifest = build_manifest(spec, frames, video_meta, index)
        target = out_dir / f"{spec['name']}.json"
        target.write_text(json.dumps(manifest) + "\n")
        print(f"{target}: {manifest['video_count']} videos, "
              f"{manifest['keyframe_count']} keyframes")


if __name__ == "__main__":
    main()
