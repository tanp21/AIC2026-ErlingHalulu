"""Build a frame-id manifest for Batch 2 2026, for re-extraction on Kaggle later.

    uv run python -m scripts.extraction.build_batch2_kaggle_manifest \
        --source data/2026/embeddings_batch2/pe-core/meta.parquet \
        --root /kaggle/input/datasets/<owner>/aic-batch2-2026-videos \
        --out data/kaggle_manifests/batch2_2026.json

Same schema as ``data/kaggle_manifests/batch1_2026.json`` (verified directly against the
uploaded ``tommr211/aic-keyframes`` Kaggle dataset, which holds only these JSON manifests --
no images):

    {"batch": "batch2", "year": "2026", "root": "...", "source": "...",
     "video_count": N, "keyframe_count": M,
     "videos": {"<video_id>": {"rel": "...", "fps": ..., "frames": [int, ...]}}}

``frames`` is the plain list of keyframe frame ids from the PE-Core embedding
``meta.parquet`` -- no PNGs are packaged or uploaded. On the Kaggle side,
``scripts/extraction/kaggle_extract_keyframes.py --layout flat`` re-derives the PNGs by
``ffmpeg``-seeking those exact frame ids out of the mounted video zip, so this only builds
after Batch 2's PE-Core embedding (and thus its ``meta.parquet``) already exists.

Unlike ``scripts/extraction/build_kaggle_manifest.py`` (built for L's multi-zip-per-group
Kaggle layout, e.g. ``Videos_L26_c``), Batch 2 ships one zip per group, so ``rel`` is just
``<group>/<video_id>.mp4`` under ``--root`` (the ``--layout flat`` resolution
``kaggle_extract_keyframes.py`` already supports) -- no zip-layout index needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def log(message):
    print(message, flush=True)


def load_frames(meta_path):
    """video_id -> sorted unique frame indices, straight from an embedding meta.parquet."""
    table = pq.read_table(meta_path, columns=["video_id", "frame_idx"])
    video_ids = np.asarray(table["video_id"].to_pylist())
    frame_idx = table["frame_idx"].to_numpy()
    order = np.argsort(video_ids, kind="stable")
    video_ids, frame_idx = video_ids[order], frame_idx[order]
    unique, starts = np.unique(video_ids, return_index=True)
    return {str(vid): sorted({int(f) for f in chunk})
           for vid, chunk in zip(unique, np.split(frame_idx, starts[1:]))}


def load_video_meta(video_meta_path):
    """video_id -> (group, fps), from indexes/video_meta.parquet."""
    table = pq.read_table(video_meta_path, columns=["video_id", "group", "fps"]).to_pydict()
    out = {}
    for video_id, group, fps in zip(table["video_id"], table["group"], table["fps"]):
        if fps is None:
            raise ValueError(f"{video_id} has no fps in {video_meta_path}")
        out[video_id] = (group, float(fps))
    return out


def build_manifest(batch, year, source, root, frames_by_video, video_meta):
    """Pure assembly of the manifest dict -- no I/O, exercised directly by unit tests."""
    videos = {}
    for video_id in sorted(frames_by_video):
        if video_id not in video_meta:
            raise KeyError(f"{video_id} is in {source} but not in video_meta")
        group, fps = video_meta[video_id]
        videos[video_id] = {
            "rel": f"{group}/{video_id}.mp4",
            "fps": fps,
            "frames": frames_by_video[video_id],
        }
    return {
        "batch": batch,
        "year": year,
        "root": root,
        "source": str(source),
        "video_count": len(videos),
        "keyframe_count": sum(len(v["frames"]) for v in videos.values()),
        "videos": videos,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="PE-Core embedding meta.parquet for batch 2")
    parser.add_argument("--video-meta-path", type=Path, default=Path("indexes/video_meta.parquet"))
    parser.add_argument("--root", required=True,
                        help="where the batch-2 video zips will be mounted on Kaggle, "
                             "e.g. /kaggle/input/datasets/<owner>/aic-batch2-2026-videos")
    parser.add_argument("--batch", default="batch2")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--out", type=Path, default=Path("data/kaggle_manifests/batch2_2026.json"))
    args = parser.parse_args(argv)

    frames_by_video = load_frames(args.source)
    video_meta = load_video_meta(args.video_meta_path)
    manifest = build_manifest(args.batch, args.year, args.source, args.root,
                              frames_by_video, video_meta)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest) + "\n")
    log(f"{args.out}: {manifest['video_count']} videos, "
        f"{manifest['keyframe_count']} keyframes (no images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
