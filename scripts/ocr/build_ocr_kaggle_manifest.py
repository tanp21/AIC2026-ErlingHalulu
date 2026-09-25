"""Build a Kaggle-mountable frame manifest for OCR extraction, straight from shots.

    uv run python -m scripts.ocr.build_ocr_kaggle_manifest \\
        --shots-dir data/2026/shots --groups M01,M02,M03,M04,M05 \\
        --out data/kaggle_manifests/ocr_batch2_m01_m05_2026.json

Unlike ``scripts/extraction/build_batch2_kaggle_manifest.py`` (which derives frame ids from
an existing PE-Core embedding ``meta.parquet``), this reads ``data/<year>/shots/<group>/*.json``
directly and recomputes the standard 3-per-shot keyframe positions with
``scripts.extraction.extract_batch2_keyframes.compute_keyframes`` -- so it has no dependency on
an embedding pass having run first. That's the point: for a group whose keyframes have never
been extracted (no ``data/<year>/keyframes/<group>/`` on disk yet, as is the case for OCR-only
work on a fresh video group), the shots are the only local artifact needed.

Same manifest schema as ``build_batch2_kaggle_manifest.py`` (``batch/year/root/source/
video_count/keyframe_count/videos: {video_id: {rel, fps, frames}}``), except ``rel`` follows
this dataset's actual mounted layout, ``Videos_<group>/videos/<video_id>.mp4`` (confirmed
against the ``yunsoo21`` Kaggle dataset for Batch 2's M-groups), not the flat ``<group>/
<video_id>.mp4`` layout ``build_batch2_kaggle_manifest.py`` targets. No image data is packaged
or uploaded -- ``frames`` is a plain list of frame ids; the Kaggle-side notebook re-derives PNGs
by ``ffmpeg``-seeking those exact frame ids out of the mounted raw video.
"""
import argparse
import json
from pathlib import Path

from scripts.extraction.extract_batch2_keyframes import compute_keyframes


def log(message):
    print(message, flush=True)


def load_shots(shots_dir, groups):
    """video_id -> {"group", "fps", "frames"}, straight from shot boundary JSON files.

    ``frames`` is the standard 3-per-shot keyframe selection (see ``compute_keyframes``).
    Pure I/O + a pure-function call, kept separate from ``build_manifest`` so the assembly
    logic can be unit-tested without touching disk.
    """
    out = {}
    for shot_path in sorted(shots_dir.glob("*/*.json")):
        group = shot_path.parent.name
        if groups and group not in groups:
            continue
        video_id = shot_path.stem
        info = json.loads(shot_path.read_text())
        out[video_id] = {
            "group": group,
            "fps": float(info["fps"]),
            "frames": compute_keyframes(info["shots"]),
        }
    return out


def build_manifest(batch, year, source, root, videos_by_id):
    """Pure assembly of the manifest dict -- no I/O, exercised directly by unit tests."""
    videos = {}
    for video_id in sorted(videos_by_id):
        entry = videos_by_id[video_id]
        videos[video_id] = {
            "rel": f"Videos_{entry['group']}/videos/{video_id}.mp4",
            "fps": entry["fps"],
            "frames": entry["frames"],
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
    parser.add_argument("--shots-dir", required=True, type=Path,
                        help="e.g. data/2026/shots")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. M01,M02")
    parser.add_argument("--root", default="",
                        help="documents the eventual /kaggle/input/<owner>/<slug> mount path; "
                             "the notebook does its own runtime discovery and does not trust "
                             "this string blindly")
    parser.add_argument("--batch", default="batch2")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    videos_by_id = load_shots(args.shots_dir, groups)
    if not videos_by_id:
        raise SystemExit(f"no shots found under {args.shots_dir}"
                         + (f" for groups {sorted(groups)}" if groups else ""))

    manifest = build_manifest(args.batch, args.year, args.shots_dir, args.root, videos_by_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest) + "\n")
    log(f"{args.out}: {manifest['video_count']} videos, "
        f"{manifest['keyframe_count']} keyframes (no images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
