#!/usr/bin/env python3
"""Extract keyframes from videos on Kaggle, driven by a manifest.

Self-contained on purpose: copy this file (or paste it into a cell) next to a manifest
from `scripts.extraction.build_kaggle_manifest` and run it. Only ffmpeg and the Python
standard library are required; tqdm is used when available.

    python kaggle_extract_keyframes.py \
        --manifest /kaggle/input/aic-keyframe-manifests/batch1_2026.json \
        --out /kaggle/working/keyframes --workers 4

Videos listed in the manifest but absent from the mount are skipped, so a notebook
with only one of the batch2 parts attached still works. Pass --strict to fail instead.

Frame index means decode order, matching how the original keyframes were cut and how
indexes/frame_maps maps frames to time. Output is <out>/<group>/<video_id>/%06d.png,
which is the layout scripts/embedding/embed_keyframes.py expects.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_valid_png(p):
    try:
        if p.stat().st_size == 0:
            return False
        with open(p, "rb") as fh:
            return fh.read(8) == PNG_SIG
    except OSError:
        return False


def extract_video_batch(video, frame_indices, out_dir, compression, fps, threads):
    """Extract all missing keyframes of one video, exact frame index.

    Uses `select='eq(n,rel)+...'` over ~100-frame chunks seeked with `-ss` so each
    chunk decodes only its slice (total ~= one pass). Frame index == decode order
    == the index TransNetV2 saw via cv2 (validated pixel-exact, including B-frames).
    Returns None on success, else the exception (for a retry pass).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = sorted({i for i in frame_indices
                      if not is_valid_png(out_dir / f"{i:06d}.png")})
    if not missing:
        return None
    CHUNK = 100  # ffmpeg select expression hits "Cannot allocate memory" beyond ~110 terms
    tmp_dir = out_dir / ".batch"
    try:
        for start in range(0, len(missing), CHUNK):
            chunk = missing[start:start + CHUNK]
            rel = [i - chunk[0] for i in chunk]
            sel = "+".join(f"eq(n\\,{i})" for i in rel)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-threads", str(threads),
                "-ss", f"{chunk[0] / fps:.6f}",
                "-t", f"{(chunk[-1] - chunk[0] + 2) / fps:.6f}",
                "-i", str(video),
                "-vf", f"select='{sel}'", "-vsync", "0",
                "-c:v", "png", "-pix_fmt", "rgb24",
                "-compression_level", str(compression),
                "-start_number", "0", str(tmp_dir / "f%06d.png"),
            ], check=True, capture_output=True, timeout=1800)
            outs = sorted(tmp_dir.glob("f*.png"))
            if len(outs) != len(chunk):
                raise RuntimeError(
                    f"select produced {len(outs)} frames, expected {len(chunk)}")
            for idx, o in zip(chunk, outs):
                o.replace(out_dir / f"{idx:06d}.png")
        return None
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return e


def group_of(video_id):
    return video_id.split("_", 1)[0]


def video_path(root, video_id, entry, layout):
    """Where a manifest entry's mp4 should live under `root`."""
    if layout == "flat":
        return root / group_of(video_id) / f"{video_id}.mp4"
    return root / entry["rel"]


def select_videos(manifest, only):
    """Manifest entries filtered by a set of group names and/or video ids."""
    videos = manifest["videos"]
    if not only:
        return dict(sorted(videos.items()))
    wanted = {t.strip() for t in only.split(",") if t.strip()}
    picked = {v: e for v, e in videos.items() if v in wanted or group_of(v) in wanted}
    unknown = wanted - set(picked) - {group_of(v) for v in picked}
    if unknown:
        raise SystemExit(f"--only matched nothing for: {', '.join(sorted(unknown))}")
    return dict(sorted(picked.items()))


def resolve(root, videos, layout, strict):
    """Split manifest entries into ones whose mp4 exists and ones that are missing."""
    found, missing = {}, []
    for video_id, entry in videos.items():
        path = video_path(root, video_id, entry, layout)
        if path.exists():
            found[video_id] = (path, entry)
        else:
            missing.append(video_id)
    if missing and strict:
        raise SystemExit(
            f"{len(missing)} videos are missing under {root}, first: {missing[0]}")
    return found, missing


def zip_group(out_dir, group, dest_dir):
    """Store (no deflate — PNGs are already compressed) one group into <group>.zip."""
    dest = dest_dir / f"{group}.zip"
    tmp = dest.with_suffix(".zip.tmp")
    src = out_dir / group
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for png in sorted(src.rglob("*.png")):
            zf.write(png, str(png.relative_to(out_dir)))
    tmp.replace(dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="/kaggle/working/keyframes")
    parser.add_argument("--root", default=None,
                        help="override the manifest's root (default: manifest root)")
    parser.add_argument("--layout", choices=("kaggle", "flat"), default="kaggle",
                        help="kaggle: <root>/<rel> from the manifest; "
                             "flat: <root>/<group>/<video_id>.mp4")
    parser.add_argument("--only", default=None,
                        help="comma-separated group names and/or video ids")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2,
                        help="ffmpeg decode threads per video")
    parser.add_argument("--compression", type=int, default=1,
                        help="PNG compression level, 0-9")
    parser.add_argument("--strict", action="store_true",
                        help="fail if any selected video is missing from the mount")
    parser.add_argument("--zip-per-group", default=None, metavar="DIR",
                        help="after extraction, write <group>.zip into DIR")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.root if args.root is not None else manifest["root"])
    out_dir = Path(args.out)

    videos = select_videos(manifest, args.only)
    found, missing = resolve(root, videos, args.layout, args.strict)
    frames_total = sum(len(e["frames"]) for _, e in found.values())
    log(f"{manifest['batch']}/{manifest['year']}: {len(found)} videos, "
        f"{frames_total} keyframes"
        + (f"; skipped {len(missing)} not on this mount" if missing else ""))
    if not found:
        return

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    def work(item):
        video_id, (path, entry) = item
        target = out_dir / group_of(video_id) / video_id
        return video_id, extract_video_batch(
            path, entry["frames"], target, args.compression, entry["fps"], args.threads)

    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in found.items()]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="video"):
            video_id, err = fut.result()
            if err is not None:
                failures[video_id] = err

    if failures:
        log(f"retrying {len(failures)} failed videos")
        retry = dict(failures)
        failures = {}
        for video_id in retry:
            _, err = work((video_id, found[video_id]))
            if err is not None:
                failures[video_id] = err

    if args.zip_per_group:
        dest_dir = Path(args.zip_per_group)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for group in sorted({group_of(v) for v in found}):
            log(f"zipped {zip_group(out_dir, group, dest_dir)}")

    if failures:
        for video_id, err in sorted(failures.items()):
            log(f"FAILED {video_id}: {err}")
        sys.exit(1)
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
