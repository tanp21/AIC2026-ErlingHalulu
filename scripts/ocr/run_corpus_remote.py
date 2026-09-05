#!/usr/bin/env python3
"""OCR the 2026 corpus straight from source video zips -- no keyframe download.

Built for a rented GPU box with limited disk: downloads one group's video zip
from the public host (the same server `scripts/extraction/extract_keyframes_remote.py`
already uses for the 2025 batch), extracts *only* the exact keyframe frame_ids
listed in the manifest via ffmpeg's decode-order `select` filter -- the same
batched approach as `notebooks/kaggle_siglip2_embed.py::extract_exact_frames`
and `scripts/extraction/extract_keyframes_remote.py::extract_video_batch` --
runs each frame through the OCR pipeline preserving per-line boxes, then
deletes the extracted frame and, once a video's frames are all done, the video
itself. Total disk use stays bounded by one zip + one video's frames
regardless of corpus size.

Meant to run sharded across several worker processes on one instance to keep
the GPU fed while ffmpeg/download work happens on other cores: pass a
disjoint `--zips` list per worker, each writing its own `--out` file. Output
is the same ``{"<group>/<video_id>/<frame_idx>": {"lines": [...]}}`` shape as
`scripts/ocr/run_corpus.py`, so `scripts/indexing/build_ocr_index.py`'s
per-video suppression/ticker-region pipeline consumes it unchanged after the
shard files are merged.

Resumable: re-run with the same `--out` and it skips any `(group, video_id,
frame_idx)` already present, and only re-extracts a video's *missing* frames
via ffmpeg -- a partially-done video does not restart from scratch.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ocr.engines import OcrConfig, OcrPipeline  # noqa: E402

BASE_URL = "https://aic-data.ledo.io.vn"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest(path):
    """``{zip_name: [{video_id, group, rel, fps, frames}, ...]}``, from batch1_2026.json."""
    raw = json.loads(Path(path).read_text())
    by_zip: dict[str, list[dict]] = {}
    for video_id, entry in raw["videos"].items():
        rel = entry["rel"]  # e.g. aic-batch1-videos/Videos_L21_a/video/L21_V001.mp4
        zip_name = rel.split("/")[1]
        group = video_id.split("_V")[0]
        by_zip.setdefault(zip_name, []).append({
            "video_id": video_id, "group": group, "rel": rel,
            "fps": float(entry["fps"]), "frames": entry["frames"],
        })
    return by_zip


def download_zip(zip_name: str, work: Path, aria_conns: int) -> Path:
    dest = work / f"{zip_name}.zip"
    ctrl = work / f"{zip_name}.zip.aria2"
    if dest.exists() and not ctrl.exists():
        return dest
    log(f"downloading {zip_name}.zip ...")
    subprocess.run([
        "aria2c", "-d", str(work), "-o", f"{zip_name}.zip",
        "-x", str(aria_conns), "-s", str(aria_conns),
        "-c", "--file-allocation=none", "--max-tries=5", "--retry-wait=10",
        "--console-log-level=warn", "--summary-interval=0",
        f"{BASE_URL}/{zip_name}.zip",
    ], check=True)
    return dest


def extract_video_from_zip(zf: zipfile.ZipFile, member: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with zf.open(member) as src, open(tmp, "wb") as sink:
        shutil.copyfileobj(src, sink, length=8 * 1024 * 1024)
    tmp.replace(dest)


def extract_exact_frames(video: Path, frame_ids, fps: float, out_dir: Path, threads: int = 2):
    """ffmpeg exact decode-order frame extraction, chunked (select expressions
    blow up past ~110 terms)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    requested = sorted({int(f) for f in frame_ids})
    chunk_size = 100
    batch = out_dir / ".batch"
    for start in range(0, len(requested), chunk_size):
        chunk = requested[start:start + chunk_size]
        relative = [f - chunk[0] for f in chunk]
        select = "+".join(f"eq(n\\,{f})" for f in relative)
        shutil.rmtree(batch, ignore_errors=True)
        batch.mkdir()
        subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", str(threads),
            "-ss", f"{chunk[0] / fps:.6f}",
            "-t", f"{(chunk[-1] - chunk[0] + 2) / fps:.6f}",
            "-i", str(video),
            "-vf", f"select='{select}'", "-vsync", "0",
            "-c:v", "png", "-pix_fmt", "rgb24", "-compression_level", "1",
            "-start_number", "0", str(batch / "f%06d.png"),
        ], check=True, capture_output=True, timeout=1800)
        frames = sorted(batch.glob("f*.png"))
        if len(frames) != len(chunk):
            raise RuntimeError(
                f"{video.name}: ffmpeg returned {len(frames)} frames for {len(chunk)} ids"
            )
        for frame_id, frame in zip(chunk, frames):
            frame.replace(out_dir / f"{frame_id:06d}.png")
    shutil.rmtree(batch, ignore_errors=True)


def load_existing(out_path: Path) -> dict:
    if out_path.exists():
        try:
            return json.loads(out_path.read_text())
        except json.JSONDecodeError:
            log(f"WARNING: {out_path} is corrupt, starting fresh")
    return {}


def process_zip(zip_name, videos, work, results, out_path, pipeline, flush_every, stats):
    import cv2

    zpath = download_zip(zip_name, work, aria_conns=16)
    videos_dir = work / "videos_tmp" / zip_name

    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
        for v in videos:
            group, video_id, fps = v["group"], v["video_id"], v["fps"]
            missing = [f for f in v["frames"]
                       if f"{group}/{video_id}/{f}" not in results]
            if not missing:
                continue

            member = v["rel"] if v["rel"] in names else None
            if member is None:
                candidates = [n for n in names if n.endswith(f"/{video_id}.mp4")]
                if len(candidates) != 1:
                    log(f"  MISSING {video_id} in {zip_name} ({len(candidates)} candidates), skip")
                    continue
                member = candidates[0]

            video_path = videos_dir / f"{video_id}.mp4"
            frames_dir = videos_dir / f"{video_id}_frames"
            try:
                extract_video_from_zip(zf, member, video_path)
                extract_exact_frames(video_path, missing, fps, frames_dir)

                for frame_id in missing:
                    img_path = frames_dir / f"{frame_id:06d}.png"
                    if not img_path.exists():
                        log(f"  missing extracted frame {video_id}/{frame_id}")
                        continue
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue
                    try:
                        lines = pipeline.run(image)
                    except Exception as exc:  # keep the run alive on a bad frame
                        log(f"  OCR error {group}/{video_id}/{frame_id}: {exc}")
                        lines = []
                    results[f"{group}/{video_id}/{frame_id}"] = {"lines": lines}
                    stats["processed"] += 1
                    if stats["processed"] % flush_every == 0:
                        out_path.write_text(json.dumps(results))
                        elapsed = time.time() - stats["t0"]
                        log(f"  {stats['processed']} frames this run, "
                            f"{stats['processed']/elapsed:.2f} fps aggregate")
            except subprocess.CalledProcessError as exc:
                log(f"  FAILED {video_id}: {exc}")
            finally:
                video_path.unlink(missing_ok=True)
                shutil.rmtree(frames_dir, ignore_errors=True)

    out_path.write_text(json.dumps(results))
    zpath.unlink(missing_ok=True)
    log(f"{zip_name} DONE ({stats['processed']} frames so far this run)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "data_manifest_2026.json"))
    parser.add_argument("--zips", required=True, help="comma-separated zip names to process")
    parser.add_argument("--work", default="/workspace/scratch")
    parser.add_argument("--out", required=True)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--rec-batch-size", type=int, default=32)
    parser.add_argument("--rec-device", default="cuda:0")
    parser.add_argument("--det-use-cuda", action="store_true")
    parser.add_argument(
        "--small-rescan", dest="small_rescan", action="store_true", default=False,
        help="re-detect small text at high magnification (~2.3x slower/frame; only "
             "recovers very small isolated text like sub-20px digits -- off by default "
             "for full-corpus throughput, see docs/reports/ocr-vast-throughput.md)",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap frames processed (0 = no cap)")
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    by_zip = load_manifest(args.manifest)
    zip_names = [z.strip() for z in args.zips.split(",") if z.strip()]

    config = replace(
        OcrConfig(), rec_batch_size=args.rec_batch_size, rec_device=args.rec_device,
        det_use_cuda=args.det_use_cuda, small_rescan=args.small_rescan,
    )
    pipeline = OcrPipeline(config)

    results = load_existing(out_path)
    stats = {"processed": 0, "t0": time.time()}

    for zip_name in zip_names:
        videos = by_zip.get(zip_name)
        if not videos:
            log(f"unknown zip {zip_name!r}, skip")
            continue
        process_zip(zip_name, videos, work, results, out_path, pipeline, args.flush_every, stats)
        if args.limit and stats["processed"] >= args.limit:
            break

    out_path.write_text(json.dumps(results))
    log(f"ALL DONE: {len(results)} total frames in {out_path}")


if __name__ == "__main__":
    main()
