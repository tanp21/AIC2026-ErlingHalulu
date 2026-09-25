"""Extract shot-based keyframes for Batch 2 2026, straight to local disk.

    uv run python -m scripts.extraction.extract_batch2_keyframes \
        --year 2026 --groups S01,M01,M02,M03,M04,M05,M06,M07,M08,M09,M10 --workers 8

Same keyframe-selection algorithm Batch 1 2026 was cut with (3 keyframes per shot at
fractional positions [0.15, 0.5, 0.85] of ``[start, end]``, see
``scripts/extraction/extract_keyframes_remote.py``'s ``compute_keyframes``), and the same
exact-frame ffmpeg extraction technique. Unlike that script, this reads shots and writes
keyframes straight from/to local disk (``data/<year>/shots`` -> ``data/<year>/keyframes``) --
no aria2c/rclone/Google Drive, since a rented GPU box already has the videos and shots local.

Output: ``data/<year>/keyframes/<group>/<video_id>/<frame_idx:06d>.png``, the layout
``scripts/embedding/embed_all.py``-style embedders read.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))

POSITIONS = (0.15, 0.5, 0.85)
CHUNK = 100  # ffmpeg select expression hits "Cannot allocate memory" beyond ~110 terms
# For sparse, long content (observed: Batch 2's S01 group -- ~2.75h videos, few widely
# spaced shots) a 100-frame chunk can span 20+ minutes of source video even though the
# term count is fine; ffmpeg then fails outright (exit 251) or silently under-produces
# frames ("select produced N frames, expected 100"). Bound each chunk's decode window in
# wall-clock time too, not just frame count.
MAX_CHUNK_SPAN_SECONDS = 120
PNG_SIG = b"\x89PNG\r\n\x1a\n"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def compute_keyframes(shots, positions=POSITIONS):
    indices = []
    for start, end in shots:
        for pos in positions:
            idx = int(start + (end - start) * pos)
            indices.append(max(start, min(end, idx)))
    return sorted(set(indices))


def is_valid_png(path):
    try:
        if path.stat().st_size == 0:
            return False
        with open(path, "rb") as handle:
            return handle.read(8) == PNG_SIG
    except OSError:
        return False


def iter_chunks(missing, fps, max_count=CHUNK, max_span_seconds=MAX_CHUNK_SPAN_SECONDS):
    """Group sorted frame indices so each group is both <= max_count frames and spans
    <= max_span_seconds of source video, whichever limit is hit first.
    """
    chunk = []
    for idx in missing:
        if chunk and (len(chunk) >= max_count
                      or (idx - chunk[0]) / fps > max_span_seconds):
            yield chunk
            chunk = []
        chunk.append(idx)
    if chunk:
        yield chunk


def extract_video_batch(video, frame_indices, out_dir, compression, fps, threads):
    """Extract all missing keyframes of one video, exact frame index.

    Returns None on success, else the exception (for a retry pass).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = sorted({i for i in frame_indices if not is_valid_png(out_dir / f"{i:06d}.png")})
    if not missing:
        return None
    tmp_dir = out_dir / ".batch"
    try:
        for chunk in iter_chunks(missing, fps):
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
            for idx, out_path in zip(chunk, outs):
                out_path.replace(out_dir / f"{idx:06d}.png")
        return None
    except Exception as error:
        return error
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def collect_jobs(shots_dir, videos_dir, keyframes_dir, groups):
    jobs = []
    for shot_path in sorted(shots_dir.glob("*/*.json")):
        group = shot_path.parent.name
        if groups and group not in groups:
            continue
        video_id = shot_path.stem
        video_path = videos_dir / group / f"{video_id}.mp4"
        if not video_path.exists():
            log(f"MISSING video for {shot_path} -- skip")
            continue
        info = json.loads(shot_path.read_text())
        frames = compute_keyframes(info["shots"])
        out_dir = keyframes_dir / group / video_id
        jobs.append((video_path, frames, out_dir, info["fps"]))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. S01,M01")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--workers", type=int, default=8, help="videos extracted in parallel")
    parser.add_argument("--threads", type=int, default=2, help="ffmpeg decode threads per video")
    parser.add_argument("--compression", type=int, default=1, help="PNG compression level, 0-9")
    args = parser.parse_args(argv)

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    shots_dir = args.data_dir / args.year / "shots"
    videos_dir = args.data_dir / args.year / "videos"
    keyframes_dir = args.data_dir / args.year / "keyframes"

    jobs = collect_jobs(shots_dir, videos_dir, keyframes_dir, groups)
    if not jobs:
        log(f"nothing to do under {shots_dir}" + (f" for groups {sorted(groups)}" if groups else ""))
        return 0
    total_frames = sum(len(frames) for _, frames, _, _ in jobs)
    log(f"{len(jobs)} videos, {total_frames} keyframes -> {keyframes_dir} ({args.workers} workers)")

    def work(job):
        video_path, frames, out_dir, fps = job
        return video_path.name, extract_video_batch(
            video_path, frames, out_dir, args.compression, fps, args.threads)

    started = time.perf_counter()
    failures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, job): job for job in jobs}
        done = 0
        for future in as_completed(futures):
            name, error = future.result()
            done += 1
            if error is not None:
                failures[name] = error
                log(f"FAIL {name}: {error}")
                continue
            if done % 10 == 0 or done == len(jobs):
                log(f"{done}/{len(jobs)} videos extracted")

    if failures:
        log(f"retrying {len(failures)} failed videos")
        retry_names = set(failures)
        failures = {}
        for job in jobs:
            if job[0].name not in retry_names:
                continue
            name, error = work(job)
            if error is not None:
                failures[name] = error

    elapsed = time.perf_counter() - started
    log(f"done: {len(jobs) - len(failures)}/{len(jobs)} videos in {elapsed / 60:.1f} min, "
        f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
