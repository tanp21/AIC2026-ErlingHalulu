"""Remux videos for fast HTTP playback start, without touching the originals.

    uv run python -m scripts.media.faststart_remux --year 2026
    uv run python -m scripts.media.faststart_remux --year 2026 --groups L21,L22

Writes `data/<year>/videos_faststart/<group>/<video_id>.mp4` -- a lossless
`-c copy -movflags +faststart` remux (moves the `moov` atom to the front of
the file; the coded picture/audio stream and its PTS/DTS values are
byte-for-byte unchanged, only the container layout moves).

Source videos here have `moov` trailing at the end of the file, so a player
must round-trip to the tail of a (frequently 100+ MB) file before it can
start playing anything -- a real, measured contributor to multi-minute
video-open delays under concurrent load. This script never reads or writes
anything under `videos/` itself, so it is invisible to
`scripts.extraction.detect_shots`, `scripts.embedding.embed_keyframes`, and
`backend.frame_maps`, all three of which only look under `videos/`. Frame_id
computation for competition submissions is therefore unaffected regardless
of when or how this script runs.

Safe to re-run: skips videos whose faststart copy already exists unless
--force. Writes to a temporary file and atomically renames into place, so a
crash mid-remux never leaves a partial/corrupt file where the server could
serve it.
"""
import argparse
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def remux_one(job):
    src, dst = Path(job[0]), Path(job[1])
    started = time.perf_counter()
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-map", "0", "-c", "copy", "-movflags", "+faststart",
            str(tmp),
        ],
        check=True,
        timeout=300,
    )
    tmp.replace(dst)  # atomic rename within the same filesystem
    return {"video": src.name, "seconds": time.perf_counter() - started}


def collect_jobs(videos_dir, out_dir, groups, force):
    jobs = []
    for video in sorted(videos_dir.rglob("*.mp4")):
        group = video.parent.name
        if groups and group not in groups:
            continue
        dst = out_dir / group / video.name
        if dst.exists() and not force:
            continue
        jobs.append((str(video), str(dst)))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026", help="dataset year under the data directory")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. L21,L22")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel ffmpeg remuxes; this is I/O-bound stream copying "
                             "(no decode/encode), so more than a handful rarely helps")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--force", action="store_true", help="redo videos already remuxed")
    args = parser.parse_args(argv)

    videos_dir = args.data_dir / args.year / "videos"
    out_dir = args.data_dir / args.year / "videos_faststart"
    if not videos_dir.is_dir():
        raise SystemExit(f"no videos under {videos_dir}")
    groups = {name.strip() for name in args.groups.split(",") if name.strip()}

    jobs = collect_jobs(videos_dir, out_dir, groups, args.force)
    if not jobs:
        log(f"nothing to do: every video under {videos_dir} already has a faststart copy")
        return 0
    log(f"{len(jobs)} videos -> {out_dir} ({args.workers} workers)")

    started = time.perf_counter()
    done = 0
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(remux_one, job): job for job in jobs}
        for future in as_completed(futures):
            name = Path(futures[future][0]).name
            try:
                result = future.result()
            except Exception as error:  # one bad file must not sink the batch
                failures.append((name, error))
                log(f"FAIL {name}: {error}")
                continue
            done += 1
            log(f"{done}/{len(jobs)} {name}: {result['seconds']:.1f}s")

    elapsed = time.perf_counter() - started
    log(f"done: {done} videos in {elapsed / 60:.1f} min, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
