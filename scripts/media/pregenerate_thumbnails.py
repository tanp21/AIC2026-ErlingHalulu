"""Pre-warm the thumbnail cache for every indexed keyframe.

    uv run python -m scripts.media.pregenerate_thumbnails
    uv run python -m scripts.media.pregenerate_thumbnails --year 2026 --workers 16
    uv run python -m scripts.media.pregenerate_thumbnails --video-ids K09_V017,K09_V019

--video-ids re-processes just the named videos regardless of --year/--groups --
handy after replacing a corrupted source file (re-download it, drop it in
place at its existing data/<year>/videos/<group>/<id>.mp4 path, then run this
with --video-ids to backfill just its cache entries instead of re-touching
the whole corpus).

Live search-result browsing hits `GET /thumb/{video_id}/{frame_idx}` (default
width 320), which decodes on demand via ffmpeg + exact PTS
(`backend.thumb.extract`). Measured on the production host (6 cores/12
threads) with each ffmpeg call pinned to `-threads 1`: bulk decode throughput
scales close to linearly up to ~12 concurrent calls (one per logical core)
and plateaus around 35 req/s beyond that -- CPU-bound video decode, not an
artificial limit. A team of 5-10 people scrolling through a full search
result page (up to `SEARCH_RESULT_LIMIT`, currently 10,000) would still take
minutes for a first-time (cache-cold) pass.

This script pays that decode cost once, offline, for every keyframe that is
actually indexed (`indexes/id_video.npy` + `id_frame.npy` +
`video_meta.parquet` -- the same source `backend.core.ProfileStore` reads),
by calling the exact same `backend.thumb.extract()` the live route calls.
Once warm, `/thumb` serves these as pure cache hits (`out.exists()` short
-circuits before any ffmpeg call) -- no code change needed in the request
path, and re-running this script is cheap/resumable since existing cache
entries are skipped.

At ~850k indexed keyframes and the measured ~35 req/s ceiling, a full corpus
run takes on the order of 7 hours. Run it during the off-season break;
--year/--groups let you shard/checkpoint the work.
"""
import argparse
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import frame_maps, thumb  # noqa: E402

INDEX_DIR = Path(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes"))


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_indexed_pairs(index_dir, years, groups, video_ids):
    id_video = np.load(index_dir / "id_video.npy")
    id_frame = np.load(index_dir / "id_frame.npy")
    rows = pq.read_table(index_dir / "video_meta.parquet").to_pylist()
    by_idx = {row["video_idx"]: row for row in rows}

    frames_by_video = defaultdict(set)
    for video_idx, frame_idx in zip(id_video, id_frame):
        row = by_idx.get(int(video_idx))
        if row is None:
            continue
        if video_ids:
            if row["video_id"] not in video_ids:
                continue
        else:
            if years and row["year"] not in years:
                continue
            if groups and row["group"] not in groups:
                continue
        frames_by_video[row["video_id"]].add(int(frame_idx))
    return frames_by_video, by_idx


def warm_one(video_id, path, frame_idx, mapping, width):
    try:
        thumb.extract(path, frame_idx, mapping, w=width)
        return None
    except Exception as error:  # one bad frame must not sink the batch
        return f"{video_id}:{frame_idx}: {error}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--year", default="", help="comma-separated years, e.g. 2025,2026")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. L21,L22")
    parser.add_argument("--video-ids", default="",
                        help="comma-separated video ids, e.g. K09_V017,K09_V019 -- "
                             "overrides --year/--groups when set")
    parser.add_argument("--width", type=int, default=320, choices=(160, 320, 640),
                        help="matches the default /thumb width used by search results")
    parser.add_argument("--workers", type=int, default=16,
                        help="measured throughput plateau on the 6-core/12-thread "
                             "production host once each ffmpeg call is pinned to "
                             "-threads 1 (see backend.thumb.ffmpeg_command); raise "
                             "only on faster/more-core hardware")
    parser.add_argument("--limit", type=int, default=0, help="cap total keyframes (0 = all)")
    args = parser.parse_args(argv)

    years = {y.strip() for y in args.year.split(",") if y.strip()}
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    video_ids = {v.strip() for v in args.video_ids.split(",") if v.strip()}

    # This is an offline batch with no live traffic to share fairly with, so run at
    # the measured ceiling regardless of the live-serving default in backend.thumb.
    thumb._ffmpeg_slots = threading.BoundedSemaphore(args.workers)

    frames_by_video, by_idx = load_indexed_pairs(args.index_dir, years, groups, video_ids)
    jobs = []
    for video_id, frame_idxs in frames_by_video.items():
        row = next(r for r in by_idx.values() if r["video_id"] == video_id)
        mapping = frame_maps.load(video_id)
        for frame_idx in sorted(frame_idxs):
            jobs.append((video_id, row["path"], frame_idx, mapping))
    if args.limit:
        jobs = jobs[: args.limit]

    if not jobs:
        log("nothing to do: no indexed keyframes matched the given filters")
        return 0
    log(f"{len(jobs)} keyframes across {len(frames_by_video)} videos, "
        f"width={args.width}, {args.workers} workers")

    started = time.perf_counter()
    done = 0
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(warm_one, video_id, path, frame_idx, mapping, args.width): video_id
            for video_id, path, frame_idx, mapping in jobs
        }
        for future in as_completed(futures):
            error = future.result()
            if error:
                failures.append(error)
                log(f"FAIL {error}")
                continue
            done += 1
            if done % 2000 == 0:
                elapsed = time.perf_counter() - started
                rate = done / elapsed
                remaining = (len(jobs) - done) / rate if rate else 0
                log(f"{done}/{len(jobs)} ({rate:.1f}/s, ~{remaining / 60:.0f} min remaining)")

    elapsed = time.perf_counter() - started
    log(f"done: {done} keyframes in {elapsed / 60:.1f} min, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
