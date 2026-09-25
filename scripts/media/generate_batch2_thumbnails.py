"""Pre-warm the thumbnail cache for one Batch 2 group's freshly embedded keyframes.

Must run BEFORE the raw video for this group is deleted -- it decodes real frames via
ffmpeg. Uses the exact same cache key scheme as `backend/thumb.py`'s live `/thumb` route
(pts + time_base + width, keyed by video filename, via `backend.thumb.extract()`), so these
pre-generated JPGs are genuine cache hits once merged into wherever `AIC_THUMB_CACHE_DIR`
points on the production host -- not a separate cache scripts.media.pregenerate_thumbnails
wouldn't recognize.

    uv run python -m scripts.media.generate_batch2_thumbnails --year 2026 --groups M01 \
        --pe-core-meta data/2026/embeddings_batch2/pe-core/meta.parquet

Requires `scripts.indexing.build_batch2_frame_maps` to have already written frame_maps for
this group (needs the exact PTS map, not fps arithmetic).
"""
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from backend import frame_maps, thumb

DATA = Path(os.environ.get("AIC_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def log(message):
    print(message, flush=True)


def frames_by_video_for_groups(meta_path, groups):
    table = pq.read_table(meta_path, columns=["video_id", "frame_idx"]).to_pylist()
    by_video = {}
    for row in table:
        video_id = row["video_id"]
        if video_id[:3] not in groups:
            continue
        by_video.setdefault(video_id, []).append(int(row["frame_idx"]))
    return by_video


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--groups", required=True, help="comma-separated groups, e.g. S01,M01")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--pe-core-meta", type=Path, required=True,
                        help="meta.parquet whose (video_id, frame_idx) rows to thumbnail")
    parser.add_argument("--width", type=int, default=320, choices=(160, 320, 640))
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    if not groups:
        raise SystemExit("--groups must name at least one group")
    videos_dir = args.data_dir / args.year / "videos"

    by_video = frames_by_video_for_groups(args.pe_core_meta, groups)
    jobs = []
    for video_id, frame_idxs in by_video.items():
        path = videos_dir / video_id[:3] / f"{video_id}.mp4"
        mapping = frame_maps.load(video_id)
        for frame_idx in frame_idxs:
            jobs.append((video_id, path, frame_idx, mapping))
    if not jobs:
        log(f"nothing to do for groups {sorted(groups)}")
        return 0
    log(f"{len(jobs)} thumbnails across {len(by_video)} videos, width={args.width}, "
        f"{args.workers} workers")

    # Offline batch, no live traffic to share fairly with -- run at the full worker count.
    thumb._ffmpeg_slots = threading.BoundedSemaphore(args.workers)

    failures = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(thumb.extract, path, frame_idx, mapping, args.width): (video_id, frame_idx)
            for video_id, path, frame_idx, mapping in jobs
        }
        for future in as_completed(futures):
            video_id, frame_idx = futures[future]
            try:
                future.result()
                done += 1
            except Exception as exc:
                failures.append(f"{video_id}:{frame_idx}: {exc}")

    log(f"thumbnails: {done} generated, {len(failures)} failed")
    if failures:
        for failure in failures[:20]:
            log(f"  FAIL {failure}")
        raise RuntimeError(f"{len(failures)} thumbnail extractions failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
