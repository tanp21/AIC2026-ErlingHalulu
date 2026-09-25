"""Build frame_maps for Batch 2 groups without needing indexes/video_meta.parquet.

Unlike scripts/indexing/build_frame_maps.py (which reads the full indexes/video_meta.parquet
catalogue), this discovers videos directly from the video group directories for the given
groups -- useful on a remote processing box that only needs frame_maps for the videos it just
downloaded, not the full production catalogue.

Both extensions are searched: the S/M/L groups ship ``.mp4``, but the N traffic-camera videos
are ``.mov`` -- an mp4-only glob finds no N video at all and quietly reports success having
done nothing.

    uv run python -m scripts.indexing.build_batch2_frame_maps --year 2026 --groups M01

Writes indexes/frame_maps/<video_id>.npz (same format scripts/indexing/build_frame_maps.py
produces), via the same probe()/save_map() functions.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.indexing.build_frame_maps import MANIFEST, probe, save_manifest, save_map, signature

DATA = Path(os.environ.get("AIC_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))


def log(message):
    print(message, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--groups", required=True, help="comma-separated groups, e.g. S01,M01")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--workers", type=int, default=8,
                        help="ffprobe is I/O-bound per video; parallelize across it")
    args = parser.parse_args(argv)

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    if not groups:
        raise SystemExit("--groups must name at least one group")
    videos_dir = args.data_dir / args.year / "videos"

    jobs = []
    for group in sorted(groups):
        jobs.extend(sorted(set((videos_dir / group).glob("*.mp4"))
                           | set((videos_dir / group).glob("*.mov"))))
    log(f"{len(jobs)} videos across {sorted(groups)}, {args.workers} workers")

    failures = []
    count = 0
    signatures = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(probe, path): path for path in jobs}
        for future in as_completed(futures):
            path = futures[future]
            video_id = path.stem
            try:
                save_map(video_id, future.result())
                signatures[video_id] = signature(path)
                count += 1
            except Exception as exc:
                failures.append(f"{video_id}: {exc}")

    if signatures:
        document = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
        save_manifest({**document.get("videos", {}), **signatures})
    log(f"frame maps: {count} written, {len(failures)} failed")
    if failures:
        raise RuntimeError("frame-map failures:\n" + "\n".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
