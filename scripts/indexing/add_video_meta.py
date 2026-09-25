"""Extend indexes/video_meta.parquet with new Batch 2 2026 videos.

    uv run python -m scripts.indexing.add_video_meta --year 2026 \
        --groups S01,M01,M02,M03,M04,M05,M06,M07,M08,M09,M10

``scripts/indexing/build_flat_profile.py`` and ``scripts/indexing/build_frame_maps.py`` both
require every video they touch to already have a row in ``indexes/video_meta.parquet``
(``video_idx, video_id, group, year, path, fps``, the same schema
``scripts/indexing/build_index.py``'s ``build_video_meta()`` writes) -- this appends rows for
new videos without rebuilding the whole catalogue (``build_index.py`` itself is hardcoded to
read only ``data/2025/embeddings/meta.parquet``, which knows nothing about new groups).

Safe to rerun: video ids already present are left untouched, and ``video_idx`` for new rows
continues after the current maximum (never renumbers or reorders existing rows, since other
on-disk arrays like a profile's ``id_video.npy`` reference ``video_idx`` positionally).
"""
import argparse
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
INDEX_DIR = Path(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes"))


def log(message):
    print(message, flush=True)


def probe_fps(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of",
        "default=noprint_wrappers=1:nokey=1", str(path),
    ], capture_output=True, text=True, check=True)
    num, den = result.stdout.strip().split("/")
    return round(float(num) / float(den), 3)


def discover_candidates(data_dir, year, groups):
    """video_id -> Path for every video under the requested groups.

    Both extensions are searched: the S/M/L groups ship ``.mp4``, but the N
    traffic-camera videos are ``.mov`` (and ``scripts/extraction/ingest_zip_videos.py``
    preserves whichever the archive used), so an mp4-only glob silently finds no N video at
    all.
    """
    videos_dir = data_dir / year / "videos"
    candidates = {}
    for group in sorted(groups):
        paths = sorted(set((videos_dir / group).glob("*.mp4"))
                       | set((videos_dir / group).glob("*.mov")))
        for path in paths:
            candidates[path.stem] = path
    return candidates


def load_existing(video_meta_path):
    if not video_meta_path.exists():
        return [], -1
    rows = pq.read_table(video_meta_path).to_pylist()
    max_idx = max((row["video_idx"] for row in rows), default=-1)
    return rows, max_idx


def plan_new_rows(existing_rows, candidates, fps_by_id, year):
    """New video_meta rows for candidates not already present, numbered after the max
    existing video_idx. Pure function, no I/O -- exercised directly by unit tests.
    """
    existing_ids = {row["video_id"] for row in existing_rows}
    max_idx = max((row["video_idx"] for row in existing_rows), default=-1)
    new_rows = []
    for video_id in sorted(candidates):
        if video_id in existing_ids:
            continue
        max_idx += 1
        new_rows.append({
            "video_idx": max_idx,
            "video_id": video_id,
            # Everything downstream derives the group as the prefix before the first
            # underscore. Slicing a fixed three characters worked for S01/M05/L21 but not for
            # the four-character N groups, where it produced "N00".
            "group": video_id.split("_", 1)[0],
            "year": year,
            "path": str(candidates[video_id]),
            "fps": fps_by_id[video_id],
        })
    return new_rows


def write_atomic(video_meta_path, rows):
    video_meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = video_meta_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp)
    tmp.replace(video_meta_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--groups", required=True, help="comma-separated groups, e.g. S01,M01")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--video-meta-path", type=Path, default=INDEX_DIR / "video_meta.parquet")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="print counts, write nothing")
    args = parser.parse_args(argv)

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    if not groups:
        raise SystemExit("--groups must name at least one group")

    candidates = discover_candidates(args.data_dir, args.year, groups)
    if not candidates:
        raise SystemExit(f"no videos found under {args.data_dir / args.year / 'videos'} "
                         f"for groups {sorted(groups)}")

    existing_rows, _ = load_existing(args.video_meta_path)
    existing_ids = {row["video_id"] for row in existing_rows}
    to_probe = {vid: path for vid, path in candidates.items() if vid not in existing_ids}
    log(f"{len(candidates)} candidates, {len(candidates) - len(to_probe)} already in "
        f"{args.video_meta_path}, {len(to_probe)} new")

    if not to_probe:
        log("nothing to add")
        return 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        fps_values = list(pool.map(probe_fps, to_probe.values()))
    fps_by_id = dict(zip(to_probe, fps_values))

    new_rows = plan_new_rows(existing_rows, candidates, fps_by_id, args.year)
    log(f"adding {len(new_rows)} rows (video_idx {new_rows[0]['video_idx']}.."
        f"{new_rows[-1]['video_idx']})")

    if args.dry_run:
        log("--dry-run: not writing")
        return 0

    write_atomic(args.video_meta_path, existing_rows + new_rows)
    log(f"wrote {args.video_meta_path} ({len(existing_rows) + len(new_rows)} total videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
