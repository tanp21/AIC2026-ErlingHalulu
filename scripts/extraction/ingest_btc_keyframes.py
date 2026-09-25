"""Ingest the organisers' own Batch 2 keyframes for the N (traffic-camera) videos.

    uv run python -m scripts.extraction.ingest_btc_keyframes \
        --zips-dir /path/to/zips --map-keyframes-dir data/2026/map-keyframes --groups N001

BTC ships ``Keyframes_N001-N010.zip`` holding ``keyframes/N001-V001/<n>.jpg``, one image
every ~2 s, and ``aic26-b2-map-keyframes.zip`` holding ``map-keyframes/<video_id>.csv`` with
the columns ``n,pts_time,fps,frame_idx``. **The image file name is the keyframe's ordinal
``n``, not a frame id** -- ``079.jpg`` is keyframe 79 of that video, whatever the row named
79 in the CSV says. So the ordinal has to be resolved through the CSV before anything
downstream can use it.

Output is the layout ``scripts/extraction/extract_batch2_keyframes.py`` already produces, so
``scripts/embedding/embed_batch2_pe_core.py`` and ``embed_batch2_naflex.py`` read it with no
change at all::

    data/<year>/keyframes/<group>/<video_id>/<frame_idx>.png

with the repo's own video id (``N001_V001``, underscore) and each image renamed to the
``frame_idx`` from the map -- because ``collect_items()`` in both embedders takes
``int(path.stem)`` as the frame id.

N001-N100 is VFR (organiser-confirmed), so the CSV's ``frame_idx`` is only a relative label
there; ``pts_time`` is the real playback time and is what a KIS/QA submission is expected to
use. Both are kept: ``frame_idx`` becomes the file name, and ``pts_ms`` (``pts_time * 1000``)
is accumulated into a side table at ``<data-dir>/<year>/keyframes_btc/points.parquet`` for
later joining into the embedding ``meta.parquet``. This mirrors how
``notebooks/kaggle_pe_core_n2026_2s.py`` carries ``pts_ms`` for the same corpus.

Run it one zip at a time (the N set is ~48 GB of JPEG): download a zip, ingest it, embed it,
delete that zip's keyframes, move to the next -- the same streaming discipline
``scripts/embedding/embed_batch2_stream.py`` uses, so the whole set never sits on disk at
once. Rerunning is safe: images already converted are skipped, and the side table is merged
rather than overwritten.
"""
import argparse
import csv
import io
import os
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))

MEMBER_RE = re.compile(r"^keyframes/(?P<video>[^/]+)/(?P<n>\d+)\.(?:jpg|jpeg|png)$",
                       re.IGNORECASE)
REQUIRED_COLUMNS = ("n", "pts_time", "fps")
POINTS_SCHEMA = ("video_id", "group", "keyframe_n", "frame_idx", "pts_ms")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def btc_video_id(member_name):
    """The BTC's own video id from a zip member path: ``N001-V001``.

    Batch 2 mixes separators -- the N videos ship hyphenated (``N001-V001``) while the M
    videos already use the repo's underscore (``M05_V031``) -- so this returns whatever the
    archive says and ``repo_video_id`` normalizes it.
    """
    match = MEMBER_RE.match(member_name)
    return match.group("video") if match else None


def repo_video_id(btc_id):
    """The repo's video id for a BTC id: ``N001-V001`` -> ``N001_V001``.

    ``scripts/extraction/ingest_zip_videos.py`` writes the video files as ``<group>_<rest>``
    for every group including the N range, and everything downstream derives
    ``group = video_id.split("_", 1)[0]``. A hyphenated BTC id has to be folded into that
    same form or the keyframes would not line up with ``data/<year>/videos/``.
    """
    return btc_id.replace("-", "_")


def group_of(video_id):
    """The repo's group label: ``N001_V001`` -> ``N001``."""
    return video_id.split("_", 1)[0]


def read_map_keyframes(path):
    """One ``map-keyframes`` CSV as ``({n: (frame_idx, pts_ms)}, derived)``.

    The ``frame_idx`` column is trusted when BTC ships it; when it is absent (or blank) the
    index is derived from the row's own ``fps`` -- ``round(pts_time * fps)``, which is exactly
    the arithmetic the shipped column already satisfies for this corpus -- and ``derived``
    comes back true so the caller can say so out loud rather than let a silently different
    definition of the id slip through.
    """
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [name for name in REQUIRED_COLUMNS if name not in columns]
        if missing:
            raise ValueError(f"{path}: missing column(s) {missing}; has {columns}")
        rows = {}
        derived = False
        for line_number, row in enumerate(reader, start=2):
            try:
                n = int(row["n"])
                pts_time = float(row["pts_time"])
                fps = float(row["fps"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: unreadable row {row!r}") from error
            raw = (row.get("frame_idx") or "").strip()
            if raw:
                frame_idx = int(raw)
            else:
                frame_idx = round(pts_time * fps)
                derived = True
            if n in rows:
                raise ValueError(f"{path}:{line_number}: duplicate n {n}")
            rows[n] = (frame_idx, round(pts_time * 1000))
    if not rows:
        raise ValueError(f"{path}: no keyframe rows")
    return rows, derived


def check_video_rows(video_id, rows):
    """Validate one video's ``n -> (frame_idx, pts_ms)`` map; returns it sorted by ordinal.

    Two checks, both of which would be silent corruption downstream rather than an error: a
    repeated ``frame_idx`` would collide two keyframes onto one file name, and a
    non-increasing ``pts_ms`` would mean the ordinals and the times disagree about which
    keyframe comes first.
    """
    ordered = sorted(rows.items())
    seen = {}
    previous = None
    for n, (frame_idx, pts_ms) in ordered:
        if frame_idx in seen:
            raise ValueError(f"{video_id}: ordinals {seen[frame_idx]} and {n} both map to "
                             f"frame_idx {frame_idx}")
        seen[frame_idx] = n
        if previous is not None and pts_ms <= previous:
            raise ValueError(f"{video_id}: pts_ms not increasing at n={n} "
                             f"({previous} -> {pts_ms})")
        previous = pts_ms
    return ordered


def zip_members(zip_path):
    """``{btc_video_id: {ordinal: member_name}}`` for every keyframe image in one archive."""
    videos = {}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            match = MEMBER_RE.match(name)
            if match is None:
                continue
            videos.setdefault(match.group("video"), {})[int(match.group("n"))] = name
    return videos


def check_images_match(video_id, members, rows):
    """Every ordinal in the map must have exactly one image, and no image may go unmapped.

    A missing image would leave a hole in the embedding set; an unmapped one would sit on
    disk for ever and never be embedded, because the file name that gets the vector is the
    ``frame_idx`` and there would be none to name it with. Checked at plan time -- before a
    single byte is extracted -- so ``--dry-run`` covers it too.
    """
    present = set(members)
    missing = sorted(set(rows) - present)
    if missing:
        raise ValueError(f"{video_id}: map-keyframes has {len(missing)} ordinal(s) with no "
                         f"image, first {missing[:5]}")
    extra = sorted(present - set(rows))
    if extra:
        raise ValueError(f"{video_id}: {len(extra)} image(s) not in map-keyframes, "
                         f"first {extra[:5]}")


def plan_archive(zip_path, members_by_video, map_keyframes_dir, groups, only):
    """The validated plan one archive contributes, as ``(btc_id, video_id, rows)`` triples."""
    plan = []
    for btc_id in sorted(members_by_video):
        video_id = repo_video_id(btc_id)
        if groups and group_of(video_id) not in groups:
            continue
        if only and video_id not in only:
            continue
        csv_path = Path(map_keyframes_dir) / f"{btc_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"{zip_path.name}: {btc_id} has keyframe images but no "
                                    f"{csv_path}")
        rows, derived = read_map_keyframes(csv_path)
        rows = check_video_rows(video_id, rows)
        check_images_match(video_id, members_by_video[btc_id], dict(rows))
        plan.append((btc_id, video_id, rows, derived))
    return plan


def convert_one(archive, member, destination, compression):
    """One ``.jpg`` member -> PNG at ``destination``, skipping a file already written."""
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as handle:
        data = handle.read()
    with Image.open(io.BytesIO(data)) as image:
        image.convert("RGB").save(destination, format="PNG", compress_level=compression,
                                  optimize=False)
    return True


def ingest_video(archive, video_id, members, rows, output_dir, workers, compression):
    """Convert one video's keyframes, renaming each image to the CSV's ``frame_idx``.

    ``rows`` and ``members`` are already known to correspond exactly (``check_images_match``
    ran at plan time), so this is purely the JPEG decode + PNG write.
    """
    jobs = [(members[n], output_dir / f"{rows[n][0]:06d}.png") for n in sorted(rows)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = sum(pool.map(
            lambda job: convert_one(archive, job[0], job[1], compression), jobs))
    return written, len(jobs)


def load_points(path):
    """Existing side-table rows, so a rerun merges instead of restarting the count."""
    if not Path(path).exists():
        return []
    return pq.read_table(path).to_pylist()


def write_points(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        name: pa.array([row[name] for row in rows],
                       type={"video_id": pa.string(), "group": pa.string(),
                             "keyframe_n": pa.int32(), "frame_idx": pa.int32(),
                             "pts_ms": pa.int64()}[name])
        for name in POINTS_SCHEMA
    })
    pq.write_table(table, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips-dir", required=True, type=Path,
                        help="directory holding Keyframes_Nxxx-Nyyy.zip archives")
    parser.add_argument("--map-keyframes-dir", required=True, type=Path,
                        help="directory holding the organisers' <video_id>.csv maps")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--groups", default="",
                        help="comma-separated group codes, e.g. N001,N002 (default: all)")
    parser.add_argument("--only", default="",
                        help="comma-separated repo video ids, for a canary run")
    parser.add_argument("--keyframes-dir", type=Path, default=None,
                        help="default: <data-dir>/<year>/keyframes")
    parser.add_argument("--points", type=Path, default=None,
                        help="default: <data-dir>/<year>/keyframes_btc/points.parquet")
    parser.add_argument("--workers", type=int, default=8,
                        help="images converted in parallel; the work is JPEG decode + PNG "
                             "encode, so match vCPUs")
    parser.add_argument("--compression", type=int, default=1,
                        help="PNG zlib level, 0-9 (1 matches extract_batch2_keyframes)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and validate the maps, extract nothing")
    args = parser.parse_args(argv)

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    only = {v.strip() for v in args.only.split(",") if v.strip()}
    keyframes_dir = args.keyframes_dir or (args.data_dir / args.year / "keyframes")
    points_path = args.points or (args.data_dir / args.year / "keyframes_btc" /
                                  "points.parquet")

    zips = sorted(Path(args.zips_dir).glob("Keyframes_*.zip"))
    if not zips:
        raise SystemExit(f"no Keyframes_*.zip under {args.zips_dir}")

    points = {(row["video_id"], row["frame_idx"]): row for row in load_points(points_path)}
    log(f"{len(points)} keyframe row(s) already in {points_path}")

    for zip_path in zips:
        members_by_video = zip_members(zip_path)
        plan = plan_archive(zip_path, members_by_video, args.map_keyframes_dir, groups, only)
        if not plan:
            continue
        derived = [video_id for _, video_id, _, is_derived in plan if is_derived]
        if derived:
            log(f"WARNING {zip_path.name}: frame_idx derived from pts_time*fps for "
                f"{len(derived)} video(s): {derived[:5]}")
        if args.dry_run:
            log(f"{zip_path.name}: {len(plan)} video(s), "
                f"{sum(len(rows) for _, _, rows, _ in plan)} keyframe(s)")
            continue

        written_total = 0
        with zipfile.ZipFile(zip_path) as archive:
            for btc_id, video_id, rows, _ in plan:
                output_dir = keyframes_dir / group_of(video_id) / video_id
                written, expected = ingest_video(archive, video_id, members_by_video[btc_id],
                                                 dict(rows), output_dir, args.workers,
                                                 args.compression)
                written_total += written
                for n, (frame_idx, pts_ms) in rows:
                    points[(video_id, frame_idx)] = {
                        "video_id": video_id, "group": group_of(video_id), "keyframe_n": n,
                        "frame_idx": frame_idx, "pts_ms": pts_ms,
                    }
                log(f"{video_id}: {written}/{expected} converted, {len(rows)} mapped")
        write_points(points_path, sorted(points.values(),
                                         key=lambda row: (row["video_id"], row["frame_idx"])))
        log(f"{zip_path.name}: {written_total} image(s) written, "
            f"{len(points)} row(s) in {points_path}")

    if args.dry_run:
        return 0
    log(f"DONE: {len(points)} keyframe row(s) -> {keyframes_dir}, {points_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
