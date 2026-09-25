"""Unzip Batch 2 2026 video archives into the repo's data layout.

    uv run python -m scripts.extraction.ingest_zip_videos \
        --zips-dir /path/to/Videos_AIC --year 2026

    uv run python -m scripts.extraction.ingest_zip_videos \
        --zips-dir /path/to/Videos_AIC --only S01,M01

Each archive is named ``Video_<GROUP>.zip`` or ``Videos_<GROUP>.zip`` and holds its videos
flat under ``videos/`` inside the zip (no group subfolder there -- the group only appears in
the destination path). Output: ``data/<year>/videos/<group>/<video_id>.<ext>``, matching the
layout ``scripts/extraction/detect_shots.py`` and everything downstream already expects.

Some archives (observed: ``Video_S01.zip``) use a hyphen between group and shot number
(``S01-V001.mp4``) while others (``Videos_M01.zip``..``M10.zip``, and the existing
``L21_V001``-style groups) use an underscore. Every downstream tool derives
``group = video_id.split("_", 1)[0]``, so any hyphen form is normalized to the underscore
form on extraction. The source extension (``.mp4`` or ``.mov``) is preserved as-is -- this
script does not transcode.

A second archive shape (observed: ``Video_N001-N010.zip``, the "N" traffic-camera videos)
names the zip after a *range* of groups instead of a single one, and holds several videos per
group (``videos/N001-V001.mov``, ``videos/N001-V002.mov``, ``videos/N001-V003.mov``, ...).
There the zip name cannot be trusted as "the" group, so each member's own group prefix is the
source of truth instead, sanity-checked against the zip's declared range.
"""
import argparse
import os
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))

ZIP_NAME_RE = re.compile(r"^Videos?_(?P<group>[A-Za-z]\d{2,3})\.zip$")
RANGE_ZIP_NAME_RE = re.compile(
    r"^Videos?_(?P<letter>[A-Za-z])(?P<start>\d{2,3})-[A-Za-z](?P<end>\d{2,3})\.zip$"
)
# group + V### with either separator; captures group loosely so the returned id is always
# rebuilt with the zip's own group (the source of truth for single-group zips), not whatever
# the entry name says.
MEMBER_RE = re.compile(r"^videos/(?P<group>[A-Za-z]\d{2,3})[-_](?P<rest>V\d+)\.(?P<ext>mp4|mov)$")


def log(message):
    print(message, flush=True)


def group_from_zip_name(zip_path):
    """The group code (e.g. ``S01``) a ``Video(s)_<GROUP>.zip`` archive covers."""
    match = ZIP_NAME_RE.match(Path(zip_path).name)
    if not match:
        raise ValueError(f"{zip_path}: name does not match Video(s)_<GROUP>.zip")
    return match.group("group")


def normalized_video_id(member_name, group):
    """The ``<group>_V<NNN>`` video id a zip member should be written as.

    Raises if the member isn't a video the ingest layout expects, so a differently shaped
    archive fails loudly instead of silently writing something into the wrong place.
    """
    match = MEMBER_RE.match(member_name)
    if not match:
        raise ValueError(f"unexpected zip member: {member_name!r}")
    return f"{group}_{match.group('rest')}"


def plan_extraction(zip_path, group):
    """(member_name, video_id) pairs a zip should be extracted as, sorted by video_id."""
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if MEMBER_RE.match(name)]
    plan = [(member, normalized_video_id(member, group)) for member in members]
    plan.sort(key=lambda item: item[1])
    return plan


def extract_member(archive, member, target):
    tmp = target.with_suffix(target.suffix + ".part")
    with archive.open(member) as source, open(tmp, "wb") as sink:
        while True:
            chunk = source.read(8 * 1024 * 1024)
            if not chunk:
                break
            sink.write(chunk)
    tmp.replace(target)


def extract_zip(zip_path, group, out_dir, force):
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan_extraction(zip_path, group)
    with zipfile.ZipFile(zip_path) as archive:
        extracted = 0
        skipped = 0
        for member, video_id in plan:
            ext = MEMBER_RE.match(member).group("ext")
            target = out_dir / f"{video_id}.{ext}"
            if target.exists() and not force:
                skipped += 1
                continue
            extract_member(archive, member, target)
            extracted += 1
    return extracted, skipped, len(plan)


def range_from_zip_name(zip_path):
    """(letter, start, end) a ``Video(s)_<L><NNN>-<L><NNN>.zip`` archive declares."""
    match = RANGE_ZIP_NAME_RE.match(Path(zip_path).name)
    if not match:
        raise ValueError(f"{zip_path}: name does not match Video(s)_<GROUP>-<GROUP>.zip")
    return match.group("letter"), int(match.group("start")), int(match.group("end"))


def plan_extraction_range(zip_path):
    """(member_name, group, video_id) triples for a range zip, sorted by video_id.

    Unlike a single-group zip, the zip name only bounds the range -- each member's own
    group prefix is the source of truth, checked against that range so a mislabeled member
    fails loudly instead of landing in the wrong directory.
    """
    letter, start, end = range_from_zip_name(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        members = [name for name in archive.namelist() if MEMBER_RE.match(name)]
    plan = []
    for member in members:
        match = MEMBER_RE.match(member)
        group = match.group("group")
        group_letter, group_num = group[0], int(group[1:])
        if group_letter != letter or not (start <= group_num <= end):
            raise ValueError(
                f"{member!r} in {zip_path}: group {group} outside declared range "
                f"{letter}{start:03d}-{letter}{end:03d}"
            )
        plan.append((member, group, f"{group}_{match.group('rest')}"))
    plan.sort(key=lambda item: item[2])
    return plan


def extract_zip_range(zip_path, videos_dir, force):
    plan = plan_extraction_range(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        extracted = 0
        skipped = 0
        for member, group, video_id in plan:
            ext = MEMBER_RE.match(member).group("ext")
            out_dir = videos_dir / group
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / f"{video_id}.{ext}"
            if target.exists() and not force:
                skipped += 1
                continue
            extract_member(archive, member, target)
            extracted += 1
    return extracted, skipped, len(plan)


def collect_zips(zips_dir, only):
    zips = []
    for path in sorted(Path(zips_dir).glob("Video*.zip")):
        match = ZIP_NAME_RE.match(path.name)
        if not match:
            continue
        group = match.group("group")
        if only and group not in only:
            continue
        zips.append((path, group))
    return zips


def collect_range_zips(zips_dir, only):
    zips = []
    for path in sorted(Path(zips_dir).glob("Video*.zip")):
        match = RANGE_ZIP_NAME_RE.match(path.name)
        if not match:
            continue
        letter, start, end = match.group("letter"), int(match.group("start")), int(match.group("end"))
        if only and not any(g[0] == letter and start <= int(g[1:]) <= end for g in only if len(g) > 1):
            continue
        zips.append(path)
    return zips


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips-dir", required=True, type=Path,
                        help="directory holding Video(s)_<GROUP>.zip archives")
    parser.add_argument("--year", default="2026")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--only", default="",
                        help="comma-separated group codes, e.g. S01,M01 (default: all "
                             "Video(s)_<GROUP>[.zip / -<GROUP>.zip] found in --zips-dir)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract videos that already exist at the destination")
    args = parser.parse_args(argv)

    only = {g.strip() for g in args.only.split(",") if g.strip()}
    zips = collect_zips(args.zips_dir, only)
    range_zips = collect_range_zips(args.zips_dir, only)
    if not zips and not range_zips:
        raise SystemExit(f"no Video(s)_<GROUP>.zip found under {args.zips_dir}"
                         + (f" matching {sorted(only)}" if only else ""))

    videos_dir = args.data_dir / args.year / "videos"
    for zip_path, group in zips:
        extracted, skipped, total = extract_zip(zip_path, group, videos_dir / group, args.force)
        log(f"{zip_path.name} -> {group}: {extracted} extracted, {skipped} already present, "
            f"{total} total")
    for zip_path in range_zips:
        extracted, skipped, total = extract_zip_range(zip_path, videos_dir, args.force)
        letter, start, end = range_from_zip_name(zip_path)
        log(f"{zip_path.name} -> {letter}{start:03d}-{letter}{end:03d}: {extracted} extracted, "
            f"{skipped} already present, {total} total")


if __name__ == "__main__":
    main()
