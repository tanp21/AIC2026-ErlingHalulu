"""Check that a machine has everything a media host serves from, before it is needed.

    # on the laptop, inside the image (no host Python needed):
    docker compose -f compose.yaml -f compose.media.yaml run --rm \
        -v ./scripts:/app/scripts:ro app python -m scripts.media.check_media_host

    # or on any host with the repo's venv:
    uv run python -m scripts.media.check_media_host --data-dir D:/aic/data

A media host answers `/`, `/thumb`, `/frame`, `/video`, `/clip`, `/frame-map` and
`/api/dres/*` from local disk and forwards everything else to the compute host
(`docs/venue-deployment.md`). So it needs the corpus, the frame maps, the shot boundaries
and the profile metadata -- and nothing else: no FAISS index, no model, no Elasticsearch.
That is a specific subset, and getting it wrong looks like success until the moment a
frame is requested.

Three failures this is built to catch, in the order they bite:

1. **A corpus that was re-downloaded instead of copied.** `backend/dres.py` refuses a
   submission whose video's size *or* mtime differs from `frame_maps/manifest.json`, so a
   freshly extracted corpus fails every DRES submission with "frame map is stale" while
   search works perfectly. `scripts.indexing.build_frame_maps --resign-only` fixes it
   without re-deriving a single map; this reports exactly which videos need it.
2. **A half-copied corpus.** Every video in the served profiles is checked for its own
   file, frame map and shot boundaries, rather than counting files in a directory.
3. **A missing keyframe store.** The N traffic-camera groups have no usable timestamps in
   their containers, so ffmpeg cannot extract a frame from them at all -- the organiser
   JPEGs under `data/<year>/keyframe_thumbs/` are the only source `/thumb` has.

Exits non-zero when anything required is missing, so it can gate a start-up.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from backend.core import ProfileStore, mounted_video_path
from backend.shots import shots_path

ROOT = Path(__file__).resolve().parents[2]

#: `/thumb` and `/frame` read these widths; `stored_keyframe_largest` walks them in this
#: order and takes the best one present, so a host may legitimately hold a subset.
KEYFRAME_WIDTHS = (1280, 640, 320, 160)


def check_corpus(store, data_dir: Path, index_dir: Path, names) -> tuple[bool, list[str]]:
    """Per-profile coverage of the four things a media host reads off disk.

    Videos and frame maps are hard requirements -- without either, a search result cannot
    be shown at all. Shot boundaries are not: they back `/clip/{id}/{shot}` (Clip Studio
    and the clip-export tab), while `/shots/` itself is forwarded to the compute host. So a
    gap there is reported loudly and does not fail the run.
    """
    map_dir = index_dir / "frame_maps"
    healthy = True
    lines: list[str] = []
    for name in names:
        try:
            _, _, _, lookup = store.load_meta(name)
        except (OSError, KeyError, ValueError) as exc:
            lines.append(f"  {name}: profile metadata unreadable -- {exc}")
            healthy = False
            continue
        rows = list(lookup.values())
        total = len(rows)
        counts = {
            "videos": sum(1 for row in rows if mounted_video_path(row, data_dir).exists()),
            "frame maps": sum(1 for row in rows if (map_dir / f"{row['video_id']}.npz").exists()),
            "shots": sum(1 for row in rows if shots_path(row, data_dir).exists()),
        }
        lines.append(
            f"  {name}: {total} videos -- "
            + ", ".join(f"{label} {present}/{total}" for label, present in counts.items())
        )
        for label in ("videos", "frame maps"):
            if counts[label] != total:
                healthy = False
                lines.append(f"    MISSING {total - counts[label]} {label}")
        if counts["shots"] != total:
            lines.append(
                f"    note: {total - counts['shots']} videos have no shot boundaries, so Clip "
                f"Studio is unavailable for them (search and playback still work)"
            )
        faststart = sum(
            1 for row in rows
            if (data_dir / str(row["year"]) / "videos_faststart" / row["group"]
                / f"{row['video_id']}.mp4").exists()
        )
        if 0 < faststart < total:
            # Not a failure: `streaming_path` falls back to the original file. It only
            # costs the player a round-trip to the tail of the file before it can start.
            lines.append(
                f"    note: {faststart}/{total} have a faststart remux; the rest still play"
            )
    return healthy, lines


def check_keyframes(store, data_dir: Path, names) -> tuple[bool, list[str]]:
    """The organiser JPEGs, for the groups that have no other thumbnail source."""
    groups: set[tuple[str, str]] = set()
    for name in names:
        try:
            _, _, _, lookup = store.load_meta(name)
        except (OSError, KeyError, ValueError):
            continue
        for row in lookup.values():
            groups.add((str(row["year"]), str(row["group"])))
    n_groups = sorted(group for year, group in groups if group.startswith("N"))
    if not n_groups:
        return True, ["  no N groups in the served profiles -- nothing needs the store"]

    lines, healthy = [], True
    for year in sorted({year for year, group in groups if group.startswith("N")}):
        # A set: two profiles can serve the same N videos (l2026-v3 and naflex-l2026-v3
        # are the same corpus under two embedding models), and counting them twice would
        # halve the reported coverage.
        ids = sorted({
            row["video_id"]
            for name in names
            for row in _rows(store, name)
            if str(row["group"]).startswith("N") and str(row["year"]) == year
        })
        if not ids:
            continue
        base = data_dir / year / "keyframe_thumbs"
        present_widths = [w for w in KEYFRAME_WIDTHS if (base / str(w)).is_dir()]
        if not present_widths:
            healthy = False
            lines.append(f"  {year}: no widths at all under {base}")
            continue
        for width in present_widths:
            present = sum(
                1 for video_id in ids
                if (base / str(width) / video_id.split("_", 1)[0] / video_id).is_dir()
            )
            if present != len(ids):
                healthy = False
            lines.append(f"  {year} @{width}px: {present}/{len(ids)} N videos")
        missing_widths = [w for w in KEYFRAME_WIDTHS if w not in present_widths]
        if missing_widths:
            lines.append(f"    note: no {', '.join(map(str, missing_widths))}px set; "
                         f"/frame falls back to a smaller one")
    return healthy, lines


def _rows(store, name):
    try:
        return list(store.load_meta(name)[3].values())
    except (OSError, KeyError, ValueError):
        return []


def check_signatures(index_dir: Path, data_dir: Path) -> tuple[bool, list[str]]:
    """Every video's size and mtime against `frame_maps/manifest.json`.

    This is the DRES precondition, and the one check that fails on a host that looks
    completely healthy otherwise.
    """
    manifest_path = index_dir / "frame_maps" / "manifest.json"
    if not manifest_path.exists():
        return False, [f"  {manifest_path} is missing -- `/api/dres/*` cannot verify a frame"]
    try:
        document = json.loads(manifest_path.read_text())
        videos = document["videos"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return False, [f"  {manifest_path} is unreadable -- {exc}"]

    if not data_dir.exists():
        return False, [f"  {data_dir} does not exist; nothing to compare the manifest against"]

    # `backend/dres.py::_source` compares the last four path components, so the manifest
    # built on another machine is understood here as long as the layout below `data/` is
    # the same. Sample rather than walk 2093 files on every run.
    stale, missing, checked = [], [], 0
    for video_id, signature in videos.items():
        parts = Path(str(signature.get("path", ""))).parts
        if len(parts) < 4:
            continue
        path = data_dir.joinpath(*parts[-4:])
        if not path.exists():
            missing.append(video_id)
            continue
        checked += 1
        stat = path.stat()
        if signature.get("size") != stat.st_size or signature.get("mtimeNs") != stat.st_mtime_ns:
            stale.append(video_id)

    if missing or stale:
        lines = [f"  {checked - len(stale)}/{checked} present videos match the manifest exactly"]
        if stale:
            lines.append(f"    size or mtime differs: {len(stale)} (e.g. {', '.join(sorted(stale)[:3])})")
        if missing:
            lines.append(f"    not at this path: {len(missing)} (e.g. {', '.join(sorted(missing)[:3])})")
        lines.append("    DRES refuses a submission whose video does not match, so every")
        lines.append("    submission of one of these fails until the manifest is re-adopted:")
        lines.append("      uv run python -m scripts.indexing.build_frame_maps")
        lines.append("    That re-verifies each map against the file's own PTS timeline and needs")
        lines.append("    only ffprobe, not a re-decode. `--resign-only` does the same job with no")
        lines.append("    ffmpeg at all, but takes a matching file size on trust, so it refuses any")
        lines.append("    video whose length changed (a re-encode or an audio remux).")
        return False, lines
    return True, [f"  {checked} videos match the manifest exactly"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--index-dir", type=Path, default=ROOT / "indexes")
    parser.add_argument("--profile", action="append", default=None,
                        help="check only this profile; repeatable (default: every one)")
    args = parser.parse_args()

    # Every directory is passed in rather than read from the module's own constants, so
    # `--data-dir` and `--index-dir` work without the AIC_* environment the server uses.
    store = ProfileStore(args.index_dir)
    names = args.profile or [item["name"] for item in store.list_profiles()]
    unknown = [name for name in names if name not in store.profiles]
    if unknown:
        raise SystemExit(f"unknown profile(s): {', '.join(unknown)}")

    print(f"data dir   {args.data_dir}")
    print(f"index dir  {args.index_dir}")
    print(f"profiles   {', '.join(names)} (default: {store.default})")

    sections = (
        ("frame-map signatures (the DRES precondition)", check_signatures(args.index_dir, args.data_dir)),
        ("corpus coverage", check_corpus(store, args.data_dir, args.index_dir, names)),
        ("stored keyframes", check_keyframes(store, args.data_dir, names)),
    )
    healthy = True
    for title, (section_healthy, lines) in sections:
        print(f"\n{title}")
        for line in lines:
            print(line)
        healthy = healthy and section_healthy

    print("\n" + ("media host is ready" if healthy else "NOT ready -- see the misses above"))
    sys.exit(0 if healthy else 1)


if __name__ == "__main__":
    main()
