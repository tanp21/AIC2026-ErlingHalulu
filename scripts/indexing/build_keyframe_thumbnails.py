#!/usr/bin/env python3
"""Turn the organisers' Batch 2 keyframe archives into the thumbnails the app serves.

    uv run python -m scripts.indexing.build_keyframe_thumbnails \
        --zips ~/Downloads --workers 8

BTC ships ``Keyframes_N001-N010.zip`` … holding ``keyframes/<video>-<V>/<n>.jpg``, one JPEG
every ~2 s, named by the keyframe's ordinal ``n``. Those are the images the N embeddings were
taken from, and for the N groups they are also the *only* workable source: those videos are VFR
with a container that carries no usable frame timestamps, so `/thumb` cannot cut one out of the
video at all (a seek decodes nothing and a `pts` match finds nothing).

Each JPEG is filed at the frame the N index would place it at -- ordinal -> ``pts_ms`` from
``data/<year>/keyframes_btc/points.parquet`` -> the frame of our copy whose PTS carries that
time, the same resolution ``build_flat_profile --frame-from pts-ms`` used. The key is therefore
the ``(video_id, frame_idx)`` a search hit reports and ``/thumb`` asks for, and
``backend.thumb.stored_keyframe`` is the lookup.

Resumable: an image that already has every requested width is skipped, so an interrupted run
continues where it stopped (and a later run can add a width without redoing the others). The
archives are read one member at a time -- the set never has to sit extracted on disk, and the
images in flight are bounded, not the whole batch.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
import io
import os
from pathlib import Path
import re
import time
import zipfile

import pyarrow.parquet as pq
from PIL import Image

from scripts.indexing.build_flat_profile import KeyframeResolver

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
MEMBER = re.compile(r"^keyframes/(?P<video>[^/]+)/(?P<n>\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)
#: The widths `/thumb` accepts (backend.app validates the set), plus the one `/frame` serves --
#: what the agent judges pixels with and the reviewer sees full-size.
WIDTHS = (160, 320, 640, 1280)
QUALITY = 82


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_points(path):
    """``{(video_id, keyframe ordinal): pts_ms}`` -- the organisers' own time for each image."""
    points = {}
    for row in pq.read_table(path).to_pylist():
        points[(row["video_id"], int(row["keyframe_n"]))] = int(row["pts_ms"])
    if not points:
        raise ValueError(f"no keyframe times in {path}")
    return points


def target_paths(out_dir, widths, video_id, frame_idx):
    group = video_id.split("_", 1)[0]
    return {w: Path(out_dir) / str(w) / group / video_id / f"{frame_idx}.jpg" for w in widths}


def resize_into(job):
    """Decode one organiser JPEG once and write every width. Runs in a worker process."""
    source, widths, targets = job
    with Image.open(io.BytesIO(source)) as image:
        image = image.convert("RGB")
        for width in widths:
            height = max(1, round(image.height * width / image.width))
            path = Path(targets[width])
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp.jpg")
            image.resize((width, height), Image.LANCZOS).save(
                temporary, format="JPEG", quality=QUALITY, optimize=True)
            temporary.replace(path)
    return len(widths)


def iter_members(zips, points):
    """``(video_id, ordinal, member_bytes)`` for every image the archives hold, in order."""
    for archive in sorted(zips):
        with zipfile.ZipFile(archive) as bundle:
            for info in bundle.infolist():
                match = MEMBER.match(info.filename)
                if match is None:
                    continue
                video_id = match.group("video").replace("-", "_")
                ordinal = int(match.group("n"))
                if (video_id, ordinal) not in points:
                    continue
                yield video_id, ordinal, bundle.read(info)


def build(zips, out_dir, points_path, base_index, widths=WIDTHS, workers=8, groups=None,
          max_in_flight=None):
    """Stream the archives through the pool, a few images in flight per worker.

    One image is ~500 KB and the set is ~88k of them: holding the batch in memory before
    submitting (the obvious way to write this) is 40+ GB and takes the host down. The archives
    are read one member at a time instead, and the pool is kept fed rather than loaded.
    """
    points = load_points(points_path)
    resolver = KeyframeResolver(base_index)
    inflight = max(1, max_in_flight or workers * 4)
    started, written, skipped, unknown, failed, seen, last_log = time.time(), 0, 0, 0, [], 0, 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for video_id, ordinal, source in iter_members(zips, points):
            if groups and video_id.split("_", 1)[0] not in groups:
                continue
            seen += 1
            try:
                frame_idx, _ = resolver.frame_for(video_id, points[(video_id, ordinal)])
            except (FileNotFoundError, OSError):
                unknown += 1
                continue
            targets = target_paths(out_dir, widths, video_id, frame_idx)
            missing = tuple(w for w, path in targets.items() if not path.exists())
            if not missing:
                skipped += 1
                continue
            # Only the widths that are actually missing: adding one later does not redo the rest.
            job = (source, missing, {w: str(targets[w]) for w in missing})
            pending[pool.submit(resize_into, job)] = job[2][max(missing)]
            if len(pending) < inflight:
                continue
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                written, failed = _collect(future, pending.pop(future), written, failed)
            if written - last_log >= 500:
                last_log = written
                log(f"{written} written, {skipped} already there, {seen} seen "
                    f"({time.time() - started:.0f}s)")
        for future in as_completed(pending):
            written, failed = _collect(future, pending[future], written, failed)

    log(f"done: {written} images written, {skipped} already there, {unknown} without a frame "
        f"map, {len(failed)} failed, in {(time.time() - started) / 60:.1f} min")
    for path, error in failed[:10]:
        log(f"FAIL {path}: {error}")
    return {"written": written, "skipped": skipped, "missing_frame_map": unknown,
            "failed": len(failed), "output": str(out_dir)}


def _collect(future, path, written, failed):
    try:
        future.result()
    except Exception as error:  # one bad image must not sink the batch
        failed.append((path, repr(error)))
        return written, failed
    return written + 1, failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zips", type=Path, required=True,
                        help="directory holding the organisers' Keyframes_*.zip archives")
    parser.add_argument("--pattern", default="Keyframes_N*.zip")
    parser.add_argument("--out-dir", type=Path, default=DATA / "2026" / "keyframe_thumbs")
    parser.add_argument("--points", type=Path, default=DATA / "2026" / "keyframes_btc" / "points.parquet")
    parser.add_argument("--base-index", type=Path, default=ROOT / "indexes")
    parser.add_argument("--widths", default=",".join(str(w) for w in WIDTHS),
                        help="thumbnail widths the app asks for (backend.app's /thumb)")
    parser.add_argument("--groups", default="", help="comma-separated groups, e.g. N001")
    parser.add_argument("--workers", type=int, default=8,
                        help="one decode per image, so this is CPU-bound; match vCPUs")
    args = parser.parse_args(argv)
    zips = sorted(args.zips.glob(args.pattern))
    if not zips:
        raise SystemExit(f"no archives matching {args.pattern} under {args.zips}")
    widths = tuple(int(w) for w in args.widths.split(",") if w.strip())
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    result = build(zips, args.out_dir, args.points, args.base_index, widths, args.workers, groups)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
