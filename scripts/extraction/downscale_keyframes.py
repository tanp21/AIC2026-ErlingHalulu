#!/usr/bin/env python3
"""Downscale all K01..K20 keyframes from 1920x1080 to 1280x720 (HD) in place.

In-place overwrite: each frame is re-encoded to the SAME inode and truncated,
so no new filesystem blocks are allocated. On a dynamically-expanding VHDX
(the dataset lives on one) this avoids growing the VHDX file on the host drive
(the `*.tmp.png` + os.replace approach allocated fresh blocks every frame).

Trade-off: NOT atomic — an interrupt mid-write can corrupt that one frame.
Re-run to finish (idempotent: frames already <=1280px wide are skipped).

Usage:
    python -m scripts.extraction.downscale_keyframes                 # default: data/2025/keyframes
    python -m scripts.extraction.downscale_keyframes --workers 12
    python -m scripts.extraction.downscale_keyframes --compression 9 # smaller files, slower
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import cv2
from tqdm import tqdm

TARGET_W = 1280
TARGET_H = 720


def init_worker():
    cv2.setNumThreads(1)


def process(path, compression):
    try:
        orig = os.path.getsize(path)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return ("err", path, 0, 0, "imread returned None")
        h, w = img.shape[:2]
        if w <= TARGET_W:
            return ("skip", path, orig, orig, "")
        resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", resized, [cv2.IMWRITE_PNG_COMPRESSION, compression])
        if not ok:
            return ("err", path, 0, 0, "imencode failed")
        data = buf.tobytes()
        with open(path, "r+b") as f:
            f.write(data)
            f.truncate(len(data))
        return ("ok", path, orig, len(data), "")
    except Exception as e:  # keep the run alive on any per-file failure
        return ("err", path, 0, 0, str(e))


def main():
    ap = argparse.ArgumentParser(description="Downscale KXX keyframes to HD")
    ap.add_argument("--root", default="data/2025/keyframes")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=200)
    ap.add_argument("--compression", type=int, default=3, help="PNG compression 0-9")
    args = ap.parse_args()

    pattern = os.path.join(args.root, "K*", "*", "*.png")
    paths = [p for p in glob.glob(pattern) if not p.endswith(".tmp.png")]
    paths.sort()
    print(f"Found {len(paths)} keyframes under {args.root}")

    if not paths:
        print("Nothing to do.")
        return

    counts = {"ok": 0, "skip": 0, "err": 0}
    total_orig = total_new = 0
    errors = []

    fn = partial(process, compression=args.compression)
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=init_worker
    ) as ex:
        for status, path, orig, new, msg in tqdm(
            ex.map(fn, paths, chunksize=args.chunksize), total=len(paths)
        ):
            counts[status] += 1
            if status == "ok":
                total_orig += orig
                total_new += new
            elif status == "skip":
                total_orig += orig
                total_new += new
            else:
                errors.append((path, msg))

    n = len(paths)
    print("=" * 60)
    print(f"Total   : {n}")
    print(f"Resized : {counts['ok']}")
    print(f"Skipped : {counts['skip']} (already <= {TARGET_W}px)")
    print(f"Errors  : {counts['err']}")
    if total_orig > 0:
        freed = total_orig - total_new
        pct = freed / total_orig * 100
        print(f"Size    : {total_orig/1e9:.2f} GB -> {total_new/1e9:.2f} GB "
              f"(freed {freed/1e9:.2f} GB, {pct:.1f}%)")
    if errors:
        with open("downscale_errors.log", "w") as f:
            for path, msg in errors:
                f.write(f"{path}\t{msg}\n")
        print(f"{len(errors)} errors written to downscale_errors.log")


if __name__ == "__main__":
    main()
