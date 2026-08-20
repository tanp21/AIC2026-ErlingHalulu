#!/usr/bin/env python3
"""Extract 1080p keyframes (K01..K20) on a cheap CPU instance, zip, upload to Drive.

Pipeline per group (Drive is the source of truth -> fully resume-safe):
  1. skip if KXX.zip already on Drive
  2. aria2c -c download Videos_KXX.zip from aic-data.ledo.io.vn
  3. unzip videos (tmp+rename), delete zip
  4. ffmpeg extract keyframes (batch per video, exact frame index via select) from shots JSON
  5. zip (ZIP_STORED, entries KXX/...), push KXX.zip -> Drive, delete keyframes+zip
  6. push videos/KXX -> Drive, delete videos

Only the zip (not individual keyframes) goes to Drive; keyframes stay local for zipping.

Shows tqdm progress + per-group ETA + upload ETA (bytes/rate based).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

BASE_URL = "https://aic-data.ledo.io.vn"
DEST = "AIC_HCMC/2026/2025_Dataset"
SHOTS_TAR = f"{DEST}/shots.tar.gz"
POSITIONS = [0.15, 0.5, 0.85]
PNG_SIG = b"\x89PNG\r\n\x1a\n"

ALL_GROUPS = [f"K{i:02d}" for i in range(1, 21)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def rclone(args, check=True, retries=5):
    cmd = ["rclone", *args]
    for attempt in range(1, retries + 1):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0:
            return r
        log(f"  rclone {' '.join(args[:2])} attempt {attempt} rc={r.returncode}: "
            f"{(r.stderr or '').strip()[:200]}")
        time.sleep(15)
    if check:
        raise RuntimeError(f"rclone failed: {' '.join(args)}")
    return r


def ensure_rclone_config():
    cfg = Path(os.path.expanduser("~/.config/rclone/rclone.conf"))
    if not cfg.exists():
        raise RuntimeError(
            f"rclone config not found at {cfg}; run `rclone config` before starting"
        )
    rclone(["version"], check=True)


def verify_remote():
    r = subprocess.run(["rclone", "lsd", "gdrive:"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"rclone gdrive check failed: {r.stderr.strip()[:300]}")


def remote_listing(remote_dir):
    r = subprocess.run(["rclone", "lsf", remote_dir], capture_output=True, text=True)
    if r.returncode == 0:
        return set(r.stdout.split())
    if r.returncode == 3:  # directory not found (expected on first run)
        return set()
    raise RuntimeError(f"rclone lsf {remote_dir} rc={r.returncode}: {r.stderr.strip()[:300]}")


def rclone_copy(src, dst, transfers, checkers, extra=None):
    cmd = ["rclone", "copy", "--progress", "--stats", "1m",
           "--size-only",
           "--transfers", str(transfers), "--checkers", str(checkers)]
    if extra:
        cmd += extra
    cmd += [src, dst]
    for attempt in range(1, 6):
        r = subprocess.run(cmd)  # stream output so --progress shows live
        if r.returncode == 0:
            return
        log(f"  rclone copy attempt {attempt} rc={r.returncode}, retry in 15s")
        time.sleep(15)
    raise RuntimeError(f"rclone copy failed: {dst}")


def dir_size_bytes(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


class UploadTracker:
    """Running upload rate + ETA, self-correcting as groups complete."""

    def __init__(self):
        self.bytes = 0
        self.seconds = 0.0
        self.groups_done = 0

    def add(self, nbytes, dt):
        self.bytes += nbytes
        self.seconds += dt

    def mark_group_done(self):
        self.groups_done += 1

    def rate(self):
        return self.bytes / self.seconds if self.seconds > 0 else 0.0

    def eta_upload(self, remaining_groups):
        rate = self.rate()
        if rate <= 0 or self.groups_done == 0:
            return None
        avg_group = self.bytes / self.groups_done
        return avg_group * remaining_groups / rate


def compute_keyframes(shots, positions=POSITIONS):
    indices = []
    for start, end in shots:
        for pos in positions:
            idx = int(start + (end - start) * pos)
            indices.append(max(start, min(end, idx)))
    return indices


def is_valid_png(p):
    try:
        if p.stat().st_size == 0:
            return False
        with open(p, "rb") as fh:
            return fh.read(8) == PNG_SIG
    except OSError:
        return False


def extract_video_batch(video, frame_indices, out_dir, compression, fps, threads):
    """Extract all missing keyframes of one video, exact frame index.

    Uses `select='eq(n,rel)+...'` over ~100-frame chunks seeked with `-ss` so each
    chunk decodes only its slice (total ~= one pass). Frame index == decode order
    == the index TransNetV2 saw via cv2 (validated pixel-exact, including B-frames).
    Returns None on success, else the exception (for a retry pass).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = sorted({i for i in frame_indices
                      if not is_valid_png(out_dir / f"{i:06d}.png")})
    if not missing:
        return None
    CHUNK = 100  # ffmpeg select expression hits "Cannot allocate memory" beyond ~110 terms
    tmp_dir = out_dir / ".batch"
    try:
        for start in range(0, len(missing), CHUNK):
            chunk = missing[start:start + CHUNK]
            rel = [i - chunk[0] for i in chunk]
            sel = "+".join(f"eq(n\\,{i})" for i in rel)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-threads", str(threads),
                "-ss", f"{chunk[0] / fps:.6f}",
                "-t", f"{(chunk[-1] - chunk[0] + 2) / fps:.6f}",
                "-i", str(video),
                "-vf", f"select='{sel}'", "-vsync", "0",
                "-c:v", "png", "-pix_fmt", "rgb24",
                "-compression_level", str(compression),
                "-start_number", "0", str(tmp_dir / "f%06d.png"),
            ], check=True, capture_output=True, timeout=1800)
            outs = sorted(tmp_dir.glob("f*.png"))
            if len(outs) != len(chunk):
                raise RuntimeError(
                    f"select produced {len(outs)} frames, expected {len(chunk)}")
            for idx, o in zip(chunk, outs):
                o.replace(out_dir / f"{idx:06d}.png")
        return None
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return e


def extract_video_zip(group, work):
    zpath = work / f"Videos_{group}.zip"
    target = work / "videos" / group
    marker = work / f".{group}.extracted"
    if not marker.exists():
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zpath) as zf:
            names = [m for m in zf.namelist() if not m.endswith("/")]
            tops = {m.split("/", 1)[0] for m in names}
            strip = len(tops) == 1 and all("/" in m for m in names)
            for m in names:
                rel = m.split("/", 1)[1] if strip else m
                dest = target / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_suffix(".part")
                with zf.open(m) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
                tmp.replace(dest)
        marker.touch()
    return sorted(target.rglob("*.mp4"))


def zip_group(group, keyframes_root, work):
    arc = work / f"{group}.zip"
    if arc.exists():
        return arc
    tmp = work / f"{group}.zip.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for p in sorted((keyframes_root / group).rglob("*.png")):
            if p.name.endswith(".tmp.png") or p.name.endswith(".part"):
                continue
            z.write(p, p.relative_to(keyframes_root))
    tmp.replace(arc)
    return arc


def fetch_shots(work):
    shots = work / "shots"
    if not shots.exists():
        rclone(["copy", f"gdrive:{SHOTS_TAR}", str(work)], check=True)
        subprocess.run(["tar", "-xzf", str(work / "shots.tar.gz"), "-C", str(work)],
                       check=True)
        (work / "shots.tar.gz").unlink(missing_ok=True)
    n = len(list(shots.rglob("*.json")))
    log(f"shots ready: {n} JSON files")


def push(src, dst, args, tracker, extra=None):
    size = dir_size_bytes(src)
    t0 = time.time()
    rclone_copy(str(src), dst, args.transfers, args.checkers, extra=extra)
    tracker.add(size, time.time() - t0)
    return size


def process_group(group, work, shots, args, tracker):
    log(f"===== {group} =====")
    videos_root = work / "videos"
    keyframes_root = work / "keyframes"

    # 1. download video zip (skip if already extracted; .aria2 control file = incomplete)
    zpath = work / f"Videos_{group}.zip"
    ctrl = work / f"Videos_{group}.zip.aria2"
    marker = work / f".{group}.extracted"
    if not marker.exists() and (not zpath.exists() or ctrl.exists()):
        subprocess.run([
            "aria2c", "-d", str(work), "-o", f"Videos_{group}.zip",
            "-x", str(args.aria), "-s", str(args.aria),
            "-c", "--file-allocation=none", "--max-tries=5", "--retry-wait=10",
            "--console-log-level=warn", "--summary-interval=30",
            f"{BASE_URL}/Videos_{group}.zip",
        ], check=True)

    # 2. unzip, delete zip
    videos = extract_video_zip(group, work)
    zpath.unlink(missing_ok=True)
    log(f"  {len(videos)} videos")

    # 3. extract keyframes (batch per video) + retry failed videos once
    shots_dir = shots / group
    tasks = []
    for v in videos:
        sf = shots_dir / f"{v.stem}.json"
        if not sf.exists():
            log(f"  MISSING shots {sf.name} — skip")
            continue
        info = json.loads(sf.read_text())
        out_dir = keyframes_root / v.relative_to(videos_root).with_suffix("")
        tasks.append((v, compute_keyframes(info["shots"]), out_dir, info["fps"]))

    for attempt in range(2):
        if not tasks:
            break
        errs = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = pool.map(
                lambda t: extract_video_batch(t[0], t[1], t[2], args.compression, t[3], args.threads),
                tasks)
            for t, r in zip(tasks, tqdm(results, total=len(tasks),
                                        desc=f"{group} extract", unit="vid")):
                if r is not None:
                    errs.append((t, r))
        if not errs:
            break
        log(f"  retry: {len(errs)} videos failed")
        tasks = [t for t, _ in errs]

    # 4. zip + upload zip (keyframes stay local only), then delete
    arc = zip_group(group, keyframes_root, work)
    zip_bytes = push(arc, f"gdrive:{DEST}/zipped/", args, tracker)
    log(f"  uploaded {group}.zip ({zip_bytes/1e9:.2f} GB)")
    shutil.rmtree(keyframes_root / group, ignore_errors=True)
    arc.unlink(missing_ok=True)

    # 5. upload videos, then delete
    vid_bytes = push(videos_root / group, f"gdrive:{DEST}/videos/{group}",
                     args, tracker, extra=["--exclude", "*.part"])
    log(f"  videos uploaded {vid_bytes/1e9:.2f} GB")
    shutil.rmtree(videos_root / group, ignore_errors=True)

    log(f"  {group} DONE")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groups", default=",".join(ALL_GROUPS),
                    help="comma-separated groups to process (default all K01..K20)")
    ap.add_argument("--work", default="/workspace", help="scratch dir")
    ap.add_argument("--workers", type=int, default=5,
                    help="videos extracted in parallel (each uses ~threads+1 cores)")
    ap.add_argument("--threads", type=int, default=3,
                    help="ffmpeg decode threads per video")
    ap.add_argument("--compression", type=int, default=3,
                    help="PNG compression level 0-9 (3 = fast/lossless)")
    ap.add_argument("--aria", type=int, default=16, help="aria2c connections")
    ap.add_argument("--transfers", type=int, default=8, help="rclone upload transfers")
    ap.add_argument("--checkers", type=int, default=16, help="rclone checkers")
    ap.add_argument("--fetch-shots-only", action="store_true",
                    help="only download shots then exit")
    args = ap.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    ensure_rclone_config()
    verify_remote()
    fetch_shots(work)
    if args.fetch_shots_only:
        return

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    done = remote_listing(f"gdrive:{DEST}/zipped")
    to_do = [g for g in groups if f"{g}.zip" not in done]
    log(f"to process: {len(to_do)} groups (skip {len(groups) - len(to_do)} already done)")

    tracker = UploadTracker()
    group_times = []
    t_all = time.time()
    for i, group in enumerate(to_do, 1):
        t0 = time.time()
        process_group(group, work, work / "shots", args, tracker)
        dt = time.time() - t0
        group_times.append(dt)
        tracker.mark_group_done()

        avg = sum(group_times) / len(group_times)
        remaining = len(to_do) - i
        log(f"[{group} {i}/{len(to_do)}] nhóm {dt/60:.1f} phút "
            f"| avg {avg/60:.1f} phút "
            f"| uploaded {tracker.bytes/1e9:.1f} GB @ {tracker.rate()/1e6:.1f} MB/s")
        if remaining > 0:
            eta_g = avg * remaining
            eta_u = tracker.eta_upload(remaining)
            msg = f"  ETA theo nhóm: ~{eta_g/3600:.1f}h"
            if eta_u is not None:
                msg += f" | ETA upload còn: ~{eta_u/3600:.1f}h"
            log(msg)

    log(f"ALL DONE in {(time.time() - t_all)/3600:.1f}h "
        f"({tracker.bytes/1e9:.1f} GB uploaded)")


if __name__ == "__main__":
    sys.exit(main())
