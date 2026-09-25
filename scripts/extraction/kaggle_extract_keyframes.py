#!/usr/bin/env python3
"""Extract keyframes from videos on Kaggle, driven by a manifest.

Self-contained on purpose: copy this file (or paste it into a cell) next to a manifest
from `scripts.extraction.build_kaggle_manifest` and run it. Only ffmpeg and the Python
standard library are required; tqdm is used when available.

    python kaggle_extract_keyframes.py \
        --manifest /kaggle/input/aic-keyframe-manifests/batch1_2026.json \
        --out /kaggle/working/keyframes --workers 4

Videos listed in the manifest but absent from the mount are skipped, so a notebook
with only one of the batch2 parts attached still works. Pass --strict to fail instead.

Frame index means decode order, matching how the original keyframes were cut and how
indexes/frame_maps maps frames to time. Output is <out>/<group>/<video_id>/%06d.png,
which is the layout scripts/embedding/embed_keyframes.py expects.

A .mov in this corpus is very likely torn: a corpus-wide survey of the N-series videos found
260 of 298 (87%) losing at least one frame internally (`ffprobe -count_packets` overstating
`-count_frames`, one video by 1223 of 13473), and all 298 emit decoder warnings. Worse, a
lost frame is not merely a gap -- extracting exact ids by seeking into the file (`-ss
<id>/fps` per chunk) can silently return the *wrong* frame once a chunk's seek point lands
past a loss, because re-seeking loses synchronization with the file's true decode-order
numbering and no local signal (a matching frame count, no ffmpeg error) catches it. Measured
directly: a video with *zero* detected internal loss still had chunked, seek-based
extraction return the wrong frame at 4 of 5 sampled positions. Extraction therefore never
seeks past the very start of the file (see `run_ffmpeg_select`): frames the file's real
decodable length falls short of are listed in <out>/undecodable.json; only a video that
fails to decode at all makes this exit non-zero.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_valid_png(p):
    try:
        if p.stat().st_size == 0:
            return False
        with open(p, "rb") as fh:
            return fh.read(8) == PNG_SIG
    except OSError:
        return False


def uniform_step(frames):
    """The constant gap between consecutive stride ids, or None if it varies.

    Most fps values give a perfectly uniform stride (25, 30, 12.5 fps: 2*fps is exact);
    only a non-integer rate like 29.97 rounds unevenly frame to frame (alternating 59/60
    gaps). A uniform step is what lets extraction run as a single unseeked pass instead of
    several explicit-id batches.
    """
    if len(frames) < 2:
        return None
    gaps = {b - a for a, b in zip(frames, frames[1:])}
    return gaps.pop() if len(gaps) == 1 else None


def run_ffmpeg_select(video, select_expr, want, tmp_dir, compression, threads):
    """One ffmpeg pass, decoding from the true start of the file -- never `-ss`.

    This is the fix. Chunked, seeked extraction (`-ss <id>/fps` per window) can silently
    return the wrong frame even when nothing about the video looks damaged: measured on a
    video with *zero* detected packet loss, seek-based chunked extraction was still wrong
    at 4 of 5 sampled positions, because re-seeking loses synchronization with the file's
    true decode-order numbering and no local check (a matching frame count, no ffmpeg
    error) catches it. Decoding straight through and letting `select` count every frame the
    decoder actually emits, from frame 0, is the only approach that matched an independent
    ground truth in that investigation -- including on the most damaged file tested (9% of
    its frames internally lost): every frame such a pass produced was correct, and the only
    gap was the real, honestly-reported tail past the file's true decodable length. `want`
    caps the output count so the pass stops once satisfied rather than reading to EOF.

    Returns the PNGs written, in order, or None if ffmpeg refused outright (a badly torn
    region can still push the decoder's error rate past its limit and abort).
    """
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", str(threads),
            "-i", str(video), "-vf", f"select='{select_expr}'", "-vsync", "0",
            "-frames:v", str(want),
            "-c:v", "png", "-pix_fmt", "rgb24",
            "-compression_level", str(compression),
            "-start_number", "0", str(tmp_dir / "f%06d.png"),
        ], capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    return sorted(tmp_dir.glob("f*.png"))


def extract_ids(video, ids, tmp_dir, out_dir, compression, threads):
    """A batch of (possibly non-uniformly spaced) ids, still always read from true 0.

    Used only when the stride is not perfectly uniform (rare non-integer fps, e.g. 29.97 --
    `uniform_step` returns None), where `select` needs an explicit OR of exact ids rather
    than one `mod` term, or as the fallback if a single-pass `mod` extraction is ever
    refused outright. Every batch still decodes from frame 0, never a fresh `-ss`, so a
    short result is unambiguously the tail of *this batch* -- there is no seek point for a
    torn region to desynchronize, so only an outright ffmpeg failure (not a short count)
    needs splitting, and the split is on the id list, never on where to seek.

    A success is moved into `out_dir` immediately, before returning: `tmp_dir` is one shared
    scratch directory across every sibling call in the recursion, and `run_ffmpeg_select`
    wipes it at the start of each call, so leaving a result for the caller to move only
    after the whole tree of calls finishes would let a later sibling delete it first.
    """
    if not ids:
        return []
    select = "+".join(f"eq(n\\,{i})" for i in ids)
    outs = run_ffmpeg_select(video, select, len(ids), tmp_dir, compression, threads)
    if outs is not None:
        for frame_id, out in zip(ids, outs):
            out.replace(out_dir / f"{frame_id:06d}.png")
        return list(ids)
    if len(ids) == 1:
        return []
    middle = len(ids) // 2
    return (extract_ids(video, ids[:middle], tmp_dir, out_dir, compression, threads)
            + extract_ids(video, ids[middle:], tmp_dir, out_dir, compression, threads))


def extract_video_batch(video, frame_indices, out_dir, compression, fps, threads):
    """Extract all missing keyframes of one video, exact frame index.

    Never seeks into the middle of the file (see `run_ffmpeg_select`): the common case (a
    uniform stride, which covers every fps in this corpus except a handful at 29.97) is one
    unseeked pass with `select='not(mod(n,step))'`; everything else -- a non-uniform stride,
    or that one pass being refused outright -- falls back to explicit-id batches, still each
    read from true 0. Frame index == decode order == the index TransNetV2 saw via cv2, and
    the same one the PE-Core notebooks embed -- this is the extractor that has to agree with
    them, so both use the same never-seek approach.

    Returns `(error, undecodable)`: `error` is an exception for a retry pass or None, and
    `undecodable` lists the ids the file's real decodable length falls short of -- reported
    rather than raised on, because one torn .mov must not fail a corpus-wide run, and the
    caller has to know which moments are missing.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted({int(i) for i in frame_indices})
    missing = sorted({i for i in frames if not is_valid_png(out_dir / f"{i:06d}.png")})
    if not missing:
        return None, []
    tmp_dir = out_dir / ".batch"
    written = []
    try:
        # The step must come from the full stride, not from `missing`: a resumed run (some
        # ids already on disk from an earlier pass) can leave `missing` a non-contiguous
        # subset whose own internal gaps are not the file's real stride spacing.
        step = uniform_step(frames)
        fast_path_ran = False
        if step is not None:
            # Through the full stride's last id, not just missing's -- the pass has to walk
            # past every already-present id on the way regardless, so this costs nothing
            # extra; outputs for ids we already have are simply not moved anywhere.
            total_positions = frames[-1] // step + 1
            outs = run_ffmpeg_select(video, f"not(mod(n\\,{step}))", total_positions,
                                     tmp_dir, compression, threads)
            if outs is not None:
                fast_path_ran = True
                missing_set = set(missing)
                for i, out in enumerate(outs):
                    frame_id = i * step
                    if frame_id in missing_set:
                        out.replace(out_dir / f"{frame_id:06d}.png")
                        written.append(frame_id)
        if not fast_path_ran:
            CHUNK = 100  # ffmpeg select hits "Cannot allocate memory" past ~110 terms
            still = [i for i in missing if i not in set(written)]
            for start in range(0, len(still), CHUNK):
                written.extend(extract_ids(video, still[start:start + CHUNK], tmp_dir,
                                           out_dir, compression, threads))
        return None, [i for i in missing if i not in set(written)]
    except Exception as e:
        return e, []
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def group_of(video_id):
    """Group label for a video id, for manifests that do not carry one.

    Both id shapes in this corpus put the group before the first separator: L21_V001 -> L21,
    N031-V001 -> N031. Only a fallback -- a manifest with its own "group" wins, because the
    repo's label there (Video_N031-N040) cannot be derived from the id alone.
    """
    return video_id.split("_", 1)[0].split("-", 1)[0]


def video_group(video_id, entry):
    return entry.get("group") or group_of(video_id)


def video_path(root, video_id, entry, layout):
    """Where a manifest entry's mp4 should live under `root`."""
    if layout == "flat":
        return root / group_of(video_id) / f"{video_id}.mp4"
    return root / entry["rel"]


def select_videos(manifest, only):
    """Manifest entries filtered by a set of group names and/or video ids."""
    videos = manifest["videos"]
    if not only:
        return dict(sorted(videos.items()))
    wanted = {t.strip() for t in only.split(",") if t.strip()}
    picked = {v: e for v, e in videos.items() if v in wanted or group_of(v) in wanted}
    unknown = wanted - set(picked) - {group_of(v) for v in picked}
    if unknown:
        raise SystemExit(f"--only matched nothing for: {', '.join(sorted(unknown))}")
    return dict(sorted(picked.items()))


def resolve(root, videos, layout, strict):
    """Split manifest entries into ones whose mp4 exists and ones that are missing."""
    found, missing = {}, []
    for video_id, entry in videos.items():
        path = video_path(root, video_id, entry, layout)
        if path.exists():
            found[video_id] = (path, entry)
        else:
            missing.append(video_id)
    if missing and strict:
        raise SystemExit(
            f"{len(missing)} videos are missing under {root}, first: {missing[0]}")
    return found, missing


def zip_group(out_dir, group, dest_dir):
    """Store (no deflate — PNGs are already compressed) one group into <group>.zip."""
    dest = dest_dir / f"{group}.zip"
    tmp = dest.with_suffix(".zip.tmp")
    src = out_dir / group
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
        for png in sorted(src.rglob("*.png")):
            zf.write(png, str(png.relative_to(out_dir)))
    tmp.replace(dest)
    return dest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="/kaggle/working/keyframes")
    parser.add_argument("--root", default=None,
                        help="override the manifest's root (default: manifest root)")
    parser.add_argument("--layout", choices=("kaggle", "flat"), default="kaggle",
                        help="kaggle: <root>/<rel> from the manifest; "
                             "flat: <root>/<group>/<video_id>.mp4")
    parser.add_argument("--only", default=None,
                        help="comma-separated group names and/or video ids")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2,
                        help="ffmpeg decode threads per video")
    parser.add_argument("--compression", type=int, default=1,
                        help="PNG compression level, 0-9")
    parser.add_argument("--strict", action="store_true",
                        help="fail if any selected video is missing from the mount")
    parser.add_argument("--zip-per-group", default=None, metavar="DIR",
                        help="after extraction, write <group>.zip into DIR")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.root if args.root is not None else manifest["root"])
    out_dir = Path(args.out)

    videos = select_videos(manifest, args.only)
    found, missing = resolve(root, videos, args.layout, args.strict)
    frames_total = sum(len(e["frames"]) for _, e in found.values())
    log(f"{manifest['batch']}/{manifest['year']}: {len(found)} videos, "
        f"{frames_total} keyframes"
        + (f"; skipped {len(missing)} not on this mount" if missing else ""))
    if not found:
        return

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(it, **kw):
            return it

    def work(item):
        video_id, (path, entry) = item
        target = out_dir / video_group(video_id, entry) / video_id
        return video_id, extract_video_batch(
            path, entry["frames"], target, args.compression, entry["fps"], args.threads)

    failures, undecodable = {}, {}

    def collect(video_id, result):
        err, absent = result
        if err is not None:
            failures[video_id] = err
        elif absent:
            undecodable[video_id] = absent

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item) for item in found.items()]
        for fut in tqdm(as_completed(futures), total=len(futures), unit="video"):
            video_id, result = fut.result()
            collect(video_id, result)

    if failures:
        log(f"retrying {len(failures)} failed videos")
        retry = dict(failures)
        failures = {}
        for video_id in retry:
            _, result = work((video_id, found[video_id]))
            collect(video_id, result)

    if undecodable:
        total = sum(len(v) for v in undecodable.values())
        log(f"{total} keyframe(s) across {len(undecodable)} video(s) have no decodable "
            f"frame in the file itself; recorded in undecodable.json")
        for video_id, ids in sorted(undecodable.items()):
            log(f"  {video_id}: {len(ids)} frame(s), first {ids[0]}")
        (out_dir / "undecodable.json").write_text(
            json.dumps(undecodable, indent=2, sort_keys=True) + "\n")

    if args.zip_per_group:
        dest_dir = Path(args.zip_per_group)
        dest_dir.mkdir(parents=True, exist_ok=True)
        for group in sorted({video_group(v, e) for v, (_, e) in found.items()}):
            log(f"zipped {zip_group(out_dir, group, dest_dir)}")

    if failures:
        for video_id, err in sorted(failures.items()):
            log(f"FAILED {video_id}: {err}")
        sys.exit(1)
    log(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
