#!/usr/bin/env python3
"""Embed the "N" traffic videos (Video_N001-N010/videos/N001-V001.mov) with PE-Core.

Self-contained on purpose, exactly like ``notebooks/kaggle_siglip2_embed.py``: paste this
file into a Kaggle script notebook, attach the video dataset, and run. Only ffmpeg, the
Python standard library and numpy are needed until the model is loaded.

One keyframe every ``STRIDE_SECONDS`` seconds, at exact decode-order frame indices::

    frames = [round(k * STRIDE_SECONDS * fps) for k in 0, 1, ... while t < duration]

so ``frame_idx / fps`` is the wall-clock time, the identity ``backend/frame_maps.py`` uses
everywhere else in the project. ``scripts/extraction/kaggle_extract_keyframes.py`` uses the
same never-seek decode for the same reason (see below).

Unlike batch 1 2026 there is no ``meta.parquet`` to read frame indices from -- the stride is
the whole point -- so the manifest is built here by probing the mounted videos with a
demux-only ffprobe (``-count_packets``, no decode). It uses the same schema as
``scripts.extraction.build_kaggle_manifest``, so it stays usable by the existing extractor
and can be published as a dataset the way ``tommr211/aic-keyframes`` was.

The datasets do not share one layout -- ``Video_N001-N010/videos/...`` in some, a bare
``videos/`` at the dataset root in others -- so the videos are found by the ``videos/``
folder itself, and the group label is read off the folder above it when there is one and
derived from the video ids when there is not.

A full part runs on its own: ``PART_COUNT=1``. The output directory and the manifest are
named after the group folders the run covers (``n2026-pe-core-N001-N010``), because the video
dataset ships in parts and a later part must not land in an earlier part's directory. If one
part is ever too large for a single session, set ``PART_COUNT`` and give each worker a
distinct ``PART_INDEX``: the partition is a deterministic greedy bin-packing over whole
videos, so every worker agrees on the split with no coordination, and each writes one
directory covering only its own videos. Merge those with
``scripts/embedding/assemble_gopt_batch1.py --dim 1280``.

A .mov in this corpus is very likely torn: a corpus-wide survey found 260 of 298 N-series
videos (87%) losing at least one frame internally (``ffprobe -count_packets`` overstating
``-count_frames``, one video by as much as 1223 of 13473), and all 298 emit decoder
warnings. Worse, a lost frame is not merely a gap -- extracting exact ids by seeking into
the file (``-ss <id>/fps`` per chunk) can silently return the *wrong* frame once a chunk's
seek point lands past a loss, because re-seeking loses synchronization with the file's true
decode-order numbering and no local signal (a matching frame count, no ffmpeg error) catches
it. Measured directly: a video with *zero* detected internal loss still had chunked,
seek-based extraction return the wrong frame at 4 of 5 sampled positions. Extraction
therefore never seeks past the very start of the file (see ``run_ffmpeg_select``): it decodes
straight through once, letting ``select`` count frames the decoder actually emits from frame
0, which is the only approach that has matched an independent ground truth on this corpus,
including on the most damaged file tested (9% of its frames lost -- every frame the pass
produced was correct, and the only gap was the real, honestly-reported tail past the file's
true length). A stride id past that true length is recorded in ``missing.json`` next to the
embeddings; only a video that fails to decode at all is listed in ``skipped.json``, so one
unreadable video does not abort a run that takes hours.
"""
import collections
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# CONFIG -- teammates forking the notebook change only this block.
# --------------------------------------------------------------------------------------
MODEL_ID = "hf-hub:timm/PE-Core-bigG-14-448"  # backend/core.py MODEL; embedding dim 1280
EMBED_DIM = 1280

STRIDE_SECONDS = 2.0  # one keyframe every 2 seconds
PART_COUNT = 1  # team run: 4. Every worker must use the same value.
PART_INDEX = 0  # this worker's slice, 0 .. PART_COUNT-1
CANARY = False  # True: embed only the first 2 videos as a smoke test
ONLY_VIDEOS = ()  # (): this worker's whole partition. A tuple of video ids embeds only those
                  # -- how a run redoes the videos an earlier run had to skip.

BATCH_SIZE = 8
SHARD_SIZE = 50_000
PROBE_WORKERS = 16  # ffprobe is demux-only; the videos are on a fast read-only mount
OUTPUT_NAME = None  # None: name the output after the group folders this run covers
# --------------------------------------------------------------------------------------

WORK = Path("/kaggle/working")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def video_dirs(*roots):
    """Every ``videos/`` folder under the mount that actually holds video files.

    The datasets do not share one layout: some nest it as ``Video_N001-N010/videos/...`` and
    at least one (N031-N040) ships a bare ``videos/`` directly at the dataset root, with no
    group folder at all. Only ``videos/`` itself is common to both, so that is what is
    searched for; the group label is worked out separately.
    """
    if not roots:
        override = os.environ.get("AIC_VIDEO_ROOT")
        roots = (Path(override),) if override else (Path("/kaggle/input"),)
    found = []
    for root in roots:
        for candidate in sorted(Path(root).rglob("videos")):
            if not candidate.is_dir():
                continue
            if any(candidate.glob("*.mov")) or any(candidate.glob("*.mp4")):
                found.append(candidate)
    if not found:
        raise SystemExit("no videos/ folder holding .mov/.mp4 under /kaggle/input -- "
                         "is the dataset attached?")
    return found


def dataset_root(videos_dir):
    """The dataset a ``videos/`` folder belongs to.

    A group folder (Video_N001-N010) sits between the dataset and its videos/ and is part of
    the relative path the repo's data layout expects; a bare videos/ has no such folder.
    """
    parent = Path(videos_dir).parent
    return parent.parent if parent.name.startswith("Video_N0") else parent


def group_name(parent_name, video_ids):
    """The repo's group label for one videos/ folder: Video_N001-N010.

    Read off the folder when the dataset has one, and derived from the ids it holds when it
    does not -- ``data/<year>/videos/<group>/<video_id>.mp4`` wants a group either way, and
    a run's output directory is named after it.
    """
    if parent_name.startswith("Video_N0"):
        return parent_name
    numbers = sorted(int(video_id[1:4]) for video_id in video_ids)
    if numbers[0] == numbers[-1]:
        return f"Video_N{numbers[0]:03d}"
    return f"Video_N{numbers[0]:03d}-N{numbers[-1]:03d}"


def discover_videos(videos_dirs, root):
    """video_id -> {rel, group, path} for every video under the given videos/ folders.

    video_id is the file stem (N001-V001). Group is the folder above videos/ when the
    dataset has one, and otherwise derived from the ids, so the manifest keeps the shape the
    repo's data/<year>/videos/<group>/<video_id>.mp4 convention expects.
    """
    videos = {}
    for videos_dir in videos_dirs:
        paths = sorted(set(videos_dir.glob("*.mov")) | set(videos_dir.glob("*.mp4")))
        ids = {}
        for path in paths:
            video_id = path.stem
            if re.fullmatch(r"N\d{3}-V\d{3}", video_id) is None:
                raise SystemExit(f"{path}: video id {video_id!r} is not the N###-V### form "
                                 f"the group label is derived from")
            if video_id in ids:
                raise ValueError(f"duplicate video id {video_id}: {ids[video_id]} and {path}")
            ids[video_id] = path
        group = group_name(videos_dir.parent.name, list(ids))
        for video_id, path in ids.items():
            if video_id in videos:
                raise ValueError(f"duplicate video id {video_id} in two videos/ folders")
            videos[video_id] = {
                "rel": str(path.relative_to(root)),
                "group": group,
                "path": path,
            }
    if not videos:
        raise SystemExit(f"no videos under {root}")
    return videos


def coverage_tag(videos):
    """Short label for the group folders a run covers, e.g. N001-N010_N041-N050.

    The video dataset ships in parts, so an output has to say which part it is: a later run
    over different groups must not land in the same directory, nor look like a replacement
    for an earlier one. Groups are the "Video_N001-N010" style folder names.
    """
    groups = sorted({entry["group"] for entry in videos.values()})
    return "_".join(group.removeprefix("Video_") for group in groups)


def probe(path):
    """(fps, frame count) for one video, without decoding a single frame.

    ``-count_packets`` demuxes only, unlike ``-count_frames`` which decodes everything.
    """
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_read_packets",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream.get("r_frame_rate") or ""
    if rate in ("", "0/0"):
        rate = stream.get("avg_frame_rate") or ""
    if rate in ("", "0/0"):
        raise ValueError(f"{path}: ffprobe returned no usable frame rate")
    numerator, denominator = rate.split("/")
    fps = float(numerator) / float(denominator)
    if fps <= 0:
        raise ValueError(f"{path}: unusable frame rate {rate}")
    count = stream.get("nb_read_packets")
    if count in (None, "N/A"):
        raise ValueError(f"{path}: ffprobe returned no packet count")
    return fps, int(count)


def stride_frames(frame_count, fps, stride_seconds):
    """Decode-order frame indices one every ``stride_seconds``: 0, 2fps, 4fps, ..."""
    step = stride_seconds * fps
    frames = []
    k = 0
    while True:
        frame = int(round(k * step))
        if frame >= frame_count:
            return frames
        if not frames or frame != frames[-1]:
            frames.append(frame)
        k += 1


def build_manifest(root, videos, stride_seconds, workers=PROBE_WORKERS):
    """Manifest in scripts.extraction.build_kaggle_manifest's schema, plus the stride."""
    probed = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(probe, entry["path"]): video_id
                for video_id, entry in videos.items()}
        for job in as_completed(jobs):
            video_id = jobs[job]
            try:
                probed[video_id] = job.result()
            except Exception as exc:  # one unreadable file must not sink the run
                failures.append(f"{video_id}: {exc}")
    if failures:
        raise RuntimeError("probe failures:\n" + "\n".join(sorted(failures)))

    out = {}
    for video_id, entry in sorted(videos.items()):
        fps, count = probed[video_id]
        out[video_id] = {"rel": entry["rel"], "group": entry["group"], "fps": fps,
                         "frames": stride_frames(count, fps, stride_seconds)}
    empty = [video_id for video_id, entry in out.items() if not entry["frames"]]
    if empty:
        log(f"{len(empty)} video(s) shorter than one stride: {', '.join(empty[:5])}")
    return {
        "batch": "n2026",
        "year": "2026",
        "stride_seconds": stride_seconds,
        "root": str(root),
        "video_count": len(out),
        "keyframe_count": sum(len(entry["frames"]) for entry in out.values()),
        "videos": out,
    }


def split_videos(manifest, part_index, part_count):
    """Assign whole videos to balanced, deterministic partitions.

    Longest video first into the currently lightest bin, so every worker computes the same
    disjoint split with no coordination. Copied from notebooks/kaggle_siglip2_embed.py.
    """
    if not 0 <= part_index < part_count:
        raise ValueError(f"part index {part_index} is outside 0..{part_count - 1}")
    bins = [[] for _ in range(part_count)]
    totals = [0] * part_count
    videos = sorted(manifest["videos"].items(),
                    key=lambda item: (-len(item[1]["frames"]), item[0]))
    for video_id, entry in videos:
        target = min(range(part_count), key=lambda index: (totals[index], index))
        bins[target].append(video_id)
        totals[target] += len(entry["frames"])
    return set(bins[part_index]), totals


def chunk_ranges(count, limit=100):
    """Split ``count`` sequential ids into evenly sized chunks of at most ``limit``.

    This no longer exists for seek reliability -- nothing here seeks. ``limit`` is purely
    ffmpeg's select expression running out of memory past roughly 110 explicit OR terms, so
    it only matters for ``extract_chunk``'s non-uniform-stride fallback below.
    """
    if count <= 0:
        return []
    chunks = -(-count // limit)
    size = -(-count // chunks)
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def uniform_step(frames):
    """The constant gap between consecutive stride ids, or ``None`` if it varies.

    Most fps values give a perfectly uniform stride (25, 30, 12.5 fps: 2*fps is exact);
    only a non-integer rate like 29.97 rounds unevenly frame to frame (alternating 59/60
    gaps -- see the self-test). A uniform step is what lets extraction run as a single
    unseeked pass instead of several explicit-id batches.
    """
    if len(frames) < 2:
        return None
    gaps = {b - a for a, b in zip(frames, frames[1:])}
    return gaps.pop() if len(gaps) == 1 else None


SHOWINFO_RE = re.compile(r"Parsed_showinfo.*?\bn:\s*(\d+)\s+pts:\s*-?\d+\s+pts_time:(-?[\d.]+)")


def parse_pts_times(stderr_bytes):
    """position -> real playback time in seconds, from a showinfo-instrumented pass.

    showinfo logs one line per frame that passes ``select``, in the same order the image2
    muxer numbers its output files (position 0 is ``f000000.png``, and so on), so this
    lines up with ``sorted(batch.glob("f*.png"))`` with no extra bookkeeping. Reading it
    off the same pass that already decoded the frame costs nothing extra: N001-N100 is
    VFR (organizer-confirmed), so ``frame_idx`` here is only a relative, reproducible
    decode-order label -- the real per-frame time has to come from PTS, not from
    ``frame_idx / fps``, and that only exists in the decoder's own accounting.
    """
    text = stderr_bytes.decode("utf-8", "replace")
    return {int(m.group(1)): float(m.group(2)) for m in SHOWINFO_RE.finditer(text)}


def run_ffmpeg_select(video, select_expr, want, batch, threads):
    """One ffmpeg pass, decoding from the true start of the file -- never ``-ss``.

    This is the fix. Chunked, seeked extraction (``-ss <id>/fps`` per window) can silently
    return the wrong frame even when nothing about the video looks damaged: measured on a
    video with *zero* detected packet loss, seek-based chunked extraction was still wrong
    at 4 of 5 sampled positions, because re-seeking loses synchronization with the file's
    true decode-order numbering in a way no local check (a matching frame count, no ffmpeg
    error) can catch. Decoding straight through and letting ``select`` count every frame
    the decoder actually emits, from frame 0, is the only approach that matched an
    independent ground truth in that investigation -- including on the most damaged file
    tested (9% of its frames internally lost): every frame such a pass produced was
    correct, and the only gap was the real, honestly-reported tail past the file's true
    decodable length. ``want`` caps the output count so the pass stops once satisfied
    rather than reading to EOF. ``showinfo`` rides along in the same filter chain so the
    real PTS of every frame comes back for free (see ``parse_pts_times``); ``-loglevel
    info`` is what makes it show up at all (``showinfo`` logs at info level).

    Returns ``(pngs, pts_seconds)`` -- the files written and their PTS by position, both in
    order -- or ``(None, None)`` if ffmpeg refused outright (a badly torn region can still
    push the decoder's error rate past its limit and abort).
    """
    shutil.rmtree(batch, ignore_errors=True)
    batch.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "info",
        "-threads", str(threads),
        "-i", str(video), "-vf", f"select='{select_expr}',showinfo", "-vsync", "0",
        "-frames:v", str(want),
        "-c:v", "png", "-pix_fmt", "rgb24", "-compression_level", "1",
        "-start_number", "0", str(batch / "f%06d.png"),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=3600)
    except subprocess.TimeoutExpired:
        return None, None
    if result.returncode != 0:
        return None, None
    outs = sorted(batch.glob("f*.png"))
    pts_by_position = parse_pts_times(result.stderr)
    return outs, [pts_by_position.get(i) for i in range(len(outs))]


def extract_chunk(video, ids, batch, threads, output):
    """A batch of (possibly non-uniformly spaced) ids, still always read from true 0.

    Used only when the stride is not perfectly uniform (rare non-integer fps, e.g. 29.97 --
    ``uniform_step`` returns ``None``), where ``select`` needs an explicit OR of exact ids
    rather than one ``mod`` term, or as the fallback if a single-pass ``mod`` extraction is
    ever refused outright. Every batch still decodes from frame 0, never a fresh ``-ss``,
    so a short result is unambiguously the tail of *this batch* -- there is no seek point
    for a torn region to desynchronize, so only an outright ffmpeg failure (not a short
    count) needs splitting, and the split is on the id list, never on where to seek.

    A success is moved into ``output`` immediately, before returning: ``batch`` is one
    shared scratch directory across every sibling call in the recursion, and
    ``run_ffmpeg_select`` wipes it at the start of each call, so leaving a result sitting
    there for the *caller* to move only after the whole tree of calls finishes would let a
    later sibling delete it first.

    Returns ``(ids, pts_seconds)`` for the ids actually written, both in the same order.
    """
    if not ids:
        return [], []
    select = "+".join(f"eq(n\\,{frame_id})" for frame_id in ids)
    outs, pts = run_ffmpeg_select(video, select, len(ids), batch, threads)
    if outs is not None:
        for frame_id, frame in zip(ids, outs):
            frame.replace(output / f"{frame_id:06d}.png")
        return list(ids), pts
    if len(ids) == 1:
        return [], []
    middle = len(ids) // 2
    left_ids, left_pts = extract_chunk(video, ids[:middle], batch, threads, output)
    right_ids, right_pts = extract_chunk(video, ids[middle:], batch, threads, output)
    return left_ids + right_ids, left_pts + right_pts


def extract_exact_frames(video, frame_ids, output, fps, threads=2):
    """Extract exact decode-order frame ids, matching the PE-Core extraction path.

    Never seeks into the middle of the file (see ``run_ffmpeg_select``): the common case
    (a uniform stride, which covers every fps in this corpus except a handful at 29.97) is
    one unseeked pass with ``select='not(mod(n,step))'``; everything else -- a non-uniform
    stride, or that one pass being refused outright -- falls back to explicit-id batches,
    still each read from true 0, split only on ffmpeg failure and never on a short count.

    Returns ``(written, pts_ms)``: the ids actually written (``frame_ids`` minus any the
    file's real decodable length falls short of), and a ``{frame_id: milliseconds}`` map of
    each one's real playback time. N001-N100 is VFR (organizer-confirmed), so ``frame_id /
    fps`` is only an approximate, relative label here, not the true time -- ``pts_ms`` is
    what a KIS/QA submission for this corpus is expected to use instead.
    """
    output.mkdir(parents=True, exist_ok=True)
    requested = sorted({int(frame_id) for frame_id in frame_ids})
    batch = output / ".batch"
    step = uniform_step(requested)
    written = None
    pts_ms = {}
    if step is not None:
        outs, pts = run_ffmpeg_select(video, f"not(mod(n\\,{step}))", len(requested), batch,
                                      threads)
        if outs is not None:
            written = []
            for i, (frame, seconds) in enumerate(zip(outs, pts)):
                frame_id = i * step
                frame.replace(output / f"{frame_id:06d}.png")
                written.append(frame_id)
                if seconds is not None:
                    pts_ms[frame_id] = round(seconds * 1000)
    if written is None:
        written = []
        for first, last in chunk_ranges(len(requested)):
            ids, pts = extract_chunk(video, requested[first:last], batch, threads, output)
            written.extend(ids)
            for frame_id, seconds in zip(ids, pts):
                if seconds is not None:
                    pts_ms[frame_id] = round(seconds * 1000)
    shutil.rmtree(batch, ignore_errors=True)
    return written, pts_ms


def rgb_digest(path):
    """Pixel fingerprint of one extracted frame, for the canary's determinism check."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        return {
            "width": image.width,
            "height": image.height,
            "sha256_rgb": hashlib.sha256(image.tobytes()).hexdigest(),
        }


class PartWriter:
    """One worker's buffered shard writer inside parts/part<PART_INDEX>.

    Shards and meta lines carry the rank so two GPUs writing the same partition never
    clobber each other. Rank suffixes live only in the intermediate parts/ tree; the merged
    output uses the plain emb_%06d.npy names the repo expects.
    """

    def __init__(self, root, part_index, rank, dimension, shard_size=SHARD_SIZE):
        self.root = Path(root) / f"part{part_index}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.dimension = dimension
        self.shard_size = shard_size
        self.buffer = np.empty((shard_size, dimension), dtype=np.float32)
        self.fill = 0
        self.shard = 0
        self.total = 0
        self.meta = open(self.root / f"meta_r{rank}.jsonl", "w", buffering=1)

    def add_batch(self, vectors, video_id, frame_ids, pts_ms=None):
        """``pts_ms``: ``{frame_id: milliseconds}``, the real playback time of each frame.

        N001-N100 is VFR (organizer-confirmed), so ``frame_idx / fps`` is only a relative
        decode-order label there, not the true time; ``pts_ms`` is what a KIS/QA submission
        for that corpus is expected to report instead. ``None`` for a video this was never
        computed for (batch 1/2's own CFR pipeline has no VFR ambiguity to solve).
        """
        pts_ms = pts_ms or {}
        for vector, frame_id in zip(vectors, frame_ids):
            self.buffer[self.fill] = vector
            self.meta.write(json.dumps({
                "rank": self.rank, "shard": self.shard, "row": self.fill,
                "video_id": video_id, "frame_idx": int(frame_id),
                "pts_ms": pts_ms.get(int(frame_id)),
            }) + "\n")
            self.fill += 1
            self.total += 1
            if self.fill == self.shard_size:
                self.flush()

    def flush(self):
        if not self.fill:
            return
        np.save(self.root / f"emb_r{self.rank}_{self.shard:06d}.npy", self.buffer[:self.fill])
        self.shard += 1
        self.fill = 0

    def close(self):
        self.flush()
        self.meta.close()


def ensure_open_clip():
    try:
        import open_clip  # noqa: F401
    except ImportError:
        log("installing open-clip-torch from git")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/mlfoundations/open_clip.git",
        ], check=True)


def load_model(device):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_ID, precision="fp16", device=device)
    model.eval()
    return model, preprocess


def encode_paths(model, preprocess, paths, device, batch_size, progress):
    """Yield (vectors, batch_size) over image paths, halving the batch on OOM."""
    import torch
    from PIL import Image

    offset = 0
    while offset < len(paths):
        size = min(batch_size, len(paths) - offset)
        while True:
            try:
                images = []
                for path in paths[offset:offset + size]:
                    with Image.open(path) as image:
                        images.append(preprocess(image.convert("RGB")))
                batch = torch.stack(images).to(device, non_blocking=True)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    vectors = model.encode_image(batch, normalize=True)
                    # Re-normalize in float32: fp16 autocast alone loses a few 1e-3.
                    vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
                vectors = vectors.cpu().numpy().astype(np.float32, copy=False)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                batch_size = min(batch_size, size)
                progress.write(f"{device}: OOM, reducing batch size to {batch_size}")
        yield vectors, batch_size
        offset += size


def run_rank(rank, nprocs, config, manifest, selected_video_ids):
    """Embed this rank's share of the partition into parts/part<PART_INDEX>."""
    import torch
    from tqdm.auto import tqdm

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    videos = [(video_id, manifest["videos"][video_id])
              for video_id in sorted(manifest["videos"]) if video_id in selected_video_ids]
    videos = videos[rank::nprocs]
    total = sum(len(entry["frames"]) for _, entry in videos)
    log(f"GPU {rank}: loading {MODEL_ID}; videos={len(videos)}; frames={total:,}")

    model, preprocess = load_model(device)
    # load_model builds the model with precision="fp16", so its weights are half tensors.
    # The probe has to go through the same autocast as the real batches, or the patch-embed
    # convolution raises "Input type (cuda.FloatTensor) and weight type (cuda.HalfTensor)".
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        probe_batch = torch.zeros(1, 3, 448, 448, device=device)
        dimension = int(model.encode_image(probe_batch, normalize=True).shape[1])
    torch.cuda.empty_cache()
    if dimension != EMBED_DIM:
        raise RuntimeError(f"{MODEL_ID} emits dim {dimension}, expected {EMBED_DIM}")

    writer = PartWriter(config["parts_dir"], config["part_index"], rank, dimension)
    batch_size = config["batch_size"]
    digests = {}
    counts = {}
    skipped = {}
    missing = {}
    progress = tqdm(total=total, desc=f"GPU {rank} extract+embed", unit="frame",
                    position=rank, dynamic_ncols=True)
    started = time.monotonic()
    for video_number, (video_id, entry) in enumerate(videos, 1):
        target = Path(config["temp_dir"]) / f"gpu{rank}" / video_id
        error = None
        written, pts_ms = [], {}
        for attempt in range(2):
            shutil.rmtree(target, ignore_errors=True)
            try:
                written, pts_ms = extract_exact_frames(
                    Path(manifest["root"]) / entry["rel"], entry["frames"], target,
                    entry["fps"])
                error = None
                break
            except Exception as exc:
                error = exc
                progress.write(f"GPU {rank} {video_id}: extraction attempt "
                               f"{attempt + 1} failed: {exc}")
        if error is not None:
            # One unreadable video must not throw away every other video's work in a run
            # that takes hours. Record it and carry on; the parent reports it loudly.
            skipped[video_id] = f"{type(error).__name__}: {error}"
            progress.write(f"GPU {rank}: SKIPPING {video_id}: {error}")
            shutil.rmtree(target, ignore_errors=True)
            continue
        if not written:
            # Every id in the video was unreadable: nothing to add, and a video with zero
            # rows would make the output check fail rather than report it as skipped.
            skipped[video_id] = "no decodable frames"
            progress.write(f"GPU {rank}: SKIPPING {video_id}: no decodable frames")
            shutil.rmtree(target, ignore_errors=True)
            continue
        if len(written) != len(entry["frames"]):
            # Which stride ids have no decodable frame, recorded rather than inferred: the
            # output check has to accept these holes, and it must not accept any other.
            missing[video_id] = sorted(set(int(f) for f in entry["frames"]) - set(written))
            progress.write(f"GPU {rank} {video_id}: wrote {len(written)} of "
                           f"{len(entry['frames'])} keyframes; "
                           f"{len(missing[video_id])} do not decode")
        counts[video_id] = len(written)

        paths = [target / f"{int(frame_id):06d}.png" for frame_id in written]
        if config["canary"]:
            digests[video_id] = {str(frame_id): rgb_digest(path)
                                 for frame_id, path in zip(written, paths)}
        consumed = 0
        for vectors, batch_size in encode_paths(
                model, preprocess, paths, device, batch_size, progress):
            frame_ids = written[consumed:consumed + len(vectors)]
            writer.add_batch(vectors, video_id, frame_ids, pts_ms)
            consumed += len(vectors)
            progress.update(len(vectors))
            elapsed = max(time.monotonic() - started, 1e-6)
            progress.set_postfix(video=video_id, img_s=f"{progress.n / elapsed:.1f}",
                                 bs=batch_size, ckpt=writer.total,
                                 mem=f"{torch.cuda.memory_allocated(rank) / 2**30:.1f}G")
        if consumed != len(written):
            raise RuntimeError(f"{video_id}: embedded {consumed}, expected {len(written)}")
        shutil.rmtree(target)
        if video_number % 5 == 0 or video_number == len(videos):
            progress.write(f"GPU {rank}: {video_number}/{len(videos)} videos, "
                           f"{writer.total:,}/{total:,} frames checkpointed")
    writer.close()
    progress.close()
    part_dir = Path(config["parts_dir"]) / f"part{config['part_index']}"
    (part_dir / f"digests_r{rank}.json").write_text(json.dumps(digests, sort_keys=True))
    # What was actually embedded, per video: the parent needs the true total to size the
    # merge, because a skipped video or a dropped tail frame makes it differ from the
    # manifest's keyframe count.
    (part_dir / f"counts_r{rank}.json").write_text(json.dumps(counts, sort_keys=True))
    (part_dir / f"skipped_r{rank}.json").write_text(json.dumps(skipped, indent=2,
                                                                sort_keys=True))
    (part_dir / f"missing_r{rank}.json").write_text(json.dumps(missing, sort_keys=True))
    log(f"GPU {rank}: complete, {writer.total:,} embeddings, "
        f"{len(skipped)} video(s) skipped, "
        f"{sum(len(v) for v in missing.values())} frame(s) undecodable")


def merge_parts(parts_dir, output_dir, expected, extra_info):
    """Merge this run's parts into the repo's sharded embedding layout.

    Carries ``pts_ms`` through into ``meta.parquet``: N001-N100 is VFR, so ``frame_idx`` is
    only a relative decode-order label there, and ``pts_ms`` (real playback milliseconds,
    from the same ffmpeg pass that decoded the frame) is what a KIS/QA submission is
    expected to report instead. Nullable, since it is ``None`` for anything embedded before
    this existed and for the rare non-uniform-stride fallback path if a specific frame's
    ``showinfo`` line was ever missed.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for meta_path in sorted(Path(parts_dir).glob("part*/meta_r*.jsonl")):
        part = int(meta_path.parent.name[len("part"):])
        with open(meta_path) as handle:
            for line in handle:
                row = json.loads(line)
                row["part"] = part
                records.append(row)
    records.sort(key=lambda row: (row["video_id"], row["frame_idx"]))
    if len(records) != expected:
        raise RuntimeError(f"merged {len(records):,} rows, expected {expected:,}")
    keys = [(row["video_id"], row["frame_idx"]) for row in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate video_id/frame_idx pairs in worker output")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    buffer = np.empty((SHARD_SIZE, EMBED_DIM), dtype=np.float32)
    fill = shard_idx = 0
    meta_rows = []
    cache_key = None
    cache = None
    for row in records:
        key = row["part"], row["rank"], row["shard"]
        if key != cache_key:
            cache = np.load(Path(parts_dir) / f"part{key[0]}" /
                            f"emb_r{key[1]}_{key[2]:06d}.npy", mmap_mode="r")
            cache_key = key
        buffer[fill] = cache[row["row"]]
        meta_rows.append((shard_idx, fill, row["video_id"], row["frame_idx"],
                          row.get("pts_ms")))
        fill += 1
        if fill == SHARD_SIZE:
            np.save(output_dir / f"emb_{shard_idx:06d}.npy", buffer)
            shard_idx += 1
            fill = 0
    if fill:
        np.save(output_dir / f"emb_{shard_idx:06d}.npy", buffer[:fill])

    pq.write_table(pa.table({
        "shard_idx": pa.array([r[0] for r in meta_rows], type=pa.int32()),
        "row_idx": pa.array([r[1] for r in meta_rows], type=pa.int32()),
        "video_id": pa.array([r[2] for r in meta_rows], type=pa.string()),
        "frame_idx": pa.array([r[3] for r in meta_rows], type=pa.int32()),
        "pts_ms": pa.array([r[4] for r in meta_rows], type=pa.int64()),
    }), output_dir / "meta.parquet")
    (output_dir / "info.json").write_text(json.dumps({
        "model": MODEL_ID, "dim": EMBED_DIM, "count": len(records),
        "shard_size": SHARD_SIZE, "normalized": True, "dtype": "float32",
        **extra_info,
    }, indent=2) + "\n")

    digests = {}
    for path in sorted(Path(parts_dir).glob("part*/digests_r*.json")):
        digests.update(json.loads(path.read_text()))
    if digests:
        (output_dir / "pixel_digests.json").write_text(json.dumps(digests, indent=2) + "\n")
    return len(records)


def verify_output(output_dir, manifest, selected_video_ids, skipped=(), missing=None):
    """Fail-closed checks that the output really is the stride we asked for.

    ``missing`` maps a video to the stride frame ids the extraction recorded as having no
    decodable frame. Those are the only holes allowed: a torn .mov really does lose frames in
    the middle of the file, and an earlier version of this check called exactly those a
    failure -- so a partially recovered video failed the whole run at the very end, after
    hours. Everything unexplained still fails.
    """
    import pyarrow.parquet as pq

    missing = missing or {}
    output_dir = Path(output_dir)
    rows = pq.read_table(output_dir / "meta.parquet").to_pylist()
    info = json.loads((output_dir / "info.json").read_text())
    stride = float(info["stride_seconds"])
    # Videos with fewer frames than one stride contribute nothing and cannot appear, and a
    # skipped video has no rows at all -- both have to be excluded from the expected set.
    active = {video_id for video_id in selected_video_ids
              if manifest["videos"].get(video_id, {}).get("frames")} - set(skipped)
    problems = []
    if info["count"] != len(rows):
        problems.append(f"info.json count {info['count']} != meta.parquet rows {len(rows)}")
    if info["dim"] != EMBED_DIM:
        problems.append(f"info.json dim {info['dim']} != {EMBED_DIM}")

    by_video = collections.defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(int(row["frame_idx"]))
    if set(by_video) != active:
        problems.append(f"videos in output {sorted(set(by_video))} != selected {sorted(active)}")
    for video_id, frames in by_video.items():
        entry = manifest["videos"].get(video_id)
        if entry is None:
            problems.append(f"{video_id}: not in the manifest")
            continue
        frames.sort()
        # The output must be the manifest's stride minus exactly the ids the worker recorded
        # as undecodable. Comparing the two lists directly says what the old position-gap
        # arithmetic said -- a 29.97 rate's alternating 59/60 steps are simply consecutive
        # entries here -- without reasoning about step sizes at all, and it accepts a hole
        # the file really has while still rejecting one nobody can account for.
        undecodable = set(missing.get(video_id, ()))
        expected = [f for f in entry["frames"] if f not in undecodable]
        if frames != expected:
            extra = [f for f in frames if f not in set(entry["frames"])]
            problems.append(
                f"{video_id}: output has {len(frames)} stride frames, expected "
                f"{len(expected)} (manifest {len(entry['frames'])} minus "
                f"{len(entry['frames']) - len(expected)} recorded undecodable)"
                + (f"; {len(extra)} are not in the stride at all" if extra else ""))

    if rows:
        shard = np.load(output_dir / "emb_000000.npy", mmap_mode="r")
        if shard.shape[1] != EMBED_DIM:
            problems.append(f"shard dim {shard.shape[1]} != {EMBED_DIM}")
        norms = np.linalg.norm(np.asarray(shard[:8], dtype=np.float32), axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            problems.append(f"sample vectors are not L2-normalized: {norms}")

    if problems:
        raise RuntimeError("output verification failed:\n" + "\n".join(problems))
    per_video = ", ".join(f"{v}:{len(f)}" for v, f in sorted(by_video.items())[:4])
    log(f"verified {len(rows):,} embeddings over {len(by_video)} videos "
        f"(dim {EMBED_DIM}, stride {stride}s); first: {per_video}")


def self_test():
    """Offline check of the stride maths, the writer and the merge -- no GPU, no Kaggle."""
    import runpy
    import tempfile
    import pyarrow.parquet as pq

    # mp.spawn re-imports this file as __mp_main__ in every child. If main() is not guarded,
    # each child re-runs the whole pipeline and recurses into another spawn -- this caught
    # exactly that on a real Kaggle run. The env flag stops the re-entrant self_test() in
    # that child, so a broken guard reports the cause instead of a RecursionError.
    if os.environ.get("AIC_SELFTEST_REENTRY") != "1":
        os.environ["AIC_SELFTEST_REENTRY"] = "1"
        try:
            runpy.run_path(__file__, run_name="__mp_main__")
        except SystemExit as exc:
            raise AssertionError(
                f"importing as __mp_main__ ran main(); mp.spawn children would recurse ({exc})"
            ) from exc
        finally:
            os.environ.pop("AIC_SELFTEST_REENTRY", None)

    assert stride_frames(300, 30.0, 2.0) == [0, 60, 120, 180, 240]
    assert stride_frames(299, 30.0, 2.0) == [0, 60, 120, 180, 240]
    assert stride_frames(60, 30.0, 2.0) == [0]
    assert stride_frames(59, 30.0, 2.0) == [0]
    assert stride_frames(10, 25.0, 2.0) == [0]
    # 29.97: rounding must not drift into a neighbouring frame.
    ntsc = stride_frames(18_000, 30000 / 1001, 2.0)
    assert ntsc[:4] == [0, 60, 120, 180], ntsc[:4]
    assert all(abs(f / (30000 / 1001) - 2 * k) < 0.05 for k, f in enumerate(ntsc))

    # A non-integer rate gives alternating frame steps (29.97 -> 59 and 60) -- the shape that
    # once made a gap check built on stride*fps report 18 legitimate steps as missing frames.
    assert sorted({b - a for a, b in zip(ntsc, ntsc[1:])}) == [59, 60]
    assert sum(1 for a, b in zip(ntsc, ntsc[1:]) if abs(b - a - 6e4 / 1001) > 0.51) == 18

    fake = {"videos": {f"N{i:03d}-V001": {"frames": list(range(0, 30 * (i + 1), 60))}
                       for i in range(1, 13)}}
    partitions = [split_videos(fake, index, 4)[0] for index in range(4)]
    assert set.union(*partitions) == set(fake["videos"])
    assert not any(left & right for i, left in enumerate(partitions)
                   for right in partitions[i + 1:])
    assert partitions == [split_videos(fake, index, 4)[0] for index in range(4)]

    assert coverage_tag({"a": {"group": "Video_N041-N050"}}) == "N041-N050"
    assert coverage_tag({"a": {"group": "Video_N011-N020"},
                         "b": {"group": "Video_N001-N010"}}) == "N001-N010_N011-N020"

    # The datasets do not share one layout: some nest Video_N001-N010/videos/, and part 3
    # (N031-N040) ships a bare videos/ at the dataset root with no group folder at all. Only
    # videos/ is common, so that is what gets searched for, and the group label is read off
    # the folder when there is one and derived from the ids when there is not -- both the
    # output directory name and the repo's data/<year>/videos/<group>/<video_id>.mp4 want it.
    assert group_name("Video_N001-N010", ["N001-V001"]) == "Video_N001-N010"
    assert group_name("d681c290396d498f3be5d-part3",
                      ["N031-V001", "N040-V003"]) == "Video_N031-N040"
    assert group_name("d681c290396d498f3be5d-part3", ["N031-V001"]) == "Video_N031"

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        nested = root / "Video_N001-N010" / "videos"
        nested.mkdir(parents=True)
        for name in ("N001-V001.mov", "N001-V002.mov"):
            (nested / name).write_bytes(b"")
        flat = root / "d681c290396d498f3be5d-part3" / "videos"
        flat.mkdir(parents=True)
        for name in ("N031-V001.mov", "N032-V002.mov", "N040-V003.mov"):
            (flat / name).write_bytes(b"")
        (root / "empty" / "videos").mkdir(parents=True)  # a videos/ holding no videos

        assert sorted(str(directory.relative_to(root)) for directory in video_dirs(root)) == [
            "Video_N001-N010/videos", "d681c290396d498f3be5d-part3/videos"]

        # Nested layout: the group is the folder, and it stays in the relative path.
        # rglob looks at descendants, so the search root is the dataset, not the videos/.
        one = video_dirs(nested.parent)
        assert dataset_root(one[0]) == root
        found = discover_videos(one, dataset_root(one[0]))
        assert set(found) == {"N001-V001", "N001-V002"}
        assert found["N001-V001"]["group"] == "Video_N001-N010"
        assert found["N001-V001"]["rel"] == "Video_N001-N010/videos/N001-V001.mov"

        # Bare layout: no folder to read, so the group comes from the ids it holds.
        one = video_dirs(flat.parent)
        assert dataset_root(one[0]) == flat.parent
        found = discover_videos(one, dataset_root(one[0]))
        assert set(found) == {"N031-V001", "N032-V002", "N040-V003"}
        assert {entry["group"] for entry in found.values()} == {"Video_N031-N040"}
        assert found["N031-V001"]["rel"] == "videos/N031-V001.mov"

    # Chunking is only for the ffmpeg select expression's ~110-term memory limit now;
    # nothing seeks, so a lone tail id is no riskier than any other single-id batch.
    assert chunk_ranges(0) == []
    assert chunk_ranges(1) == [(0, 1)]
    assert chunk_ranges(100) == [(0, 100)]
    assert chunk_ranges(101) == [(0, 51), (51, 101)]
    assert chunk_ranges(301) == [(0, 76), (76, 152), (152, 228), (228, 301)]
    for count in (1, 2, 99, 100, 101, 199, 301, 1000, 35217):
        ranges = chunk_ranges(count)
        assert ranges[0][0] == 0 and ranges[-1][1] == count
        assert all(b - a <= 100 for a, b in ranges), (count, ranges)
        assert all(nxt[0] == cur[1] for cur, nxt in zip(ranges, ranges[1:])), count
        if count > 1:
            assert ranges[-1][1] - ranges[-1][0] > 1, (count, ranges)

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = {
            "root": str(root), "videos": {
                "N001-V001": {"rel": "Video_N001-N010/videos/N001-V001.mov",
                              "group": "Video_N001-N010", "fps": 30.0,
                              "frames": [0, 60, 120]},
                "N001-V002": {"rel": "Video_N001-N010/videos/N001-V002.mov",
                              "group": "Video_N001-N010", "fps": 30.0,
                              "frames": [0, 60]},
            }}
        # Two parts, disjoint videos -- exactly how run_rank partitions a real selection.
        # shard_size=2 makes the 3-frame video roll over into a second part shard.
        pts_by_video = {"N001-V001": {0: 0, 60: 2000, 120: 4001},
                       "N001-V002": {0: 0, 60: 2000}}
        for part_index, video_ids in ((0, ["N001-V001"]), (1, ["N001-V002"])):
            writer = PartWriter(root / "parts", part_index, 0, EMBED_DIM, shard_size=2)
            for video_id in video_ids:
                frames = manifest["videos"][video_id]["frames"]
                vectors = np.full((len(frames), EMBED_DIM), part_index + 1.0, np.float32)
                vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
                writer.add_batch(vectors, video_id, frames, pts_by_video[video_id])
            writer.close()
            (root / "parts" / f"part{part_index}" / "digests_r0.json").write_text("{}")

        selected = set(manifest["videos"])
        count = merge_parts(root / "parts", root / "out", 5,
                            {"stride_seconds": STRIDE_SECONDS, "part_index": 0,
                             "part_count": 1})
        assert count == 5
        table = pq.read_table(root / "out" / "meta.parquet")
        assert table.num_rows == 5
        assert table.column_names == ["shard_idx", "row_idx", "video_id", "frame_idx",
                                      "pts_ms"]
        assert [p.name for p in sorted((root / "out").glob("emb_*.npy"))] == ["emb_000000.npy"]
        assert np.load(root / "out" / "emb_000000.npy").shape == (5, EMBED_DIM)
        merged = list(zip(table["video_id"].to_pylist(), table["frame_idx"].to_pylist()))
        assert merged == sorted(merged), "merge must sort by (video_id, frame_idx)"
        assert merged[0] == ("N001-V001", 0) and merged[-1] == ("N001-V002", 60)
        # pts_ms rides along with its own (video_id, frame_idx), independent of merge order.
        by_key = {(r["video_id"], r["frame_idx"]): r["pts_ms"] for r in table.to_pylist()}
        assert by_key[("N001-V001", 120)] == 4001
        assert by_key[("N001-V002", 60)] == 2000
        verify_output(root / "out", manifest, selected)

    # A stride frame the file cannot decode is expected on this corpus, not a failure. The
    # old check derived holes from the frame numbers themselves and called every one of them
    # a problem -- so once extraction learned to recover the rest of a torn video, the run
    # would have died at the final check after hours of GPU time. What is verified instead is
    # that the output is the manifest's stride minus exactly the ids the worker recorded, and
    # that an unrecorded hole is still caught.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        manifest = {"root": str(root), "videos": {
            "N001-V001": {"rel": "x", "group": "Video_N001-N010", "fps": 30.0,
                          "frames": [0, 60, 120, 180, 240]}}}
        writer = PartWriter(root / "parts", 0, 0, EMBED_DIM, shard_size=8)
        vectors = np.zeros((4, EMBED_DIM), np.float32)
        vectors[:, 0] = 1.0
        writer.add_batch(vectors, "N001-V001", [0, 60, 180, 240])  # 120 does not decode
        writer.close()
        (root / "parts" / "part0" / "digests_r0.json").write_text("{}")
        merge_parts(root / "parts", root / "out", 4,
                    {"stride_seconds": STRIDE_SECONDS, "part_index": 0, "part_count": 1})
        verify_output(root / "out", manifest, {"N001-V001"}, (), {"N001-V001": [120]})
        try:
            verify_output(root / "out", manifest, {"N001-V001"})
        except RuntimeError as exc:
            assert "N001-V001" in str(exc), exc
        else:
            raise AssertionError("a hole nobody recorded must fail verification")

    assert uniform_step([0]) is None
    assert uniform_step([0, 60, 120, 180]) == 60
    assert uniform_step(ntsc) is None  # the 29.97 fixture above: alternating 59/60

    # The one invariant every code path must keep: nothing ever seeks. A chunked, seeked
    # extractor was measured wrong at 4 of 5 sampled positions on a video with *zero*
    # detected internal frame loss, so seeking is never safe here, torn file or not.
    # subprocess.run is faked (no real ffmpeg needed): it records the command, asserts no
    # "-ss", and writes exactly the "-frames:v" count of PNGs so run_ffmpeg_select's own
    # glob sees a clean success, the same as a real pass finding everything it asked for.
    real_run = subprocess.run
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        assert "-ss" not in command, f"a seek slipped back in: {command}"
        batch_dir = Path(command[-1]).parent
        batch_dir.mkdir(parents=True, exist_ok=True)
        want = int(command[command.index("-frames:v") + 1])
        for i in range(want):
            (batch_dir / f"f{i:06d}.png").write_bytes(b"")

        class _Done:
            returncode = 0
            stderr = b""

        return _Done()

    subprocess.run = fake_run
    try:
        # Fast path: a uniform stride is one pass with a mod() term, no OR-list at all.
        with tempfile.TemporaryDirectory() as temp:
            written, pts_ms = extract_exact_frames(Path("clean.mov"), [0, 60, 120, 180],
                                                    Path(temp), 30.0)
    finally:
        subprocess.run = real_run
    assert written == [0, 60, 120, 180], written
    assert pts_ms == {}, pts_ms  # the fake never emits showinfo lines; absent, not wrong
    assert len(commands) == 1, f"a uniform stride must need exactly one pass: {commands}"
    assert "select='not(mod(n\\,60))',showinfo" in " ".join(commands[0]), commands[0]
    assert "-loglevel" in commands[0] and commands[0][commands[0].index("-loglevel") + 1] \
        == "info", "showinfo logs at info level; -loglevel error would hide it"

    # A short result from the fast pass is not a mislabel -- it can only be the real tail,
    # because nothing seeks. Modelled directly on run_ffmpeg_select's contract (uses a real
    # temp file so the ".mov" extension and image2 muxer path both work end to end).
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        batch = root / ".batch"

        def fake_run_ffmpeg_select(video, select_expr, want, batch, threads):
            # A uniform-stride request for `want` ids, but the "file" only really has
            # `available` of them -- exactly what a torn file whose packet count overstates
            # its true decodable length looks like from this function's point of view.
            available = 3
            batch.mkdir(parents=True, exist_ok=True)
            outs, pts = [], []
            for i in range(min(want, available)):
                p = batch / f"f{i:06d}.png"
                p.write_text(str(i))
                outs.append(p)
                pts.append(i * 2.0)  # matches the real step*fps timing for this fixture
            return outs, pts

        real_select = globals()["run_ffmpeg_select"]
        globals()["run_ffmpeg_select"] = fake_run_ffmpeg_select
        try:
            written, pts_ms = extract_exact_frames(Path("torn.mov"), [0, 60, 120, 180, 240],
                                                    root / "out", 30.0)
        finally:
            globals()["run_ffmpeg_select"] = real_select
        assert written == [0, 60, 120], written  # exactly the tail is missing, never a gap
        assert pts_ms == {0: 0, 60: 2000, 120: 4000}, pts_ms

    # Non-uniform stride (NTSC-style): explicit id batches, still never seeked, split on
    # outright ffmpeg failure only -- modelled by making one specific id "poison" the whole
    # batch it lands in, recoverable only by halving the *id list*, never a seek point.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        poison = 178
        calls = []

        def fake_select(video, select_expr, want, batch, threads):
            calls.append(want)
            ids = [int(term.split(",")[1].rstrip(")")) for term in select_expr.split("+")]
            if poison in ids:
                return None, None  # the whole batch is refused, like an ffmpeg abort
            # Match run_ffmpeg_select's real contract: wipe the batch dir first, so
            # filenames from one recursive call never collide with a sibling's leftovers.
            shutil.rmtree(batch, ignore_errors=True)
            batch.mkdir(parents=True, exist_ok=True)
            outs, pts = [], []
            for i, frame_id in enumerate(ids):
                p = batch / f"f{i:06d}.png"
                p.write_text(str(frame_id))
                outs.append(p)
                pts.append(frame_id / 30.0)  # a plausible, if not VFR-jittered, PTS
            return outs, pts

        # Alternating 59/60 gaps, exactly the shape 29.97 fps produces: uniform_step is None,
        # so extract_exact_frames must go straight to the explicit-id batch path below.
        ntsc_like = [0, 59, 119, poison, 238, 297]
        assert uniform_step(ntsc_like) is None
        real_select = globals()["run_ffmpeg_select"]
        globals()["run_ffmpeg_select"] = fake_select
        try:
            written, pts_ms = extract_exact_frames(Path("ntsc.mov"), ntsc_like, root / "out",
                                                    30.0)
        finally:
            globals()["run_ffmpeg_select"] = real_select
        assert written == [i for i in ntsc_like if i != poison], written
        assert pts_ms == {i: round(i / 30.0 * 1000) for i in written}, pts_ms
        for frame_id in written:
            held = (root / "out" / f"{frame_id:06d}.png").read_text()
            assert held == str(frame_id), f"{frame_id} holds frame {held}: mislabelled"
        assert len(calls) <= 4 * len(ntsc_like), f"splitting ran away: {calls}"
    print("self-test OK")


def main():
    if os.environ.get("AIC_SELFTEST_REENTRY") == "1":
        # self_test() re-imports this file as __mp_main__ with this flag set. Reaching main()
        # means the guard at the bottom is gone, so every mp.spawn child would re-run the
        # whole pipeline and recurse into another spawn.
        raise AssertionError("main() ran while the file was re-imported as __mp_main__")

    if "--self-test" in sys.argv:
        self_test()
        return

    if not Path("/kaggle/input").exists():
        raise SystemExit("full runs are only supported inside Kaggle (or use --self-test)")

    # A run costs hours, so the cheap offline checks go first: a truncated upload or a slip in
    # this file then fails in seconds with a clear message, rather than somewhere deep in a
    # run nobody can afford to repeat.
    self_test()

    part_index = int(os.environ.get("AIC_PART_INDEX", PART_INDEX))
    part_count = int(os.environ.get("AIC_PART_COUNT", PART_COUNT))
    canary = os.environ.get("AIC_CANARY", "1" if CANARY else "0") == "1"
    stride = float(os.environ.get("AIC_STRIDE_SECONDS", STRIDE_SECONDS))

    ensure_open_clip()
    dirs = video_dirs()
    roots = {dataset_root(directory) for directory in dirs}
    if len(roots) != 1:
        raise SystemExit(f"videos come from more than one dataset: "
                         f"{sorted(str(root) for root in roots)}")
    root = roots.pop()
    videos = discover_videos(dirs, root)
    log(f"video root: {root} ({len(dirs)} videos/ folder(s))")
    log(f"probing {len(videos)} videos for fps + frame count (demux only)")
    manifest = build_manifest(root, videos, stride)
    log(f"manifest: {manifest['video_count']} videos, "
        f"{manifest['keyframe_count']:,} keyframes at {stride}s")

    # Name the run after the groups it covers: this dataset ships in parts, and a later
    # run must not land in the same directory as an earlier one.
    tag = coverage_tag(manifest["videos"])
    name = OUTPUT_NAME or f"n2026-pe-core-{tag}"

    manifest_path = WORK / "manifest" / f"n2026_{int(stride)}s_{tag}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
    log(f"wrote {manifest_path}")

    selected_video_ids, partition_totals = split_videos(manifest, part_index, part_count)
    only = os.environ.get("AIC_ONLY_VIDEOS")
    only_ids = {v.strip() for v in only.split(",") if v.strip()} if only else set(ONLY_VIDEOS)
    if only_ids:
        # Narrowing the selection must not move the output: a redo of the skipped videos has
        # to land in the same directory as the run it completes, so the name keeps coming
        # from the partition and only which videos get embedded changes.
        unknown = sorted(only_ids - set(manifest["videos"]))
        if unknown:
            raise SystemExit(f"ONLY_VIDEOS are not in this dataset: {unknown}")
        selected_video_ids = only_ids
        log(f"ONLY_VIDEOS: embedding just these {len(only_ids)} video(s)")
    if canary:
        selected_video_ids = set(sorted(selected_video_ids)[:2])
    expected = sum(len(manifest["videos"][video_id]["frames"])
                   for video_id in selected_video_ids)

    output = WORK / "embeddings" / name
    parts = WORK / ".pe_core_parts" / name
    temp = WORK / ".pe_core_frames" / name
    for path in (output, parts, temp):
        shutil.rmtree(path, ignore_errors=True)

    import torch

    nprocs = max(1, min(2, torch.cuda.device_count()))
    log(f"part {part_index}/{part_count}; coverage={tag}; "
        f"selected_videos={len(selected_video_ids)}; selected_frames={expected:,}; "
        f"partition_frames={partition_totals}; gpus={nprocs}; canary={canary}")
    config = {
        "part_index": part_index, "parts_dir": str(parts), "temp_dir": str(temp),
        "batch_size": int(os.environ.get("AIC_BATCH_SIZE", BATCH_SIZE)),
        "canary": canary,
    }
    # spawn needs a real file to re-import; a plain notebook kernel has no __file__.
    spawnable = getattr(sys.modules.get("__main__"), "__file__", None) is not None
    if nprocs == 1 or not spawnable:
        if nprocs > 1:
            log(f"{nprocs} GPUs but no importable __main__; running ranks sequentially")
        for rank in range(nprocs):
            run_rank(rank, nprocs, config, manifest, selected_video_ids)
    else:
        import torch.multiprocessing as mp
        mp.spawn(run_rank, args=(nprocs, config, manifest, selected_video_ids),
                 nprocs=nprocs, join=True)

    counts = {}
    for path in sorted(Path(parts).glob("part*/counts_r*.json")):
        counts.update(json.loads(path.read_text()))
    skipped = {}
    for path in sorted(Path(parts).glob("part*/skipped_r*.json")):
        skipped.update(json.loads(path.read_text()))
    missing = {}
    for path in sorted(Path(parts).glob("part*/missing_r*.json")):
        missing.update(json.loads(path.read_text()))
    if missing:
        log(f"{sum(len(v) for v in missing.values())} stride frame(s) across "
            f"{len(missing)} video(s) have no decodable frame; recorded in missing.json")
    if skipped:
        log(f"WARNING: {len(skipped)} of {len(selected_video_ids)} video(s) skipped -- "
            f"their keyframes are not in the output:")
        for video_id, reason in sorted(skipped.items()):
            log(f"  {video_id}: {reason}")
    unaccounted = sorted(set(selected_video_ids) - set(counts) - set(skipped))
    if unaccounted:
        raise RuntimeError(f"{len(unaccounted)} video(s) neither embedded nor skipped: "
                           f"{unaccounted[:10]}")
    # Size the merge by what the workers actually wrote, not by the manifest: a skipped
    # video or a dropped tail frame makes the two differ.
    expected = sum(counts.values())

    merge_parts(parts, output, expected, {
        "stride_seconds": stride, "part_index": part_index, "part_count": part_count,
        "groups": sorted({entry["group"] for entry in manifest["videos"].values()}),
        "skipped": sorted(skipped),
        "undecodable_frames": sum(len(v) for v in missing.values()),
    })
    if skipped:
        (Path(output) / "skipped.json").write_text(
            json.dumps(skipped, indent=2, sort_keys=True) + "\n")
    if missing:
        (Path(output) / "missing.json").write_text(
            json.dumps(missing, indent=2, sort_keys=True) + "\n")
    shutil.rmtree(parts)
    shutil.rmtree(temp, ignore_errors=True)
    verify_output(output, manifest, selected_video_ids, skipped, missing)
    log(f"DONE: {expected:,} embeddings over "
        f"{len(selected_video_ids) - len(skipped)} videos "
        f"({sum(len(v) for v in missing.values())} keyframes undecodable) -> {output}")
    log(f"manifest to publish as a dataset: {manifest_path}")


# Run when Kaggle executes this file as a script or pastes it into a notebook cell -- both
# set __name__ to __main__. Never run in an mp.spawn child, which re-imports this file under
# the name __mp_main__ and would otherwise re-run the whole pipeline. The plain guard also
# keeps the module importable, which is how --self-test exercises the pieces.
if __name__ == "__main__":
    main()
