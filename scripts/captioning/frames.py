"""Sample a caption unit's frames straight out of the source MP4 as JPEGs.

Adapted from `caption_shots_gemini.sample_video_frames`, which already gets the awkward
parts right: seek with `-ss` before `-i` so ffmpeg does not decode the whole file, pick
exact decode-order indices inside the seeked segment, and stream MJPEG down a pipe
rather than writing temporary files. The additions here are a window (rather than a
whole shot) and a resolution knob, which is the single biggest cost lever in the run.

No temporary clips are rendered, so this stays consistent with how `caption_shots.py`
and `backend/shots.py` treat a shot: a time window on the source clock.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import numpy as np


DEFAULT_QUALITY = 2
DEFAULT_TIMEOUT = 180
SOI, EOI = b"\xff\xd8", b"\xff\xd9"


MIN_FRAMES = 4
# A window of D seconds at F fps does not reliably decode round(D*F) frames: the `-ss`
# seek lands on a frame boundary that rounding cannot predict, so the last index is
# often absent and `select=eq(n,last)` then matches nothing. Staying one frame short of
# the computed tail costs at most one frame of coverage and removes the whole class of
# "decoded 23/24 frames" failures.
TAIL_MARGIN = 1
# TransNetV2's cut boundary is not exact -- the frames right at the start or end of a
# shot can still show a sliver of the scene before/after it (observed: L26_V448 shot 39
# opens on a frame or two of the previous kitchen-wide-shot before the actual peppercorn
# close-up that is this shot's real content). Sampling only the inset middle keeps the
# few frames the model sees on-content instead of on the transition. 12% off each end
# leaves a 76% core; EDGE_INSET_FLOOR keeps very short shots from losing coverage they
# cannot spare.
EDGE_INSET_FRACTION = 0.12
EDGE_INSET_FLOOR = 4  # frames


def frame_positions(unit):
    """Decode-order indices to keep, relative to the start of the seeked window."""
    available = max(1, int(unit.duration_sec * unit.fps) - TAIL_MARGIN)
    count = min(unit.frame_count, available)
    inset = min(int(available * EDGE_INSET_FRACTION), (available - 1) // 2)
    if available < EDGE_INSET_FLOOR:
        inset = 0  # nothing to spare on a handful of frames
    lo, hi = inset, max(inset, available - 1 - inset)
    return np.rint(np.linspace(lo, hi, count)).astype(int).tolist()


def ffmpeg_command(unit, positions, width=0, quality=DEFAULT_QUALITY):
    """`width=0` keeps the source resolution; otherwise scale down, never up."""
    filters = ["+".join(f"eq(n\\,{position})" for position in positions)]
    filters[0] = f"select={filters[0]}"
    if width:
        # -2 keeps the aspect ratio on an even height; min() never upscales a
        # source that is already narrower than the target.
        filters.append(f"scale=w=min(iw\\,{width}):h=-2")
    return [
        "ffmpeg", "-loglevel", "error",
        "-ss", f"{unit.window_start_sec:.6f}",
        "-i", unit.video_path,
        "-t", f"{unit.duration_sec:.6f}",
        "-vf", ",".join(filters),
        "-fps_mode", "vfr", "-frames:v", str(len(positions)),
        "-an", "-c:v", "mjpeg", "-q:v", str(quality),
        "-f", "image2pipe", "pipe:1",
    ]


def split_jpeg_stream(payload):
    """MJPEG over a pipe has no container; cut on the JPEG start/end markers."""
    frames, offset = [], 0
    while (start := payload.find(SOI, offset)) >= 0:
        end = payload.find(EOI, start + 2)
        if end < 0:
            break
        frames.append(payload[start:end + 2])
        offset = end + 2
    return frames


def sample_frames(unit, width=0, quality=DEFAULT_QUALITY, timeout=DEFAULT_TIMEOUT):
    positions = frame_positions(unit)
    result = subprocess.run(
        ffmpeg_command(unit, positions, width, quality),
        check=True, capture_output=True, timeout=timeout,
    )
    frames = split_jpeg_stream(result.stdout)
    # A frame or two short is a boundary artefact, not a broken window: the frames that
    # did arrive are still the uniform sample, minus its tail. Losing a third of them is
    # a real decode problem and should fail the unit.
    floor = max(MIN_FRAMES, len(positions) - 2)
    if len(frames) < min(floor, len(positions)):
        raise OSError(
            f"decoded {len(frames)}/{len(positions)} frames for {unit.unit_id}"
        )
    return frames


def cache_key(unit, width, quality):
    """Identifies the exact pixels, so an A/B can replay them bit for bit."""
    digest = hashlib.sha256(
        f"{unit.video_path}|{unit.window_start_sec:.6f}|{unit.duration_sec:.6f}"
        f"|{unit.frame_count}|{width}|{quality}".encode()
    ).hexdigest()
    return f"{unit.video_id}_{unit.shot_index:05d}_{unit.window_index:03d}_{digest[:12]}"


def sample_frames_cached(unit, cache_dir, width=0, quality=DEFAULT_QUALITY,
                         timeout=DEFAULT_TIMEOUT):
    """Frame cache for the pilot sweeps.

    The KV-dtype A/B is only meaningful if both passes see identical input, and
    re-running ffmpeg is not guaranteed to be bit-identical across builds. The full run
    does not use this -- it would need 100k+ files on disk for no benefit.
    """
    if cache_dir is None:
        return sample_frames(unit, width, quality, timeout)
    directory = Path(cache_dir) / cache_key(unit, width, quality)
    if directory.is_dir():
        paths = sorted(directory.glob("*.jpg"))
        if paths:
            return [path.read_bytes() for path in paths]
    frames = sample_frames(unit, width, quality, timeout)
    staging = directory.with_name(f".{directory.name}.tmp")
    staging.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        (staging / f"{index:03d}.jpg").write_bytes(frame)
    staging.replace(directory)
    return frames
