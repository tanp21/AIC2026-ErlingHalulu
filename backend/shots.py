"""TransNetV2 shot boundaries, and exact-PTS clip extraction via ffmpeg.

Shot boundaries are produced offline by `scripts.extraction.detect_shots` and read
here as `data/<year>/shots/<group>/<video_id>.json`, mirroring the layout of
`core.mounted_video_path`. Frame indices are decode order, the same identity
`frame_maps.FrameMap` uses, so every frame -> time conversion goes through the map.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))
MAX_CLIP_SECONDS = float(os.environ.get("AIC_CLIP_MAX_SECONDS", 120))
CLIP_TIMEOUT_SECONDS = 300
_clip_slots = threading.BoundedSemaphore(2)


@dataclass(frozen=True)
class ShotIndex:
    """Inclusive [start_frame, end_frame] bounds, ordered by start frame."""

    fps: float
    bounds: np.ndarray

    def __post_init__(self):
        bounds = np.asarray(self.bounds, dtype=np.int64).reshape(-1, 2)
        if len(bounds) and (
            np.any(bounds[:, 1] < bounds[:, 0])
            or np.any(np.diff(bounds[:, 0]) <= 0)
            or bounds[0, 0] < 0
        ):
            raise ValueError("shot bounds must be non-negative, ordered, and strictly increasing")
        object.__setattr__(self, "bounds", bounds)

    def __len__(self):
        return len(self.bounds)

    def clamped(self, frame_count):
        """Trim shots that run past the frame map.

        TransNetV2 counts frames through cv2 while FrameMap counts PTS entries; the two
        can disagree by a frame or two at the tail of a file.
        """
        kept = self.bounds[self.bounds[:, 0] < frame_count].copy()
        if len(kept):
            np.minimum(kept[:, 1], frame_count - 1, out=kept[:, 1])
        return ShotIndex(self.fps, kept)

    def shot_at(self, frame_idx):
        """Index of the shot containing `frame_idx`, or None if it falls in a gap."""
        if not len(self.bounds) or frame_idx < 0:
            return None
        position = int(np.searchsorted(self.bounds[:, 0], frame_idx, side="right")) - 1
        if position < 0 or frame_idx > int(self.bounds[position, 1]):
            return None
        return position


def shots_path(row, data_dir=DATA):
    return Path(data_dir) / str(row["year"]) / "shots" / row["group"] / f"{row['video_id']}.json"


def scan_media(data_dir=DATA):
    """Every video on the data disk, keyed by video id.

    The search index only covers the profiles that were embedded, but shot boundaries
    exist for older datasets too. Clip export works off whatever is on disk, so it needs
    a catalogue that does not depend on `video_meta.parquet`.
    """
    catalogue = {}
    for path in sorted(Path(data_dir).glob("*/videos/*/*.mp4")):
        catalogue[path.stem] = {
            "video_id": path.stem,
            "group": path.parent.name,
            "year": path.parent.parent.parent.name,
            "path": str(path),
        }
    return catalogue


@lru_cache(maxsize=1)
def _media_catalogue(data_dir):
    return scan_media(data_dir)


def media_catalogue(data_dir=DATA):
    """Cached `scan_media`. Videos do not appear at runtime; their shot JSON does."""
    return _media_catalogue(str(data_dir))


def load_path(path):
    data = json.loads(Path(path).read_text())
    return ShotIndex(
        float(data.get("fps") or 0.0),
        np.asarray(data.get("shots") or [], dtype=np.int64).reshape(-1, 2),
    )


@lru_cache(maxsize=64)
def _load_cached(path):
    return load_path(path)


def load(row, data_dir=DATA):
    return _load_cached(str(shots_path(row, data_dir)))


def has_shots(row, data_dir=DATA):
    return shots_path(row, data_dir).exists()


def frame_duration(mapping, frame_idx):
    """Seconds one source frame occupies, from the neighbouring PTS delta."""
    pts = mapping.pts
    if frame_idx + 1 < len(pts):
        delta = int(pts[frame_idx + 1]) - int(pts[frame_idx])
    elif frame_idx > 0:
        delta = int(pts[frame_idx]) - int(pts[frame_idx - 1])
    else:
        return 0.0
    return delta * mapping.time_base_num / mapping.time_base_den


def shot_bounds_seconds(mapping, start_frame, end_frame):
    """Source-clock cut window for the inclusive frame range, matching `-copyts`.

    Both edges sit half a frame outside the shot rather than exactly on a frame's PTS:
    landing on a PTS makes inclusion depend on how `%.6f` happens to round, which costs
    or gains a frame at the boundary.
    """
    start = max(0.0, mapping.source_time(start_frame) - frame_duration(mapping, start_frame) / 2)
    end = mapping.source_time(end_frame) + frame_duration(mapping, end_frame) / 2
    return start, end


def clip_command(video_path, start_time, end_time, output):
    """Re-encode the exact window; `-copyts` keeps -ss/-to on the source clock."""
    return [
        "ffmpeg", "-y", "-loglevel", "error", "-copyts",
        "-ss", f"{start_time:.6f}", "-i", str(video_path), "-to", f"{end_time:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-avoid_negative_ts", "make_zero", "-movflags", "+faststart", str(output),
    ]


def export(video_path, start_time, end_time, output):
    with _clip_slots:
        subprocess.run(
            clip_command(video_path, start_time, end_time, output),
            check=True,
            timeout=CLIP_TIMEOUT_SECONDS,
        )
    return output
