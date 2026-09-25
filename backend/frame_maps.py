"""Exact source-frame lookup using presentation timestamps (PTS)."""
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent


def map_directory(index_dir):
    return Path(index_dir) / "frame_maps"


MAP_DIR = map_directory(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes"))


@dataclass(frozen=True)
class FrameMap:
    pts: np.ndarray
    time_base_num: int
    time_base_den: int

    def __post_init__(self):
        pts = np.asarray(self.pts, dtype=np.int64)
        if not len(pts) or np.any(np.diff(pts) <= 0):
            raise ValueError("frame PTS must be non-empty and strictly increasing")
        if self.time_base_num <= 0 or self.time_base_den <= 0:
            raise ValueError("time base must be positive")
        object.__setattr__(self, "pts", pts)

    def frame_time(self, frame_id):
        return (int(self.pts[frame_id]) - int(self.pts[0])) * self.time_base_num / self.time_base_den

    def times(self):
        """Every frame's playback time in seconds, the array a time->frame search sorts.

        One allocation per map, so a caller resolving many times against the same video
        should hold on to it (`scripts.indexing.build_flat_profile.KeyframeResolver` does).
        """
        return (self.pts - self.pts[0]) * (self.time_base_num / self.time_base_den)

    def source_time(self, frame_id):
        return int(self.pts[frame_id]) * self.time_base_num / self.time_base_den

    def frame_at_media_time(self, media_time):
        target = int(self.pts[0]) + round(float(media_time) * self.time_base_den / self.time_base_num)
        pos = int(np.searchsorted(self.pts, target))
        return pos if pos < len(self.pts) and int(self.pts[pos]) == target else None

    def nearest_frame(self, media_time, frame_ids):
        ids = np.asarray(frame_ids, dtype=np.int64)
        if not len(ids):
            raise KeyError("no indexed frames")
        times = (self.pts[ids] - self.pts[0]) * (self.time_base_num / self.time_base_den)
        pos = int(np.searchsorted(times, float(media_time)))
        candidates = ids[max(0, pos - 1):min(len(ids), pos + 1)]
        return int(min(candidates, key=lambda frame: abs(self.frame_time(frame) - media_time)))

    def payload(self, video_id):
        return {
            "videoId": video_id,
            "timeBaseNumerator": self.time_base_num,
            "timeBaseDenominator": self.time_base_den,
            "firstPts": int(self.pts[0]),
            "ptsDeltas": np.diff(self.pts).astype(int).tolist(),
            "frameCount": len(self.pts),
        }


def load_path(path):
    with np.load(path) as data:
        return FrameMap(
            data["pts"],
            int(data["time_base_num"]),
            int(data["time_base_den"]),
        )


@lru_cache(maxsize=None)
# Unbounded: the corpus is currently ~1500 videos at ~30KB/map (well under 100MB
# total), and a single search pools up to SEARCH_RESULT_LIMIT (10,000) results
# spanning potentially hundreds of distinct videos -- a small bound here (previously
# 32) meant that loop thrashed the cache on nearly every call, measured at ~1.6ms
# per load under thrashing vs. a cache hit, i.e. up to ~16s added to a single
# search request. There is no eviction concern worth trading that away for.
def load(video_id):
    return load_path(MAP_DIR / f"{video_id}.npz")
