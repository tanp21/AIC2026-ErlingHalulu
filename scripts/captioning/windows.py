"""Caption units: one per shot, or several fixed-overlap windows for long shots.

TransNetV2 only cuts where the picture changes, so a lecture filmed from one locked-off
camera comes back as a single 662-second "shot". Captioning that as one unit samples a
frame every 40 seconds and produces a description no query can match -- and four of the
ten total misses in `docs/reports/visual-search-recall-failure-analysis-2026.md` are
exactly that shape.

Shots at or under `WINDOW_SECONDS` stay one unit, so 98.5% of the 2026 corpus keeps the
existing one-row-per-shot contract. Longer shots are covered by equal-length windows
that overlap by `OVERLAP_SECONDS`, because a hard edge cuts actions in half.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


WINDOW_SECONDS = 20.0
OVERLAP_SECONDS = 2.0
# 8 frames, flat, regardless of window length -- this is the exact recipe of the
# "1024-default-8frames" run the user picked as the quality baseline (it is what
# correctly read "Mộc Châu" on K17_V003 shot 102). Windowing already keeps every unit
# at or under 20 s, so 8 frames there is at least as dense as 8 frames ever was on the
# shots that config was validated on (all well under 20 s).
FIXED_FRAME_COUNT = 8


@dataclass(frozen=True)
class CaptionUnit:
    """One inference request: a time window inside one shot of one video."""

    year: str
    group: str
    video_id: str
    shot_index: int
    start_frame: int
    end_frame: int
    fps: float
    video_path: str
    window_index: int
    window_count: int
    window_start_sec: float
    window_end_sec: float
    frame_count: int

    @property
    def duration_sec(self):
        return self.window_end_sec - self.window_start_sec

    @property
    def shot_start_sec(self):
        return self.start_frame / self.fps

    @property
    def shot_end_sec(self):
        return (self.end_frame + 1) / self.fps

    @property
    def shot_duration_sec(self):
        return (self.end_frame - self.start_frame + 1) / self.fps

    @property
    def unit_id(self):
        return f"{self.video_id}/{self.shot_index}/{self.window_index}"


def frames_for(duration_sec):
    """Frames to sample from a window of this length. Flat by design; see FIXED_FRAME_COUNT."""
    return FIXED_FRAME_COUNT


def window_count(duration_sec, window_seconds=WINDOW_SECONDS,
                 overlap_seconds=OVERLAP_SECONDS):
    if duration_sec <= window_seconds:
        return 1
    stride = window_seconds - overlap_seconds
    return math.ceil((duration_sec - window_seconds) / stride) + 1


def window_spans(duration_sec, window_seconds=WINDOW_SECONDS,
                 overlap_seconds=OVERLAP_SECONDS):
    """Offsets of equal-length windows covering [0, duration] exactly.

    The window length is shrunk to `(duration + (n-1) * overlap) / n` rather than pinned
    at 20 s, which keeps every window the same length, holds the overlap at exactly
    `overlap_seconds`, and avoids the degenerate case a fixed stride produces: a 20.5 s
    shot would otherwise get two windows overlapping by 19.5 s, i.e. two captions of the
    same thing. The result is always <= `window_seconds`.
    """
    count = window_count(duration_sec, window_seconds, overlap_seconds)
    if count == 1:
        return [(0.0, duration_sec)]
    length = (duration_sec + (count - 1) * overlap_seconds) / count
    stride = length - overlap_seconds
    spans = [(index * stride, index * stride + length) for index in range(count - 1)]
    # Pin the last window to the shot end instead of accumulating float drift.
    spans.append((duration_sec - length, duration_sec))
    return spans


def build_units(records, window_seconds=WINDOW_SECONDS, overlap_seconds=OVERLAP_SECONDS):
    """Expand `caption_shots.ShotRecord`s into `CaptionUnit`s."""
    units = []
    for record in records:
        duration = record.duration_sec
        spans = window_spans(duration, window_seconds, overlap_seconds)
        # Half a frame outside the shot at both ends, for the same reason ShotRecord
        # pads: landing exactly on a frame's timestamp makes inclusion depend on how
        # the %.6f in the ffmpeg command happens to round.
        padding = 0.25 / record.fps
        shot_start = record.start_frame / record.fps
        for index, (start, end) in enumerate(spans):
            first, last = index == 0, index == len(spans) - 1
            units.append(CaptionUnit(
                year=record.year,
                group=record.group,
                video_id=record.video_id,
                shot_index=record.shot_index,
                start_frame=record.start_frame,
                end_frame=record.end_frame,
                fps=record.fps,
                video_path=record.video_path,
                window_index=index,
                window_count=len(spans),
                window_start_sec=max(0.0, shot_start + start - (padding if first else 0.0)),
                window_end_sec=shot_start + end + (padding if last else 0.0),
                frame_count=frames_for(end - start),
            ))
    return units


def summarize(units):
    """Counts for the dry-run gate: shots in, units out, how many shots were split."""
    shots = {(unit.video_id, unit.shot_index) for unit in units}
    split = {(unit.video_id, unit.shot_index) for unit in units if unit.window_count > 1}
    return {
        "shots": len(shots),
        "units": len(units),
        "split_shots": len(split),
        "max_windows": max((unit.window_count for unit in units), default=0),
        "total_frames": sum(unit.frame_count for unit in units),
    }
