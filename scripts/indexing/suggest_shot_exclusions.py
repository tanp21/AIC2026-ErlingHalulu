#!/usr/bin/env python3
"""Suggest non-story shots from existing image embeddings; never auto-exclude them.

The first 2026 run has a few audited L21 seeds. Later runs learn from reviewed
``presenter`` and ``intro`` labels, so the same command can be reused for K groups.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

from backend import shot_labels


ROOT = Path(__file__).resolve().parents[2]
PRESENTER_SEEDS = (
    ("L21_V001", 1), ("L21_V001", 12), ("L21_V001", 44), ("L21_V001", 54),
    ("L21_V003", 0), ("L21_V003", 13), ("L21_V003", 24), ("L21_V003", 59),
)
PRESENTER_VARIANT_SEEDS = (("L21_V002", 86), ("L21_V002", 87))
INTRO_SEEDS = (("L21_V001", 0), ("L21_V001", 11), ("L21_V003", 12))
OUTRO_SEEDS = (("L21_V006", 265), ("L22_V001", 362))
OPENING_FIXED_LAST_SHOT = {"L25": 0, "L26": 5, "L27": 15, "L29": 23}
OPENING_ANCHOR_RULES = {
    "L28": {
        "seeds": (("L28_V001", 4), ("L28_V002", 3),
                  ("L28_V003", 4), ("L28_V024", 4)),
        "max_sec": 60,
        "threshold": 0.95,
        "include_anchor": False,
    },
    "L30": {
        "seeds": (("L30_V001", 2), ("L30_V050", 1)),
        "max_sec": 20,
        "threshold": 0.90,
    },
}
PRESENTER_BOUNDARY_THRESHOLD = 0.78
PRESENTER_LAYOUT_THRESHOLD = 0.60
PRESENTER_WEAK_THRESHOLD = 0.77
PRESENTER_WEAK_LAYOUT_THRESHOLD = 0.65
GLOBAL_INTRO_THRESHOLD = 0.90
OUTRO_THRESHOLD = 0.95
OUTRO_MAX_SEC = 8
ADJACENT_BUMPER_THRESHOLD = 0.80
PREVIEW_OVERLAY_GROUPS = {"L21", "L22"}
PREVIEW_OVERLAY_MIN_SHOTS = 4
PREVIEW_OVERLAY_MAX_MEDIAN_SEC = 3.0
PREVIEW_PHRASES = tuple(
    tuple(phrase.split())
    for phrase in (
        "phần sau",
        "ngay sau đây trong chương trình",
        "tiếp theo trong chương trình",
        "tiếp theo trong sáu mươi giây",
        "trong phần tiếp theo",
    )
)


def shot_at(bounds: np.ndarray, frame_idx: int) -> int | None:
    position = int(np.searchsorted(bounds[:, 0], frame_idx, side="right")) - 1
    return position if position >= 0 and frame_idx <= int(bounds[position, 1]) else None


def load_shot_vectors(source: Path, shots_dir: Path, groups: set[str]):
    boundaries = {}
    fps = {}
    for group in sorted(groups):
        for path in sorted((shots_dir / group).glob("*.json")):
            payload = json.loads(path.read_text())
            boundaries[path.stem] = np.asarray(payload["shots"], dtype=np.int64)
            fps[path.stem] = float(payload.get("fps") or 30.0)
    if not boundaries:
        raise SystemExit(f"no shot JSON found for {', '.join(sorted(groups))}")

    rows = pq.read_table(source / "meta.parquet").to_pylist()
    sums, counts, frames = {}, defaultdict(int), defaultdict(list)
    shard_idx = None
    shard = None
    for row in rows:
        video_id = row["video_id"]
        bounds = boundaries.get(video_id)
        if bounds is None:
            continue
        position = shot_at(bounds, int(row["frame_idx"]))
        if position is None:
            continue
        if row["shard_idx"] != shard_idx:
            shard_idx = row["shard_idx"]
            shard = np.load(source / f"emb_{shard_idx:06d}.npy", mmap_mode="r")
        key = (video_id, position)
        vector = np.asarray(shard[row["row_idx"]], dtype=np.float32)
        sums[key] = sums.get(key, np.zeros_like(vector)) + vector
        counts[key] += 1
        frames[key].append(int(row["frame_idx"]))
    vectors = {}
    for key, total in sums.items():
        vector = total / counts[key]
        norm = np.linalg.norm(vector)
        if norm:
            vectors[key] = vector / norm
    representative_frames = {
        key: sorted(values)[len(values) // 2] for key, values in frames.items()
    }
    return vectors, boundaries, fps, representative_frames


def reviewed_seeds(vectors, boundaries, label_dir: Path, reason: str):
    seeds = []
    for video_id, bounds in boundaries.items():
        document = shot_labels.load(video_id, bounds, label_dir)
        for raw_index, label in document["labels"].items():
            key = (video_id, int(raw_index))
            if label["state"] == "excluded" and label["reason"] == reason and key in vectors:
                seeds.append(vectors[key])
    return seeds


def centroid(vectors, keys, extra=()):
    seeds = [vectors[key] for key in keys if key in vectors] + list(extra)
    if not seeds:
        return None
    result = np.mean(seeds, axis=0)
    norm = np.linalg.norm(result)
    return result / norm if norm else None


def _normalise_token(token) -> str:
    return "".join(character for character in str(token).lower() if character.isalnum())


def load_preview_anchors(path: Path, video_ids: set[str]) -> dict[str, list[float]]:
    """Transcript times that explicitly announce an upcoming preview block."""
    if not path.exists():
        return {}
    required = {"audio_key", "offset", "token"}
    missing = required - set(pq.read_schema(path).names)
    if missing:
        raise ValueError(f"transcript is missing columns: {', '.join(sorted(missing))}")
    grouped = defaultdict(list)
    for row in pq.read_table(path, columns=sorted(required)).to_pylist():
        if row["audio_key"] not in video_ids or not isinstance(row["token"], str):
            continue
        try:
            offset = float(row["offset"])
        except (TypeError, ValueError):
            continue
        token = _normalise_token(row["token"])
        if token and math.isfinite(offset) and offset >= 0:
            grouped[row["audio_key"]].append((offset, token))

    anchors = {}
    for video_id, rows in grouped.items():
        words = [(token, offset) for offset, token in sorted(rows)]
        found = set()
        for phrase in PREVIEW_PHRASES:
            width = len(phrase)
            for index in range(len(words) - width + 1):
                if tuple(token for token, _ in words[index:index + width]) == phrase:
                    found.add(words[index][1])
        if found:
            anchors[video_id] = sorted(found)
    return anchors


def load_layout_vectors(keyframes_dir: Path, representative_frames, keys) -> dict:
    """Tiny colour-layout descriptors for separating studio shots from bumpers."""
    layouts = {}
    for video_id, shot_index in keys:
        frame_idx = representative_frames.get((video_id, shot_index))
        if frame_idx is None:
            continue
        path = keyframes_dir / video_id[:3] / video_id / f"{frame_idx:06d}.png"
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = cv2.resize(image, (24, 14), interpolation=cv2.INTER_AREA)
        vector = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32).reshape(-1)
        vector = (vector - vector.mean()) / (vector.std() + 1e-6)
        norm = np.linalg.norm(vector)
        if norm:
            layouts[(video_id, shot_index)] = vector / norm
    return layouts


def load_preview_overlay_matches(keyframes_dir: Path, representative_frames) -> set:
    """Shots carrying the fixed red ``TIẾP THEO`` graphics used by 60 Giây."""
    matches = set()
    for key, frame_idx in representative_frames.items():
        video_id, _ = key
        if video_id[:3] not in PREVIEW_OVERLAY_GROUPS:
            continue
        path = keyframes_dir / video_id[:3] / video_id / f"{frame_idx:06d}.png"
        image = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)
        if image is None:
            continue
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        red = (
            ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
            & (hsv[:, :, 1] > 110)
            & (hsv[:, :, 2] > 60)
        )
        top_left = red[
            int(0.06 * height):int(0.24 * height), :int(0.22 * width)
        ]
        lower_third = red[int(0.72 * height):int(0.92 * height)]
        if float(top_left.mean()) >= 0.45 and float(lower_third.mean()) >= 0.50:
            matches.add(key)
    return matches


def preview_overlay_runs(matches, bounds, fps):
    """Join one short missed shot, then retain only fast four-shot montages."""
    indices = sorted(matches)
    candidates = []
    run = []
    gaps = 0
    for index in indices:
        distance = index - run[-1] if run else None
        gap_is_short = (
            distance == 2
            and gaps == 0
            and (bounds[index - 1, 1] - bounds[index - 1, 0] + 1) / fps <= 3
        )
        if not run or distance == 1:
            run.append(index)
        elif gap_is_short:
            run.extend((index - 1, index))
            gaps += 1
        else:
            candidates.append(run)
            run, gaps = [index], 0
    if run:
        candidates.append(run)

    matched = set(indices)
    return [
        run for run in candidates
        if len(matched.intersection(run)) >= PREVIEW_OVERLAY_MIN_SHOTS
        and float(np.median(
            (bounds[run, 1] - bounds[run, 0] + 1) / fps
        )) <= PREVIEW_OVERLAY_MAX_MEDIAN_SEC
    ]


def suggest_opening_intro(video_id, bounds, fps, vectors, last_shot=None,
                          prototypes=None, max_sec=0, threshold=0,
                          include_anchor=True):
    """Suggest a reviewed program-opening prefix; fail closed without an anchor."""
    confidence = 0.95
    if last_shot is not None:
        if last_shot >= len(bounds):
            return {}
    else:
        candidates = [
            (float(np.max(prototypes @ vector)), index)
            for (candidate, index), vector in vectors.items()
            if candidate == video_id and bounds[index, 0] / fps <= max_sec
        ]
        if not candidates:
            return {}
        confidence, last_shot = max(candidates)
        if confidence < threshold:
            return {}
    return {
        index: {"reason": "intro", "confidence": min(1.0, confidence)}
        for index in range(last_shot + int(include_anchor))
    }


def suggest_video(video_id, bounds, fps, vectors, presenter, intro,
                  presenter_threshold=0.84, intro_threshold=0.78, preview_max_sec=50,
                  preview_anchors=(), layouts=None, preview_overlay_matches=(), outro=None):
    layouts = layouts or {}
    presenter = np.asarray(presenter)
    scores = {
        index: float(np.max(presenter @ vector)) if presenter.ndim == 2
        else float(vector @ presenter)
        for (candidate, index), vector in vectors.items()
        if candidate == video_id
    }
    intro_scores = {} if intro is None else {
        index: float(vectors[(video_id, index)] @ intro)
        for index in range(len(bounds)) if (video_id, index) in vectors
    }
    high_presenters = {
        index for index, score in scores.items()
        if score >= presenter_threshold
        and (intro is None or score >= intro_scores.get(index, -1))
    }
    layout_prototypes = [
        layouts[(video_id, index)] for index in high_presenters
        if (video_id, index) in layouts
    ]
    presenter_matches = set(high_presenters)
    if layout_prototypes:
        prototype_matrix = np.stack(layout_prototypes)
        for index, score in scores.items():
            layout = layouts.get((video_id, index))
            layout_threshold = (
                PRESENTER_LAYOUT_THRESHOLD
                if score >= PRESENTER_BOUNDARY_THRESHOLD
                else PRESENTER_WEAK_LAYOUT_THRESHOLD
            )
            if (
                index not in presenter_matches
                and score >= min(presenter_threshold, PRESENTER_WEAK_THRESHOLD)
                and layout is not None
                and float(np.max(prototype_matrix @ layout)) >= layout_threshold
            ):
                presenter_matches.add(index)

    suggestions = {
        index: {"reason": "presenter", "confidence": min(1.0, score)}
        for index, score in scores.items() if index in presenter_matches
    }
    if intro is not None:
        bumpers = [
            index for index, score in intro_scores.items()
            if 8 <= bounds[index][0] / fps <= preview_max_sec and score >= intro_threshold
        ]
        if bumpers:
            bumper = max(bumpers)
            first_story = bumper + 1
            while (
                first_story < len(bounds)
                and first_story in presenter_matches
            ):
                suggestions.setdefault(first_story, {
                    "reason": "presenter", "confidence": scores[first_story],
                })
                first_story += 1
            for index in range(first_story):
                if index in suggestions:
                    continue
                reason = "intro" if intro_scores.get(index, -1) >= intro_threshold else "preview"
                suggestions[index] = {"reason": reason, "confidence": 0.9}

    for anchor_sec in preview_anchors:
        anchor_shot = shot_at(bounds, int(anchor_sec * fps))
        if anchor_shot is None:
            continue
        presenters = [
            index for index in range(anchor_shot + 1, len(bounds))
            if bounds[index][0] / fps <= anchor_sec + 25
            and index in presenter_matches
        ]
        if not presenters:
            continue
        first, last = anchor_shot + 1, presenters[0] - 1
        if last - first + 1 < 4 or (bounds[last][1] + 1) / fps - anchor_sec > 25:
            continue
        durations = (bounds[first:last + 1, 1] - bounds[first:last + 1, 0] + 1) / fps
        if float(np.median(durations)) > 3:
            continue
        presenter_index = presenters[0]
        suggestions.setdefault(presenter_index, {
            "reason": "presenter", "confidence": scores[presenter_index],
        })
        for index in range(first, last + 1):
            suggestions[index] = {"reason": "preview", "confidence": 0.9}
    for bumper, intro_score in intro_scores.items():
        if intro_score < GLOBAL_INTRO_THRESHOLD:
            continue
        presenters = [
            index for index in range(bumper + 1, len(bounds))
            if bounds[index][0] / fps <= bounds[bumper][0] / fps + 25
            and index in presenter_matches
        ]
        if not presenters:
            continue
        first, last = bumper + 1, presenters[0] - 1
        if last - first + 1 < 4:
            continue
        durations = (bounds[first:last + 1, 1] - bounds[first:last + 1, 0] + 1) / fps
        if float(np.median(durations)) > 3:
            continue
        for index in range(first, last + 1):
            suggestions[index] = {"reason": "preview", "confidence": 0.9}
    overlay_matches = {
        index for candidate, index in preview_overlay_matches
        if candidate == video_id
    }
    for run in preview_overlay_runs(overlay_matches, bounds, fps):
        for index in run:
            suggestions[index] = {"reason": "preview", "confidence": 0.9}
        for index in (run[0] - 1, run[-1] + 1):
            score = intro_scores.get(index, -1)
            if index not in presenter_matches and score >= ADJACENT_BUMPER_THRESHOLD:
                suggestions[index] = {
                    "reason": "intro", "confidence": min(1.0, score),
                }
    for index, score in intro_scores.items():
        if score >= GLOBAL_INTRO_THRESHOLD:
            suggestions[index] = {
                "reason": "intro", "confidence": min(1.0, score),
            }
    if outro is not None and video_id[:3] in PREVIEW_OVERLAY_GROUPS:
        index = len(bounds) - 1
        vector = vectors.get((video_id, index))
        duration = (bounds[index, 1] - bounds[index, 0] + 1) / fps
        if vector is not None and duration <= OUTRO_MAX_SEC:
            score = float(vector @ outro)
            if score >= OUTRO_THRESHOLD:
                suggestions[index] = {
                    "reason": "outro", "confidence": min(1.0, score),
                }
    return suggestions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--groups", default="L21,L22")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--transcript-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--label-dir", type=Path, default=shot_labels.LABEL_DIR)
    parser.add_argument("--presenter-threshold", type=float, default=0.84)
    parser.add_argument("--intro-threshold", type=float, default=0.78)
    parser.add_argument("--preview-max-sec", type=float, default=50)
    args = parser.parse_args(argv)
    groups = {item.strip() for item in args.groups.split(",") if item.strip()}
    source = args.source or args.data_dir / args.year / "embeddings"
    vectors, boundaries, fps, representative_frames = load_shot_vectors(
        source, args.data_dir / args.year / "shots", groups
    )
    opening_groups = set(OPENING_FIXED_LAST_SHOT) | set(OPENING_ANCHOR_RULES)
    story_boundaries = {
        video_id: bounds for video_id, bounds in boundaries.items()
        if video_id[:3] not in opening_groups
    }
    opening_prototypes = {}
    for group, rule in OPENING_ANCHOR_RULES.items():
        if group not in groups:
            continue
        seeds = [vectors[key] for key in rule["seeds"] if key in vectors]
        if len(seeds) != len(rule["seeds"]):
            raise SystemExit(f"opening intro seeds are unavailable for {group}")
        opening_prototypes[group] = np.stack(seeds)

    preview_anchors, layouts, preview_overlay_matches = {}, {}, set()
    presenter = intro = outro = None
    if story_boundaries:
        transcript_path = (
            args.transcript_path
            or args.data_dir / args.year / "audio" / "step_data.parquet"
        )
        preview_anchors = load_preview_anchors(transcript_path, set(story_boundaries))
        if not transcript_path.exists():
            print(f"[suggest] transcript not found, skipping mid-program previews: {transcript_path}")
        presenter_seeds = reviewed_seeds(
            vectors, story_boundaries, args.label_dir, "presenter"
        )
        presenter_prototypes = [
            centroid(vectors, PRESENTER_SEEDS),
            centroid(vectors, PRESENTER_VARIANT_SEEDS),
            *presenter_seeds,
        ]
        presenter_prototypes = [
            vector for vector in presenter_prototypes if vector is not None
        ]
        if not presenter_prototypes:
            raise SystemExit("mark at least one presenter shot before generating suggestions")
        presenter = np.stack(presenter_prototypes)
        layout_keys = {
            key for key, vector in vectors.items()
            if key[0] in story_boundaries
            and float(np.max(presenter @ vector)) >= min(
                args.presenter_threshold, PRESENTER_WEAK_THRESHOLD
            )
        }
        layouts = load_layout_vectors(
            args.data_dir / args.year / "keyframes", representative_frames, layout_keys
        )
        preview_overlay_matches = load_preview_overlay_matches(
            args.data_dir / args.year / "keyframes", representative_frames
        )
        intro = centroid(
            vectors, INTRO_SEEDS,
            reviewed_seeds(vectors, story_boundaries, args.label_dir, "intro"),
        )
        outro = centroid(
            vectors, OUTRO_SEEDS,
            reviewed_seeds(vectors, story_boundaries, args.label_dir, "outro"),
        )

    total = 0
    for video_id, bounds in sorted(boundaries.items()):
        group = video_id[:3]
        if group in OPENING_FIXED_LAST_SHOT:
            suggestions = suggest_opening_intro(
                video_id, bounds, fps[video_id], vectors,
                last_shot=OPENING_FIXED_LAST_SHOT[group],
            )
        elif group in OPENING_ANCHOR_RULES:
            rule = OPENING_ANCHOR_RULES[group]
            suggestions = suggest_opening_intro(
                video_id, bounds, fps[video_id], vectors,
                prototypes=opening_prototypes[group],
                max_sec=rule["max_sec"], threshold=rule["threshold"],
                include_anchor=rule.get("include_anchor", True),
            )
        else:
            suggestions = suggest_video(
                video_id, bounds, fps[video_id], vectors, presenter, intro,
                args.presenter_threshold, args.intro_threshold,
                args.preview_max_sec, preview_anchors.get(video_id, ()), layouts,
                preview_overlay_matches, outro,
            )
        shot_labels.replace_suggestions(video_id, bounds, suggestions, args.label_dir)
        total += len(suggestions)
        print(f"[suggest] {video_id}: {len(suggestions)}", flush=True)
    print(f"[suggest] {total} unreviewed candidates across {len(boundaries)} videos")


if __name__ == "__main__":
    main()
