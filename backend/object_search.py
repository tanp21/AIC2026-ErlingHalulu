"""Polars-backed object search over Parquet-aligned FAISS keyframes."""
from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import heapq
import json
import os
from pathlib import Path

import polars as pl
from pydantic import BaseModel, Field, field_validator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get(
    "AIC_OBJECT_DATA_DIR",
    Path(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes")) / "objects" / "current",
))
SEARCH_CANDIDATE_LIMIT = 100_000
EMPTY_DETECTIONS = pl.DataFrame(schema={
    "frame_id": pl.UInt32, "profile_mask": pl.UInt64, "score": pl.Float64,
    "x0": pl.Float64, "y0": pl.Float64, "x1": pl.Float64, "y1": pl.Float64,
    "cx": pl.Float64, "cy": pl.Float64, "det_idx": pl.UInt16,
})


class ObjectClause(BaseModel):
    label: str = Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z ]*$")
    region: tuple[float, float, float, float]
    min_count: int = Field(default=1, ge=1, le=10)

    @field_validator("region")
    @classmethod
    def valid_region(cls, region):
        x0, y0, x1, y1 = region
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("object region must be inside the image and have positive area")
        return region


class ObjectQuery(BaseModel):
    clauses: list[ObjectClause] = Field(min_length=1, max_length=6)
    min_confidence: float = Field(default=0.35, ge=0.25, le=0.95)


class ObjectStore:
    def __init__(self, directory: Path):
        self.directory = directory
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"object data is unavailable: {directory}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1:
            raise RuntimeError(f"unsupported object data schema: {manifest.get('schema_version')}")
        self.labels = manifest["labels"]
        self.profiles = manifest["profiles"]
        self.alignment_max_seconds = manifest["alignment_max_seconds"]
        frames = pl.read_parquet(directory / "keyframes.parquet")
        self.by_id = {}
        self.by_key = {}
        for row in frames.iter_rows(named=True):
            self.by_id[row["id"]] = row
            self.by_key[(row["video_id"], row["frame_idx"])] = row
        self.detections = {}
        for index, label in enumerate(self.labels):
            path = directory / "labels" / f"{index:03d}.parquet"
            self.detections[label] = pl.read_parquet(path) if path.is_file() else EMPTY_DETECTIONS

    def profile_mask(self, profile):
        if profile not in self.profiles:
            raise ValueError(f"unknown object profile: {profile}")
        return self.profiles[profile]["mask"]


@lru_cache(maxsize=1)
def _load_store(directory: str):
    return ObjectStore(Path(directory))


def _store():
    return _load_store(str(DATA_DIR.resolve()))


def warmup():
    if DATA_DIR.exists():
        _store()


def catalog():
    store = _store()
    return {"labels": store.labels,
            "profiles": [{"profile": name, "keyframes": item["keyframes"]}
                         for name, item in sorted(store.profiles.items())],
            "alignment_max_seconds": store.alignment_max_seconds}


def video_detections(video_id: str):
    store = _store()
    directory = store.directory / "playback" / f"video_id={video_id}"
    if not directory.is_dir():
        return {"video_id": video_id, "frames": []}
    rows = pl.scan_parquet(str(directory / "*.parquet")).sort(["time", "frame_idx"]).collect().to_dicts()
    frames = []
    for row in rows:
        detections = []
        for label, score, box in zip(row["labels"], row["scores"], row["boxes"]):
            y0, x0, y1, x1 = box
            detections.append({"label": label, "confidence": round(score, 6),
                               "box": [x0, y0, x1, y1]})
        detections.sort(key=lambda detection: -detection["confidence"])
        frames.append({"frame_idx": row["frame_idx"], "time": row["time"],
                       "detections": detections})
    return {"video_id": video_id, "frames": frames}


def _check_labels(store, query):
    unknown = {clause.label for clause in query.clauses} - set(store.labels)
    if unknown:
        raise ValueError(f"unknown object label: {', '.join(sorted(unknown))}")


def _candidates(store, profile_mask, clause, min_confidence):
    x0, y0, x1, y1 = clause.region
    return store.detections[clause.label].filter(
        ((pl.col("profile_mask") & pl.lit(profile_mask)) != 0)
        & (pl.col("score") >= min_confidence)
        & (pl.col("cx") >= x0) & (pl.col("cx") <= x1)
        & (pl.col("cy") >= y0) & (pl.col("cy") <= y1)
    )


def _top_distinct(rows, n):
    if rows.is_empty():
        return rows
    budget = min(rows.height, max(64, n * 2))
    while True:
        selected = (rows.top_k(budget, by=["score", "frame_id", "det_idx"],
                               reverse=[False, True, True])
                    .sort(["score", "frame_id", "det_idx"], descending=[True, False, False])
                    .unique(subset="frame_id", keep="first", maintain_order=True))
        if selected.height >= n or budget == rows.height:
            return selected.head(n)
        budget = min(rows.height, budget * 2)


def _matched_detection(label, row):
    return {"label": label, "confidence": round(row["score"], 6),
            "box": [row[key] for key in ("x0", "y0", "x1", "y1")]}


def _hit(store, frame_id, score, mean_score, matched_objects):
    frame = store.by_id[frame_id]
    return {**frame, "score": round(score, 6), "mean_score": round(mean_score, 6),
            "delta_ms": round(frame["delta"] * 1000, 1), "matched_objects": matched_objects}


def _match(detections, query):
    """Assign distinct detections to all requested object slots on one keyframe."""
    slots = [clause for clause in query.clauses for _ in range(clause.min_count)]
    choices = []
    for clause in slots:
        x0, y0, x1, y1 = clause.region
        choices.append(sorted((index for index, row in enumerate(detections)
                               if row["label"] == clause.label
                               and row["score"] >= query.min_confidence
                               and x0 <= row["cx"] <= x1 and y0 <= row["cy"] <= y1),
                              key=lambda index: (-detections[index]["score"], detections[index]["det_idx"])))
    if any(not options for options in choices):
        return None
    owner = {}
    slot_order = sorted(range(len(slots)), key=lambda index: len(choices[index]))

    def assign(slot, visited):
        for index in choices[slot]:
            if index in visited:
                continue
            visited.add(index)
            if index not in owner or assign(owner[index], visited):
                owner[index] = slot
                return True
        return False

    if not all(assign(slot, set()) for slot in slot_order):
        return None
    matched = [detections[index] for index in owner]
    score = min(row["score"] for row in matched)
    mean = sum(row["score"] for row in matched) / len(matched)
    return {"score": round(score, 6), "mean_score": round(mean, 6),
            "matched_objects": [_matched_detection(row["label"], row) for row in matched]}


def _search_single(store, profile_mask, query, k):
    clause = query.clauses[0]
    rows = _candidates(store, profile_mask, clause, query.min_confidence)
    if clause.min_count == 1:
        selected = _top_distinct(rows, k + 1).to_dicts()
        hits = [_hit(store, row["frame_id"], row["score"], row["score"],
                     [_matched_detection(clause.label, row)]) for row in selected[:k]]
        return {"hits": hits, "truncated": len(selected) > k}
    if rows.is_empty():
        return {"hits": [], "truncated": False}
    count = clause.min_count
    grouped = (rows.group_by("frame_id").agg(
        pl.struct(["score", "x0", "y0", "x1", "y1", "det_idx"])
          .sort_by(["score", "det_idx"], descending=[True, False]).head(count).alias("objects"),
        pl.col("score").sort(descending=True).head(count).alias("scores"),
        pl.len().alias("count"),
    ).filter(pl.col("count") >= count).with_columns(
        pl.col("scores").list.get(count - 1).alias("rank_score"),
        pl.col("scores").list.mean().alias("mean_score"),
    ).top_k(k + 1, by=["rank_score", "frame_id"], reverse=[False, True])
     .sort(["rank_score", "frame_id"], descending=[True, False]))
    selected = grouped.to_dicts()
    hits = [_hit(store, row["frame_id"], row["rank_score"], row["mean_score"],
                 [_matched_detection(clause.label, obj) for obj in row["objects"]])
            for row in selected[:k]]
    return {"hits": hits, "truncated": len(selected) > k}


def _rows_for_frames(store, labels, frame_ids):
    by_frame = defaultdict(list)
    if not frame_ids:
        return by_frame
    for label in labels:
        rows = store.detections[label].filter(pl.col("frame_id").is_in(frame_ids))
        for row in rows.iter_rows(named=True):
            by_frame[row["frame_id"]].append({**row, "label": label})
    return by_frame


def search(profile: str, query: ObjectQuery, k: int):
    if not 1 <= k <= 10_000:
        raise ValueError("object search limit must be 1 to 10000")
    store = _store()
    mask = store.profile_mask(profile)
    _check_labels(store, query)
    if len(query.clauses) == 1:
        return _search_single(store, mask, query, k)
    # Anchor with the rarest label/region, then check all clauses on those frames.
    anchor = min(query.clauses, key=lambda clause: (
        store.detections[clause.label].height
        * (clause.region[2] - clause.region[0]) * (clause.region[3] - clause.region[1]),
        clause.label,
    ))
    anchor_rows = _candidates(store, mask, anchor, query.min_confidence)
    if anchor_rows.is_empty():
        return {"hits": [], "truncated": False}
    budget = min(512, SEARCH_CANDIDATE_LIMIT)
    processed = 0
    truncated = False
    top = []
    while True:
        anchors = _top_distinct(anchor_rows, min(budget + 1, SEARCH_CANDIDATE_LIMIT + 1)).to_dicts()
        batch = anchors[processed:budget]
        ids = [row["frame_id"] for row in batch]
        by_frame = _rows_for_frames(store, {clause.label for clause in query.clauses}, ids)
        for frame_id in ids:
            match = _match(by_frame.get(frame_id, []), query)
            if match is None:
                continue
            item = _hit(store, frame_id, match["score"], match["mean_score"], match["matched_objects"])
            order = (item["score"], item["mean_score"], -frame_id)
            if len(top) < k:
                heapq.heappush(top, (order, item))
            elif order > top[0][0]:
                heapq.heapreplace(top, (order, item))
                truncated = True
            else:
                truncated = True
        processed = len(ids) + processed
        if processed < len(anchors) and len(top) >= k and anchors[processed]["score"] < top[0][0][0]:
            truncated = True
            break
        if processed >= SEARCH_CANDIDATE_LIMIT or len(anchors) <= processed:
            truncated |= len(anchors) > processed
            break
        budget = min(budget * 2, SEARCH_CANDIDATE_LIMIT)
    return {"hits": [item for _, item in sorted(top, key=lambda pair: pair[0], reverse=True)],
            "truncated": truncated}


def match_frames(profile: str, query: ObjectQuery, points: list[dict]):
    if len(points) > 500:
        raise ValueError("object frame batch exceeds 500")
    store = _store()
    mask = store.profile_mask(profile)
    _check_labels(store, query)
    frame_ids = []
    for point in points:
        frame = store.by_key.get((point["video_id"], point["frame_idx"]))
        frame_ids.append(frame["id"] if frame and frame["profile_mask"] & mask else None)
    by_frame = _rows_for_frames(store, {clause.label for clause in query.clauses},
                                [frame_id for frame_id in frame_ids if frame_id is not None])
    result = []
    for frame_id in frame_ids:
        match = _match(by_frame.get(frame_id, []), query) if frame_id is not None else None
        result.append(None if match is None else {
            **match, "delta_ms": round(store.by_id[frame_id]["delta"] * 1000, 1),
        })
    return result
