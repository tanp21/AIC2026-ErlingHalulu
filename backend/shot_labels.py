"""Reviewed shot exclusions shared by the reviewer and offline index builders."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading


ROOT = Path(__file__).resolve().parent.parent
LABEL_DIR = Path(os.environ.get("AIC_SHOT_LABEL_DIR", ROOT / "artifacts" / "shot-labels"))
VIDEO_ID = re.compile(r"^[A-Z]\d{2}[_-]V\d{3}$")
STATES = {"suggested", "excluded", "included"}
REASONS = {"intro", "preview", "presenter", "outro", "other"}
_write_lock = threading.Lock()


class LabelConflict(ValueError):
    pass


def bounds_digest(bounds) -> str:
    canonical = json.dumps(
        [[int(start), int(end)] for start, end in bounds], separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def label_path(video_id: str, root=LABEL_DIR) -> Path:
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError("invalid video id")
    return Path(root) / video_id[:3] / f"{video_id}.json"


def empty_document(video_id: str, bounds) -> dict:
    return {
        "version": 1,
        "video_id": video_id,
        "shots_digest": bounds_digest(bounds),
        "revision": 0,
        "labels": {},
    }


def load(video_id: str, bounds, root=LABEL_DIR) -> dict:
    path = label_path(video_id, root)
    if not path.exists():
        return empty_document(video_id, bounds)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1 or document.get("video_id") != video_id:
        raise ValueError(f"malformed shot labels: {path}")
    if document.get("shots_digest") != bounds_digest(bounds):
        raise LabelConflict(f"shot boundaries changed since labels were created: {video_id}")
    labels = document.get("labels")
    if not isinstance(document.get("revision"), int) or document["revision"] < 0:
        raise ValueError(f"malformed shot-label revision: {path}")
    if not isinstance(labels, dict):
        raise ValueError(f"malformed shot labels: {path}")
    for raw_index, label in labels.items():
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid shot label index in {path}") from exc
        if raw_index != str(index) or index < 0 or index >= len(bounds) or not isinstance(label, dict):
            raise ValueError(f"invalid shot label index in {path}: {raw_index}")
        if label.get("state") not in STATES or label.get("reason") not in REASONS:
            raise ValueError(f"invalid shot label in {path}: {raw_index}")
        confidence = label.get("confidence")
        if confidence is not None and (
            not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1
        ):
            raise ValueError(f"invalid shot confidence in {path}: {raw_index}")
    return document


def _write(document: dict, root=LABEL_DIR) -> None:
    path = label_path(document["video_id"], root)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}-", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def update(video_id: str, bounds, expected_revision: int, updates: list[dict], actor: str,
           root=LABEL_DIR) -> dict:
    with _write_lock:
        document = load(video_id, bounds, root)
        if document["revision"] != expected_revision:
            raise LabelConflict("shot labels changed; reload before saving")
        labels = dict(document["labels"])
        now = datetime.now(timezone.utc).isoformat()
        for change in updates:
            index = int(change["shot_index"])
            if index < 0 or index >= len(bounds):
                raise ValueError(f"shot {index} is outside {video_id}")
            state = change.get("state")
            if state is None:
                labels.pop(str(index), None)
                continue
            reason = change.get("reason")
            if state not in {"excluded", "included"} or reason not in REASONS:
                raise ValueError("reviewed labels require a valid state and reason")
            previous = labels.get(str(index), {})
            labels[str(index)] = {
                "state": state,
                "reason": reason,
                **({"confidence": previous["confidence"]} if "confidence" in previous else {}),
                "reviewed_by": actor,
                "reviewed_at": now,
            }
        document["labels"] = labels
        document["revision"] += 1
        _write(document, root)
        return document


def replace_suggestions(video_id: str, bounds, suggestions: dict[int, dict], root=LABEL_DIR) -> dict:
    """Replace unreviewed suggestions while preserving every reviewed decision."""
    with _write_lock:
        document = load(video_id, bounds, root)
        labels = {
            key: value for key, value in document["labels"].items()
            if value["state"] != "suggested"
        }
        for index, suggestion in suggestions.items():
            key = str(index)
            if key in labels:
                continue
            reason = suggestion["reason"]
            confidence = float(suggestion["confidence"])
            if reason not in REASONS or not 0 <= confidence <= 1:
                raise ValueError("invalid automatic suggestion")
            labels[key] = {
                "state": "suggested", "reason": reason,
                "confidence": round(confidence, 4),
            }
        document["labels"] = labels
        document["revision"] += 1
        _write(document, root)
        return document


def excluded_shots(video_id: str, bounds, root=LABEL_DIR) -> set[int]:
    document = load(video_id, bounds, root)
    return {
        int(index) for index, label in document["labels"].items()
        if label["state"] == "excluded"
    }
