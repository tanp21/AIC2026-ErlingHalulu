"""BTC 2026 DRES payloads. These functions never perform HTTP requests."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

from .frame_maps import FrameMap


VIDEO_ID = re.compile(r"^[A-Z]\d{2,3}_V\d{3}$")


def video_id_from_source(path: str | Path, expected: str) -> str:
    source = Path(path)
    if source.suffix.lower() not in {".mp4", ".mov"} or not VIDEO_ID.fullmatch(source.stem):
        raise ValueError("original video filename is invalid")
    if source.stem != expected:
        raise ValueError("video ID does not match original filename")
    return source.stem


def _frame_id(value: int, mapping: FrameMap) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < len(mapping.pts):
        raise ValueError("selected frame is outside the original video")
    return value


def frame_timestamp_ms(mapping: FrameMap, frame_id: int) -> int:
    """Nearest integer millisecond of a frame on the video's zero-based timeline.

    Work with PTS integers throughout: `frame_time()` returns a float for UI/search,
    but rounding that float again can move a submission by a millisecond.
    """
    frame_id = _frame_id(frame_id, mapping)
    numerator = (int(mapping.pts[frame_id]) - int(mapping.pts[0])) * mapping.time_base_num * 1000
    return (2 * numerator + mapping.time_base_den) // (2 * mapping.time_base_den)


def _milliseconds(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("frame timestamp must be a non-negative integer millisecond")
    return str(value)


def _answer_set(answer: dict) -> dict:
    return {"answerSets": [{"answers": [answer]}]}


def build_kis(video_id: str, frame_timestamp_ms: int) -> dict:
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError("video ID is invalid")
    timestamp = _milliseconds(frame_timestamp_ms)
    payload = _answer_set({"mediaItemName": video_id, "start": timestamp, "end": timestamp})
    validate_kis_payload(payload, frame_timestamp_ms)
    return payload


def validate_kis_payload(payload: dict, selected_frame_timestamp_ms: int) -> None:
    try:
        answer = payload["answerSets"][0]["answers"][0]
        start, end = answer["start"], answer["end"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("KIS payload is malformed") from exc
    if start != end or start != _milliseconds(selected_frame_timestamp_ms):
        raise ValueError("KIS start and end must equal the selected frame timestamp")


def normalize_answer(answer: str) -> str:
    """Uppercase, whitespace-free, with Vietnamese folded to ASCII -- the answer form the organizers
    compare in. Whatever the person typed stays a Vietnamese or English answer, only its spelling
    changes: an English answer has no diacritics to lose and is just closed up and uppercased.

    NFD splits every Vietnamese letter into its base plus combining marks except `đ`, which has no
    canonical decomposition and is folded by hand.
    """
    folded = "".join(character for character in unicodedata.normalize("NFD", answer)
                     if not unicodedata.combining(character))
    return "".join(folded.replace("đ", "d").replace("Đ", "D").split()).upper()


def build_qa(answer: str, video_id: str, frame_timestamp_ms: int) -> dict:
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Q&A answer is required")
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError("video ID is invalid")
    return _answer_set({"text": f"QA-{normalize_answer(answer)}-{video_id}-{_milliseconds(frame_timestamp_ms)}"})


def build_trake(video_id: str, frame_ids: list[int], event_count: int | None = None) -> dict:
    if not VIDEO_ID.fullmatch(video_id):
        raise ValueError("video ID is invalid")
    if event_count is not None and (isinstance(event_count, bool) or not isinstance(event_count, int) or event_count < 1):
        raise ValueError("TRAKE event count is invalid")
    if not isinstance(frame_ids, list) or not frame_ids or len(frame_ids) > 100 or (event_count is not None and len(frame_ids) != event_count):
        raise ValueError("TRAKE needs 1–100 ordered event frames")
    if any(isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 for frame in frame_ids):
        raise ValueError("TRAKE frame IDs must be non-negative integers")
    if len(set(frame_ids)) != len(frame_ids):
        raise ValueError("TRAKE frame IDs must be unique")
    return _answer_set({"text": f"TR-{video_id}-{','.join(map(str, frame_ids))}"})


def canonical_payload(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def payload_fingerprint(payload: dict) -> str:
    return hashlib.sha256(canonical_payload(payload).encode("utf-8")).hexdigest()
