"""Server-side DRES adapter for the organisers' 2026 final-round format."""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
import time
import csv
from bisect import bisect_left
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from typing import Literal

from . import frame_maps
from .dres_payload import (
    build_kis, build_qa, build_trake, canonical_payload, frame_timestamp_ms,
    payload_fingerprint, video_id_from_source,
)


logger = logging.getLogger("uvicorn.error")
Kind = Literal["kis", "qa", "trake"]


class DresError(Exception):
    def __init__(self, status_code: int, detail: str, category: str = "not_sent"):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.category = category


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_message(response: httpx.Response, session: str, password: str = "") -> str:
    try:
        data = response.json()
    except ValueError:
        data = None
    if isinstance(data, dict):
        value = next((data[key] for key in ("message", "detail", "error") if isinstance(data.get(key), str)), "")
    else:
        value = ""
    if not value:
        value = response.reason_phrase or "DRES response"
    for secret in (session, password):
        if secret:
            value = value.replace(secret, "[redacted]")
    return value[:300]


class DresConfig:
    def __init__(self, base_url: str, username: str = "", password: str = ""):
        base_url = base_url.strip().rstrip("/")
        if base_url:
            parsed = urlsplit(base_url)
            local = parsed.hostname in {"127.0.0.1", "localhost", "::1"}
            if (parsed.scheme != "https" and not (parsed.scheme == "http" and local)
                    or not parsed.hostname or parsed.username or parsed.password
                    or parsed.path or parsed.query or parsed.fragment):
                raise ValueError("AIC_DRES_BASE_URL must be an HTTPS origin (HTTP loopback for mocks)")
        self.base_url = base_url
        self.username = username
        self.password = password

    @classmethod
    def from_env(cls):
        return cls(
            os.environ.get("AIC_DRES_BASE_URL", ""),
            os.environ.get("AIC_DRES_USERNAME", ""),
            os.environ.get("AIC_DRES_PASSWORD", ""),
        )


class DresClient:
    """No redirect or automatic retry; only this class knows DRES's URL shape."""

    def __init__(self, config: DresConfig, client: httpx.Client | None = None):
        self.config = config
        # httpx INFO lines include full URLs, including DRES's required session query param.
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        self.http = client or httpx.Client(
            base_url=config.base_url,
            transport=httpx.HTTPTransport(retries=0),
            timeout=httpx.Timeout(12.0, connect=3.0, read=12.0, write=5.0, pool=2.0),
            follow_redirects=False,
            trust_env=False,
        )

    def close(self):
        self.http.close()

    def login(self) -> str:
        if not self.config.username or not self.config.password:
            raise DresError(409, "DRES username/password are not configured")
        try:
            response = self.http.post("/api/v2/login", json={
                "username": self.config.username, "password": self.config.password,
            })
        except httpx.RequestError as exc:
            raise DresError(503, f"DRES login connection failed ({type(exc).__name__})") from None
        if response.status_code >= 400:
            raise DresError(502, f"DRES login HTTP {response.status_code}: " +
                            _safe_message(response, "", self.config.password))
        try:
            data = response.json()
        except ValueError as exc:
            raise DresError(502, "DRES login returned malformed JSON") from exc
        if not isinstance(data, dict) or data.get("role") != "PARTICIPANT" or not isinstance(data.get("sessionId"), str) or not data["sessionId"].strip():
            raise DresError(502, "DRES login response has no participant sessionId")
        return data["sessionId"]

    def evaluations(self, session: str) -> list[dict]:
        try:
            response = self.http.get("/api/v2/client/evaluation/list", params={"session": session})
        except httpx.RequestError as exc:
            raise DresError(503, f"DRES evaluation list failed ({type(exc).__name__})") from None
        if response.status_code in (401, 403):
            raise DresError(401, "DRES session was rejected")
        if response.status_code >= 400:
            raise DresError(502, f"DRES evaluation list HTTP {response.status_code}: " +
                            _safe_message(response, session))
        try:
            data = response.json()
        except ValueError as exc:
            raise DresError(502, "DRES evaluation list returned malformed JSON") from exc
        if not isinstance(data, list) or any(not isinstance(item, dict) for item in data):
            raise DresError(502, "DRES evaluation list has an unexpected schema")
        eligible = [item for item in data if item.get("status") == "ACTIVE" and item.get("type") == "SYNCHRONOUS"]
        if any(not isinstance(item.get("id"), str) or not item["id"] for item in eligible):
            raise DresError(502, "DRES active evaluation has no ID")
        return [{"id": item["id"], "name": str(item.get("name", ""))} for item in eligible]

    def submit(self, evaluation_id: str, session: str, payload: dict) -> httpx.Response:
        return self.http.post(
            "/api/v2/submit/" + quote(evaluation_id, safe=""),
            params={"session": session}, json=payload,
        )


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    video_id: str = Field(min_length=1, max_length=80)
    frame_idx: StrictInt | None = Field(default=None, ge=0)
    frame_source: Literal["indexed_keyframe", "original_frame"] = "indexed_keyframe"
    frame_ids: list[StrictInt] | None = Field(default=None, max_length=100)
    answer: str | None = None


class ContextPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(ge=0)
    kind: Kind
    label: str = Field(min_length=1, max_length=160)
    event_count: int | None = Field(default=None, ge=1, le=100)


class PrepareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate: Candidate


class SubmitRequest(PrepareRequest):
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_evaluation_id: str
    expected_context_revision: int = Field(ge=0)
    retry_uncertain: bool = False


class EvaluationChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evaluation_id: str | None = None


class DresSubmissionStore:
    """Shared SQLite ledger; credentials and session never enter it."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS dres_context (
                    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                    kind TEXT NOT NULL, label TEXT NOT NULL, event_count INTEGER,
                    revision INTEGER NOT NULL, updated_by TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dres_attempts (
                    id INTEGER PRIMARY KEY, evaluation_id TEXT NOT NULL,
                    context_revision INTEGER NOT NULL, context_label TEXT NOT NULL,
                    task_type TEXT NOT NULL, payload_json TEXT NOT NULL,
                    fingerprint TEXT NOT NULL, video_id TEXT NOT NULL,
                    frame_idx INTEGER, frame_ids_json TEXT,
                    actor TEXT NOT NULL, submitted_at TEXT NOT NULL,
                    status TEXT NOT NULL, http_status INTEGER, latency_ms REAL,
                    server_message TEXT
                );
                CREATE INDEX IF NOT EXISTS dres_attempt_lookup
                    ON dres_attempts(evaluation_id, fingerprint, id);
            """)
            db.execute("UPDATE dres_attempts SET status = 'uncertain', server_message = 'App stopped while submit was pending' WHERE status = 'pending'")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def context(self):
        with self._connect() as db:
            row = db.execute("SELECT * FROM dres_context WHERE singleton = 1").fetchone()
        return dict(row) if row else None

    def set_context(self, patch: ContextPatch, actor: str):
        label = patch.label.strip()
        if not label or (patch.kind != "trake" and patch.event_count is not None):
            raise DresError(422, "Task label is required; only TRAKE may specify an event count")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT revision FROM dres_context WHERE singleton = 1").fetchone()
            revision = int(row[0]) if row else 0
            if revision != patch.expected_revision:
                raise DresError(409, "Task context changed; refresh before editing")
            db.execute("INSERT OR REPLACE INTO dres_context VALUES (1, ?, ?, ?, ?, ?, ?)",
                       (patch.kind, label, patch.event_count, revision + 1, actor, _now()))
        return self.context()

    def begin(self, prepared: dict, actor: str, retry_uncertain: bool = False) -> int:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            busy = db.execute("SELECT id FROM dres_attempts WHERE evaluation_id = ? AND status = 'pending' LIMIT 1",
                              (prepared["evaluation_id"],)).fetchone()
            if busy:
                raise DresError(409, "A DRES submission is already pending")
            previous = db.execute(
                "SELECT status, http_status FROM dres_attempts WHERE evaluation_id = ? AND fingerprint = ? ORDER BY id DESC LIMIT 1",
                (prepared["evaluation_id"], prepared["fingerprint"]),
            ).fetchone()
            if previous and not (previous["status"] == "not_sent"
                                 or previous["status"] == "uncertain" and retry_uncertain
                                 or previous["status"] == "rejected" and previous["http_status"] in (401, 403)):
                raise DresError(409, "This exact DRES payload was already attempted in this evaluation")
            row = db.execute("""INSERT INTO dres_attempts
                (evaluation_id, context_revision, context_label, task_type, payload_json,
                 fingerprint, video_id, frame_idx, frame_ids_json, actor, submitted_at, status)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""", (
                prepared["evaluation_id"], prepared["context_revision"], prepared["context_label"],
                prepared["kind"], canonical_payload(prepared["payload"]), prepared["fingerprint"],
                prepared["video_id"], prepared.get("frame_idx"),
                json.dumps(prepared.get("frame_ids")), actor, _now(),
            ))
            return row.lastrowid

    def finish(self, attempt_id: int, status: str, http_status: int | None, latency_ms: float, message: str):
        with self._connect() as db:
            db.execute("UPDATE dres_attempts SET status=?, http_status=?, latency_ms=?, server_message=? WHERE id=?",
                       (status, http_status, round(latency_ms, 1), message[:300], attempt_id))
        return self.get(attempt_id)

    def get(self, attempt_id: int):
        with self._connect() as db:
            row = db.execute("SELECT * FROM dres_attempts WHERE id = ?", (attempt_id,)).fetchone()
        return self._public(row)

    def history(self, limit: int = 20):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM dres_attempts ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._public(row) for row in rows]

    @staticmethod
    def _public(row):
        result = dict(row)
        result["payload"] = json.loads(result.pop("payload_json"))
        result["frame_ids"] = json.loads(result.pop("frame_ids_json"))
        return result


class DresService:
    def __init__(self, config: DresConfig, store: DresSubmissionStore, resolve_video,
                 load_map=None, index_dir: str | Path = frame_maps.MAP_DIR.parent,
                 client: DresClient | None = None):
        self.config = config
        self.store = store
        self.resolve_video = resolve_video
        self.manifest_path = Path(index_dir) / "frame_maps" / "manifest.json"
        self.load_map = load_map or (lambda video_id: frame_maps.load_path(
            self.manifest_path.parent / f"{video_id}.npz"))
        self.client = client or (DresClient(config) if config.base_url else None)
        self.session: str | None = None
        self.evaluations: list[dict] = []
        self.evaluation_id: str | None = None
        self.context_evaluation_id: str | None = None
        self.manual_evaluation_id: str | None = None
        self.lock = threading.RLock()

    @staticmethod
    def _organizer_keyframes(video_path: Path, video_id: str):
        """Return BTC map-keyframes as frame_idx -> (pts_ms, official FRAME_ID).

        The organizer's `n` column is only the image-file ordinal. `frame_idx` is
        the identifier used by our indexed image paths and is kept verbatim for
        TRAKE. `pts_time` is authoritative for those sampled frames, especially
        for VFR Batch 2 videos where frame_idx is only a relative label.
        """
        map_dir = video_path.parents[2] / "map-keyframes"
        csv_path = next((candidate for candidate in map_dir.glob("*.csv")
                         if candidate.stem.replace("-", "_") == video_id), None)
        if csv_path is None:
            return None
        found: dict[int, list[tuple[int, int]]] = {}
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if not {"n", "pts_time", "frame_idx"}.issubset(reader.fieldnames or []):
                    raise ValueError("organizer keyframe map has an unexpected schema")
                for row in reader:
                    frame_idx = int(row["frame_idx"])
                    ordinal = int(row["n"])
                    pts_ms = int((Decimal(row["pts_time"]) * 1000).quantize(
                        Decimal("1"), rounding=ROUND_HALF_UP))
                    found.setdefault(frame_idx, []).append((pts_ms, ordinal))
        except (OSError, ValueError, InvalidOperation, TypeError) as exc:
            raise DresError(503, "Organizer keyframe map is malformed") from exc
        return found

    @staticmethod
    def _selected_timestamp(mapping, organizer_keyframes, video_id: str, frame_idx: int,
                            frame_source: str = "indexed_keyframe") -> int:
        if frame_source == "original_frame":
            return frame_timestamp_ms(mapping, frame_idx)
        rows = (organizer_keyframes or {}).get(frame_idx, [])
        if len(rows) == 1:
            return rows[0][0]
        if len(rows) > 1:
            return frame_timestamp_ms(mapping, frame_idx)
        if organizer_keyframes is not None and video_id.split("_", 1)[0].startswith("N"):
            raise ValueError("Selected Batch 2 VFR frame is absent from the organizer PTS map")
        return frame_timestamp_ms(mapping, frame_idx)

    def frame_timestamp(self, video_id: str, frame_idx: int) -> int:
        normalized, mapping = self._source(video_id)
        row = self.resolve_video(normalized)
        organizer_keyframes = self._organizer_keyframes(Path(row["path"]), normalized)
        try:
            return self._selected_timestamp(mapping, organizer_keyframes, normalized, frame_idx)
        except ValueError as exc:
            raise DresError(422, str(exc)) from exc

    def official_frame_id(self, video_id: str, source_frame_idx: int) -> int:
        normalized, mapping = self._source(video_id)
        row = self.resolve_video(normalized)
        organizer_keyframes = self._organizer_keyframes(Path(row["path"]), normalized)
        if not organizer_keyframes:
            raise DresError(422, "Official TRAKE FRAME_ID map is unavailable for this video")
        try:
            timestamp_ms = frame_timestamp_ms(mapping, source_frame_idx)
        except ValueError as exc:
            raise DresError(422, str(exc)) from exc
        matches = {frame_idx for frame_idx, rows in organizer_keyframes.items()
                   if any(pts_ms == timestamp_ms for pts_ms, _ordinal in rows)}
        if len(matches) != 1:
            raise DresError(422, "Player frame is not a unique official TRAKE keyframe")
        return matches.pop()

    def official_keyframes(self, video_id: str) -> list[dict]:
        """Expose BTC's semantic keyframes for deliberate TRAKE selection."""
        normalized, mapping = self._source(video_id)
        row = self.resolve_video(normalized)
        keyframes = self._organizer_keyframes(Path(row["path"]), normalized)
        if not keyframes:
            raise DresError(422, "Official TRAKE FRAME_ID map is unavailable for this video")
        result = []
        for frame_id, rows in keyframes.items():
            timestamps = {timestamp_ms for timestamp_ms, _ordinal in rows}
            # Some organizer maps repeat a FRAME_ID at adjacent initial samples. Their ID
            # remains explicit and unambiguous; use the verified original PTS map for the
            # thumbnail/player location, matching _selected_timestamp's duplicate-row rule.
            if len(timestamps) == 1:
                timestamp_ms = timestamps.pop()
            else:
                try:
                    timestamp_ms = frame_timestamp_ms(mapping, frame_id)
                except ValueError as exc:
                    raise DresError(422, f"Official TRAKE FRAME_ID {frame_id} has conflicting timestamps") from exc
            source_pts = mapping.pts
            target_pts = int(source_pts[0]) + timestamp_ms * mapping.time_base_den / (1000 * mapping.time_base_num)
            pos = bisect_left(source_pts, target_pts)
            choices = [index for index in (pos - 1, pos) if 0 <= index < len(source_pts)]
            source_frame_idx = min(choices, key=lambda index: abs(
                (int(source_pts[index]) - int(source_pts[0])) * mapping.time_base_num * 1000
                - timestamp_ms * mapping.time_base_den))
            result.append({"frame_id": frame_id, "source_frame_idx": source_frame_idx,
                           "timestamp_ms": timestamp_ms})
        return sorted(result, key=lambda item: (item["timestamp_ms"], item["frame_id"]))

    def close(self):
        if self.client:
            self.client.close()

    def status(self):
        context = self.store.context()
        return {
            "configured": bool(self.client),
            "credentials_configured": bool(self.config.username and self.config.password),
            "host": urlsplit(self.config.base_url).hostname or "",
            "connected": bool(self.session),
            "evaluations": self.evaluations,
            "evaluation_id": self.evaluation_id,
            "evaluation_manual": self.manual_evaluation_id == self.evaluation_id and self.evaluation_id is not None,
            "context": context,
            "context_valid": bool(context and self.evaluation_id and self.context_evaluation_id == self.evaluation_id),
        }

    def _require_client(self):
        if not self.client:
            raise DresError(409, "DRES base URL is not configured")
        return self.client

    def connect(self, *, username: str | None = None, password: str | None = None):
        with self.lock:
            previous_evaluation = self.evaluation_id
            previous_context_evaluation = self.context_evaluation_id
            client = self._require_client()
            if username is not None or password is not None:
                if not isinstance(username, str) or not username.strip() or not isinstance(password, str) or not password:
                    raise DresError(422, "DRES username and password are required")
                self.config.username, self.config.password = username.strip(), password
            try:
                self.session = client.login()
            except DresError:
                self.session = None
                raise
            self.evaluations = []
            self.evaluation_id = None
            self.refresh_evaluations()
            if self.evaluation_id == previous_evaluation:
                self.context_evaluation_id = previous_context_evaluation
            return self.status()

    def reset(self):
        with self.lock:
            self.session = None
            self.evaluations = []
            self.evaluation_id = None
            self.context_evaluation_id = None
            self.manual_evaluation_id = None
            return self.status()

    def refresh_evaluations(self):
        with self.lock:
            if not self.session:
                raise DresError(409, "Connect to DRES first")
            try:
                self.evaluations = self._require_client().evaluations(self.session)
            except DresError as exc:
                if exc.status_code == 401:
                    self.reset()
                raise
            previous_evaluation = self.evaluation_id
            context_was_valid = bool(previous_evaluation and self.context_evaluation_id == previous_evaluation)
            eligible_ids = {item["id"] for item in self.evaluations}
            if self.manual_evaluation_id not in eligible_ids:
                self.manual_evaluation_id = None
            if len(self.evaluations) == 1:
                context = self.store.context()
                try:
                    self.evaluation_id = self._evaluation_for_kind(context["kind"]) if context else self.evaluations[0]["id"]
                except DresError:
                    self.evaluation_id = None
                self.manual_evaluation_id = None
            elif self.manual_evaluation_id:
                self.evaluation_id = self.manual_evaluation_id
            else:
                context = self.store.context()
                try:
                    self.evaluation_id = self._evaluation_for_kind(context["kind"]) if context else None
                except DresError:
                    self.evaluation_id = None
            if self.evaluation_id != previous_evaluation:
                self.context_evaluation_id = self.evaluation_id if context_was_valid and self.evaluation_id else None
            return self.status()

    def select_evaluation(self, evaluation_id: str | None):
        with self.lock:
            if evaluation_id is not None and evaluation_id not in {item["id"] for item in self.evaluations}:
                raise DresError(422, "Selected evaluation is not ACTIVE and SYNCHRONOUS")
            previous = self.evaluation_id
            self.manual_evaluation_id = evaluation_id
            if evaluation_id:
                self.evaluation_id = evaluation_id
            elif len(self.evaluations) == 1:
                context = self.store.context()
                try:
                    self.evaluation_id = self._evaluation_for_kind(context["kind"]) if context else self.evaluations[0]["id"]
                except DresError:
                    self.evaluation_id = None
            else:
                context = self.store.context()
                try:
                    self.evaluation_id = self._evaluation_for_kind(context["kind"]) if context else None
                except DresError:
                    self.evaluation_id = None
            if previous != self.evaluation_id:
                self.context_evaluation_id = None
            return self.status()

    def _evaluation_for_kind(self, kind: str) -> str:
        pattern = {
            "kis": r"(?<![A-Z0-9])KIS(?![A-Z0-9])",
            "qa": r"(?<![A-Z0-9])(?:QA|Q\s*&\s*A)(?![A-Z0-9])",
            "trake": r"(?<![A-Z0-9])TRAKE(?![A-Z0-9])",
        }
        matches = [item for item in self.evaluations if re.search(pattern[kind], item["name"], re.IGNORECASE)]
        if len(matches) != 1:
            named = [item for item in self.evaluations if any(re.search(p, item["name"], re.IGNORECASE) for p in pattern.values())]
            if len(self.evaluations) == 1 and not named:
                return self.evaluations[0]["id"]
            raise DresError(409, f"Could not uniquely match task type {kind.upper()} to an ACTIVE SYNCHRONOUS DRES evaluation name")
        return matches[0]["id"]

    def set_context(self, patch: ContextPatch, actor: str):
        with self.lock:
            if not self.session or not self.evaluations:
                raise DresError(409, "Connect to DRES and refresh ACTIVE SYNCHRONOUS evaluations before setting its task")
            eligible_ids = {item["id"] for item in self.evaluations}
            if self.manual_evaluation_id in eligible_ids:
                evaluation_id = self.manual_evaluation_id
            else:
                evaluation_id = self._evaluation_for_kind(patch.kind)
            context = self.store.set_context(patch, actor)
            if self.evaluation_id != evaluation_id:
                self.context_evaluation_id = None
            self.evaluation_id = evaluation_id
            self.context_evaluation_id = self.evaluation_id
            return context

    def _source(self, video_id: str):
        row = self.resolve_video(video_id)
        path = Path(row["path"])
        try:
            normalized = video_id_from_source(path, video_id)
        except ValueError as exc:
            raise DresError(422, str(exc)) from exc
        try:
            document = json.loads(self.manifest_path.read_text())
            signature = document["videos"][normalized]
            stat = path.stat()
        except (OSError, KeyError, ValueError, TypeError) as exc:
            raise DresError(503, "Original-video frame map has no verified source signature") from exc
        recorded = signature.get("path") if isinstance(signature, dict) else None
        recorded_path = Path(recorded) if isinstance(recorded, str) else Path("")
        # The offline builder records its host path; the production bind mount has a
        # different root, but retains year/videos/group/filename.
        same_source = (recorded_path == path or len(recorded_path.parts) >= 4 and len(path.parts) >= 4
                       and recorded_path.parts[-4:] == path.parts[-4:])
        if (not isinstance(signature, dict) or not same_source
                or signature.get("size") != stat.st_size or signature.get("mtimeNs") != stat.st_mtime_ns):
            raise DresError(503, "Original-video frame map is stale; rebuild it before DRES submission")
        try:
            return normalized, self.load_map(normalized)
        except (OSError, ValueError) as exc:
            raise DresError(503, "Original-video frame map is unavailable") from exc

    def prepare(self, candidate: Candidate):
        with self.lock:
            if not self.session or not self.evaluation_id:
                raise DresError(409, "DRES session and active evaluation are required")
            context = self.store.context()
            if not context or self.context_evaluation_id != self.evaluation_id:
                raise DresError(409, "Set the current task for this evaluation first")
            video_id, mapping = self._source(candidate.video_id)
            organizer_keyframes = self._organizer_keyframes(Path(self.resolve_video(video_id)["path"]), video_id)
            kind = context["kind"]
            frame_idx = None
            frame_ids = None
            timestamp_ms = None
            try:
                if kind == "kis":
                    if candidate.frame_ids is not None or candidate.answer is not None:
                        raise ValueError("KIS accepts only one selected frame")
                    if candidate.frame_idx is None:
                        raise ValueError("Select an exact frame for KIS")
                    frame_idx = candidate.frame_idx
                    timestamp_ms = self._selected_timestamp(mapping, organizer_keyframes, video_id,
                                                            frame_idx, candidate.frame_source)
                    payload = build_kis(video_id, timestamp_ms)
                elif kind == "qa":
                    if candidate.frame_ids is not None:
                        raise ValueError("Q&A accepts one selected frame")
                    if candidate.frame_idx is None:
                        raise ValueError("Select an exact frame for Q&A")
                    frame_idx = candidate.frame_idx
                    timestamp_ms = self._selected_timestamp(mapping, organizer_keyframes, video_id,
                                                            frame_idx, candidate.frame_source)
                    payload = build_qa(candidate.answer, video_id, timestamp_ms)
                else:
                    if candidate.frame_idx is not None or candidate.answer is not None:
                        raise ValueError("TRAKE accepts only ordered FRAME_IDs")
                    frame_ids = candidate.frame_ids
                    if frame_ids is None:
                        raise ValueError("Select TRAKE event frames")
                    # A TRAKE event is the source-video frame of the moment, not the organizer keyframe
                    # nearest to it, so any frame of the video is a valid event: their evaluation maps
                    # the submitted index onto its own keyframes. Only the video's own bounds are
                    # checked here, and the organizer map stays a navigation aid (`official_keyframes`).
                    frame_count = len(mapping.pts)
                    for frame in frame_ids:
                        if frame >= frame_count:
                            raise ValueError(f"FRAME_ID {frame} is outside the original video")
                    payload = build_trake(video_id, frame_ids)
            except (ValueError, TypeError) as exc:
                raise DresError(422, str(exc)) from exc
            answer = payload["answerSets"][0]["answers"][0]
            return {
                "evaluation_id": self.evaluation_id,
                "context_revision": context["revision"], "context_label": context["label"],
                "kind": kind, "video_id": video_id,
                "frame_idx": frame_idx, "frame_ids": frame_ids,
                "timestamp_ms": timestamp_ms,
                "payload": payload, "preview": answer.get("text") or canonical_payload(payload),
                "fingerprint": payload_fingerprint(payload),
            }

    def submit(self, request: SubmitRequest, actor: str):
        with self.lock:
            self.refresh_evaluations()
            prepared = self.prepare(request.candidate)
            if (prepared["fingerprint"] != request.fingerprint
                    or prepared["context_revision"] != request.expected_context_revision):
                raise DresError(409, "DRES preview changed; prepare it again before submitting")
            try:
                attempt_id = self.store.begin(prepared, actor, request.retry_uncertain)
            except sqlite3.Error as exc:
                raise DresError(503, "Local DRES ledger unavailable; request was not sent") from exc
            started = time.perf_counter()
            session = self.session
            try:
                response = self._require_client().submit(prepared["evaluation_id"], session, prepared["payload"])
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                status, http_status, message = "not_sent", None, f"DRES connection failed before sending ({type(exc).__name__})"
            except httpx.RequestError as exc:
                status, http_status, message = "uncertain", None, f"DRES response is uncertain ({type(exc).__name__})"
            else:
                http_status = response.status_code
                message = _safe_message(response, session, self.config.password)
                if 200 <= response.status_code < 300:
                    try:
                        data = response.json()
                    except ValueError:
                        data = None
                    rejected = isinstance(data, dict) and (
                        data.get("status") in ("REJECTED", "rejected", "ERROR", "error", "FAILED", "failed") or data.get("success") is False
                        or data.get("error") not in (None, "", False)
                    )
                    status = "rejected" if rejected else "accepted" if isinstance(data, dict) else "uncertain"
                elif 400 <= response.status_code < 500:
                    status = "rejected"
                    if response.status_code in (401, 403):
                        self.reset()
                else:
                    status = "uncertain"
            latency_ms = (time.perf_counter() - started) * 1000
            try:
                result = self.store.finish(attempt_id, status, http_status, latency_ms, message)
            except sqlite3.Error as exc:
                raise DresError(503, "DRES request may have arrived, but the local ledger could not record its result; check DRES before retry", "uncertain") from exc
            logger.info("dres_submit host=%s evaluation=%s task=%s video=%s frame=%s time_ms=%s frames=%s status=%s http=%s latency_ms=%.1f",
                        urlsplit(self.config.base_url).hostname, prepared["evaluation_id"],
                        prepared["kind"], prepared["video_id"], prepared["frame_idx"],
                        prepared["timestamp_ms"], prepared["frame_ids"], status, http_status, latency_ms)
            return result


def install_dres_routes(app: FastAPI):
    def service(request: Request) -> DresService:
        return request.app.state.dres_service

    def admin(request: Request):
        if not request.state.is_admin:
            raise DresError(403, "Only the owner can manage DRES authentication and evaluation")

    @app.exception_handler(DresError)
    async def dres_error(_request, exc: DresError):
        return JSONResponse({"detail": exc.detail, "category": exc.category}, status_code=exc.status_code)

    @app.get("/api/dres/status")
    async def status(request: Request):
        return service(request).status()

    @app.get("/api/dres/frame/{video_id}/{frame_idx}")
    async def frame_timestamp(video_id: str, frame_idx: int, request: Request):
        return {"timestamp_ms": await run_in_threadpool(service(request).frame_timestamp, video_id, frame_idx)}

    @app.get("/api/dres/frame-id/{video_id}/{frame_idx}")
    async def official_frame_id(video_id: str, frame_idx: int, request: Request):
        return {"frame_id": await run_in_threadpool(service(request).official_frame_id, video_id, frame_idx)}

    @app.get("/api/dres/keyframes/{video_id}")
    async def official_keyframes(video_id: str, request: Request):
        return {"frames": await run_in_threadpool(service(request).official_keyframes, video_id)}

    @app.post("/api/dres/connect")
    async def connect(request: Request):
        admin(request)
        return await run_in_threadpool(service(request).connect)

    @app.post("/api/dres/relogin")
    async def relogin(request: Request):
        admin(request)
        return await run_in_threadpool(service(request).connect)

    @app.post("/api/dres/reset")
    async def reset(request: Request):
        admin(request)
        return service(request).reset()

    @app.post("/api/dres/evaluations/refresh")
    async def refresh(request: Request):
        admin(request)
        return await run_in_threadpool(service(request).refresh_evaluations)

    @app.post("/api/dres/evaluation")
    async def evaluation(choice: EvaluationChoice, request: Request):
        return service(request).select_evaluation(choice.evaluation_id)

    @app.post("/api/dres/context")
    async def context(patch: ContextPatch, request: Request):
        return await run_in_threadpool(service(request).set_context, patch, request.state.login)

    @app.post("/api/dres/prepare")
    async def prepare(payload: PrepareRequest, request: Request):
        return await run_in_threadpool(service(request).prepare, payload.candidate)

    @app.post("/api/dres/submit")
    async def submit(payload: SubmitRequest, request: Request):
        return await run_in_threadpool(service(request).submit, payload, request.state.login)

    @app.get("/api/dres/history")
    async def history(request: Request):
        return {"attempts": service(request).store.history()}
