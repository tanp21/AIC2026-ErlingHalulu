"""Small SQLite-backed collaboration workspace."""
from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import unicodedata
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from . import video_ids


MAX_ARCHIVE_BYTES = 1024 * 1024
MAX_EXPANDED_BYTES = 2 * 1024 * 1024
MAX_QUERY_BYTES = 64 * 1024
MAX_QUERY_FILES = 200
QUERY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*-(kis|qa|trake)\.txt$")
VIDEO_ID = video_ids.VIDEO_ID
_MISSING = object()


class CollaborationError(Exception):
    def __init__(self, status_code: int, detail: str, current=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.current = current


class QueryPatch(BaseModel):
    expected_version: int = Field(ge=0)
    assignee: str | None = None
    answers: list[dict] | None = None
    event_count: int | None = Field(default=None, ge=2, le=100)


class ReviewPatch(BaseModel):
    expected_version: int = Field(ge=0)
    reviewed: bool


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _natural_key(value: str):
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", value)]


class CollaborationStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self):
        with self._connect() as db:
            db.execute("PRAGMA journal_mode = WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS rounds (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    active INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS one_active_round
                    ON rounds(active) WHERE active = 1;
                CREATE TABLE IF NOT EXISTS queries (
                    id INTEGER PRIMARY KEY,
                    round_id INTEGER NOT NULL REFERENCES rounds(id) ON DELETE CASCADE,
                    filename TEXT NOT NULL,
                    kind TEXT NOT NULL CHECK(kind IN ('kis', 'qa', 'trake')),
                    prompt TEXT NOT NULL,
                    event_count INTEGER,
                    assignee TEXT,
                    answers_json TEXT NOT NULL DEFAULT '[]',
                    version INTEGER NOT NULL DEFAULT 0,
                    reviewed_by TEXT,
                    reviewed_at TEXT,
                    updated_by TEXT,
                    updated_at TEXT,
                    UNIQUE(round_id, filename COLLATE NOCASE)
                );
                PRAGMA user_version = 1;
            """)

    def import_round(self, name: str, payload: bytes, actor: str):
        name = name.strip()
        if not name or len(name) > 80:
            raise CollaborationError(422, "round name must contain 1-80 characters")
        parsed = self._parse_archive(payload)
        created = _now()
        try:
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute("UPDATE rounds SET active = 0, revision = revision + 1 WHERE active = 1")
                cursor = db.execute(
                    "INSERT INTO rounds(name, active, created_by, created_at) VALUES (?, 1, ?, ?)",
                    (name, actor, created),
                )
                round_id = cursor.lastrowid
                db.executemany(
                    """INSERT INTO queries(round_id, filename, kind, prompt, event_count,
                                             updated_by, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [(round_id, *query, actor, created) for query in parsed],
                )
        except sqlite3.IntegrityError as exc:
            raise CollaborationError(409, "round name already exists") from exc
        return self.get_round(round_id)

    def _parse_archive(self, payload: bytes):
        if not payload or len(payload) > MAX_ARCHIVE_BYTES:
            raise CollaborationError(422, "ZIP must be between 1 byte and 1 MiB")
        try:
            archive = zipfile.ZipFile(io.BytesIO(payload))
        except (zipfile.BadZipFile, OSError) as exc:
            raise CollaborationError(422, "invalid ZIP archive") from exc
        with archive:
            infos = [info for info in archive.infolist() if not info.is_dir()]
            if not 1 <= len(infos) <= MAX_QUERY_FILES:
                raise CollaborationError(422, "ZIP must contain 1-200 query files")
            if sum(info.file_size for info in infos) > MAX_EXPANDED_BYTES:
                raise CollaborationError(422, "expanded ZIP exceeds 2 MiB")
            paths = []
            for info in infos:
                path = PurePosixPath(info.filename)
                mode = info.external_attr >> 16
                if path.is_absolute() or ".." in path.parts or len(path.parts) not in (1, 2):
                    raise CollaborationError(422, f"unsafe ZIP path: {info.filename}")
                if mode and (mode & 0o170000) == 0o120000:
                    raise CollaborationError(422, "ZIP symlinks are not allowed")
                if info.file_size > MAX_QUERY_BYTES:
                    raise CollaborationError(422, f"query file exceeds 64 KiB: {path.name}")
                paths.append((path, info))
            depths = {len(path.parts) for path, _ in paths}
            if len(depths) != 1:
                raise CollaborationError(422, "query files must be flat or share one root folder")
            if depths == {2} and len({path.parts[0] for path, _ in paths}) != 1:
                raise CollaborationError(422, "query files must share one root folder")

            seen = set()
            parsed = []
            for path, info in paths:
                match = QUERY_NAME.fullmatch(path.name)
                if not match:
                    raise CollaborationError(422, f"invalid query filename: {path.name}")
                folded = path.name.casefold()
                if folded in seen:
                    raise CollaborationError(422, f"duplicate query filename: {path.name}")
                seen.add(folded)
                try:
                    prompt = archive.read(info).decode("utf-8-sig")
                except UnicodeDecodeError as exc:
                    raise CollaborationError(422, f"query is not UTF-8: {path.name}") from exc
                except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
                    raise CollaborationError(422, f"cannot read query from ZIP: {path.name}") from exc
                if not prompt.strip():
                    raise CollaborationError(422, f"query is empty: {path.name}")
                kind = match.group(1).lower()
                parsed.append((path.stem, kind, prompt, None))
        return sorted(parsed, key=lambda row: _natural_key(row[0]))

    def list_rounds(self):
        with self._connect() as db:
            rows = db.execute("SELECT * FROM rounds ORDER BY id DESC").fetchall()
            return [self._round_summary(db, row) for row in rows]

    def _round_summary(self, db, row):
        queries = db.execute("SELECT * FROM queries WHERE round_id = ?", (row["id"],)).fetchall()
        counts = {status: 0 for status in ("todo", "in_progress", "ready", "reviewed")}
        for query in queries:
            counts[self._query_dict(query)["status"]] += 1
        return {
            "id": row["id"], "name": row["name"], "active": bool(row["active"]),
            "revision": row["revision"], "created_by": row["created_by"],
            "created_at": row["created_at"], "counts": counts, "total": len(queries),
        }

    def get_round(self, round_id: int):
        with self._connect() as db:
            row = db.execute("SELECT * FROM rounds WHERE id = ?", (round_id,)).fetchone()
            if row is None:
                raise CollaborationError(404, "round not found")
            queries = db.execute("SELECT * FROM queries WHERE round_id = ?", (round_id,)).fetchall()
            query_dicts = sorted((self._query_dict(query) for query in queries),
                                 key=lambda query: _natural_key(query["filename"]))
            return {**self._round_summary(db, row), "queries": query_dicts,
                    "etag": f'"round-{round_id}-{row["revision"]}"'}

    def activate_round(self, round_id: int):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            target = db.execute("SELECT active FROM rounds WHERE id = ?", (round_id,)).fetchone()
            if target is None:
                raise CollaborationError(404, "round not found")
            if target["active"]:
                return self.get_round(round_id)
            db.execute("UPDATE rounds SET active = 0, revision = revision + 1 WHERE active = 1")
            db.execute("UPDATE rounds SET active = 1, revision = revision + 1 WHERE id = ?", (round_id,))
        return self.get_round(round_id)

    def update_query(self, query_id: int, expected_version: int, actor: str,
                     allowed_users: frozenset[str], *, assignee=_MISSING, answers=_MISSING,
                     event_count=_MISSING):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._query_for_update(db, query_id)
            if row["version"] != expected_version:
                raise CollaborationError(409, "stale query version", self._query_dict(row))
            values = {}
            if assignee is not _MISSING:
                if assignee is not None:
                    assignee = str(assignee).strip().lower()
                    if assignee not in allowed_users:
                        raise CollaborationError(422, "assignee is not an allowed teammate")
                values["assignee"] = assignee
            if answers is not _MISSING:
                normalized = self._normalize_answers(row["kind"], row["event_count"], answers)
                values.update(answers_json=json.dumps(normalized, ensure_ascii=False),
                              reviewed_by=None, reviewed_at=None)
            if event_count is not _MISSING:
                if row["kind"] != "trake":
                    raise CollaborationError(422, "only TRAKE has an event count")
                if json.loads(row["answers_json"]):
                    raise CollaborationError(422, "clear TRAKE answers before changing its event count")
                values["event_count"] = event_count
            if not values:
                return self._query_dict(row)
            values.update(version=row["version"] + 1, updated_by=actor, updated_at=_now())
            assignments = ", ".join(f"{key} = ?" for key in values)
            db.execute(f"UPDATE queries SET {assignments} WHERE id = ?",
                       (*values.values(), query_id))
            db.execute("UPDATE rounds SET revision = revision + 1 WHERE id = ?", (row["round_id"],))
            updated = db.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
        return self._query_dict(updated)

    def set_review(self, query_id: int, expected_version: int, actor: str, reviewed: bool):
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._query_for_update(db, query_id)
            if row["version"] != expected_version:
                raise CollaborationError(409, "stale query version", self._query_dict(row))
            query = self._query_dict(row)
            if reviewed and query["status"] != "ready":
                raise CollaborationError(422, "query must have valid answers before review")
            now = _now()
            db.execute(
                """UPDATE queries SET reviewed_by = ?, reviewed_at = ?, version = version + 1,
                                      updated_by = ?, updated_at = ? WHERE id = ?""",
                (actor if reviewed else None, now if reviewed else None, actor, now, query_id),
            )
            db.execute("UPDATE rounds SET revision = revision + 1 WHERE id = ?", (row["round_id"],))
            updated = db.execute("SELECT * FROM queries WHERE id = ?", (query_id,)).fetchone()
        return self._query_dict(updated)

    def _query_for_update(self, db, query_id):
        row = db.execute(
            "SELECT q.* FROM queries q JOIN rounds r ON r.id = q.round_id WHERE q.id = ? AND r.active = 1",
            (query_id,),
        ).fetchone()
        if row is None:
            raise CollaborationError(409, "query is not in the active round")
        return row

    def _normalize_answers(self, kind, event_count, answers):
        if not isinstance(answers, list) or len(answers) > 100:
            raise CollaborationError(422, "answers must be a list with at most 100 rows")
        normalized = []
        for answer in answers:
            if not isinstance(answer, dict):
                raise CollaborationError(422, "each answer must be an object")
            video_id = answer.get("video_id", "")
            if not isinstance(video_id, str):
                raise CollaborationError(422, "video_id must be a string")
            if kind in ("kis", "qa"):
                frame_id = answer.get("frame_id")
                if frame_id is not None and (isinstance(frame_id, bool) or not isinstance(frame_id, int) or frame_id < 0):
                    raise CollaborationError(422, "frame_id must be a non-negative integer or null")
                row = {"video_id": video_id, "frame_id": frame_id}
                if kind == "qa":
                    text = answer.get("answer", "")
                    if not isinstance(text, str) or len(text) > 100:
                        raise CollaborationError(422, "QA answer must be a string of at most 100 characters")
                    row["answer"] = text
            else:
                if event_count is None:
                    raise CollaborationError(422, "an admin must set the TRAKE event count first")
                frames = answer.get("frame_ids", [])
                if not isinstance(frames, list) or len(frames) > event_count:
                    raise CollaborationError(422, f"TRAKE requires at most {event_count} frame slots")
                for frame in frames:
                    if frame is not None and (isinstance(frame, bool) or not isinstance(frame, int) or frame < 0):
                        raise CollaborationError(422, "TRAKE frames must be non-negative integers or null")
                row = {"video_id": video_id, "frame_ids": frames + [None] * (event_count - len(frames))}
            if answer.get("auto_filled") is True:
                row["auto_filled"] = True
            elif "auto_filled" in answer and not isinstance(answer["auto_filled"], bool):
                raise CollaborationError(422, "auto_filled must be a boolean")
            normalized.append(row)
        return normalized

    def _query_dict(self, row):
        answers = json.loads(row["answers_json"])
        valid = bool(answers) and all(self._valid_answer(row["kind"], row["event_count"], answer)
                                      for answer in answers)
        if row["reviewed_by"] and valid:
            status = "reviewed"
        elif valid:
            status = "ready"
        elif row["assignee"] or answers:
            status = "in_progress"
        else:
            status = "todo"
        return {
            "id": row["id"], "round_id": row["round_id"], "filename": row["filename"],
            "kind": row["kind"], "prompt": row["prompt"], "event_count": row["event_count"],
            "assignee": row["assignee"], "answers": answers, "version": row["version"],
            "reviewed_by": row["reviewed_by"], "reviewed_at": row["reviewed_at"],
            "updated_by": row["updated_by"], "updated_at": row["updated_at"], "status": status,
        }

    def _valid_answer(self, kind, event_count, answer):
        video_id = answer.get("video_id")
        if not isinstance(video_id, str) or VIDEO_ID.fullmatch(video_id) is None:
            return False
        if kind in ("kis", "qa"):
            frame = answer.get("frame_id")
            if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
                return False
            if kind == "qa":
                text = answer.get("answer")
                if not isinstance(text, str) or not text.strip() or len(text) > 100:
                    return False
            return True
        frames = answer.get("frame_ids")
        return (isinstance(frames, list) and len(frames) == event_count
                and all(isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0 for frame in frames)
                and all(left < right for left, right in zip(frames, frames[1:])))

    def export_preview(self, round_id: int):
        return self._preview_from_snapshot(self.get_round(round_id))

    def _preview_from_snapshot(self, snapshot):
        preview = {"missing": [], "invalid": [], "unreviewed": []}
        for query in snapshot["queries"]:
            if not query["answers"]:
                preview["missing"].append(query["filename"])
            elif query["status"] == "in_progress":
                preview["invalid"].append(query["filename"])
            elif not query["reviewed_by"]:
                preview["unreviewed"].append(query["filename"])
        return preview

    def build_export(self, round_id: int, allow_incomplete: bool):
        snapshot = self.get_round(round_id)
        preview = self._preview_from_snapshot(snapshot)
        if any(preview.values()) and not allow_incomplete:
            raise CollaborationError(409, "export has warnings", preview)
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for query in snapshot["queries"]:
                if query["status"] not in ("ready", "reviewed"):
                    continue
                stream = io.StringIO(newline="")
                writer = csv.writer(stream, lineterminator="\n")
                for answer in query["answers"]:
                    if query["kind"] == "kis":
                        writer.writerow([answer["video_id"], answer["frame_id"]])
                    elif query["kind"] == "qa":
                        quoted_answer = answer["answer"].replace('"', '""')
                        stream.write(
                            f'{answer["video_id"]},{answer["frame_id"]},"{quoted_answer}"\n'
                        )
                    else:
                        writer.writerow([answer["video_id"], *answer["frame_ids"]])
                archive.writestr(f"submission/{query['filename']}.csv", stream.getvalue().encode("utf-8"))
        normalized_name = unicodedata.normalize(
            "NFKD", snapshot["name"].replace("Đ", "D").replace("đ", "d")
        )
        safe_name = "".join(character for character in normalized_name if character.isascii() and character.isalnum()) or "submission"
        return output.getvalue(), f"{safe_name}.zip"


def install_collaboration_routes(app: FastAPI) -> None:
    router = app

    def store(request: Request) -> CollaborationStore:
        return request.app.state.collaboration_store

    @app.exception_handler(CollaborationError)
    async def collaboration_error(_request, exc: CollaborationError):
        body = {"detail": exc.detail}
        if exc.current is not None:
            body["current"] = exc.current
        return JSONResponse(body, status_code=exc.status_code)

    @router.get("/api/collab/rounds")
    async def rounds(request: Request):
        return store(request).list_rounds()

    @router.post("/api/collab/rounds", status_code=201)
    async def create_round(request: Request, name: str = Query(min_length=1, max_length=80)):
        if not request.state.is_admin:
            raise CollaborationError(403, "only the owner can upload rounds")
        return store(request).import_round(name, await request.body(), request.state.login)

    @router.put("/api/collab/rounds/{round_id}/active")
    async def activate(round_id: int, request: Request):
        if not request.state.is_admin:
            raise CollaborationError(403, "only the owner can activate rounds")
        return store(request).activate_round(round_id)

    @router.get("/api/collab/rounds/{round_id}")
    async def round_snapshot(round_id: int, request: Request):
        snapshot = store(request).get_round(round_id)
        etag = snapshot.pop("etag")
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers={"ETag": etag})
        return JSONResponse(snapshot, headers={"ETag": etag})

    @router.patch("/api/collab/queries/{query_id}")
    async def patch_query(query_id: int, patch: QueryPatch, request: Request):
        fields = patch.model_fields_set
        kwargs = {}
        if "assignee" in fields:
            kwargs["assignee"] = patch.assignee
        if "answers" in fields:
            kwargs["answers"] = patch.answers
        if "event_count" in fields:
            if not request.state.is_admin:
                raise CollaborationError(403, "only the owner can set the TRAKE event count")
            kwargs["event_count"] = patch.event_count
        return store(request).update_query(
            query_id, patch.expected_version, request.state.login,
            request.app.state.access_config.allowed_users, **kwargs,
        )

    @router.put("/api/collab/queries/{query_id}/review")
    async def review_query(query_id: int, patch: ReviewPatch, request: Request):
        return store(request).set_review(
            query_id, patch.expected_version, request.state.login, patch.reviewed
        )

    @router.get("/api/collab/rounds/{round_id}/export-preview")
    async def export_preview(round_id: int, request: Request):
        return store(request).export_preview(round_id)

    @router.get("/api/collab/rounds/{round_id}/export")
    async def export(round_id: int, request: Request, allow_incomplete: bool = False):
        payload, filename = store(request).build_export(round_id, allow_incomplete)
        return Response(
            payload, media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
