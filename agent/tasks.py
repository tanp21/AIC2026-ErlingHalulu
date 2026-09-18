"""Competition query files and answer keys (see RULES_GUIDELINES.md)."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import io
from pathlib import Path
import re
import zipfile

KIND_PATTERN = re.compile(r"-(kis|qa|trake)$", re.IGNORECASE)
# "E1: ...", "E2 ...", "Cảnh 1: ..." -- one line per TRAKE event.
EVENT_LINE = re.compile(r"^\s*(?:E\s*\d+|Cảnh\s*\d+)\s*[:.)-]?\s+\S", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class Task:
    query_id: str
    kind: str
    text: str

    @property
    def event_count(self) -> int | None:
        """Events a TRAKE answer row must contain, when the query lists them."""
        if self.kind != "trake":
            return None
        count = len(EVENT_LINE.findall(self.text))
        return count or None


def task_from_text(query_id: str, text: str, kind: str | None = None) -> Task:
    match = KIND_PATTERN.search(query_id)
    kind = (kind or (match.group(1) if match else "kis")).lower()
    return Task(query_id, kind, text.strip())


def _query_sort_key(query_id):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", query_id)]


def load_tasks(path) -> list[Task]:
    """One `.txt` file, a directory of them, or the organisers' zip."""
    path = Path(path)
    tasks = []
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.lower().endswith(".txt") and not name.endswith("/"):
                    text = archive.read(name).decode("utf-8-sig")
                    tasks.append(task_from_text(Path(name).stem, text))
    elif path.is_dir():
        for file in path.glob("*.txt"):
            tasks.append(task_from_text(file.stem, file.read_text(encoding="utf-8-sig")))
    else:
        tasks.append(task_from_text(path.stem, path.read_text(encoding="utf-8-sig")))
    return sorted(tasks, key=lambda task: _query_sort_key(task.query_id))


@dataclass(frozen=True)
class GoldRow:
    video_id: str
    frames: tuple[int, ...]
    answer: str | None = None


def _gold_row(kind, fields):
    fields = [field.strip() for field in fields]
    video_id = fields[0]
    if kind == "qa":
        return GoldRow(video_id, (int(fields[1]),), fields[2] if len(fields) > 2 else None)
    frames = tuple(int(field) for field in fields[1:] if field)
    return GoldRow(video_id, frames if kind == "trake" else frames[:1])


def load_answers(path) -> dict[str, list[GoldRow]]:
    """Answer keys as `{query_id: [GoldRow, ...]}`.

    Accepts the single-CSV form (`query-id,video,frame,...`) or a zip of
    `submission/<query-id>.csv` files, where several KIS rows list the accepted
    frames of one clip.
    """
    path = Path(path)
    answers: dict[str, list[GoldRow]] = {}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".csv"):
                    continue
                query_id = Path(name).stem
                kind = task_from_text(query_id, "").kind
                text = archive.read(name).decode("utf-8-sig")
                rows = [row for row in csv.reader(io.StringIO(text)) if row and row[0].strip()]
                answers[query_id] = [_gold_row(kind, row) for row in rows]
        return answers
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or not row[0].strip():
                continue
            query_id = row[0].strip()
            kind = task_from_text(query_id, "").kind
            answers.setdefault(query_id, []).append(_gold_row(kind, row[1:]))
    return answers
