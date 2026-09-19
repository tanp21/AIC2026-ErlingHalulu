"""Agent results -> the organisers' CSV/zip submission format (RULES_GUIDELINES.md)."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile

MAX_ROWS = 100
MAX_ANSWER_CHARS = 100


def result_rows(result: dict, max_rows: int = MAX_ROWS) -> list[list[str]]:
    """Rows for one query from `AgentResult.to_dict()`; no header, best first."""
    kind = result["kind"]
    if kind == "trake":
        rows, seen = [], set()
        for sequence in result.get("sequences", []):
            key = (sequence["video_id"], tuple(sequence["frames"]))
            if key not in seen:
                seen.add(key)
                rows.append([sequence["video_id"], *map(str, sequence["frames"])])
        return rows[:max_rows]
    rows, seen = [], set()
    answer = (result.get("answer") or "").strip()[:MAX_ANSWER_CHARS]
    for candidate in result.get("ranking", []):
        key = (candidate["video_id"], candidate["frame_idx"])
        if key in seen:
            continue
        seen.add(key)
        row = [candidate["video_id"], str(candidate["frame_idx"])]
        if kind == "qa":
            row.append(answer)
        rows.append(row)
    return rows[:max_rows]


def to_csv(rows: list[list[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def write_submission(results: list[dict], directory: Path, max_rows: int = MAX_ROWS) -> Path:
    """`<directory>/submission/<query>.csv` for each result, zipped next to it."""
    directory = Path(directory)
    folder = directory / "submission"
    folder.mkdir(parents=True, exist_ok=True)
    archive_path = directory / "submission.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for result in results:
            text = to_csv(result_rows(result, max_rows))
            (folder / f"{result['query_id']}.csv").write_text(text, encoding="utf-8", newline="")
            archive.writestr(f"submission/{result['query_id']}.csv", text)
    return archive_path
