"""Score saved agent results against an answer key.

The organisers do not publish their frame tolerance, so a KIS/QA row counts as a
hit when its video matches and its frame lies within `tolerance` frames of the gold
range (several gold rows for one query mark the accepted clip's extent). Treat the
numbers as a relative signal between agent versions, not a leaderboard estimate.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata

from .submission import result_rows

DEFAULT_TOLERANCE = 50
CUTOFFS = (1, 5, 20, 100)


@dataclass(frozen=True)
class QueryScore:
    query_id: str
    kind: str
    hit_rank: int | None
    video_rank: int | None
    answer_ok: bool | None
    answer: str | None
    gold_answer: str | None


def normalize_answer(text: str | None) -> str:
    text = unicodedata.normalize("NFC", (text or "")).casefold()
    text = re.sub(r"[\"'“”‘’.,;:!?()\[\]]", " ", text)
    return " ".join(text.split())


def _frame_ok(frame, gold_frames, tolerance):
    return min(gold_frames) - tolerance <= frame <= max(gold_frames) + tolerance


def score_result(result: dict, gold_rows, tolerance: int = DEFAULT_TOLERANCE) -> QueryScore:
    kind = result["kind"]
    rows = result_rows(result)
    gold_videos = {gold.video_id for gold in gold_rows}
    hit_rank = video_rank = None
    for rank, row in enumerate(rows, 1):
        video_id = row[0]
        if video_id in gold_videos and video_rank is None:
            video_rank = rank
        if hit_rank is not None:
            continue
        if kind == "trake":
            frames = [int(value) for value in row[1:]]
            if any(
                gold.video_id == video_id and len(gold.frames) == len(frames)
                and all(abs(a - b) <= tolerance for a, b in zip(frames, gold.frames))
                for gold in gold_rows
            ):
                hit_rank = rank
        else:
            frames = [gold.frames[0] for gold in gold_rows if gold.video_id == video_id]
            if frames and _frame_ok(int(row[1]), frames, tolerance):
                hit_rank = rank
    answer_ok = gold_answer = None
    if kind == "qa":
        gold_answer = next((gold.answer for gold in gold_rows if gold.answer), None)
        answer_ok = normalize_answer(result.get("answer")) == normalize_answer(gold_answer)
    return QueryScore(result["query_id"], kind, hit_rank, video_rank, answer_ok,
                      result.get("answer"), gold_answer)


def summarize(scores: list[QueryScore]) -> dict:
    summary = {"queries": len(scores)}
    for label, subset in (("all", scores), *(
        (kind, [s for s in scores if s.kind == kind]) for kind in ("kis", "qa", "trake")
    )):
        if not subset:
            continue
        entry = {"n": len(subset)}
        for cutoff in CUTOFFS:
            entry[f"R@{cutoff}"] = round(
                sum(1 for s in subset if s.hit_rank and s.hit_rank <= cutoff) / len(subset), 3)
        entry["video_R@1"] = round(sum(1 for s in subset if s.video_rank == 1) / len(subset), 3)
        entry["MRR"] = round(sum(1 / s.hit_rank for s in subset if s.hit_rank) / len(subset), 3)
        answered = [s for s in subset if s.answer_ok is not None]
        if answered:
            entry["answer_acc"] = round(sum(s.answer_ok for s in answered) / len(answered), 3)
        summary[label] = entry
    return summary
