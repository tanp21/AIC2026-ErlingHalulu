"""The autonomous judge: every candidate is looked at as a moment, not as one keyframe.

`Inspector.verify` shows a vision model a candidate's single keyframe, five candidates to a
call. That was Gemini's job. The team moved judging to `qwen/qwen3.7-flash` on OpenRouter
(cost), and with it to the shape the Search tab's "Check by AI" already uses
(`agent/checker.py`): ONE call per candidate, holding its own keyframe plus the keyframes of
the shots around it, the speech and on-screen text of the window, and the query split into
checkable details. A query is often a short story over several shots; a single frame shows one
of them, and judging a frame against the whole story is how a right moment scored 5.

Measured live on a cooking query (2026-09-25): a call looks at ~9 keyframes of a ±60 s window
and takes 13-16 s; six in flight take the same wall time as one. So the fan-out is wide and the
run's time budget is longer than Gemini's fast mode (see `STORY_BUDGETS` in `agent/orchestrator`).

The same call also reads a Q&A answer off the moment (`CHECK_ANSWER_RULE`), so every judged
candidate of a Q&A query carries a reading; the orchestrator counts the readings of the
best-scored moments as votes once judging is over (`_Run._read_answer`).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import logging

from .checker import CheckError, CheckQuery, CheckTarget, check_moment
from .evidence import EvidenceStore, QueryDetails
from .pool import Verdict
from .web import is_unknown

logger = logging.getLogger("agent")

#: Seconds either side of the moment. ±60 s holds the shots a story query's other beats sit in
#: without reaching the next story of a news bulletin; the checker's ±300 s is for a person
#: asking about a whole segment.
JUDGE_WINDOW_SECONDS = 60.0
#: `(dense, most, count)` for the shot walk: every keyframe (up to 3) of the nearest two shots a
#: side, one of each further shot, at most 8 frames a side -- about 9-15 images a call.
JUDGE_CONTEXT = (2, 8, 3)
#: Calls in flight. OpenRouter serves them in parallel: six took the time of one.
JUDGE_WORKERS = 8
#: Seconds a round of judging may take. A call is 13-16 s; what drags past this is a slow
#: compute host serving the keyframes (measured: 16 candidates took 190 s while the venue
#: tunnel dropped requests), and a candidate left unjudged costs less than the whole run.
JUDGE_DEADLINE_SECONDS = 60.0


def judge_name(client) -> str:
    """A short name for who judged, for the UI and the run JSON: `qwen`, `gemini`, ..."""
    model = str(getattr(client, "model", "") or "").lower()
    family = model.split("/")[-1].split("-")[0].split(":")[0]
    return "".join(ch for ch in family if ch.isalpha()) or "model"


class MomentJudge:
    """`verify()` with `Inspector.verify`'s contract, one checker call per candidate."""

    def __init__(self, client, search, *, workers: int = JUDGE_WORKERS,
                 window: float = JUDGE_WINDOW_SECONDS, context: tuple[int, int, int] = JUDGE_CONTEXT,
                 store: EvidenceStore | None = None, deadline: float = JUDGE_DEADLINE_SECONDS):
        self.client = client
        self.search = search
        self.workers = max(1, int(workers))
        self.window = float(window)
        self.context = context
        self.store = store or EvidenceStore(search)
        self.name = judge_name(client)
        self.deadline = float(deadline)

    def query(self, task, plan, question=None) -> CheckQuery:
        """What the judge is told: the organisers' query, the frame to find and, for Q&A, the question."""
        visual = ""
        if plan is not None and plan.scenes:
            scene = plan.scenes[min(max(plan.target_scene, 0), len(plan.scenes) - 1)]
            visual = scene.visual_en[0] if scene.visual_en else ""
        if question is None and task.kind == "qa":
            question = (plan.question_vi if plan is not None else None) or task.text
        return CheckQuery(visual_text=visual, kind=task.kind, question=question, query_text=task.text)

    def verify(self, task, plan, candidates, question=None,
               details: QueryDetails | list | None = None) -> tuple[dict[str, Verdict], list[str]]:
        """`({cid: Verdict}, errors)`; a candidate whose call failed is left out, like a failed chunk."""
        if not candidates:
            return {}, []
        query = self.query(task, plan, question)
        entries = details if details is not None else QueryDetails()

        def one(candidate):
            target = CheckTarget(candidate.video_id, int(candidate.frame_idx), float(candidate.time))
            for attempt in range(2):
                try:
                    return candidate.cid, check_moment(
                        self.client, self.search, self.store, query=query, target=target, details=entries,
                        window_seconds=self.window, context=self.context)
                except CheckError as exc:
                    # Measured on the first runs: about one reply in fifteen came back as JSON the
                    # parser could not repair. The same call again is usually fine.
                    if attempt or "invalid JSON" not in str(exc):
                        raise

        verdicts, errors = {}, []
        # Not a `with` block: that would wait for the calls the deadline gave up on.
        pool = ThreadPoolExecutor(min(self.workers, len(candidates)))
        try:
            futures = {pool.submit(one, candidate): candidate for candidate in candidates}
            done, late = wait(futures, timeout=self.deadline)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        for future, candidate in futures.items():
            if future in late:
                errors.append(f"{candidate.cid}: no verdict within {self.deadline:.0f} s")
                continue
            try:
                cid, checked = future.result()
            except Exception as exc:  # one moment unjudged is a gap, not a failed round
                errors.append(f"{type(exc).__name__}: {exc}"[:200])
                continue
            verdicts[cid] = self._verdict(checked, asked=query.asked is not None)
        return verdicts, errors

    def _verdict(self, checked: dict, *, asked: bool) -> Verdict:
        overall = checked.get("verdict") or {}
        answer = checked.get("answer") if asked else None
        text = (answer or {}).get("text") or None
        if is_unknown(text):
            text = None  # "không rõ" is the judge saying this moment does not answer it
        details = checked.get("details") or []
        checks = []
        for check in checked.get("checks") or []:
            index = check.get("detail")
            if not isinstance(index, int) or not 0 <= index < len(details):
                continue
            checks.append({"text": details[index]["text"], "status": check.get("status"),
                           "source": check.get("source"), "note": check.get("note") or "",
                           "quote": check.get("quote") or ""})
        return Verdict(
            score=float(overall.get("score") or 0.0), reason=str(overall.get("reason") or ""),
            answer=text, by=self.name,
            answer_confidence=float((answer or {}).get("confidence") or 0.0) if text else None,
            checks=checks, frames_seen=checked.get("frames_seen"),
        )
