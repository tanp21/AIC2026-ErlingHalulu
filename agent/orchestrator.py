"""One query, end to end: plan -> fan-out -> verify -> (rewrite | agent loop) -> ranking.

Three modes share every step. `fast` (the default, ~10-20 s a query) runs round 1,
at most one DeepSeek rewrite round without thinking, and does every Gemini call it
can in parallel, with web lookups in the background. `deep` hands a weak round 1 to
the multi-step DeepSeek tool loop below (70-90 s on average, several minutes on the
hardest queries) for when there is time to spare. `auto` is fast, then the deep loop
only for a query fast mode left unsure.

Round 1 is deterministic (the plan's channels, run as one server batch, then the top
candidates verified by Gemini) because it is where most queries are won and it must
be fast. Only when round 1 does not produce a confident match does DeepSeek take
over with tools, reformulating and drilling into promising videos -- the loop the VBS
agent literature found most useful (rewrite on failure, search inside candidate
videos, verify before committing).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field, replace
import json
import logging
import re
import time

from .evaluate import normalize_answer
from .llm import LLMError
from .plan import (
    TRAKE_TIGHT_GAP, Plan, analysis_messages, expected_events, initial_queries, parse_plan,
    transcript_text,
)
from .pool import VERIFIED_MATCH, Candidate, CandidatePool, Sequence, local_path
from .prompts import AGENT_SYSTEM, LOOKUP_PROMPT, REWRITE_SYSTEM
from .tasks import Task
from .vision import Inspector, caption_at, transcript_window
from .web import cooking_lookup, fact_lookup, is_cooking, recipe_lookup, similar, tagged

logger = logging.getLogger("agent")
VIDEO_ID = re.compile(r"^[A-Z]\d{2}_V\d{3}$")
OBSERVE_TOP = 12
# A reading whose evidence is the narration: its names went through ASR.
SPOKEN_EVIDENCE = re.compile(r"lời|thoại|nói|phát thanh|thuyết minh|giọng|transcript|speech", re.IGNORECASE)


@dataclass
class AgentOptions:
    mode: str = "fast"  # fast | deep | auto
    max_steps: int = 6
    time_budget: float = 300.0
    verify_top: int = 12
    verify_chunk: int = 8
    verify_per_step: int = 8
    accept_score: float = 9.0  # fast: a verified match this good skips the rewrite round
    auto_web: bool = True  # background web searches for cooking queries and Q&A answers
    escalate_below: float = 7.0  # auto: a best verdict under this goes on to the deep loop
    escalate_budget: float = 120.0  # auto: total seconds allowed once escalated
    early_stop_score: float = 9.0
    falsify: bool = True
    answer_moments: int = 3
    trake_check: int = 6
    trake_refine: int = 2
    neighbour_radius: int = 6
    max_rows: int = 100
    plan_thinking: bool = False
    spoken_numbers: bool = False

    @classmethod
    def for_mode(cls, mode="fast", **overrides):
        """Defaults tuned per mode; explicit overrides win.

        One Gemini call takes ~4 s whether it judges 2 or 8 frames, so fast mode
        splits work into more, smaller calls run side by side instead of fewer
        large ones run in turn.
        """
        defaults = FAST_DEFAULTS if mode in ("fast", "auto") else {}
        return cls(mode=mode, **{**defaults, **overrides})


FAST_DEFAULTS = dict(
    time_budget=20.0, verify_top=15, verify_chunk=5,
    answer_moments=2, trake_check=8, trake_refine=1,
)


@dataclass
class AgentResult:
    task: Task
    plan: Plan | None = None
    ranking: list[Candidate] = field(default_factory=list)
    sequences: list[Sequence] = field(default_factory=list)
    answer: str | None = None
    answer_evidence: str | None = None
    lookups: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    notes: str = ""
    elapsed: float = 0.0
    error: str | None = None

    def to_dict(self):
        def candidate(c):
            return {
                "id": c.cid, "video_id": c.video_id, "frame_idx": c.frame_idx,
                "time": round(c.time, 3), "fused": round(c.fused, 5),
                "verdict": None if c.verdict is None else vars(c.verdict),
                "evidence": [vars(e) for e in c.evidence.values()],
            }

        return {
            "query_id": self.task.query_id, "kind": self.task.kind, "query": self.task.text,
            "plan": self.plan.to_dict() if self.plan else None,
            "answer": self.answer, "answer_evidence": self.answer_evidence,
            "ranking": [candidate(c) for c in self.ranking],
            "sequences": [vars(s) for s in self.sequences],
            "lookups": self.lookups, "trace": self.trace, "notes": self.notes,
            "elapsed": round(self.elapsed, 1), "error": self.error,
        }


TOOLS = [
    {"type": "function", "function": {
        "name": "search",
        "description": (
            "Run up to 8 retrieval queries as one batch. New results merge into the "
            "candidate list and the best unverified candidates are verified automatically."
        ),
        "parameters": {"type": "object", "properties": {"queries": {
            "type": "array", "items": {"type": "object", "properties": {
                "kind": {"type": "string",
                         "enum": ["visual", "audio", "caption", "ocr", "temporal", "similar"]},
                "text": {"type": "string", "description": (
                    "visual: English caption of one frame; audio: Vietnamese spoken phrase; "
                    "caption: English sentence as an offline shot caption would word it; "
                    "ocr: exact on-screen text")},
                "paraphrases": {"type": "array", "items": {"type": "string"},
                                "description": "visual only: up to 4 same-meaning rewordings of text"},
                "method": {"type": "string", "enum": ["keyword", "semantic"],
                           "description": "audio only; keyword (BM25) by default"},
                "events": {"type": "array", "items": {"type": "string"},
                           "description": "temporal only: 2-6 English captions in order"},
                "audio_text": {"type": "string", "description": "temporal only: optional spoken phrase"},
                "candidate_id": {"type": "string", "description": "similar only: search from this candidate's frame"},
                "video_ids": {"type": "array", "items": {"type": "string"},
                              "description": "visual/similar/caption only: search inside these videos"},
            }, "required": ["kind"]},
        }}, "required": ["queries"]},
    }},
    {"type": "function", "function": {
        "name": "inspect",
        "description": "Have Gemini look at specific candidates, optionally with a specific question.",
        "parameters": {"type": "object", "properties": {
            "candidate_ids": {"type": "array", "items": {"type": "string"}},
            "question": {"type": "string"},
        }, "required": ["candidate_ids"]},
    }},
    {"type": "function", "function": {
        "name": "neighbours",
        "description": (
            "Add the keyframes just before and after a candidate as new candidates and "
            "verify them. Use to find the exact moment or a clearer view."
        ),
        "parameters": {"type": "object", "properties": {
            "candidate_id": {"type": "string"},
            "radius": {"type": "integer", "minimum": 1, "maximum": 10},
        }, "required": ["candidate_id"]},
    }},
    {"type": "function", "function": {
        "name": "read_transcript",
        "description": "Read the speech transcript of a video around a time.",
        "parameters": {"type": "object", "properties": {
            "video_id": {"type": "string"},
            "time": {"type": "number"},
            "seconds": {"type": "number", "description": "half-window, default 60"},
        }, "required": ["video_id", "time"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Google Search (via Gemini) to resolve names, facts or cross-check an answer.",
        "parameters": {"type": "object", "properties": {
            "question": {"type": "string"},
        }, "required": ["question"]},
    }},
    {"type": "function", "function": {
        "name": "finish",
        "description": "Stop and submit the final ranking (best first).",
        "parameters": {"type": "object", "properties": {
            "ranking": {"type": "array", "items": {"type": "string"}},
            "answer": {"type": "string", "description": "qa only: the literal answer"},
            "notes": {"type": "string", "description": "short Vietnamese note for the human checker"},
        }, "required": ["ranking"]},
    }},
]


FINISH_TOOL = next(tool for tool in TOOLS if tool["function"]["name"] == "finish")


class Agent:
    def __init__(self, deepseek, gemini, search, options: AgentOptions | None = None,
                 clock=time.monotonic):
        self.deepseek = deepseek
        self.gemini = gemini
        self.search = search
        self.options = options or AgentOptions()
        self.inspector = Inspector(gemini, search, chunk_size=self.options.verify_chunk)
        self.clock = clock

    def run(self, task: Task) -> AgentResult:
        return _Run(self, task).execute()


class _Run:
    def __init__(self, agent: Agent, task: Task):
        self.agent = agent
        self.task = task
        self.options = agent.options
        self.pool = CandidatePool()
        self.result = AgentResult(task)
        self.started = agent.clock()
        self.plan: Plan | None = None
        self.searched: set = set()
        self.cues: dict[str, list] = {}
        self.shot_captions: dict[str, list] = {}
        self.finished: dict | None = None
        self.step = 0
        self.cooking = agent.options.auto_web and is_cooking(task.text)
        self.background = ThreadPoolExecutor(max_workers=3)
        self.pending: list = []
        self.budget = agent.options.time_budget

    # -- bookkeeping ---------------------------------------------------------------
    def elapsed(self):
        return self.agent.clock() - self.started

    def over_budget(self):
        return self.elapsed() > self.budget

    def remaining(self):
        return self.budget - self.elapsed()

    def log(self, tool, args, summary, started):
        entry = {"step": self.step, "tool": tool, "args": args, "summary": summary,
                 "seconds": round(self.agent.clock() - started, 2)}
        self.result.trace.append(entry)
        logger.info("%s step=%d %s %s", self.task.query_id, self.step, tool, summary)

    # -- main flow -----------------------------------------------------------------
    def execute(self) -> AgentResult:
        try:
            if self.options.mode in ("fast", "auto"):
                self._fast()
                if (self.options.mode == "auto" and self.task.kind != "trake"
                        and self._best_score() < self.options.escalate_below):
                    self.budget = self.options.escalate_budget
                    self.log("escalate", {}, f"best={self._best_score():.0f} -> deep loop", self.agent.clock())
                    self._loop()
            else:
                self._plan()
                self._round_one()
                if not self._confident():
                    self._loop()
            self._finalize()
        except Exception as exc:
            logger.exception("%s failed", self.task.query_id)
            self.result.error = f"{type(exc).__name__}: {exc}"
            if self.result.plan is not None and not self.result.ranking:
                self.result.ranking = self.pool.ordered()
        finally:
            # Lookups still running are dropped, not waited for: they only ever add hints.
            self._collect_lookups(0)
            self.background.shutdown(wait=False, cancel_futures=True)
        self.result.elapsed = self.elapsed()
        return self.result

    # -- fast mode -----------------------------------------------------------------
    def _fast(self):
        """Plan, one batch, parallel verification, and one rewrite only if needed.

        Web lookups (the plan's, and for cooking queries a recipe search from the
        raw description, started before planning) run in the background and are
        waited for only where their result is used.
        """
        if self.cooking:
            self._lookup_later(cooking_lookup(self.task.text, None), "món ăn (Món ngon mỗi ngày)")
        self._plan()
        for question in self.plan.lookups:
            self._lookup_later(LOOKUP_PROMPT.format(question=question), question)
        self._run_search(initial_queries(self.plan, self.task, self.options.spoken_numbers), "round1")
        if self.task.kind == "trake":
            return  # paths are checked event by event in _finalize_sequences
        self._verify(self.pool.to_verify(self.options.verify_top))
        best = next((c for c in self.pool.ordered() if c.verdict), None)
        # A 9-10 must survive the falsifier (~5 s): on round 2, fast mode's misses were
        # mostly confident look-alikes it would have questioned. ~8 s is one DeepSeek
        # call, one batch and one parallel verification.
        confident = (best is not None and best.verdict.score >= self.options.accept_score
                     and not self._falsified(best))
        if not confident and self.remaining() > 8:
            self._rewrite_round()

    def _best_score(self):
        return max((c.verdict.score for c in self.pool.candidates.values() if c.verdict), default=0.0)

    def _lookup_later(self, prompt, label):
        self.pending.append(self.background.submit(self._lookup, label, prompt))

    def _collect_lookups(self, timeout):
        """Move finished background lookups into the result, waiting up to `timeout` s."""
        if not self.pending:
            return
        if timeout > 0:
            wait(self.pending, timeout=timeout)
        for future in [f for f in self.pending if f.done()]:
            self.pending.remove(future)
            if not future.cancelled() and future.exception() is None:
                self.result.lookups.append(future.result())

    def _web_notes(self):
        return "\n".join(
            f"- {entry['question']}: {entry.get('answer') or entry.get('error')}"
            for entry in self.result.lookups
        ) or "(none)"

    def _rewrite_round(self):
        """DeepSeek (no thinking, one call) rewrites what round 1 missed; runs and verifies it."""
        self.step = 1
        started = self.agent.clock()
        # A cooking lookup started before planning is usually back by now; its dish
        # name is what the transcript channel needs.
        self._collect_lookups(min(3.0, max(0.0, self.remaining() - 10)))
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": (
                f"Task type: {self.task.kind}\nQuery:\n{self.task.text}\n\n"
                f"Round-1 plan:\n{json.dumps(self.plan.to_dict(), ensure_ascii=False)}\n\n"
                f"Web notes:\n{self._web_notes()}\n\nRound-1 result:\n{self.observation()}"
            )},
        ]
        try:
            data = self.agent.deepseek.chat_json(messages, thinking=False, max_tokens=2000)
        except LLMError as exc:
            data = {}
            self.log("rewrite", {}, f"failed: {exc}", started)
        raw_queries = data.get("queries") if isinstance(data, dict) else None
        queries = [self._server_query(raw, index)
                   for index, raw in enumerate((raw_queries or [])[:6]) if isinstance(raw, dict)]
        queries = [query for query in queries if query]
        # Right video, wrong moment is the commonest miss: look for the target scene
        # inside the videos round 1 favoured.
        caption, *paraphrases = self.plan.scenes[self.plan.target_scene].visual_en
        scoped = {"kind": "visual", "id": "s1.scoped", "text": caption,
                  "video_ids": self.pool.video_ranking()[:5], "limit": 100}
        if paraphrases:
            scoped["paraphrases"] = paraphrases[:4]
        queries.append(scoped)
        self.log("rewrite", {"queries": queries},
                 "; ".join(str(q.get("text") or q.get("events"))[:60] for q in queries), started)
        self._run_search(queries, "rewrite")
        fresh = self._best_neighbours()
        self._verify(fresh + self.pool.to_verify(self.options.verify_top - len(fresh)))

    def _best_neighbours(self, radius=2):
        """Unverified keyframes next to a promising-but-unsure match, for the exact moment."""
        best = next((c for c in self.pool.ordered() if c.verdict), None)
        if best is None or not 5 <= best.verdict.score < self.options.accept_score:
            return []
        try:
            frames = self.agent.search.neighbours(best.video_id, best.frame_idx, radius)
        except Exception:
            return []
        added = [self.pool.add_frame(f["video_id"], f["frame_idx"], f["time"], source=f"n.{best.cid}")
                 for f in frames]
        return [c for c in added if c.verdict is None][:4]

    def _plan(self):
        started = self.agent.clock()
        messages = analysis_messages(self.task)
        last_error = None
        for _ in range(2):
            try:
                data = self.agent.deepseek.chat_json(messages, thinking=self.options.plan_thinking)
                self.plan = parse_plan(data, self.task)
                break
            except (LLMError, ValueError) as exc:
                last_error = exc
        if self.plan is None:
            raise RuntimeError(f"planning failed: {last_error}")
        self.result.plan = self.plan
        self.log("plan", {}, f"{len(self.plan.scenes)} scenes, "
                 f"{len(self.plan.transcript_vi)} transcript, {len(self.plan.caption_en)} caption, "
                 f"{len(self.plan.ocr)} ocr, {len(self.plan.lookups)} lookups", started)

    def _round_one(self):
        queries = initial_queries(self.plan, self.task, self.options.spoken_numbers)
        with ThreadPoolExecutor(max_workers=3) as pool:
            lookups = [pool.submit(self._lookup, question) for question in self.plan.lookups]
            self._run_search(queries, "round1")
            for future in lookups:
                self.result.lookups.append(future.result())
        self._verify(self.pool.to_verify(self.options.verify_top))

    def _lookup(self, question, prompt=None):
        started = self.agent.clock()
        try:
            answer = self.agent.gemini.web_search(prompt or LOOKUP_PROMPT.format(question=question))
        except LLMError as exc:
            entry = {"question": question, "error": str(exc)}
        else:
            entry = {"question": question, "answer": answer.text,
                     "sources": [uri for _, uri in answer.sources]}
        self.log("web_search", {"question": question},
                 (entry.get("answer") or entry.get("error", ""))[:120], started)
        return entry

    def _confident(self):
        best = next((c for c in self.pool.ordered() if c.verdict), None)
        if best is None or best.verdict.score < self.options.early_stop_score:
            return False
        if self.task.kind == "qa" and not best.verdict.answer:
            return False
        if self.task.kind == "trake":
            return False
        return not self._falsified(best)

    def _falsified(self, candidate):
        """Skeptical re-check before stopping early; a rejection sends the run to the loop.

        On the first live round-1 run, 2 of 13 early stops were confident wrong
        matches (a different lion dance, a different bridge). A separate falsifier
        caught a third of wrong submissions in the VBS agent study. Its first live
        run here rejected two correct matches over details a few keyframes cannot
        show (finishing order), hence the prompt's not_visible rule and a rejection
        that never lowers the score.
        """
        if not self.options.falsify:
            return False
        started = self.agent.clock()
        try:
            neighbours = self.agent.search.neighbours(candidate.video_id, candidate.frame_idx, 2)
            rejected, reason, contradicted = self.agent.inspector.falsify(
                self.task, candidate, neighbours)
        except Exception as exc:  # no second opinion is not a rejection
            self.log("falsify", {"candidate": candidate.cid}, f"failed: {exc}", started)
            return False
        if rejected:
            # Keep the score: on the live round-1 run both rejections hit correct matches
            # (a finish order no single keyframe can show). A rejection only denies the
            # early stop; the objection rides along for the loop and the human to weigh.
            objection = "; ".join(contradicted) or reason
            candidate.verdict.reason = f"{candidate.verdict.reason} | Phản biện: {objection}"[:300]
        self.log("falsify", {"candidate": candidate.cid},
                 f"{'reject' if rejected else 'accept'}: {contradicted or reason}", started)
        return rejected

    # -- tools -----------------------------------------------------------------------
    def _run_search(self, queries, label):
        started = self.agent.clock()
        fresh, repeated = [], []
        for query in queries:
            signature = json.dumps({k: v for k, v in query.items() if k not in ("id", "limit")},
                                   sort_keys=True, ensure_ascii=False)
            if signature in self.searched:
                repeated.append(query.get("text") or query.get("events") or query["kind"])
            else:
                self.searched.add(signature)
                fresh.append(query)
        if not fresh:
            self.log("search", {"label": label}, "all queries already run", started)
            return "All of these queries were already run; change the wording or channel."
        results = self.agent.search.search(fresh)
        created = self.pool.add_results(results)
        errors = {qid: r["error"] for qid, r in results.items() if "error" in r}
        counts = {qid: len(r.get("hits", [])) for qid, r in results.items() if "hits" in r}
        self.log("search", {"label": label, "queries": fresh},
                 f"{len(fresh)} queries, {len(created)} new candidates, errors={errors}", started)
        note = f"{len(fresh)} queries run ({counts}); {len(created)} new candidates."
        if repeated:
            note += f" Skipped as already run: {repeated}."
        if errors:
            note += f" Errors: {errors}"
        return note

    def _verify(self, candidates, question=None):
        if not candidates:
            return
        started = self.agent.clock()
        verdicts, errors = self.agent.inspector.verify(self.task, self.plan, candidates, question)
        for cid, verdict in verdicts.items():
            candidate = self.pool.get(cid)
            previous = candidate.verdict
            if question and previous is not None:
                # A follow-up question is judged against that question, not the whole
                # query; keep the match score and take only its reading.
                verdict.score = previous.score
                verdict.answer = verdict.answer or previous.answer
            candidate.verdict = verdict
        best = max((v.score for v in verdicts.values()), default=None)
        self.log("verify", {"ids": [c.cid for c in candidates], "question": question},
                 f"{len(verdicts)} verdicts, best={best}"
                 + (f", errors={[e[:120] for e in errors]}" if errors else ""), started)

    def _server_query(self, raw, index):
        kind = raw.get("kind")
        qid = f"s{self.step}.{index}"
        video_ids = [v for v in (raw.get("video_ids") or []) if isinstance(v, str) and VIDEO_ID.match(v)][:50]
        text = " ".join(str(raw.get("text") or "").split())[:300]
        if kind == "visual" and text:
            query = {"kind": "visual", "id": qid, "text": text, "limit": 100 if video_ids else 200}
            paraphrases = [" ".join(str(p).split())[:300] for p in raw.get("paraphrases") or []
                           if str(p).strip()][:4]
            if paraphrases:
                query["paraphrases"] = paraphrases
            if video_ids:
                query["video_ids"] = video_ids
            return query
        if kind == "audio" and text:
            method = raw.get("method") if raw.get("method") in ("keyword", "semantic") else "keyword"
            return {"kind": "audio", "id": qid, "text": transcript_text(text, self.options.spoken_numbers),
                    "method": method, "limit": 100}
        if kind == "ocr" and text:
            return {"kind": "ocr", "id": qid, "text": text, "limit": 50}
        if kind == "caption" and text:
            query = {"kind": "caption", "id": qid, "text": text, "limit": 100}
            if video_ids:
                query["video_ids"] = video_ids
            return query
        if kind == "temporal":
            events = [" ".join(str(e).split())[:300] for e in raw.get("events") or [] if str(e).strip()]
            if 2 <= len(events) <= 6:
                query = {"kind": "temporal", "id": qid, "events": events, "limit": 20,
                         "max_gap_seconds": 300}
                if raw.get("audio_text"):
                    query["audio_text"] = transcript_text(str(raw["audio_text"]), self.options.spoken_numbers)[:300]
                return query
        if kind == "similar":
            candidate = self.pool.get(str(raw.get("candidate_id", "")))
            if candidate is not None:
                query = {"kind": "similar", "id": qid, "video_id": candidate.video_id,
                         "frame_idx": candidate.frame_idx, "limit": 100}
                if video_ids:
                    query["video_ids"] = video_ids
                return query
        return None

    def tool_search(self, args):
        raw_queries = args.get("queries") or []
        queries, rejected = [], []
        for index, raw in enumerate(raw_queries[:8]):
            query = self._server_query(raw if isinstance(raw, dict) else {}, index)
            (queries if query else rejected).append(query or raw)
        if not queries:
            return f"No valid query. Rejected: {rejected}"
        note = self._run_search(queries, f"step{self.step}")
        self._verify(self.pool.to_verify(self.options.verify_per_step))
        if rejected:
            note += f" Rejected invalid queries: {rejected}"
        return note

    def tool_inspect(self, args):
        candidates = [self.pool.get(str(cid)) for cid in args.get("candidate_ids") or []]
        candidates = [c for c in candidates if c is not None][:16]
        if not candidates:
            return "No known candidate ids."
        self._verify(candidates, (args.get("question") or "").strip() or None)
        return "Inspected " + ", ".join(c.cid for c in candidates)

    def tool_neighbours(self, args):
        candidate = self.pool.get(str(args.get("candidate_id", "")))
        if candidate is None:
            return "Unknown candidate id."
        radius = min(max(int(args.get("radius") or self.options.neighbour_radius), 1), 10)
        frames = self.agent.search.neighbours(candidate.video_id, candidate.frame_idx, radius)
        added = {}
        for frame in frames:
            neighbour = self.pool.add_frame(frame["video_id"], frame["frame_idx"], frame["time"],
                                            source=f"n.{candidate.cid}")
            added[neighbour.cid] = neighbour
        self._verify([c for c in added.values() if c.verdict is None])
        return "Neighbour candidates: " + ", ".join(added)

    def tool_read_transcript(self, args):
        video_id = str(args.get("video_id", ""))
        if not VIDEO_ID.match(video_id):
            return "Invalid video id."
        if video_id not in self.cues:
            self.cues[video_id] = self.agent.search.transcript(video_id)
        seconds = min(max(float(args.get("seconds") or 60), 10), 300)
        return transcript_window(self.cues[video_id], float(args.get("time") or 0), seconds)

    def tool_web_search(self, args):
        entry = self._lookup(str(args.get("question", ""))[:500])
        self.result.lookups.append(entry)
        return entry.get("answer") or f"error: {entry.get('error')}"

    def tool_finish(self, args):
        self.finished = args
        return "Finished."

    # -- the agent loop ----------------------------------------------------------------
    def observation(self):
        lines = ["Candidates (current order, best first):"]
        for candidate in self.pool.ordered()[:OBSERVE_TOP]:
            sources = ", ".join(
                f"{e.kind}#{e.rank}" for e in sorted(candidate.evidence.values(), key=lambda e: e.rank)[:5]
            )
            verdict = candidate.verdict
            judged = (f"verdict={verdict.score:.0f} \"{verdict.reason}\""
                      + (f" answer=\"{verdict.answer}\"" if verdict and verdict.answer else "")
                      if verdict else "unverified")
            lines.append(f"{candidate.cid} {candidate.video_id} frame={candidate.frame_idx} "
                         f"t={candidate.time:.1f}s fused={candidate.fused:.4f} {judged} [{sources}]")
        lines.append("Top videos: " + ", ".join(self.pool.video_ranking()[:8]))
        if self.pool.sequences:
            best = sorted(self.pool.sequences, key=lambda s: s.rank)[:3]
            lines.append("Temporal paths: " + "; ".join(
                f"{s.video_id} frames={s.frames} candidates={s.candidate_ids}" for s in best))
        lines.append(f"Budget: step {self.step}/{self.options.max_steps}, "
                     f"{self.elapsed():.0f}s/{self.budget:.0f}s used.")
        return "\n".join(lines)

    def _initial_message(self):
        lookups = "\n".join(
            f"- {entry['question']}: {entry.get('answer') or entry.get('error')}"
            for entry in self.result.lookups
        ) or "(none)"
        return (
            f"Task type: {self.task.kind}\nQuery id: {self.task.query_id}\n"
            f"Query:\n{self.task.text}\n\n"
            f"Plan used for round 1:\n{json.dumps(self.plan.to_dict(), ensure_ascii=False)}\n\n"
            f"Web lookups:\n{lookups}\n\n"
            f"Round 1 result:\n{self.observation()}\n\n"
            "Continue with tools, or finish if the evidence is conclusive."
        )

    def _loop(self):
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": self._initial_message()},
        ]
        handlers = {
            "search": self.tool_search, "inspect": self.tool_inspect,
            "neighbours": self.tool_neighbours, "read_transcript": self.tool_read_transcript,
            "web_search": self.tool_web_search, "finish": self.tool_finish,
        }
        nudged = False
        while self.step < self.options.max_steps and not self.over_budget() and self.finished is None:
            self.step += 1
            started = self.agent.clock()
            message = self.agent.deepseek.chat(messages, tools=TOOLS)
            messages.append(message)
            calls = message.get("tool_calls") or []
            if not calls:
                if nudged:
                    self.log("loop", {}, "model stopped without finish", started)
                    break
                nudged = True
                messages.append({"role": "user", "content": "Call a tool, or call finish with your ranking."})
                continue
            for call in calls:
                name = call.get("function", {}).get("name", "")
                try:
                    args = json.loads(call.get("function", {}).get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                handler = handlers.get(name)
                call_started = self.agent.clock()
                try:
                    content = handler(args) if handler else f"Unknown tool {name}."
                except Exception as exc:  # a tool failure is an observation, not a crash
                    content = f"Tool error: {type(exc).__name__}: {exc}"
                if name not in ("search", "web_search"):  # these log themselves
                    self.log(name, args, content[:160], call_started)
                if name != "finish":
                    content = f"{content}\n\n{self.observation()}"
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})
        if self.finished is None and self.step:
            self._wrap_up(messages)

    def _wrap_up(self, messages):
        """Out of budget without finish: one last turn that may only finish.

        Without it, what the loop learned but never committed (a name found by
        web_search, the model's own ranking) would be lost to the fallback ordering.
        """
        started = self.agent.clock()
        request = "The budget is spent. Call finish now with your best ranking"
        if self.task.kind == "qa":
            request += " and the literal answer"
        messages.append({"role": "user", "content": request + "."})
        try:
            message = self.agent.deepseek.chat(messages, tools=[FINISH_TOOL])
        except LLMError as exc:
            self.log("wrap_up", {}, f"failed: {exc}", started)
            return
        for call in message.get("tool_calls") or []:
            if call.get("function", {}).get("name") == "finish":
                try:
                    self.finished = json.loads(call["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    continue
        self.log("wrap_up", {}, "finished" if self.finished else "no finish call", started)

    # -- final answer ------------------------------------------------------------------
    def _finalize(self):
        ordered = self.pool.ordered()
        chosen = []
        if self.finished:
            for cid in self.finished.get("ranking") or []:
                candidate = self.pool.get(str(cid))
                if candidate is not None and candidate not in chosen:
                    chosen.append(candidate)
            self.result.notes = str(self.finished.get("notes") or "")[:500]
        # The model's picks lead only if they are not contradicted by verification.
        chosen = [c for c in chosen if c.verdict is None or c.verdict.score >= VERIFIED_MATCH / 2]
        self.result.ranking = chosen + [c for c in ordered if c not in chosen]
        if self.task.kind == "qa":
            self._finalize_answer()
        if self.task.kind == "trake":
            self._finalize_sequences()

    def _finalize_answer(self):
        """Agent's committed answer, else a dedicated reading of the top moment.

        A verdict's `answer` field is only a last resort: verification judges the
        whole query, and on a live run Gemini filled it with a visible detail
        ("can màu xanh lá") instead of the asked-for pass name.
        """
        answer = str((self.finished or {}).get("answer") or "").strip()
        evidence = "agent (finish)"
        if not answer and self.result.ranking:
            if self.options.mode != "deep":
                answer, evidence = self._answer_with_web()
            else:
                answer, evidence, _ = self._read_answer()
        if not answer:
            top = next((c for c in self.result.ranking if c.verdict and c.verdict.answer), None)
            if top is not None:
                answer, evidence = top.verdict.answer, f"verdict on {top.cid} (unconfirmed)"
        self.result.answer = answer[:100] or None
        self.result.answer_evidence = evidence

    def _read_answer(self):
        """Read the answer at up to `answer_moments` moments of the top video.

        One keyframe can hide what is asked (a pointer over the map, a legend cut
        off); the answer most readings agree on wins, ties going to confidence.
        """
        top_video = self.result.ranking[0].video_id
        moments = [c for c in self.result.ranking if c.video_id == top_video][:self.options.answer_moments]
        started = self.agent.clock()
        try:
            cues = self.cues.get(top_video) or self.agent.search.transcript(top_video)
        except Exception:
            cues = []
        hints = "\n".join(f"- {entry['question']}: {entry['answer'][:300]}"
                          for entry in self.result.lookups if entry.get("answer"))
        question = self.plan.question_vi or self.task.text

        captions = self._captions(top_video)

        def read(candidate):
            neighbours = self.agent.search.neighbours(candidate.video_id, candidate.frame_idx, 3)
            return self.agent.inspector.answer(self.task, question, candidate, neighbours, cues,
                                               hints, captions)

        readings, errors = [], []
        with ThreadPoolExecutor(max_workers=len(moments)) as pool:
            for candidate, future in [(c, pool.submit(read, c)) for c in moments]:
                try:
                    reply = future.result()
                except Exception as exc:
                    errors.append(str(exc)[:120])
                    continue
                text = str(reply.get("answer") or "").strip()
                if text:
                    readings.append((text, float(reply.get("confidence") or 0), candidate.cid,
                                     str(reply.get("evidence") or "")))
        if not readings:
            self.log("answer", {"moments": [c.cid for c in moments]}, f"no reading; errors={errors}", started)
            return "", "no reading", 0.0
        votes = {}
        for text, confidence, _, _ in readings:
            key = normalize_answer(text)
            count, best = votes.get(key, (0, 0.0))
            votes[key] = (count + 1, max(best, confidence))
        winner = max(readings, key=lambda r: (*votes[normalize_answer(r[0])], r[1]))
        text, confidence, cid, evidence = winner
        summary = f"{text!r} from {cid} ({votes[normalize_answer(text)][0]}/{len(readings)} readings agree)"
        self.log("answer", {"readings": [(r[0], r[1], r[2]) for r in readings]}, summary, started)
        return text, f"read from {cid}: {evidence}", confidence

    def _answer_with_web(self):
        """Read the answer from the video while the web answers the same question.

        Cooking shows get a recipe search from the top video's own opening narration
        (where the host names the dish); spoken and world-knowledge answers get a
        news search from the narration around the moment, since ASR misspells names
        ("Đèo Bứa" for "Đèo Tà Pứa"). Both run beside the reading, after round 1.
        """
        top = self.result.ranking[0]
        question = self.plan.question_vi or self.task.text
        recipe = None
        # Every Q&A but a count: the planner's answer_source is a guess (the spoken pass
        # name of p1-17 was planned as on_screen_text), and it runs beside the reading.
        if self.cooking or (self.options.auto_web and self.plan.answer_source != "visual_count"):
            try:
                cues = self.cues.get(top.video_id) or self.agent.search.transcript(top.video_id)
                self.cues[top.video_id] = cues
            except Exception:
                cues = []
            if self.cooking:
                label = f"công thức ({top.video_id})"
                prompt = recipe_lookup(self.task.text, question, transcript_window(cues, 45.0, 45.0),
                                       transcript_window(cues, top.time, 30.0))
            else:
                label = f"tra cứu đáp án ({top.video_id})"
                prompt = fact_lookup(self.task.text, question, transcript_window(cues, top.time, 60.0))
            recipe = self.background.submit(self._lookup, label, prompt)
        # Earlier lookups are hints for the reading when they are back in time.
        self._collect_lookups(min(3.0, max(0.0, self.remaining() - 8)))
        video, evidence, confidence = self._read_answer()
        web, web_source = None, None
        if recipe is not None:
            try:
                entry = recipe.result(timeout=max(2.0, self.remaining() + 8))
            except Exception:
                entry = {}
            if entry:
                self.result.lookups.append(entry)
                web, web_source = tagged(entry.get("answer", ""), "ĐÁP ÁN"), "recipe"
        if web is None:
            self._collect_lookups(0)
            web = next((tagged(e.get("answer", ""), "ĐÁP ÁN") for e in self.result.lookups
                        if tagged(e.get("answer", ""), "ĐÁP ÁN")), None)
            web_source = "web" if web else None
        spoken = (self.plan.answer_source == "speech"
                  or bool(SPOKEN_EVIDENCE.search(evidence or "")))
        return reconcile_answer(video, evidence, confidence, web, web_source, spoken=spoken)

    def _captions(self, video_id):
        """This video's shot captions, fetched once; empty if the server has none."""
        if video_id not in self.shot_captions:
            try:
                self.shot_captions[video_id] = self.agent.search.captions(video_id)
            except Exception:
                self.shot_captions[video_id] = []
        return self.shot_captions[video_id]

    def _finalize_sequences(self):
        """Check the best paths event by event, then pin the best ones' frames.

        Paths come fused over the tight and loose temporal queries and the other
        channels' support for their video; all checks and all per-event picks of a
        path run side by side (one Gemini call is ~4 s whatever its size).
        """
        count = expected_events(self.plan, self.task)
        sequences = self.pool.ranked_sequences(count)
        events = (self.plan.events_en or [scene.visual_en[0] for scene in self.plan.scenes])[:count]
        events += [""] * (count - len(events))
        sequences = self._local_sequences(sequences, events) + sequences
        order = {id(s): index for index, s in enumerate(sequences)}

        started = self.agent.clock()
        checked, errors = sequences[:self.options.trake_check], []
        with ThreadPoolExecutor(max_workers=max(1, len(checked))) as pool:
            futures = [(s, pool.submit(self.agent.inspector.check_sequence,
                                       self.task, events, s.video_id, s.frames)) for s in checked]
            for sequence, future in futures:
                try:
                    sequence.event_scores = future.result()
                except Exception as exc:  # an unchecked path keeps its fused rank
                    errors.append(str(exc)[:120])

        def key(sequence):
            if sequence.event_scores is None:
                return (1, 0.0, order[id(sequence)])
            return (0, -sum(sequence.event_scores) / len(sequence.event_scores), order[id(sequence)])

        sequences.sort(key=key)
        self.log("check_sequences", {"checked": len(checked)},
                 "; ".join(f"{s.video_id}={s.event_scores}" for s in sequences[:3])
                 + (f" errors={errors}" if errors else ""), started)
        final = []
        for index, sequence in enumerate(sequences):
            # Fast mode always refines: its picks run in parallel (~5 s) and a TRAKE row
            # is only right if every event is.
            if index < self.options.trake_refine and (self.options.mode != "deep" or not self.over_budget()):
                original = replace(sequence, frames=list(sequence.frames), times=list(sequence.times))
                self._refine(sequence, events)
                final.append(sequence)
                if original.frames != sequence.frames:
                    final.append(original)  # the unrefined path is the next-best guess
            else:
                final.append(sequence)
        self.result.sequences = final

    def _local_sequences(self, sequences, events, videos=4):
        """One tight path per leading video, from each event searched inside that video.

        The server's DP ranks videos well but can take an event from minutes away
        (p1-16 found the gold video with its first event 2 min early). Searching each
        event within the video and keeping consecutive events <= 45 s apart puts the
        path where the events actually happen together.
        """
        if not all(events):
            return []
        leading = []
        for sequence in sequences:
            if sequence.video_id not in leading:
                leading.append(sequence.video_id)
        leading = leading[:min(videos, 12 // len(events))]
        if not leading:
            return []
        started = self.agent.clock()
        queries = [{"kind": "visual", "id": f"lp{v}.{e}", "text": event, "video_ids": [video], "limit": 40}
                   for v, video in enumerate(leading) for e, event in enumerate(events)]
        try:
            results = self.agent.search.search(queries)
        except Exception as exc:
            self.log("local_paths", {}, f"failed: {exc}", started)
            return []
        existing = {(s.video_id, tuple(s.frames)) for s in sequences}
        found = []
        for v, video in enumerate(leading):
            hits = [results.get(f"lp{v}.{e}", {}).get("hits", []) for e in range(len(events))]
            best = local_path(hits, TRAKE_TIGHT_GAP)
            if best is None:
                continue
            value, path = best
            frames = [hit["frame_idx"] for hit in path]
            if (video, tuple(frames)) not in existing:
                found.append(Sequence("local", v + 1, video, frames, [hit["time"] for hit in path], value, []))
        self.log("local_paths", {"videos": leading},
                 "; ".join(f"{s.video_id}={s.frames}" for s in found) or "none", started)
        return found

    def _refine(self, sequence, events):
        """Move each event to the first neighbouring keyframe that shows it, in order."""
        started = self.agent.clock()

        def pick(index):
            neighbours = self.agent.search.neighbours(
                sequence.video_id, sequence.frames[index], self.options.neighbour_radius)
            event = events[index] if index < len(events) else ""
            picked, confidence, _ = self.agent.inspector.pick_frame(self.task.text, event, neighbours)
            return neighbours, (picked if picked is not None and confidence >= 5 else None)

        with ThreadPoolExecutor(max_workers=max(1, len(sequence.frames))) as pool:
            picks = list(pool.map(pick, range(len(sequence.frames))))
        previous_time = -1.0
        for index, (neighbours, picked) in enumerate(picks):
            # TRAKE rows must stay in time order: a pick at or before the previous event
            # falls back to the server's frame, else to the first later neighbour.
            if picked is None or picked["time"] <= previous_time:
                picked = None if sequence.times[index] > previous_time else next(
                    (f for f in neighbours if f["time"] > previous_time), None)
            if picked is not None:
                sequence.frames[index], sequence.times[index] = picked["frame_idx"], picked["time"]
            previous_time = sequence.times[index]
        sequence.refined = True
        self.log("refine", {"video_id": sequence.video_id}, f"frames={sequence.frames}", started)


def _same_answer(a, b):
    """Equal ignoring case, punctuation and spacing ("300 g" is "300g"); "cá" is not "cá sòng"."""
    a, b = normalize_answer(a).replace(" ", ""), normalize_answer(b).replace(" ", "")
    return bool(a) and a == b


def reconcile_answer(video, evidence, confidence, web, web_source, spoken=False):
    """(answer, evidence) from the video reading and a web answer, for the human to weigh.

    The answer key is what the video shows, so a confident reading wins; the web
    decides only when the reading is empty or unsure (a recipe's quantity or dish
    name the frames do not make legible), or, for a spoken answer, when it is the
    same name spelled differently -- the reading heard it through ASR.
    """
    if video and web and _same_answer(video, web):
        return video, f"{evidence}; {web_source} đồng ý"
    if spoken and video and web and similar(video, web) >= 0.5:
        return web, f"{web_source} sửa chính tả lời thoại: video đọc được {video!r}; {evidence}"
    if video and (confidence >= 7 or not web):
        return video, evidence + (f"; {web_source} nói: {web}" if web else "")
    if web:
        return web, f"{web_source}: {web}; video đọc được: {video or '-'} ({confidence:.0f}/10)"
    return video, evidence
