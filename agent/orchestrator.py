"""One query, end to end: plan -> fan-out -> verify -> agent loop -> final ranking.

Round 1 is deterministic (the plan's channels, run as one server batch, then the top
candidates verified by Gemini) because it is where most queries are won and it must
be fast. Only when round 1 does not produce a confident match does DeepSeek take
over with tools, reformulating and drilling into promising videos -- the loop the VBS
agent literature found most useful (rewrite on failure, search inside candidate
videos, verify before committing).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import re
import time

from .llm import LLMError
from .plan import Plan, analysis_messages, expected_events, initial_queries, parse_plan, transcript_variants
from .pool import VERIFIED_MATCH, Candidate, CandidatePool, Sequence
from .prompts import AGENT_SYSTEM, LOOKUP_PROMPT
from .tasks import Task
from .vision import Inspector, transcript_window

logger = logging.getLogger("agent")
VIDEO_ID = re.compile(r"^[A-Z]\d{2}_V\d{3}$")
OBSERVE_TOP = 12


@dataclass
class AgentOptions:
    max_steps: int = 6
    time_budget: float = 300.0
    verify_top: int = 12
    verify_per_step: int = 8
    early_stop_score: float = 9.0
    trake_refine: int = 2
    neighbour_radius: int = 6
    max_rows: int = 100
    plan_thinking: bool = False


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
                "kind": {"type": "string", "enum": ["visual", "audio", "ocr", "temporal", "similar"]},
                "text": {"type": "string", "description": (
                    "visual: English caption of one frame; audio: Vietnamese spoken phrase; "
                    "ocr: exact on-screen text")},
                "method": {"type": "string", "enum": ["keyword", "semantic"],
                           "description": "audio only; keyword (BM25) by default"},
                "events": {"type": "array", "items": {"type": "string"},
                           "description": "temporal only: 2-6 English captions in order"},
                "audio_text": {"type": "string", "description": "temporal only: optional spoken phrase"},
                "candidate_id": {"type": "string", "description": "similar only: search from this candidate's frame"},
                "video_ids": {"type": "array", "items": {"type": "string"},
                              "description": "visual/similar only: search inside these videos"},
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


class Agent:
    def __init__(self, deepseek, gemini, search, options: AgentOptions | None = None,
                 clock=time.monotonic):
        self.deepseek = deepseek
        self.gemini = gemini
        self.search = search
        self.options = options or AgentOptions()
        self.inspector = Inspector(gemini, search)
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
        self.finished: dict | None = None
        self.step = 0

    # -- bookkeeping ---------------------------------------------------------------
    def elapsed(self):
        return self.agent.clock() - self.started

    def over_budget(self):
        return self.elapsed() > self.options.time_budget

    def log(self, tool, args, summary, started):
        entry = {"step": self.step, "tool": tool, "args": args, "summary": summary,
                 "seconds": round(self.agent.clock() - started, 2)}
        self.result.trace.append(entry)
        logger.info("%s step=%d %s %s", self.task.query_id, self.step, tool, summary)

    # -- main flow -----------------------------------------------------------------
    def execute(self) -> AgentResult:
        try:
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
        self.result.elapsed = self.elapsed()
        return self.result

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
                 f"{len(self.plan.transcript_vi)} transcript, {len(self.plan.ocr)} ocr, "
                 f"{len(self.plan.lookups)} lookups", started)

    def _round_one(self):
        queries = initial_queries(self.plan, self.task)
        with ThreadPoolExecutor(max_workers=3) as pool:
            lookups = [pool.submit(self._lookup, question) for question in self.plan.lookups]
            self._run_search(queries, "round1")
            for future in lookups:
                self.result.lookups.append(future.result())
        self._verify(self.pool.unverified(self.options.verify_top))

    def _lookup(self, question):
        started = self.agent.clock()
        try:
            answer = self.agent.gemini.web_search(LOOKUP_PROMPT.format(question=question))
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
        return self.task.kind != "trake"

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
                 f"{len(verdicts)} verdicts, best={best}, errors={len(errors)}", started)

    def _server_query(self, raw, index):
        kind = raw.get("kind")
        qid = f"s{self.step}.{index}"
        video_ids = [v for v in (raw.get("video_ids") or []) if isinstance(v, str) and VIDEO_ID.match(v)][:50]
        text = " ".join(str(raw.get("text") or "").split())[:300]
        if kind == "visual" and text:
            query = {"kind": "visual", "id": qid, "text": text, "limit": 100 if video_ids else 200}
            if video_ids:
                query["video_ids"] = video_ids
            return query
        if kind == "audio" and text:
            method = raw.get("method") if raw.get("method") in ("keyword", "semantic") else "keyword"
            return {"kind": "audio", "id": qid, "text": transcript_variants(text)[0],
                    "method": method, "limit": 100}
        if kind == "ocr" and text:
            return {"kind": "ocr", "id": qid, "text": text, "limit": 50}
        if kind == "temporal":
            events = [" ".join(str(e).split())[:300] for e in raw.get("events") or [] if str(e).strip()]
            if 2 <= len(events) <= 6:
                query = {"kind": "temporal", "id": qid, "events": events, "limit": 20,
                         "max_gap_seconds": 300}
                if raw.get("audio_text"):
                    query["audio_text"] = transcript_variants(str(raw["audio_text"]))[0][:300]
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
        self._verify(self.pool.unverified(self.options.verify_per_step))
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
                     f"{self.elapsed():.0f}s/{self.options.time_budget:.0f}s used.")
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
                if name not in ("search",):
                    self.log(name, args, content[:160], call_started)
                if name != "finish":
                    content = f"{content}\n\n{self.observation()}"
                messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": content})

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
        answer = str((self.finished or {}).get("answer") or "").strip()
        evidence = "agent"
        if not answer:
            top = next((c for c in self.result.ranking if c.verdict and c.verdict.answer), None)
            if top is not None:
                answer, evidence = top.verdict.answer, f"verdict on {top.cid}"
        if not answer and self.result.ranking:
            top = self.result.ranking[0]
            started = self.agent.clock()
            try:
                neighbours = self.agent.search.neighbours(top.video_id, top.frame_idx, 3)
                cues = self.cues.get(top.video_id) or self.agent.search.transcript(top.video_id)
                reply = self.agent.inspector.answer(
                    self.task, self.plan.question_vi or self.task.text, top, neighbours, cues,
                )
                answer = str(reply.get("answer") or "").strip()
                evidence = str(reply.get("evidence") or "")
            except Exception as exc:
                evidence = f"answer step failed: {exc}"
            self.log("answer", {"candidate": top.cid}, f"{answer!r} ({evidence})", started)
        self.result.answer = answer[:100] or None
        self.result.answer_evidence = evidence

    def _finalize_sequences(self):
        count = expected_events(self.plan, self.task)
        sequences = [s for s in self.pool.sequences if len(s.frames) == count]

        def strength(sequence):
            scores = [self.pool.get(cid).verdict.score if self.pool.get(cid).verdict else 5.0
                      for cid in sequence.candidate_ids]
            return (-sum(scores) / len(scores), sequence.rank)

        sequences.sort(key=strength)
        events = self.plan.events_en or [scene.visual_en[0] for scene in self.plan.scenes]
        for sequence in sequences[:self.options.trake_refine]:
            if self.over_budget():
                break
            self._refine(sequence, events)
        self.result.sequences = sequences

    def _refine(self, sequence, events):
        """Move each event to the first neighbouring keyframe that shows it, in order."""
        started = self.agent.clock()
        previous_time = -1.0
        for index, (frame, time_) in enumerate(zip(list(sequence.frames), list(sequence.times))):
            neighbours = self.agent.search.neighbours(sequence.video_id, frame, self.options.neighbour_radius)
            neighbours = [f for f in neighbours if f["time"] > previous_time]
            event = events[index] if index < len(events) else ""
            picked, confidence, _ = self.agent.inspector.pick_frame(self.task.text, event, neighbours)
            if picked is None or confidence < 5:
                # Keep the server's frame unless an earlier event moved past it; TRAKE
                # rows must stay in time order.
                picked = None if time_ > previous_time else (neighbours[0] if neighbours else None)
            if picked is not None:
                sequence.frames[index], sequence.times[index] = picked["frame_idx"], picked["time"]
            previous_time = sequence.times[index]
        sequence.refined = True
        self.log("refine", {"video_id": sequence.video_id}, f"frames={sequence.frames}", started)
