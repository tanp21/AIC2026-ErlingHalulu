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
from .evidence import (
    TIE_BAND, EvidenceStore, QueryDetails, detail_weights, evidence_of, query_details, suggest_answers,
    summarize_all, video_coverage,
)
from .llm import LLMError
from .plan import (
    TRAKE_TIGHT_GAP, Plan, analysis_messages, expected_events, initial_queries, parse_plan,
    transcript_text,
)
from .pool import VERIFIED_MATCH, Candidate, CandidatePool, Sequence, Verdict, local_path
from .prompts import AGENT_SYSTEM, HUMAN_REWRITE_NOTE, LOOKUP_PROMPT, REWRITE_SYSTEM
from .review import HUMAN_ACCEPT, HUMAN_REJECT, HumanReview
from .tasks import Task
from .vision import Inspector, caption_at, label_context, transcript_window
from .web import cooking_lookup, dish_name, fact_lookup, is_cooking, recipe_lookup, similar, tagged

logger = logging.getLogger("agent")
VIDEO_ID = re.compile(r"^[A-Z]\d{2}_V\d{3}$")
OBSERVE_TOP = 12
#: How many candidates the live review view shows (every pending one is added on top).
PROGRESS_TOP = 24
#: A person judging scans a grid rather than a handful, so the view carries far more.
PROGRESS_TOP_HUMAN = 80
#: How many candidates get a checklist. Every one shown, not a leading slice of the retrieval order:
#: a p2-6 report found the gold candidate at rank 25 by retrieval fusion sitting with no checklist at
#: all, because only the top 20 of that order ever got one -- the ordering this checklist itself is
#: meant to fix never got the chance to see it. Retrieval only proposes candidates and can rank the
#: right one anywhere in what is shown, so the checklist has to reach all of it, not a leading slice.
SUMMARY_TOP = PROGRESS_TOP_HUMAN
#: Calls in flight at once (each is one video's candidates -- captions, speech and OCR together, and
#: up to three of that video's candidates judged by vision in parallel within it). Raised alongside
#: `SUMMARY_TOP`: checklisting every shown candidate instead of a leading 20 is roughly four times the
#: work, and this is what keeps it roughly the same wall-clock time instead of four times as long.
SUMMARY_WORKERS = 12
#: How often a wait on a model call looks at the human's decisions.
REVIEW_POLL_SECONDS = 0.25
# A reading whose evidence is the narration: its names went through ASR.
SPOKEN_EVIDENCE = re.compile(r"lời|thoại|nói|phát thanh|thuyết minh|giọng|transcript|speech", re.IGNORECASE)


@dataclass
class AgentOptions:
    mode: str = "fast"  # fast | deep | auto
    max_steps: int = 6
    time_budget: float = 300.0
    verify_top: int = 12
    verify_chunk: int = 8
    #: How many chunks are judged at once. Small chunks with more of them in flight beat
    #: large ones: measured on one query's twenty candidates, a 20-image call scored the
    #: gold moment 6/9/8/3 where four 5-image calls scored it 10/10/9 -- the judge loses
    #: the candidate when it looks at too much at once. More workers, same chunk, is the
    #: way to more coverage: 30 candidates at chunk 5 take 5.8 s at eight workers against
    #: 9.5 s at four, and score the gold 10 either way.
    verify_workers: int = 4
    #: How far down the fused order a second, wider pass reaches when the first one left
    #: the run unsure. The official set's misses sat between fused rank 30 and 100 for a
    #: third of the queries, and the judge scores the right moment highest 93% of the
    #: time it is shown one, so this is the cheapest recovery there is -- and it costs
    #: judge calls rather than seconds, because the chunks overlap.
    verify_wide: int = 120
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
    #: `model` sends frames to a vision model to decide; `human` never does for a decision -- a
    #: person looks at every candidate through the UI and the run waits for them (see agent/review.py).
    judge: str = "model"
    #: `human` only: seconds of the reviewer doing nothing before the run finishes with
    #: what it has, so an abandoned tab does not hold a worker forever.
    human_idle_timeout: float = 900.0
    #: `human` only: also show DeepSeek the pixels of each candidate's own moment while building the
    #: checklist (`agent.evidence.vision_summarize`). A caption is one model's guess about a frame, made
    #: once, offline, and can misname or miscount; the candidate's own pixels, plus the frames of the
    #: shots around it, are ground truth the text-only checklist never had. This never decides anything
    #: by itself -- the checklist stays a hint, and the person still accepts, rejects or scores every
    #: candidate -- and it costs one more DeepSeek call per candidate, server to DeepSeek, not the
    #: reviewer's own bandwidth.
    checklist_vision: bool = True

    @classmethod
    def for_mode(cls, mode="fast", **overrides):
        """Defaults tuned per mode; explicit overrides win.

        One Gemini call takes ~4 s whether it judges 2 or 8 frames, so fast mode
        splits work into more, smaller calls run side by side instead of fewer
        large ones run in turn.

        `verify_chunk` is therefore a *network* knob and does nothing for
        `AIC_AGENT_VISION=local`: a local backend fans a chunk out into one call per
        frame and holds a lock across them, so the work is serial whatever this is
        set to. Retuning it will not make the local path faster.
        """
        defaults = FAST_DEFAULTS if mode in ("fast", "auto") else {}
        return cls(mode=mode, **{**defaults, **overrides})


FAST_DEFAULTS = dict(
    # Back to fifteen, and no wide pass, because judging more measured *worse*: over the
    # 113-query set the wider field (median 30 judged, up to 150) found the right moment
    # for seven more queries but the judge's own discrimination fell from 64/82 to 59/89 --
    # the max of a noisy score over a bigger field is a bigger noise. Net, top-1 went from
    # 67 to 63. Bounding the ranking to the pool's own top-N afterwards does not recover
    # it (58/57/60 against 61 unbounded), so the field itself is what has to stay small.
    time_budget=20.0, verify_top=15, verify_chunk=5, verify_workers=4, verify_wide=15,
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
    #: Every reviewer action with its timestamp (human judge only), for measuring the person.
    review_log: list[dict] = field(default_factory=list)

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
            **({"review_log": self.review_log} if self.review_log else {}),
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
                 clock=time.monotonic, vision=None):
        # `gemini` is still the web-search client whatever `vision` is: Google
        # Search grounding has no local equivalent, and the vision backend only
        # ever sees pixels.
        self.deepseek = deepseek
        self.gemini = gemini
        self.vision = vision or gemini
        self.search = search
        self.options = options or AgentOptions()
        self.inspector = Inspector(
            self.vision, search, chunk_size=self.options.verify_chunk,
            workers=self.options.verify_workers,
        )
        self.clock = clock

    def run(self, task: Task, review: HumanReview | None = None, progress=None) -> AgentResult:
        """One task; `review` carries a teammate's decisions, `progress(snapshot)` the live view.

        Both are optional: the CLI runs without a human, the UI supplies both.
        """
        return _Run(self, task, review, progress).execute()


class _Run:
    def __init__(self, agent: Agent, task: Task, review: HumanReview | None = None, progress=None):
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
        self.human = agent.options.judge == "human"
        self.review = review or HumanReview(judge=agent.options.judge)
        self.progress = progress
        #: Candidates the reviewer asked for by hand (neighbours, exact frames), kept in
        #: the view however low the fusion ranks them.
        self.pinned: list[str] = []
        #: The query as checkable details, and DeepSeek's per-candidate checklist against
        #: them (agent/evidence.py). Written by one background job at a time.
        self.details: QueryDetails | None = None
        self.summaries: dict[str, dict] = {}
        self.suggestions: list[dict] = []
        self.summarizing = False
        self._summary_tried: set[str] = set()
        self._suggest_tried = False
        self.evidence = EvidenceStore(agent.search)
        #: A stage code the UI translates: planning, searching, judging, second_look,
        #: rewriting, agent_loop, reading_answer, checking_paths, finalizing, done.
        self.stage = "planning"
        #: Candidates whose images are with Gemini right now, for the review view.
        self.judging: set[str] = set()
        #: Gemini's own verdicts, kept so withdrawing a human decision restores them.
        self.model_verdicts: dict[str, Verdict | None] = {}
        # Model calls a human decision may abandon run here, never under a `with`
        # block that would wait for them on exit.
        self.waiter = ThreadPoolExecutor(max_workers=4)

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
        self._publish("planning")
        try:
            if self.human:
                self._human()
            elif self.options.mode in ("fast", "auto"):
                self._fast()
                if (self.options.mode == "auto" and self.task.kind != "trake"
                        and not self.review.decided()
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
            self.waiter.shutdown(wait=False, cancel_futures=True)
        self.result.elapsed = self.elapsed()
        self.result.review_log = self.review.log() if self.human else []
        self.stage = "done"
        self._publish()
        return self.result

    # -- the human in the loop -------------------------------------------------------
    def _publish(self, stage=None):
        """Hand the live view to whoever is watching; never lets the view fail the run."""
        if stage is not None:
            self.stage = stage
        if self.progress is None:
            return
        try:
            self.progress(self.snapshot())
        except Exception:
            logger.exception("%s progress callback failed", self.task.query_id)

    def snapshot(self) -> dict:
        """What Gemini is shown, as it is shown, for a teammate to judge alongside it."""
        ordered = self.pool.ordered()
        agent_rank = {c.cid: index + 1 for index, c in enumerate(ordered)}
        weights, common = self._detail_weights()
        coverage = self._video_coverage(weights)
        if self.human and not self.summarizing:
            # Not while the checklist is still arriving: candidates would jump about under the cursor.
            ordered = self._by_evidence(ordered, weights, coverage)
        shown = ordered[:PROGRESS_TOP_HUMAN if self.human else PROGRESS_TOP]
        shown += [c for c in ordered if c.cid in self.judging and c not in shown]
        shown += [c for c in ordered if c.cid in self.pinned and c not in shown]
        plan = None
        if self.plan is not None:
            plan = {
                "summary_vi": self.plan.summary_vi,
                "scenes": [scene.description_vi or scene.visual_en[0] for scene in self.plan.scenes],
                "target_scene": self.plan.target_scene,
                "question_vi": self.plan.question_vi,
                "events": self.plan.events_en,
            }
            if self.human:
                plan["keywords"] = self._keywords()

        def candidate(c):
            item = {
                "id": c.cid, "video_id": c.video_id, "frame_idx": c.frame_idx,
                "time": round(c.time, 3), "judging": c.cid in self.judging,
                "verdict": None if c.verdict is None else vars(c.verdict),
                # The second argument is the caption the verify step fetched for this
                # frame, if it fetched one: a reviewer has to read what the judge read.
                "context": label_context(
                    c, self.agent.inspector.caption_for(c.video_id, c.frame_idx)),
                "sources": sorted({e.kind for e in c.evidence.values()}),
            }
            if self.human:
                # Which query found it and how high: a moment three channels agree on is
                # a different bet from one visual hit at rank 90.
                item["channels"] = sorted(
                    ({"kind": e.kind, "source": e.source, "rank": e.rank}
                     for e in c.evidence.values() if e.kind != "neighbour"),
                    key=lambda channel: channel["rank"])
                item["fused"] = round(c.fused, 5)
                item["pinned"] = c.cid in self.pinned
                item["summary"] = self.summaries.get(c.cid)
                item["evidence"] = evidence_of(item["summary"], weights, c.time, coverage.get(c.video_id, 0.0))
                item["agent_rank"] = agent_rank[c.cid]
            return item

        snapshot = {
            "stage": self.stage,
            "step": self.step,
            "elapsed": round(self.elapsed(), 1),
            "plan": plan,
            "candidates": [candidate(c) for c in shown],
            "sequences": [{
                "video_id": s.video_id, "frames": list(s.frames),
                "event_scores": s.event_scores, "refined": s.refined,
                **({"source": s.source, "times": list(s.times)} if self.human else {}),
            } for s in self.result.sequences or self.pool.ranked_sequences()[:8]],
            "answer": self.result.answer,
            "review": self.review.snapshot(),
        }
        if self.human:
            snapshot["lookups"] = [
                {"question": entry["question"], "answer": entry.get("answer"),
                 "error": entry.get("error"), "sources": entry.get("sources", [])}
                for entry in self.result.lookups]
            snapshot["total_candidates"] = len(self.pool)
            snapshot["details"] = self.details.texts if self.details else []
            snapshot["detail_weights"], snapshot["common_details"] = weights, common
            snapshot["summarizing"] = self.summarizing
            if self.task.kind == "qa":
                snapshot["suggested_answers"] = self._answer_suggestions()
        return snapshot

    def _keywords(self) -> dict:
        """The words a reviewer should see lit up in a transcript, an OCR line or a caption."""
        plan = self.plan
        vietnamese = [*plan.transcript_vi, *plan.ocr, plan.question_vi or ""]
        english = [*plan.caption_en, *plan.readings_en]
        for scene in plan.scenes:
            english += scene.visual_en[:1]
        if self.details:  # the words the checklist looked for, so what is lit is what was searched
            for detail in self.details.details:
                vietnamese += detail["terms_vi"]
                english += detail["terms_en"]
        return {"vi": [t for t in dict.fromkeys(vietnamese) if t],
                "en": [t for t in dict.fromkeys(english) if t]}

    def _apply_review(self):
        """A teammate's decisions become verdicts no model reply can overwrite."""
        self.review.changed.clear()
        decisions = self.review.verdicts()
        for cid, value in decisions.items():
            candidate = self.pool.get(cid)
            if candidate is None:
                continue
            if candidate.verdict is None or candidate.verdict.by != "human":
                self.model_verdicts[cid] = candidate.verdict
            if value >= HUMAN_ACCEPT:
                reason = "Accepted by the reviewer"
            elif value <= HUMAN_REJECT:
                reason = "Rejected by the reviewer"
            else:
                reason = f"Scored {value:.0f}/10 by the reviewer"
            candidate.verdict = Verdict(value, reason, by="human")
        for cid, verdict in list(self.model_verdicts.items()):
            if cid not in decisions:  # a withdrawn decision gives the model its verdict back
                candidate = self.pool.get(cid)
                if candidate is not None and candidate.verdict and candidate.verdict.by == "human":
                    candidate.verdict = verdict
                del self.model_verdicts[cid]

    def _human_says(self, cid) -> bool:
        return cid in self.review.verdicts()

    def _await(self, futures, give_up):
        """Wait for model calls, re-applying decisions, until done or `give_up()`.

        Returns the futures that finished; the rest keep running unseen. This is
        the whole point of the review: on a congested venue network a Gemini call
        can take far longer than a teammate needs to say "that one".
        """
        futures = list(futures)
        while True:
            done, pending = wait(futures, timeout=REVIEW_POLL_SECONDS)
            if self.review.changed.is_set():
                self._apply_review()
                self._publish()
            if not pending or give_up():
                return done

    # -- human judge mode ------------------------------------------------------------
    def _human(self):
        """Plan and search, then let a person judge; no model ever sees a frame.

        Round 1 is the same batch every mode runs. After it the run waits for the
        reviewer, who steers: `more` asks DeepSeek for a rewrite that takes their
        verdicts and note into account, `scope` searches inside one video,
        `neighbours` and `add_frame` put exact keyframes in front of them. It ends
        on `finish`/`stop`, or after `human_idle_timeout` seconds of silence.
        """
        if self.cooking:
            self._lookup_later(cooking_lookup(self.task.text, None),
                               "món ăn (Món ngon mỗi ngày)", about_the_query=True)
        self._plan()
        for question in self.plan.lookups:
            self._lookup_later(LOOKUP_PROMPT.format(question=question), question, about_the_query=True)
        self._run_search(initial_queries(self.plan, self.task, self.options.spoken_numbers,
                                         skip_kinds=self._unavailable_channels()), "round1")
        if self.task.kind == "trake":
            self._local_paths_for_review()
        self._kick_summaries()
        self._publish("awaiting_review")
        self._human_loop()

    def _human_loop(self):
        quiet_since, seen = self.agent.clock(), len(self.review.log())
        while not self.review.decided():
            self._collect_lookups(0)
            woken = self.review.changed.wait(REVIEW_POLL_SECONDS)
            # A background summary also wakes the loop, but only the person counts as activity.
            if len(self.review.log()) != seen:
                seen, quiet_since = len(self.review.log()), self.agent.clock()
            elif self.agent.clock() - quiet_since > self.options.human_idle_timeout:
                self.log("human", {}, "no reviewer activity; finishing with the current ranking",
                         self.agent.clock())
                return
            if not woken:
                continue
            self._apply_review()
            for command in self.review.take_commands():
                if self.review.decided():
                    break
                self._run_command(command)
            self._kick_summaries()
            self._publish("awaiting_review")

    def _run_command(self, command):
        """One reviewer request to search; a failure is a note in the trace, never the run."""
        started = self.agent.clock()
        kind = command.get("type")
        try:
            if kind == "more":
                self._rewrite_round(hint=command.get("hint"))
            elif kind == "scope":
                video_id = str(command.get("video_id", ""))
                if VIDEO_ID.match(video_id):
                    self._publish("searching")
                    self._run_search(self._scoped_queries(room=6, only=[video_id]), f"scope.{video_id}")
            elif kind == "neighbours":
                candidate = self.pool.get(str(command.get("cid", "")))
                if candidate is not None:
                    self._publish("searching")
                    radius = min(max(int(command.get("radius") or self.options.neighbour_radius), 1), 10)
                    for frame in self.agent.search.neighbours(candidate.video_id, candidate.frame_idx, radius):
                        added = self.pool.add_frame(frame["video_id"], frame["frame_idx"], frame["time"],
                                                    source=f"n.{candidate.cid}")
                        if added.cid not in self.pinned:
                            self.pinned.append(added.cid)
            elif kind == "add_frame":
                video_id = str(command.get("video_id", ""))
                if VIDEO_ID.match(video_id):
                    added = self.pool.add_frame(video_id, int(command["frame_idx"]),
                                                float(command.get("time") or 0.0), source="human")
                    if added.cid not in self.pinned:
                        self.pinned.append(added.cid)
                    if command.get("accept"):
                        self.review.decide(added.cid, "accept")
                    self._apply_review()
            self.log("human", command, "ok", started)
        except Exception as exc:  # a search that fails must not end the review
            logger.exception("%s reviewer command %s failed", self.task.query_id, kind)
            self.log("human", command, f"failed: {type(exc).__name__}: {exc}", started)

    def _kick_summaries(self):
        """Start the text checklist for candidates that have none, unless one is running.

        Runs on the background pool and reads only words: the transcript, OCR text and
        caption around each moment. The orchestrator thread does the publishing -- a
        background thread that published would walk the pool while a command is
        adding to it -- so the job only nudges the loop to look again.
        """
        if not self.human or self.summarizing:
            return
        ordered = self.pool.ordered()
        wanted = ordered[:SUMMARY_TOP] + [c for c in ordered if c.cid in self.pinned]
        fresh = [c for c in wanted if c.cid not in self._summary_tried]
        suggest = self.task.kind == "qa" and not self._suggest_tried
        if not fresh and not suggest:
            return
        items = [{"id": c.cid, "video_id": c.video_id, "frame_idx": c.frame_idx, "time": c.time} for c in fresh]
        self._summary_tried.update(item["id"] for item in items)
        self._suggest_tried = self._suggest_tried or suggest
        top = ([{"id": c.cid, "video_id": c.video_id, "frame_idx": c.frame_idx, "time": c.time} for c in ordered[:2]]
               if suggest else [])
        self.summarizing = True
        self.background.submit(self._summary_job, items, top)

    def _summary_job(self, items, top):
        started = self.agent.clock()
        try:
            if self.details is None:
                self.details = query_details(self.agent.deepseek, self.task, self.plan)

            def store(found):
                self.summaries.update(found)
                self.review.changed.set()  # the loop publishes; see _kick_summaries

            vision = self.agent.deepseek if self.options.checklist_vision else None
            summarize_all(self.agent.deepseek, self.details, self.evidence, items, workers=SUMMARY_WORKERS,
                          on_batch=store, vision=vision, search=self.agent.search)
            if top:
                question = (self.plan.question_vi if self.plan else None) or self.task.text
                self.suggestions = suggest_answers(self.agent.deepseek, question, self.details, self.evidence, top)
        except Exception:  # a checklist is a hint; the person can judge without it
            logger.exception("%s evidence summary failed", self.task.query_id)
        finally:
            self.summarizing = False
            self.review.changed.set()
            self.log("summaries", {"n": len(items)}, f"{len(self.summaries)} summarised", started)

    def _answer_suggestions(self) -> list[dict]:
        """Answers for a Q&A question that the words around the top moments or the web offer."""
        found = [{**s, "source": s.get("source") or "video"} for s in self.suggestions]
        for entry in self.result.lookups:
            web = tagged(entry.get("answer", "") or "", "ĐÁP ÁN")
            if web:
                found.append({"answer": web, "source": "web", "quote": entry["question"],
                              "cid": None, "confidence": 0.0})
        return found

    def _local_paths_for_review(self):
        """TRAKE: paths found by searching each event inside the leading videos, as candidates.

        The reviewer assembles the sequence from frames, so these paths' frames have to
        be in the grid, and the paths themselves offered as a starting point.
        """
        count = expected_events(self.plan, self.task)
        sequences = self.pool.ranked_sequences(count)
        events = (self.plan.events_en or [scene.visual_en[0] for scene in self.plan.scenes])[:count]
        events += [""] * (count - len(events))
        local = self._local_sequences(sequences, events)
        for sequence in local:
            for frame, time in zip(sequence.frames, sequence.times):
                self.pool.add_frame(sequence.video_id, frame, time, source="local", kind="temporal")
        self.result.sequences = local + sequences

    # -- fast mode -----------------------------------------------------------------
    def _fast(self):
        """Plan, one batch, parallel verification, and one rewrite only if needed.

        Web lookups (the plan's, and for cooking queries a recipe search from the
        raw description, started before planning) run in the background and are
        waited for only where their result is used.
        """
        if self.cooking:
            # About the query, not about a video the agent has already chosen: those
            # two kinds are kept apart because only this one is safe to search for.
            self._lookup_later(cooking_lookup(self.task.text, None),
                               "món ăn (Món ngon mỗi ngày)", about_the_query=True)
        self._plan()
        for question in self.plan.lookups:
            self._lookup_later(LOOKUP_PROMPT.format(question=question), question, about_the_query=True)
        self._run_search(initial_queries(self.plan, self.task, self.options.spoken_numbers,
                                         skip_kinds=self._unavailable_channels()), "round1")
        if self.task.kind == "trake":
            return  # paths are checked event by event in _finalize_sequences
        self._verify(self.pool.to_verify(self.options.verify_top))
        if self.review.decided():
            return
        best = next((c for c in self.pool.ordered() if c.verdict), None)
        # A 9-10 must survive the falsifier (~5 s): on round 2, fast mode's misses were
        # mostly confident look-alikes it would have questioned. ~8 s is one DeepSeek
        # call, one batch and one parallel verification.
        checked = best.cid if best is not None else None
        falsified = (best is not None and best.verdict.score >= self.options.accept_score
                     and self._falsified(best))
        confident = (best is not None and best.verdict.score >= self.options.accept_score
                     and not falsified)
        # Widen before rewriting: show the judge more of the order it already ranked.
        if (not confident and not self.review.decided() and not self.over_budget()
                and self.options.verify_wide > self.options.verify_top):
            self._verify(self.pool.to_verify(self.options.verify_wide))
            best = next((c for c in self.pool.ordered() if c.verdict), None)
            if best is not None and best.cid != checked:
                # A different candidate now leads, so the falsifier has not seen it; the
                # one it did see keeps its verdict rather than paying for it twice.
                falsified = (best.verdict.score >= self.options.accept_score
                             and self._falsified(best))
            confident = (best is not None and best.verdict.score >= self.options.accept_score
                         and not falsified)
        # The rewrite is the only recovery this mode has, so it is not gated on the
        # budget being comfortable: a 12 s round 1 once cancelled it with 8.0 s left
        # and the run finished on a wrong video the rewrite would have questioned.
        if not confident and not self.over_budget() and not self.review.decided():
            self._rewrite_round()

    def _best_score(self):
        return max((c.verdict.score for c in self.pool.candidates.values() if c.verdict), default=0.0)

    def _unavailable_channels(self) -> frozenset:
        """Channels this deployment cannot serve, so the batch does not spend slots on them.

        The plan names channels, not deployments: a caption query against a server with no
        caption index is a slot taken from a live channel and then lost. Probed once per
        run, and a client that cannot be asked (tests, an older build) is assumed capable.
        """
        probe = getattr(self.agent.search, "caption_channel_available", None)
        if probe is None or probe():
            return frozenset()
        return frozenset({"caption"})

    def _lookup_later(self, prompt, label, about_the_query=False):
        self.pending.append(self.background.submit(self._lookup, label, prompt, about_the_query))

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

    def _rewrite_round(self, hint=None):
        """DeepSeek (no thinking, one call) rewrites what round 1 missed; runs and verifies it.

        With a human judge nothing is verified: the new candidates go to the reviewer,
        `hint` is what they typed, and the round can repeat, so it gets its own step.
        """
        self.step = self.step + 1 if self.human else 1
        self._publish("rewriting")
        started = self.agent.clock()
        # A cooking lookup started before planning is usually back by now; its dish
        # name is what the transcript channel needs. Waited for even when the budget is
        # spent: a lookup averages 11.5 s (measured over 18 runs) and dropping it cost 16
        # of the 34 cooking queries their only source of the dish name.
        self._collect_lookups(min(3.0, max(1.5, self.remaining() - 10)))
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": (
                f"Task type: {self.task.kind}\nQuery:\n{self.task.text}\n\n"
                f"Round-1 plan:\n{json.dumps(self.plan.to_dict(), ensure_ascii=False)}\n\n"
                f"Web notes:\n{self._web_notes()}\n\nRound-1 result:\n{self.observation()}"
                + (HUMAN_REWRITE_NOTE.format(hint=hint or "(none)") if self.human else "")
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
        # Before the scoped visuals: a dish name is a key the corpus certainly has, and
        # it is the one thing here that was measured to rank the answer's video first.
        queries += self._dish_queries(room=2)
        queries += self._scoped_queries(room=8 - len(queries))
        self.log("rewrite", {"queries": queries},
                 "; ".join(str(q.get("text") or q.get("events"))[:60] for q in queries), started)
        self._run_search(queries, "rewrite")
        if self.human:
            return
        fresh = self._best_neighbours()
        self._verify(fresh + self.pool.to_verify(self.options.verify_top - len(fresh)))

    def _dish_queries(self, room):
        """The dish names the web lookups found, as transcript queries.

        The prompts already ask a lookup to end with `TỪ KHÓA: <tên món>; …`, and the
        rewrite prompt is handed the same lookups as prose -- but the prose is what the
        model has to notice, and across runs it used the quantities the question
        mentions instead ("hai muỗng nước tương") rather than the name. Measured on the
        round-3 soy-sauce query: the lookup named "Cá hồi áp chảo xốt tiêu", an audio
        search for that name returns its video **first**, and the query's own terms
        ranked it 30th of 84. The transcript says the dish out loud, so the name is a
        key the corpus has and the description does not.
        """
        names = []
        for entry in self.result.lookups:
            # Only the lookups that describe the *query*. A lookup anchored to a video
            # the agent already chose (the recipe search for the current top candidate)
            # names that video's dish whatever it is, and searching it would spend a
            # query reinforcing a choice that may be the mistake being corrected.
            if not entry.get("about_the_query"):
                continue
            name = dish_name(entry.get("answer"))
            if name and name not in names:
                names.append(name)
        return [{"kind": "audio", "id": f"s1.k{index}", "text": name,
                 "method": "keyword", "limit": 100}
                for index, name in enumerate(names[:max(0, room)])]

    def _scoped_queries(self, room, videos=3, only=None):
        """Each scene searched inside the videos round 1 favoured.

        Right video, wrong moment is the commonest miss, and on a multi-scene query
        the moment to submit is often not the scene that found the video: the one
        that did can be generic ("two women with a phone") while the distinctive
        scene ("a Black woman walking past") is never searched inside it.
        """
        leading = list(only) if only else self.pool.video_ranking()[:videos]
        if not leading or room <= 0:
            return []
        order = [self.plan.target_scene] + [index for index in range(len(self.plan.scenes))
                                            if index != self.plan.target_scene]
        scoped = []
        for index in order[:max(1, room)]:
            caption, *paraphrases = self.plan.scenes[index].visual_en
            query = {"kind": "visual", "id": f"s1.scoped{index}", "text": caption,
                     "video_ids": leading, "limit": 100}
            if paraphrases:
                query["paraphrases"] = paraphrases[:4]
            scoped.append(query)
        return scoped

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
        self._publish("searching")
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

    def _lookup(self, question, prompt=None, about_the_query=False):
        started = self.agent.clock()
        try:
            answer = self.agent.gemini.web_search(prompt or LOOKUP_PROMPT.format(question=question))
        except LLMError as exc:
            entry = {"question": question, "error": str(exc), "about_the_query": about_the_query}
        else:
            entry = {"question": question, "answer": answer.text,
                     "sources": [uri for _, uri in answer.sources],
                     "about_the_query": about_the_query}
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
        if not self.options.falsify or self.review.decided() or self.human:
            return False
        started = self.agent.clock()
        self.judging = {candidate.cid}
        self._publish("second_look")

        def second_look():
            neighbours = self.agent.search.neighbours(candidate.video_id, candidate.frame_idx, 2)
            return self.agent.inspector.falsify(self.task, candidate, neighbours)

        future = self.waiter.submit(second_look)
        done = self._await([future], self.review.decided)
        self.judging = set()
        if future not in done:
            self.log("falsify", {"candidate": candidate.cid}, "abandoned for a human decision", started)
            return False
        try:
            rejected, reason, contradicted = future.result()
        except Exception as exc:  # no second opinion is not a rejection
            self.log("falsify", {"candidate": candidate.cid}, f"failed: {exc}", started)
            return False
        if candidate.verdict is None or candidate.verdict.by == "human":
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
        self._apply_review()
        self._publish()
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
        """Gemini judges `candidates` while the review view shows them as being judged.

        A candidate a teammate already decided is not sent at all, and a decision
        that ends the run stops the wait: the late reply is discarded.
        """
        self._apply_review()
        candidates = [c for c in candidates if not self._human_says(c.cid)]
        if not candidates or self.review.decided() or self.human:
            return
        started = self.agent.clock()
        self.judging = {c.cid for c in candidates}
        self._publish("judging")
        future = self.waiter.submit(self.agent.inspector.verify, self.task, self.plan, candidates, question)
        done = self._await([future], self.review.decided)
        self.judging = set()
        if future not in done:
            self.log("verify", {"ids": [c.cid for c in candidates]}, "abandoned for a human decision", started)
            self._publish()
            return
        verdicts, errors = future.result()
        for cid, verdict in verdicts.items():
            if self._human_says(cid):
                continue  # a teammate decided while Gemini was still looking
            candidate = self.pool.get(cid)
            previous = candidate.verdict
            if question and previous is not None:
                # A follow-up question is judged against that question, not the whole
                # query; keep the match score and take only its reading.
                verdict.score = previous.score
                verdict.answer = verdict.answer or previous.answer
            candidate.verdict = verdict
        best = max((v.score for v in verdicts.values()), default=None)
        self._publish()
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
        if not self.human:
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
        self._publish("agent_loop")
        while (self.step < self.options.max_steps and not self.over_budget()
               and self.finished is None and not self.review.decided()):
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
        if self.finished is None and self.step and not self.review.decided():
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
        self._apply_review()
        self._publish("finalizing")
        ordered = self.pool.ordered()
        chosen = []
        if self.finished:
            for cid in self.finished.get("ranking") or []:
                candidate = self.pool.get(str(cid))
                if candidate is not None and candidate not in chosen:
                    chosen.append(candidate)
            self.result.notes = str(self.finished.get("notes") or "")[:500]
        # The model's picks lead only if they are not contradicted by verification,
        # and never ahead of a candidate a teammate accepted.
        chosen = [c for c in chosen if c.verdict is None or c.verdict.score >= VERIFIED_MATCH / 2]
        accepted = [c for c in ordered if c.verdict and c.verdict.by == "human"
                    and c.verdict.score >= VERIFIED_MATCH]
        chosen = accepted + [c for c in chosen if c not in accepted]
        self.result.ranking = chosen + [c for c in ordered if c not in chosen]
        if self.human:
            # The order on the reviewer's screen if they sent it; else the evidence's, never retrieval's alone.
            self.result.ranking = (self._in_the_order_seen(self.result.ranking) if self.review.order
                                   else self._by_evidence(self.result.ranking))
        if self.task.kind == "qa":
            self._finalize_answer()
        if self.task.kind == "trake":
            self._finalize_sequences_human() if self.human else self._finalize_sequences()

    def _detail_weights(self):
        entries = self.details.details if self.details else []
        return detail_weights(self.summaries.values(), len(entries), entries)

    @staticmethod
    def _split(ranking):
        """Accepted candidates, the ones nobody has settled, and rejected ones, each in its given order.

        What a reviewer accepted leads and what they rejected trails whatever any other order says.
        """
        def bucket(candidate):
            verdict = candidate.verdict
            if verdict is not None and verdict.by == "human":
                return 0 if verdict.score >= VERIFIED_MATCH else 2 if verdict.score <= 3 else 1
            return 1

        return ([c for c in ranking if bucket(c) == 0], [c for c in ranking if bucket(c) == 1],
                [c for c in ranking if bucket(c) == 2])

    def _video_coverage(self, weights):
        return video_coverage([(c.video_id, self.summaries[c.cid]) for c in self.pool.candidates.values()
                               if c.cid in self.summaries], weights)

    def _by_evidence(self, ranking, weights=None, coverage=None):
        """The unsettled candidates ordered by what the words around them support, not by retrieval.

        The searches (visual, audio, OCR, temporal) only find candidates, and each of them can be
        wrong: three visual queries ranked a frame of blue bowls first for a query about clams
        beside a vase of ginger. So a candidate is ranked by its checklist -- the weighted share of
        the query's details the transcript, OCR and captions around it support, and the share its
        whole video supports -- and retrieval order is only what separates candidates the evidence
        cannot: those within `TIE_BAND` of each other, and those with a checklist that says nothing
        or none. Only support counts; a contradiction is shown but does not lower a candidate
        (see `agent.evidence.STATUS_VALUE`).

        Within a tie, the candidate's OWN share settles it before retrieval does -- video coverage
        LIFTS a tie, it does not decide one. Reported on p2-6: two videos of the same real event, one
        with the query's distinctive graphic (a candidate's own share .66) and one without it (a
        different candidate's own share .63, from a duplicate broadcast whose own narration happened
        to state a headcount the first video's did not); the second video's every candidate still
        scored the detail it lacks as `not_mentioned`, so it added nothing there, but scored fuller on
        two others, and the blended score put it in the same band, one place ahead by retrieval alone.
        Retrieval had no opinion on either video's story -- it is not evidence of which one is right --
        so it is not what should break a tie the checklist can: the candidate whose own moment supports
        more of the query leads, and only candidates tied on that too fall back to retrieval.
        """
        weights = self._detail_weights()[0] if weights is None else weights
        coverage = self._video_coverage(weights) if coverage is None else coverage
        head, middle, tail = self._split(ranking)

        def key(pair):
            position, candidate = pair
            found = evidence_of(self.summaries.get(candidate.cid), weights, candidate.time,
                                coverage.get(candidate.video_id, 0.0))
            if found is None or found["tier"] == 1:
                return (1, 0, 0, position)
            # Bands, not raw scores: a difference smaller than TIE_BAND is a tie for retrieval to settle.
            return (0, -round(found["score"] / TIE_BAND), -found["share"], position)

        return head + [c for _, c in sorted(enumerate(middle), key=key)] + tail

    def _in_the_order_seen(self, ranking):
        """Candidates nobody judged, in the order the reviewer's screen showed them.

        The view re-ranks those by their checklist; a submission that ignored it would put on
        top what the reviewer was looking at fifth. What they accepted still leads, what they
        rejected still trails, and a candidate the screen never listed follows the listed ones
        in the agent's own order.
        """
        position = {cid: index for index, cid in enumerate(self.review.order)}
        head, middle, tail = self._split(ranking)
        middle = sorted(enumerate(middle), key=lambda pair: (0, position[pair[1].cid]) if pair[1].cid in position else (1, pair[0]))
        return head + [c for _, c in middle] + tail

    def _finalize_answer(self):
        """Agent's committed answer, else a dedicated reading of the top moment.

        A verdict's `answer` field is only a last resort: verification judges the
        whole query, and on a live run Gemini filled it with a visible detail
        ("can màu xanh lá") instead of the asked-for pass name.
        """
        if self.review.answer:
            self.result.answer = self.review.answer
            self.result.answer_evidence = "typed by the reviewer"
            return
        answer = str((self.finished or {}).get("answer") or "").strip()
        evidence = "agent (finish)"
        if not answer and self.result.ranking and not self.review.decided() and not self.human:
            if self.options.mode != "deep":
                answer, evidence = self._answer_with_web()
            else:
                answer, evidence, _ = self._read_answer()
        if not answer and self.human:
            # The person submitted without typing one: a suggestion read from the words
            # around the top moments beats an empty answer cell, and is labelled as unconfirmed.
            suggestion = next(iter(self._answer_suggestions()), None)
            if suggestion:
                answer, evidence = suggestion["answer"], f"suggested from {suggestion['source']} (unconfirmed)"
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
        self._publish("reading_answer")
        futures = [(c, self.waiter.submit(read, c)) for c in moments]
        done = self._await([f for _, f in futures],
                           lambda: bool(self.review.answer) or self.review.stopped)
        for candidate, future in futures:
            if future not in done:
                errors.append("abandoned for a human decision")
                continue
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
        # Earlier lookups are hints for the reading when they are back in time -- and
        # waited for, for the same reason as above: a 1.5 s floor costs less than the
        # answer they carry.
        self._collect_lookups(min(3.0, max(1.5, self.remaining() - 8)))
        video, evidence, confidence = self._read_answer()
        if self.review.answer:
            return self.review.answer, "typed by the reviewer"
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
        checked, errors = ([] if self.review.stopped else sequences[:self.options.trake_check]), []
        self.result.sequences = sequences
        self._publish("checking_paths")
        futures = [(s, self.waiter.submit(self.agent.inspector.check_sequence,
                                          self.task, events, s.video_id, s.frames)) for s in checked]
        done = self._await([f for _, f in futures], lambda: self.review.stopped)
        for sequence, future in futures:
            if future not in done:
                errors.append("abandoned: stopped by the reviewer")
                continue
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
            if (index < self.options.trake_refine and not self.review.stopped
                    and (self.options.mode != "deep" or not self.over_budget())):
                original = replace(sequence, frames=list(sequence.frames), times=list(sequence.times))
                self._refine(sequence, events)
                final.append(sequence)
                if original.frames != sequence.frames:
                    final.append(original)  # the unrefined path is the next-best guess
            else:
                final.append(sequence)
        self.result.sequences = final

    def _finalize_sequences_human(self):
        """The reviewer's path first; the paths the run found follow, as alternatives.

        Nothing is checked or refined: a model picking frames is what the person is
        replacing, and re-picking their frames would undo the point of choosing them.
        """
        found = self.result.sequences or self.pool.ranked_sequences(expected_events(self.plan, self.task))
        chosen = self.review.sequence
        if chosen:
            by_frame = {(c.video_id, c.frame_idx): c.time for c in self.pool.candidates.values()}
            times = chosen.get("times") or [by_frame.get((chosen["video_id"], f), 0.0) for f in chosen["frames"]]
            found = [Sequence("human", 0, chosen["video_id"], list(chosen["frames"]),
                              [float(t) for t in times], 10.0, [])] + [
                s for s in found if (s.video_id, tuple(s.frames)) != (chosen["video_id"], tuple(chosen["frames"]))]
        self.result.sequences = found

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
