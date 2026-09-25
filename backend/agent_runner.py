"""Server-hosted retrieval-agent runs, driven from the UI (`/api/agent/*`).

The `agent/` package is normally a CLI HTTP client of this server. Here it runs
inside the server on a bounded thread pool so a teammate can start a query from
the browser: the orchestrator still talks to this same process over loopback
`/agent/search`, `/thumb`, `/frame`, `/context` and `/transcript`.

API keys arrive with the start request, live in memory only for the duration of
the run, and are never written to disk or logged. Output files (run JSON,
`report.html`, `submission.zip`) go to `AIC_AGENT_OUT_DIR`, which must be a
writable mount (`/workspace/agent` in the container, `artifacts/agent` locally).

Run state is in-process: the container runs a single uvicorn worker, and a
restart drops in-flight runs (the client can start a new one).

While a run is in progress its latest `progress` snapshot -- the candidates
Gemini is judging, with the same speech, OCR and caption context Gemini gets --
rides along on every poll, and `POST .../review` lets the owner accept or reject
a candidate, type a Q&A answer, or stop the run. The orchestrator applies those
decisions as it goes and stops waiting on Gemini once one of them settles the
query (`agent/review.py`): at a venue whose network slows Gemini to a crawl, a
teammate looking at the same frames is often the faster judge.
"""
from __future__ import annotations

import logging
import os
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from agent.cli import save_run
from agent.config import VISION_ENV, AgentSettings, deployment_vision, openrouter_key
from agent.judge import MomentJudge
from agent.llm import DeepSeekClient, GeminiClient, OfflineGemini, Usage
from agent.orchestrator import Agent, AgentOptions
from agent.review import HumanReview
from agent.search_client import SearchClient
from agent.tasks import Task, task_from_text
from agent.vlm import build_vision

from .core import ROOT

logger = logging.getLogger("uvicorn.error")

AGENT_RUN_TTL_SECONDS = 3600
# One run already fans out its own pools; the bound keeps the loopback search batches
# from crowding out human searches. A run with a human judge holds its worker for as
# long as the person takes, so it is wider than the model-only runs ever needed.
_AGENT_RUN_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="agent-run")
# from_env() falls back to the repository .env when it can; server-hosted runs
# must use exactly the keys from the request.
_NO_DOTENV = Path("/nonexistent")


VIDEO_ID_PATTERN = r"^[A-Z]\d{2}[_-]V\d{3}$"


class AgentRunConflict(RuntimeError):
    """A user already has a run in progress."""


class AgentRunReq(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    kind: Literal["kis", "qa", "trake"] = "kis"
    mode: Literal["fast", "auto", "deep"] = "fast"
    # `human`: no model sees a frame, the person does (agent/review.py). Needs no Gemini
    # key; a key given anyway is only used for the Google-Search lookups.
    judge: Literal["model", "human"] = "model"
    # Both may be empty: the owner's runs fall back to the keys this deployment
    # holds (see `_deployment_key`), so a field left blank is a question the server
    # answers, not a client-side error.
    deepseek_api_key: str = Field(default="", max_length=512)
    gemini_api_key: str = Field(default="", max_length=512)
    # The team's shared key (see `_SHARED_KEYS`): anyone's run falls back to the deployment's,
    # so this is only for a teammate who wants the run billed to their own OpenRouter account.
    openrouter_api_key: str = Field(default="", max_length=512)
    max_steps: int = Field(default=6, ge=1, le=12)
    time_budget: float | None = Field(default=None, ge=30.0, le=900.0)
    deepseek_model: str | None = Field(default=None, max_length=100)
    gemini_model: str | None = Field(default=None, max_length=100)


class AgentSequenceReq(BaseModel):
    """The TRAKE path a human assembled: one frame per event, in order."""
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)
    frames: list[int] = Field(min_length=1, max_length=12)
    times: list[float] | None = Field(default=None, max_length=12)

    @field_validator("frames")
    @classmethod
    def _non_negative(cls, frames):
        if any(frame < 0 for frame in frames):
            raise ValueError("frame indices must not be negative")
        return frames

    @model_validator(mode="after")
    def _one_time_each(self):
        if self.times is not None and len(self.times) != len(self.frames):
            raise ValueError("one time per frame")
        return self


class AgentCommandReq(BaseModel):
    """A search a human asks the run to do: more, inside a video, around a frame, or an exact frame."""
    type: Literal["more", "scope", "neighbours", "add_frame"]
    hint: str | None = Field(default=None, max_length=300)
    video_id: str | None = Field(default=None, pattern=VIDEO_ID_PATTERN)
    cid: str | None = Field(default=None, pattern=r"^C\d{1,6}$")
    frame_idx: int | None = Field(default=None, ge=0)
    time: float | None = Field(default=None, ge=0)
    accept: bool = False

    @model_validator(mode="after")
    def _has_what_it_needs(self):
        needs = {"more": (), "scope": ("video_id",), "neighbours": ("cid",),
                 "add_frame": ("video_id", "frame_idx")}[self.type]
        missing = [name for name in needs if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.type} needs {', '.join(missing)}")
        return self


class AgentReviewReq(BaseModel):
    """One reviewer action; any combination of the fields may be sent at once."""
    candidate_id: str | None = Field(default=None, pattern=r"^C\d{1,6}$")
    decision: Literal["accept", "reject", "clear"] | None = None
    #: A 0-10 verdict between accept and reject (a human judge only).
    score: float | None = Field(default=None, ge=0, le=10)
    answer: str | None = Field(default=None, max_length=100)
    sequence: AgentSequenceReq | None = None
    command: AgentCommandReq | None = None
    stop: bool = False
    #: Submit: the reviewer is done (a human judge only).
    finish: bool = False
    #: The candidate order on the reviewer's screen when they submit or stop (a human judge only).
    order: list[str] | None = Field(default=None, max_length=300)

    @field_validator("order")
    @classmethod
    def _candidate_ids(cls, order):
        if order is not None and any(not re.fullmatch(r"C\d{1,6}", cid) for cid in order):
            raise ValueError("order holds candidate ids")
        return order


@dataclass
class AgentRun:
    run_id: str
    owner: str
    task: Task
    output_dir: Path
    report_url: str
    started: float
    expires_at: float
    status: str = "running"
    result: dict | None = None
    usage: dict | None = None
    error: str | None = None
    review: HumanReview = field(default_factory=HumanReview)
    progress: dict | None = None
    #: Set when the owner started this run, which is the only case a request may
    #: spend the deployment's own model keys. Decided from the request identity at
    #: creation, never from the request body, and carried on the run so the executor
    #: thread cannot be talked into a different answer.
    deployment_keys: bool = False

    def to_payload(self, now: float):
        return {
            "run_id": self.run_id,
            "status": self.status,
            "query_id": self.task.query_id,
            "kind": self.task.kind,
            "query": self.task.text,
            "elapsed": round((self.result or {}).get("elapsed", now - self.started), 1),
            "result": self.result,
            "usage": self.usage,
            "error": self.error,
            "progress": self.progress,
            "review": self.review.snapshot(),
        }


class AgentRunStore:
    """Owner-scoped run registry with a TTL, mirroring `SearchSnapshotStore`."""

    def __init__(self, ttl_seconds=AGENT_RUN_TTL_SECONDS, clock=time.monotonic):
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._lock = threading.Lock()
        self._runs: dict[str, AgentRun] = {}

    def create(self, run_id, owner, task, output_dir, report_url, deployment_keys=False,
               judge="model"):
        with self._lock:
            self._purge()
            if any(run.owner == owner and run.status == "running"
                   for run in self._runs.values()):
                raise AgentRunConflict("an agent run is already in progress")
            now = self.clock()
            run = AgentRun(
                run_id=run_id, owner=owner, task=task, output_dir=output_dir,
                report_url=report_url, started=now, expires_at=now + self.ttl_seconds,
                deployment_keys=deployment_keys, review=HumanReview(judge=judge),
            )
            self._runs[run_id] = run
            return run

    def get(self, owner, run_id):
        with self._lock:
            self._purge()
            run = self._runs.get(run_id)
            if run is None or run.owner != owner:
                raise LookupError("agent run not found")
            run.expires_at = self.clock() + self.ttl_seconds
            return run

    def active(self, owner):
        """The owner's live run, or None.

        A reloaded page (or a second tab) holds no run id to poll, and without this its
        only way back to a run it started is a 409 from the start route.
        """
        with self._lock:
            self._purge()
            for run in self._runs.values():
                if run.owner == owner and run.status == "running":
                    return run
            return None

    def update_progress(self, run_id, snapshot):
        with self._lock:
            run = self._runs.get(run_id)
            if run is not None:
                run.progress = snapshot

    def finish(self, run_id, *, result, usage, error):
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return
            run.result = result
            run.usage = usage
            run.error = error
            run.status = "error" if error and not result else "done"

    def _purge(self):
        now = self.clock()
        for run_id, run in list(self._runs.items()):
            if run.expires_at <= now:
                del self._runs[run_id]


def run_query(task: Task, settings: AgentSettings, mode: str, max_steps: int,
              time_budget: float | None, *, review: HumanReview | None = None, progress=None,
              judge: str = "model", human_idle_timeout: float | None = None):
    """Build the clients and run one task; split out so tests can stub it."""
    usage = Usage()
    deepseek = DeepSeekClient(
        settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url,
        thinking=settings.deepseek_thinking, usage=usage,
    )
    # Same tuning the CLI gives a fast run, and it matters more here: a teammate's key
    # may be a free-tier one, where a rate-limited call retries five times and sleeps
    # 2+4+8+16 s doing it. Runs from the UI default to fast, so fast is what they get.
    # No `hedge_after`: hedging races a duplicate request, which doubles the quota a
    # free-tier key spends and is the opposite of what a congested one needs.
    fast = mode in ("fast", "auto")
    gemini = (GeminiClient(
        settings.gemini_api_key, settings.gemini_model,
        thinking_level=settings.gemini_thinking_level, usage=usage,
        **({"timeout": 40.0, "retries": 2, "max_backoff": 4.0} if fast else {}),
    ) if settings.gemini_api_key else OfflineGemini())
    # A human judge never reaches a vision model, so none is built: OfflineGemini raises
    # if anything ever tried to hand it a frame, which is the guarantee the mode makes.
    vision = (OfflineGemini() if judge == "human" else
              build_vision(settings, usage=usage, gemini=gemini if settings.gemini_api_key else None))
    search = SearchClient.from_settings(settings)
    overrides = {
        "max_steps": max_steps,
        "plan_thinking": settings.deepseek_plan_thinking,
        "spoken_numbers": settings.spoken_numbers,
    }
    # A model-judged run on OpenRouter judges each candidate as a moment (its keyframe and
    # the shots around it, `agent/judge.py`), which also sets the longer budgets that needs.
    judge_style = settings.judge_style if judge != "human" else "keyframe"
    overrides["judge_style"] = judge_style
    if time_budget is not None:
        overrides["time_budget"] = time_budget
    if judge == "human":
        overrides["judge"] = "human"
        overrides["checklist_vision"] = settings.checklist_vision
        overrides["checklist_vision_model"] = settings.checklist_vision_model
        if human_idle_timeout is not None:
            overrides["human_idle_timeout"] = human_idle_timeout
    try:
        agent = Agent(deepseek, gemini, search, AgentOptions.for_mode(mode, **overrides),
                      vision=vision,
                      judge=MomentJudge(vision, search) if judge_style == "moment" else None)
        result = agent.run(task, review=review, progress=progress).to_dict()
    finally:
        search.close()
    return result, usage


#: The keys this deployment holds for UI runs, read from the app's own environment
#: (compose `env_file: .env.secure`). They belong to the owner, so only an
#: admin-started run may spend them -- a teammate's run always supplies its own, and
#: nothing in the API can reach these.
_DEPLOYMENT_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY",
                       "openrouter": "OPENROUTER_API_KEY"}
#: ...with one deliberate exception. `OPENROUTER_API_KEY` is the TEAM's key, not the owner's
#: personal quota: it is what the Search tab's "Check by AI" already spends for whoever presses
#: the button, at well under a cent a call, and it is the fallback for the whole tab now that
#: the Gemini quota is gone. Admin-gating it would leave four of the five teammates unable to
#: run the agent at all. It still never leaves the server -- a run reads it here, the API
#: cannot set it, and nothing echoes it back.
_SHARED_KEYS = frozenset({"openrouter"})


def _deployment_key(name: str) -> str:
    if name == "openrouter":
        return openrouter_key()  # OPENROUTER_API_KEY, or the QWEN_API_KEY it was handed round as
    return os.environ.get(_DEPLOYMENT_KEY_ENV[name], "").strip()


def execute_run(store: AgentRunStore, run: AgentRun, req: AgentRunReq, internal_url: str):
    try:
        def key(supplied: str, name: str) -> str:
            """The request's key, or this deployment's -- for the owner, or for a shared key."""
            if supplied.strip():
                return supplied.strip()
            if run.deployment_keys or name in _SHARED_KEYS:
                return _deployment_key(name)
            return ""

        env = {
            "DEEPSEEK_API_KEY": key(req.deepseek_api_key, "deepseek"),
            "GEMINI_API_KEY": key(req.gemini_api_key, "gemini"),
            "OPENROUTER_API_KEY": key(req.openrouter_api_key, "openrouter"),
            "AIC_AGENT_URL": internal_url,
            "AIC_AGENT_TRUSTED_USER": run.owner,
            # The server has /agent/search; skip the compat fallback.
            "AIC_AGENT_SEARCH_API": "batch",
        }
        if req.deepseek_model:
            env["DEEPSEEK_MODEL"] = req.deepseek_model
        if req.gemini_model:
            env["GEMINI_MODEL"] = req.gemini_model
        # The vision backend belongs to the deployment, not to the requester.
        for name in VISION_ENV:
            if os.environ.get(name):
                env[name] = os.environ[name]
        # Which client reads the checklist's pixels is also the deployment's call, same
        # reasoning as the vision backend above -- a teammate's pasted key just pays for it.
        if os.environ.get("AIC_AGENT_CHECKLIST_MODEL"):
            env["AIC_AGENT_CHECKLIST_MODEL"] = os.environ["AIC_AGENT_CHECKLIST_MODEL"]
        if req.judge == "human":
            # `AgentSettings` demands a Gemini key for a Gemini vision backend, and a
            # human-judged run has no vision backend at all (`run_query` builds none).
            env["AIC_AGENT_VISION"] = "deepseek"
        settings = AgentSettings.from_env(env, dotenv=_NO_DOTENV)
        result, usage = run_query(
            run.task, settings, req.mode, req.max_steps, req.time_budget,
            review=run.review,
            progress=lambda snapshot: store.update_progress(run.run_id, snapshot),
            **({"judge": "human", "human_idle_timeout": human_idle_timeout()}
               if req.judge == "human" else {}),
        )
        save_run([result], run.output_dir, run.report_url, usage)
        store.finish(run.run_id, result=result, usage=usage.as_dict(), error=None)
    except Exception as exc:  # reported to the owner; message never includes the keys
        logger.warning("agent run %s failed: %s", run.run_id, type(exc).__name__)
        store.finish(run.run_id, result=None, usage=None, error=f"{type(exc).__name__}: {exc}")


def human_idle_timeout() -> float:
    """Seconds of reviewer silence after which a human-judged run finishes on its own."""
    try:
        return max(30.0, float(os.environ.get("AIC_AGENT_HUMAN_TIMEOUT", "900")))
    except ValueError:
        return 900.0


def agent_output_dir(run_id: str) -> Path:
    return Path(os.environ.get("AIC_AGENT_OUT_DIR", ROOT / "artifacts" / "agent")) / run_id


def agent_internal_url() -> str:
    return os.environ.get("AIC_AGENT_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")


def shutdown_agent_runs():
    _AGENT_RUN_EXECUTOR.shutdown(wait=False, cancel_futures=True)


def install_agent_routes(app) -> None:
    def store(request: Request) -> AgentRunStore:
        return request.app.state.agent_runs

    def owned(request: Request, run_id: str) -> AgentRun:
        try:
            return store(request).get(request.state.login, run_id)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/agent/runs")
    async def start_run(req: AgentRunReq, request: Request):
        # A key may come from the request or -- for the owner's own runs -- from this
        # deployment's environment, which is why the owner does not have to paste one
        # at all. A teammate never falls back: those keys are the owner's, and
        # spending them is the one thing this gate exists to prevent.
        admin = bool(getattr(request.state, "is_admin", False))

        def usable(supplied: str, name: str) -> bool:
            return bool(supplied.strip()) or (admin and bool(_deployment_key(name)))

        if not usable(req.deepseek_api_key, "deepseek"):
            raise HTTPException(422, "a DeepSeek API key is required")
        # A Gemini key is only needed when this deployment judges with Gemini: on a
        # local vision backend a run works without one, and only the web lookup goes
        # without.
        if (req.judge == "model" and deployment_vision() == "gemini"
                and not usable(req.gemini_api_key, "gemini")):
            raise HTTPException(422, "a Gemini API key is required")
        # OpenRouter is the team's shared key, so every teammate may fall back to the
        # deployment's -- unlike the two above. Only a deployment with no key at all fails.
        if (req.judge == "model" and deployment_vision() == "openrouter"
                and not (req.openrouter_api_key.strip() or _deployment_key("openrouter"))):
            raise HTTPException(422, "an OpenRouter API key is required")
        run_id = secrets.token_hex(8)
        task = task_from_text(run_id, req.text, req.kind)
        runs = store(request)
        try:
            run = runs.create(
                run_id, request.state.login, task, agent_output_dir(run_id),
                str(request.base_url).rstrip("/"), deployment_keys=admin, judge=req.judge,
            )
        except AgentRunConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        _AGENT_RUN_EXECUTOR.submit(execute_run, runs, run, req, agent_internal_url())
        return JSONResponse(run.to_payload(time.monotonic()), status_code=202)

    @app.get("/api/agent/runs/active")
    async def active_run(request: Request):
        """The caller's own live run, so a reloaded page adopts it instead of hitting the
        start route's "already in progress" wall with nothing to look at."""
        run = store(request).active(request.state.login)
        return {"run": run.to_payload(time.monotonic()) if run else None}

    @app.get("/api/agent/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        return owned(request, run_id).to_payload(time.monotonic())

    @app.post("/api/agent/runs/{run_id}/review")
    async def review_run(run_id: str, req: AgentReviewReq, request: Request):
        """The owner's decision on a running query, applied by the agent as it goes."""
        run = owned(request, run_id)
        if run.status != "running":
            raise HTTPException(409, "the agent run has already finished")
        human = run.review.judge == "human"
        if (req.decision is not None or req.score is not None) and req.candidate_id is None:
            raise HTTPException(422, "a decision needs a candidate_id")
        if req.score is not None or req.sequence or req.command or req.finish or req.order is not None:
            if not human:
                raise HTTPException(422, "scores, sequences, commands and finish need a human judge")
        if req.decision is not None:
            run.review.decide(req.candidate_id, req.decision)
        if req.score is not None:
            run.review.score(req.candidate_id, req.score)
        if req.answer is not None:
            run.review.set_answer(req.answer)
        if req.sequence is not None:
            run.review.set_sequence(req.sequence.video_id, req.sequence.frames, req.sequence.times)
        if req.command is not None:
            run.review.push(req.command.model_dump(exclude_none=True))
        if req.order is not None:
            run.review.set_order(req.order)  # before finish or stop, which end the wait
        if req.finish:
            run.review.finish()
        if req.stop:
            run.review.stop()
        return run.to_payload(time.monotonic())

    @app.get("/api/agent/runs/{run_id}/submission.zip")
    async def download_submission(run_id: str, request: Request):
        run = owned(request, run_id)
        if run.status != "done":
            raise HTTPException(409, "the agent run has not finished")
        archive = run.output_dir / "submission.zip"
        if not archive.exists():
            raise HTTPException(404, "this run produced no submission")
        return FileResponse(
            archive, media_type="application/zip",
            filename=f"{run.task.query_id}-submission.zip",
        )
