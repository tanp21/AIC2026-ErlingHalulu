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
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from agent.cli import save_run
from agent.config import VISION_ENV, AgentSettings, deployment_vision
from agent.llm import DeepSeekClient, GeminiClient, OfflineGemini, Usage
from agent.orchestrator import Agent, AgentOptions
from agent.review import HumanReview
from agent.search_client import SearchClient
from agent.tasks import Task, task_from_text
from agent.vlm import build_vision

from .core import ROOT

logger = logging.getLogger("uvicorn.error")

AGENT_RUN_TTL_SECONDS = 3600
# One run already fans out its own pools; two keeps the loopback search batches
# from crowding out human searches.
_AGENT_RUN_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent-run")
# from_env() falls back to the repository .env when it can; server-hosted runs
# must use exactly the keys from the request.
_NO_DOTENV = Path("/nonexistent")


class AgentRunConflict(RuntimeError):
    """A user already has a run in progress."""


class AgentRunReq(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    kind: Literal["kis", "qa", "trake"] = "kis"
    mode: Literal["fast", "auto", "deep"] = "fast"
    # Both may be empty: the owner's runs fall back to the keys this deployment
    # holds (see `_deployment_key`), so a field left blank is a question the server
    # answers, not a client-side error.
    deepseek_api_key: str = Field(default="", max_length=512)
    gemini_api_key: str = Field(default="", max_length=512)
    max_steps: int = Field(default=6, ge=1, le=12)
    time_budget: float | None = Field(default=None, ge=30.0, le=900.0)
    deepseek_model: str | None = Field(default=None, max_length=100)
    gemini_model: str | None = Field(default=None, max_length=100)


class AgentReviewReq(BaseModel):
    """One reviewer action; any combination of the fields may be sent at once."""
    candidate_id: str | None = Field(default=None, pattern=r"^C\d{1,6}$")
    decision: Literal["accept", "reject", "clear"] | None = None
    answer: str | None = Field(default=None, max_length=100)
    stop: bool = False


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

    def create(self, run_id, owner, task, output_dir, report_url, deployment_keys=False):
        with self._lock:
            self._purge()
            if any(run.owner == owner and run.status == "running"
                   for run in self._runs.values()):
                raise AgentRunConflict("an agent run is already in progress")
            now = self.clock()
            run = AgentRun(
                run_id=run_id, owner=owner, task=task, output_dir=output_dir,
                report_url=report_url, started=now, expires_at=now + self.ttl_seconds,
                deployment_keys=deployment_keys,
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
              time_budget: float | None, *, review: HumanReview | None = None, progress=None):
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
    vision = build_vision(settings, usage=usage, gemini=gemini if settings.gemini_api_key else None)
    search = SearchClient.from_settings(settings)
    overrides = {
        "max_steps": max_steps,
        "plan_thinking": settings.deepseek_plan_thinking,
        "spoken_numbers": settings.spoken_numbers,
    }
    if time_budget is not None:
        overrides["time_budget"] = time_budget
    try:
        agent = Agent(deepseek, gemini, search, AgentOptions.for_mode(mode, **overrides),
                      vision=vision)
        result = agent.run(task, review=review, progress=progress).to_dict()
    finally:
        search.close()
    return result, usage


#: The keys this deployment holds for UI runs, read from the app's own environment
#: (compose `env_file: .env.secure`). They belong to the owner, so only an
#: admin-started run may spend them -- a teammate's run always supplies its own, and
#: nothing in the API can reach these.
_DEPLOYMENT_KEY_ENV = {"deepseek": "DEEPSEEK_API_KEY", "gemini": "GEMINI_API_KEY"}


def _deployment_key(name: str) -> str:
    return os.environ.get(_DEPLOYMENT_KEY_ENV[name], "").strip()


def execute_run(store: AgentRunStore, run: AgentRun, req: AgentRunReq, internal_url: str):
    try:
        def key(supplied: str, name: str) -> str:
            """The request's key, or this deployment's for the owner's own run."""
            if supplied.strip():
                return supplied.strip()
            return _deployment_key(name) if run.deployment_keys else ""

        env = {
            "DEEPSEEK_API_KEY": key(req.deepseek_api_key, "deepseek"),
            "GEMINI_API_KEY": key(req.gemini_api_key, "gemini"),
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
        settings = AgentSettings.from_env(env, dotenv=_NO_DOTENV)
        result, usage = run_query(
            run.task, settings, req.mode, req.max_steps, req.time_budget,
            review=run.review,
            progress=lambda snapshot: store.update_progress(run.run_id, snapshot),
        )
        save_run([result], run.output_dir, run.report_url, usage)
        store.finish(run.run_id, result=result, usage=usage.as_dict(), error=None)
    except Exception as exc:  # reported to the owner; message never includes the keys
        logger.warning("agent run %s failed: %s", run.run_id, type(exc).__name__)
        store.finish(run.run_id, result=None, usage=None, error=f"{type(exc).__name__}: {exc}")


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
        if deployment_vision() == "gemini" and not usable(req.gemini_api_key, "gemini"):
            raise HTTPException(422, "a Gemini API key is required")
        run_id = secrets.token_hex(8)
        task = task_from_text(run_id, req.text, req.kind)
        runs = store(request)
        try:
            run = runs.create(
                run_id, request.state.login, task, agent_output_dir(run_id),
                str(request.base_url).rstrip("/"), deployment_keys=admin,
            )
        except AgentRunConflict as exc:
            raise HTTPException(409, str(exc)) from exc
        _AGENT_RUN_EXECUTOR.submit(execute_run, runs, run, req, agent_internal_url())
        return JSONResponse(run.to_payload(time.monotonic()), status_code=202)

    @app.get("/api/agent/runs/{run_id}")
    async def get_run(run_id: str, request: Request):
        return owned(request, run_id).to_payload(time.monotonic())

    @app.post("/api/agent/runs/{run_id}/review")
    async def review_run(run_id: str, req: AgentReviewReq, request: Request):
        """The owner's decision on a running query, applied by the agent as it goes."""
        run = owned(request, run_id)
        if run.status != "running":
            raise HTTPException(409, "the agent run has already finished")
        if req.decision is not None:
            if req.candidate_id is None:
                raise HTTPException(422, "a decision needs a candidate_id")
            run.review.decide(req.candidate_id, req.decision)
        if req.answer is not None:
            run.review.set_answer(req.answer)
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
