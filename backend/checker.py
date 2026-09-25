"""`POST /api/check` -- the Search tab's "Check by AI", on demand, one moment at a time.

A person scanning the results grid has one question about the frame in front of them: *is this
the moment?* The retrieval agent can answer it, but only as part of a whole run. This answers it
for one frame, when they press the button, in a few seconds, and remembers the answer.

The orchestration is `agent.checker`; everything here is the surface around it:

- **The key is the SHOT, not the frame.** Every keyframe of one shot, for one query, is one
  cache entry -- because the window of shots a check looks at is essentially the same for any of
  them, so judging them separately would be the same call twice. This is what makes stepping
  through the filmstrip free after the first press, and what lets a teammate reuse an answer
  someone else already paid for. `force` re-reads one exact frame when a person wants that.
- **The checklist is cached per query**, so checking twenty candidates of one query costs one
  text call and twenty vision calls, not forty.
- **One shared key.** `OPENROUTER_API_KEY` lives on the server and is never sent to a browser;
  `/session` says only whether it is there, which is what hides the button. Unlike the Retrieval
  agent tab, nobody pastes anything: a check costs well under a cent and the point is that it is
  one press.
- **The words and pixels come back over loopback**, through the same `SearchClient` the agent
  uses, so a split-host deployment (`AIC_CORE_URL`) works unchanged -- `/thumb`, `/shots`,
  `/transcript` and `/ocr` are all already proxied to the compute host.
"""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import logging
import os
from pathlib import Path
import threading
import time

from fastapi import HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator
from starlette.concurrency import run_in_threadpool
from typing import Literal

from agent.checker import (
    CHECK_MAX_SECONDS, CHECK_MIN_SECONDS, CHECK_WINDOW_SECONDS, CheckError, CheckQuery,
    CheckTarget, check_moment,
)
from agent.config import read_dotenv
from agent.evidence import EvidenceStore, QueryDetails, query_details, span_around
from agent.openrouter import DEFAULT_BASE_URL, DEFAULT_MODEL, OpenRouterClient
from agent.prompts import CHECK_DETAILS_SYSTEM
from agent.search_client import SearchClient

from . import search_core, video_ids

logger = logging.getLogger("aic.checker")
ROOT = Path(__file__).resolve().parent.parent

VIDEO_ID_PATTERN = video_ids.VIDEO_ID_PATTERN
#: How long a check and a checklist stay usable. A competition round is shorter than this, and a
#: query's answer does not change while it runs.
CHECK_TTL_SECONDS = int(os.environ.get("AIC_CHECK_TTL_SECONDS", "3600"))
CHECK_CACHE_MAX = 512
DETAILS_CACHE_MAX = 256
#: Checks one person may have in flight, and how many the whole server runs at once. Both are
#: about the wait, not the money: a second press while the first is still running is almost
#: always the same person changing their mind, and the client aborts the first request anyway.
PER_USER_LIMIT = 2
GLOBAL_LIMIT = 6
#: Without shot boundaries a moment is bucketed by time instead, so scrubbing still reuses.
FALLBACK_BUCKET_SECONDS = 10.0


# -- configuration ----------------------------------------------------------------------------
def _dotenv() -> dict[str, str]:
    """The repository `.env`, read once. The environment always wins over it.

    A deployment sets `OPENROUTER_API_KEY` in its compose environment; reading `.env` as a
    fallback is what makes `uvicorn backend.app:app` work from a clone without exporting it
    first, the same fallback `AgentSettings.from_env` gives the CLI. This is deliberately NOT
    what `backend/agent_runner.py` does -- a hosted agent run must use exactly the keys the
    request carried, because those are one teammate's personal credentials. This key is the
    team's, and the deployment is the only place it comes from.
    """
    global _DOTENV
    if _DOTENV is None:
        _DOTENV = read_dotenv(ROOT / ".env")
    return _DOTENV


_DOTENV: dict[str, str] | None = None


def _setting(name: str, default: str = "") -> str:
    return (os.environ.get(name) or _dotenv().get(name, "") or default).strip()


def checker_key() -> str:
    return _setting("OPENROUTER_API_KEY")


def checker_model() -> str:
    return _setting("OPENROUTER_MODEL", DEFAULT_MODEL)


def default_window() -> float:
    try:
        value = float(_setting("AIC_CHECK_WINDOW_SECONDS", str(CHECK_WINDOW_SECONDS)))
    except ValueError:
        return CHECK_WINDOW_SECONDS
    return max(CHECK_MIN_SECONDS, min(CHECK_MAX_SECONDS, value))


def check_config() -> dict:
    """What `/session` tells the UI, so the button is hidden rather than failing when pressed."""
    return {"available": bool(checker_key()), "model": checker_model(),
            "window_seconds": default_window()}


def internal_url() -> str:
    return _setting("AIC_AGENT_INTERNAL_URL", "http://127.0.0.1:8000").rstrip("/")


# -- the clients, one per process ---------------------------------------------------------------
_CLIENT_LOCK = threading.Lock()
_JUDGE: tuple[str, str, OpenRouterClient] | None = None
_SEARCH: SearchClient | None = None
_STORE: EvidenceStore | None = None


def judge_client() -> OpenRouterClient:
    """The OpenRouter client, rebuilt when the key or model changes underneath it."""
    global _JUDGE
    key, model = checker_key(), checker_model()
    if not key:
        raise HTTPException(503, "AI check is not configured (OPENROUTER_API_KEY is not set)")
    with _CLIENT_LOCK:
        if _JUDGE is None or _JUDGE[0] != key or _JUDGE[1] != model:
            if _JUDGE is not None:
                _JUDGE[2].close()
            _JUDGE = (key, model, OpenRouterClient(
                key, model, _setting("OPENROUTER_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL))
        return _JUDGE[2]


def _internal_identity(request: Request) -> str:
    """The login the checker's own loopback fetches run under.

    One identity for the whole process, not the person who pressed the button: these are
    server-internal reads of frames and transcripts on behalf of a check, they share one
    connection pool and one per-video text cache, and the audit line that names the real person
    is the one for `/api/check` itself. None of these routes go through `LatestWinsCoordinator`,
    so a check can never supersede anyone's search.
    """
    config = getattr(request.app.state, "access_config", None)
    return getattr(config, "admin_user", None) or getattr(request.state, "login", "")


def search_client(request: Request) -> SearchClient:
    global _SEARCH
    with _CLIENT_LOCK:
        if _SEARCH is None:
            _SEARCH = SearchClient(internal_url(), trusted_user=_internal_identity(request),
                                   api="batch", captions=False)
        return _SEARCH


def evidence_store(request: Request) -> EvidenceStore:
    global _STORE
    client = search_client(request)
    with _CLIENT_LOCK:
        if _STORE is None or _STORE.search is not client:
            _STORE = EvidenceStore(client, maxsize=192)
        return _STORE


def shutdown_checks() -> None:
    """Close the process-wide clients; called from the app's lifespan shutdown."""
    global _JUDGE, _SEARCH, _STORE
    with _CLIENT_LOCK:
        if _JUDGE is not None:
            _JUDGE[2].close()
        if _SEARCH is not None:
            _SEARCH.close()
        _JUDGE = _SEARCH = _STORE = None


# -- the cache ----------------------------------------------------------------------------------
@dataclass
class _Entry:
    value: dict
    expires_at: float


class CheckStore:
    """Checks by cache key, with a TTL and a single-flight slot per key.

    Deliberately NOT owner-scoped, unlike `SearchSnapshotStore` and `AgentRunStore`: what a check
    says about a shot is a fact about the video and the query, not about the person who asked, so
    the second teammate to look at the same candidate gets the first one's answer for free. The
    key already carries the query, so two people working different queries never collide.

    `claim` is what stops the same call being made twice at once: the first caller in gets the
    slot and runs, and anyone else on that key waits on the same future instead of paying for a
    duplicate.
    """

    def __init__(self, ttl_seconds: float = CHECK_TTL_SECONDS, maxsize: int = CHECK_CACHE_MAX,
                 clock=time.monotonic):
        self.ttl = float(ttl_seconds)
        self.maxsize = int(maxsize)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _Entry] = {}
        self._running: dict[str, Future] = {}

    def get(self, key: str) -> dict | None:
        with self._lock:
            self._purge()
            entry = self._entries.get(key)
            return dict(entry.value) if entry else None

    def running(self, key: str) -> bool:
        with self._lock:
            return key in self._running

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._purge()
            self._entries[key] = _Entry(dict(value), self._clock() + self.ttl)
            while len(self._entries) > self.maxsize:
                oldest = min(self._entries, key=lambda name: self._entries[name].expires_at)
                self._entries.pop(oldest, None)

    def claim(self, key: str) -> tuple[Future, bool]:
        """`(future, mine)`. `mine` means this caller has to run the work and settle the future."""
        with self._lock:
            existing = self._running.get(key)
            if existing is not None:
                return existing, False
            future: Future = Future()
            self._running[key] = future
            return future, True

    def settle(self, key: str, future: Future, value: dict | None, error: BaseException | None):
        with self._lock:
            self._running.pop(key, None)
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(value)

    def _purge(self) -> None:
        now = self._clock()
        for key in [name for name, entry in self._entries.items() if entry.expires_at <= now]:
            self._entries.pop(key, None)


class DetailsStore:
    """The query's checklist, cached by query, so N candidates cost one details call."""

    def __init__(self, ttl_seconds: float = CHECK_TTL_SECONDS, maxsize: int = DETAILS_CACHE_MAX,
                 clock=time.monotonic):
        self.ttl = float(ttl_seconds)
        self.maxsize = int(maxsize)
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[QueryDetails, float]] = {}

    def get_or_make(self, key: str, build) -> QueryDetails:
        now = self._clock()
        with self._lock:
            found = self._entries.get(key)
            if found and found[1] > now:
                return found[0]
        details = build()
        with self._lock:
            self._entries[key] = (details, self._clock() + self.ttl)
            while len(self._entries) > self.maxsize:
                oldest = min(self._entries, key=lambda name: self._entries[name][1])
                self._entries.pop(oldest, None)
        return details


def _store(request: Request) -> CheckStore:
    store = getattr(request.app.state, "check_results", None)
    if store is None:                       # a test app that never ran the lifespan
        store = CheckStore()
        request.app.state.check_results = store
    return store


def _details_store(request: Request) -> DetailsStore:
    store = getattr(request.app.state, "check_details", None)
    if store is None:
        store = DetailsStore()
        request.app.state.check_details = store
    return store


# -- keys ----------------------------------------------------------------------------------------
def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]


def normalise_query(text: str) -> str:
    return " ".join((text or "").split()).lower()


def query_key(query: CheckQuery) -> str:
    return _digest(query.kind, normalise_query(query.visual_text), normalise_query(query.audio_text),
                   normalise_query(query.ocr_text), normalise_query(query.question or ""))


def check_key(query: CheckQuery, video_id: str, bucket: str, window: float, model: str) -> str:
    return _digest(query_key(query), video_id, bucket, f"{window:.0f}", model)


@dataclass(frozen=True)
class _Placement:
    """Where a frame sits: which shot holds it, and which frames that answer therefore covers."""
    bucket: str
    shot_index: int | None
    covers: dict | None = field(default=None)


def place(search, video_id: str, frame_idx: int, at: float) -> _Placement:
    """The shot a frame belongs to, or a time bucket when the video has no shot boundaries.

    The shot is the unit a check is cached by, because it is the unit the check actually reads:
    the window it looks at is built shot by shot outward from the moment, so two keyframes of one
    shot would be handed almost exactly the same images. A video with no boundaries (the 2025 `K*`
    videos, or a 2026 video before `detect_shots` has run) falls back to a coarse time bucket, so
    scrubbing still reuses instead of calling on every seek.
    """
    try:
        shots = search.shots(video_id) or []
    except Exception as exc:
        logger.warning("check: no shots for %s (%s)", video_id, type(exc).__name__)
        shots = []
    for shot in shots:
        if shot["start_frame"] <= frame_idx <= shot["end_frame"]:
            return _Placement(f"s{shot['index']}", shot["index"], {
                "start_frame": shot["start_frame"], "end_frame": shot["end_frame"],
                "start": shot["start"], "end": shot["end"],
            })
    bucket = int(at // FALLBACK_BUCKET_SECONDS)
    return _Placement(f"t{bucket}", None, {
        "start_frame": None, "end_frame": None,
        "start": bucket * FALLBACK_BUCKET_SECONDS,
        "end": (bucket + 1) * FALLBACK_BUCKET_SECONDS,
    })


# -- requests -------------------------------------------------------------------------------------
class CheckReq(BaseModel):
    """The same three channels a real search request carries (`backend.app.SearchReq`), not one
    flattened sentence: a check has to know whether it is reading a scene description, words to
    find spoken, or a string to find written on screen, and those mean different things to the
    judge (`agent.checker.CheckQuery`). The Search tab fills these from whichever field the
    person actually searched with, not from whatever the query box currently shows.
    """
    visual_text: str = Field(default="", max_length=2000)
    audio_text: str = Field(default="", max_length=2000)
    ocr_text: str = Field(default="", max_length=2000)
    kind: Literal["kis", "qa", "trake"] = "kis"
    question: str | None = Field(default=None, max_length=500)
    video_id: str = Field(pattern=VIDEO_ID_PATTERN)
    #: One of the two. The player cannot always resolve an exact frame while playing
    #: (`frameMap.resolveFrameId` answers null off an exact PTS), so it sends the time instead
    #: and the server maps it to the nearest indexed keyframe.
    frame_idx: int | None = Field(default=None, ge=0)
    time: float | None = Field(default=None, ge=0)
    window_seconds: float = Field(default=0, ge=0, le=CHECK_MAX_SECONDS)
    profile: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    force: bool = False

    @model_validator(mode="after")
    def require_a_moment(self):
        if self.frame_idx is None and self.time is None:
            raise ValueError("either frame_idx or time is required")
        if not (self.visual_text.strip() or self.audio_text.strip() or self.ocr_text.strip()):
            raise ValueError("at least one of visual_text, audio_text or ocr_text is required")
        return self


def _as_query(req: CheckReq) -> CheckQuery:
    return CheckQuery(visual_text=req.visual_text.strip(), audio_text=req.audio_text.strip(),
                      ocr_text=req.ocr_text.strip(), kind=req.kind, question=req.question)


def _resolve_moment(video_id: str, frame_idx: int | None, at: float | None,
                    profile: str | None) -> CheckTarget:
    """A request's `frame_idx` or `time` as one indexed moment, with its exact time.

    `search_core.annotate` is what `/agent/search` already uses to put a caption hit's time on a
    keyframe, and it is remote-aware, so this works on a split-host teammate's app too.
    """
    point = {"video_id": video_id}
    if frame_idx is not None:
        point["frame_idx"] = int(frame_idx)
    else:
        point["time"] = float(at)
    try:
        mark = (search_core.annotate([point], profile or search_core.default_profile()) or [None])[0]
    except Exception as exc:
        logger.warning("check: annotate failed for %s (%s)", video_id, type(exc).__name__)
        mark = None
    if mark is None:
        if frame_idx is None:
            raise HTTPException(404, f"{video_id} has no indexed keyframe near that time")
        # A frame the person stepped to in the player may sit outside the index, and the frame
        # itself is still cuttable -- judge exactly what they are looking at.
        return CheckTarget(video_id, int(frame_idx), float(at or 0.0))
    return CheckTarget(video_id, int(mark["frame_idx"]),
                       float(mark.get("time", at if at is not None else 0.0)))


# -- limits ---------------------------------------------------------------------------------------
_INFLIGHT_LOCK = threading.Lock()
_INFLIGHT: dict[str, int] = {}


class _Slot:
    """A per-user and global in-flight slot, released however the request ends."""

    def __init__(self, login: str):
        self.login = login

    def __enter__(self):
        with _INFLIGHT_LOCK:
            if sum(_INFLIGHT.values()) >= GLOBAL_LIMIT:
                raise HTTPException(429, "too many checks running; try again in a moment")
            if _INFLIGHT.get(self.login, 0) >= PER_USER_LIMIT:
                raise HTTPException(429, "you already have a check running")
            _INFLIGHT[self.login] = _INFLIGHT.get(self.login, 0) + 1
        return self

    def __exit__(self, *_):
        with _INFLIGHT_LOCK:
            remaining = _INFLIGHT.get(self.login, 1) - 1
            if remaining > 0:
                _INFLIGHT[self.login] = remaining
            else:
                _INFLIGHT.pop(self.login, None)
        return False


# -- the work --------------------------------------------------------------------------------------
def run_check(request: Request, req: CheckReq) -> dict:
    """One check, cache first. Runs on a worker thread; raises `HTTPException` for the route."""
    login = getattr(request.state, "login", "")
    query = _as_query(req)
    model = checker_model()
    window = req.window_seconds or default_window()
    client = judge_client()
    search = search_client(request)
    target = _resolve_moment(req.video_id, req.frame_idx, req.time, req.profile)
    placement = place(search, target.video_id, target.frame_idx, target.time)
    # `force` asks about this exact frame rather than its shot, so it gets a key of its own --
    # it neither reads nor overwrites the shot's answer, which other frames are still using.
    bucket = f"f{target.frame_idx}" if req.force else placement.bucket
    key = check_key(query, target.video_id, bucket, window, model)

    store = _store(request)
    if not req.force and (cached := store.get(key)) is not None:
        return {**cached, "cached": True, "latency_ms": 0.0}

    future, mine = store.claim(key)
    if not mine:
        # Someone else is already asking exactly this. Wait for their answer instead of paying
        # for the same call again; their failure is ours too.
        value = future.result()
        return {**value, "cached": True, "latency_ms": 0.0}

    started = time.perf_counter()
    try:
        # The query's checklist and the moment's transcript/OCR window are independent of each
        # other, and on a cold check they are the two costs BEFORE either model call can start --
        # measured at ~5.7 s and ~2.3 s. Fetching them concurrently hides the shorter under the
        # longer, where the serial default paid both. `check_moment` then runs its own text/vision
        # pair in parallel as before.
        span_start = max(CHECK_MIN_SECONDS, min(CHECK_MAX_SECONDS, float(window)))
        with ThreadPoolExecutor(2) as pool:
            details_future = pool.submit(
                _details_store(request).get_or_make, query_key(query),
                lambda: query_details(client, _DetailsTask(query.description, query.kind),
                                      system=CHECK_DETAILS_SYSTEM))
            span_future = pool.submit(span_around, evidence_store(request),
                                      target.video_id, target.time - span_start,
                                      target.time + span_start)
            details = details_future.result()
            span = span_future.result()
        result = check_moment(client, search, evidence_store(request),
                              query=query, target=target, details=details,
                              window_seconds=window, span=span)
    except CheckError as exc:
        store.settle(key, future, None, HTTPException(502, f"the AI check failed: {exc}"))
        raise HTTPException(502, f"the AI check failed: {exc}") from exc
    except HTTPException as exc:
        store.settle(key, future, None, exc)
        raise
    except Exception as exc:
        logger.warning("check failed for %s: %s", req.video_id, type(exc).__name__)
        failure = HTTPException(502, "the AI check failed")
        store.settle(key, future, None, failure)
        raise failure from exc
    value = {
        **result, "key": key, "model": model, "by": login,
        "shot_index": placement.shot_index,
        "covers": None if req.force else placement.covers,
        "checked_at": time.time(),
    }
    store.put(key, value)
    store.settle(key, future, value, None)
    return {**value, "cached": False, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}


@dataclass(frozen=True)
class _DetailsTask:
    """The two fields `evidence.query_details` reads off an agent `Task`.

    `text` is `CheckQuery.description` here -- the channels the person actually searched with,
    each labelled in Vietnamese, not one flattened sentence (see `CheckReq`).
    """
    text: str
    kind: str


def install_check_routes(app) -> None:
    """Register `/api/check` routes.

    `add_api_route` rather than a router, for the same reason the rest of the backend does it:
    a router object would hide these from `app.routes` iteration.
    """

    async def check(request: Request, req: CheckReq):
        if not checker_key():
            raise HTTPException(503, "AI check is not configured (OPENROUTER_API_KEY is not set)")
        with _Slot(getattr(request.state, "login", "")):
            return await run_in_threadpool(run_check, request, req)

    async def lookup(
        request: Request,
        video_id: str = Query(pattern=VIDEO_ID_PATTERN),
        visual_text: str = Query(default="", max_length=2000),
        audio_text: str = Query(default="", max_length=2000),
        ocr_text: str = Query(default="", max_length=2000),
        kind: Literal["kis", "qa", "trake"] = Query(default="kis"),
        question: str | None = Query(default=None, max_length=500),
        frame_idx: int | None = Query(default=None, ge=0),
        time_: float | None = Query(default=None, alias="time", ge=0),
        window_seconds: float = Query(default=0, ge=0, le=CHECK_MAX_SECONDS),
        profile: str | None = Query(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$"),
    ):
        """Has this moment already been checked? Never calls a model.

        What a reloaded page asks, and what tells one teammate that another is already checking
        the same shot rather than starting a second call for the same answer.
        """
        if frame_idx is None and time_ is None:
            raise HTTPException(422, "either frame_idx or time is required")
        if not (visual_text.strip() or audio_text.strip() or ocr_text.strip()):
            raise HTTPException(422, "at least one of visual_text, audio_text or ocr_text is required")
        if not checker_key():
            return {"status": "unavailable", "result": None}

        def find():
            search = search_client(request)
            target = _resolve_moment(video_id, frame_idx, time_, profile)
            placement = place(search, target.video_id, target.frame_idx, target.time)
            query = CheckQuery(visual_text=visual_text.strip(), audio_text=audio_text.strip(),
                               ocr_text=ocr_text.strip(), kind=kind, question=question)
            key = check_key(query, target.video_id, placement.bucket,
                            window_seconds or default_window(), checker_model())
            store = _store(request)
            found = store.get(key)
            if found is not None:
                return {"status": "hit", "result": {**found, "cached": True, "latency_ms": 0.0}}
            return {"status": "running" if store.running(key) else "miss", "result": None}

        return await run_in_threadpool(find)

    app.add_api_route("/api/check", check, methods=["POST"])
    app.add_api_route("/api/check/lookup", lookup, methods=["GET"])
