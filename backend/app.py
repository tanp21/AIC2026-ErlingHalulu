"""FastAPI VBS-style search app."""
import os
import json
import logging
import secrets
import subprocess
import tempfile
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Literal, Union

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Path, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
from starlette.background import BackgroundTask

from . import core, thumb, shots, shot_labels, audio_search, ocr_search, frame_maps
from .temporal_search import (
    SCORING_METHODS, build_sequences, deduplicate_candidates, fuse_audio, temporal_nms,
)
from .access_control import AccessConfig, LatestWinsCoordinator, install_access_control
from .collaboration import CollaborationStore, install_collaboration_routes

ROOT = core.ROOT
FRONTEND_DIST = ROOT / "static" / "dist"
logger = logging.getLogger("uvicorn.error")
SEARCH_PAGE_SIZE = 100
SEARCH_RESULT_LIMIT = 10_000
SEARCH_SNAPSHOT_TTL_SECONDS = 600
TEMPORAL_PAGE_SIZE = 20
TEMPORAL_RESULT_LIMIT = 10_000
MAX_QUERY_IMAGE_BYTES = 6 * 1024 * 1024

AGENT_BATCH_MAX_QUERIES = 12
AGENT_RESULT_LIMIT = 500
AGENT_TEMPORAL_VIDEO_LIMIT = 50
AGENT_SNIPPET_CHARS = 400

# Shared across hybrid/OCR-filtered requests instead of a fresh pool per call;
# sized for the worst case of 5 concurrent requests x 2 workers each (visual+audio
# or primary+ocr fan-out).
_SEARCH_FANOUT_EXECUTOR = ThreadPoolExecutor(max_workers=10, thread_name_prefix="search-fanout")
# Separate from the fan-out pool: an agent batch's temporal+audio sub-query itself
# submits to _SEARCH_FANOUT_EXECUTOR, so sharing one pool could starve that inner work.
_AGENT_BATCH_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-batch")


class SearchReq(BaseModel):
    text: str | None = Field(default=None, max_length=2000)
    visual_text: str | None = Field(default=None, max_length=2000)
    audio_text: str | None = Field(default=None, max_length=2000)
    audio_method: Literal["semantic", "keyword"] = "semantic"
    ocr_text: str | None = Field(default=None, max_length=2000)
    pe_weight: float = Field(default=.75, ge=0, le=1)
    mode: Literal["visual", "audio", "hybrid", "ocr"] = "visual"
    index: str = "flat"
    profile: Literal["l2026"] = "l2026"
    nprobe: int = Field(default=0, ge=0, le=4096)
    ef: int = Field(default=0, ge=0, le=4096)


class ImageReference(BaseModel):
    video_id: str = Field(pattern=r"^[A-Z]\d{2}_V\d{3}$")
    frame_idx: int = Field(ge=0)


class ImageSearchReq(SearchReq):
    mode: Literal["visual", "hybrid"] = "visual"
    image_weight: float = Field(default=0.5, ge=0, le=1)
    reference: ImageReference | None = None


class SearchPageReq(BaseModel):
    cursor: str = Field(min_length=1, max_length=128)


class TemporalSearchReq(BaseModel):
    events: list[str] = Field(min_length=2, max_length=6)
    audio_text: str | None = Field(default=None, max_length=2000)
    audio_method: Literal["semantic", "keyword"] = "semantic"
    pe_weight: float = Field(default=.75, ge=0, le=1)
    max_gap_seconds: int = Field(default=120, ge=1, le=600)
    profile: Literal["l2026"] = "l2026"

    @field_validator("events")
    @classmethod
    def validate_events(cls, events):
        stripped = [event.strip() for event in events]
        if any(not event or len(event) > 2000 for event in stripped):
            raise ValueError("each temporal event must contain 1 to 2000 characters")
        return stripped

    @field_validator("audio_text")
    @classmethod
    def normalize_audio_text(cls, text):
        return text.strip() if text is not None else None


class CompareReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=50, ge=1, le=200)


VideoId = Annotated[str, Field(pattern=r"^[A-Z]\d{2}_V\d{3}$")]
AgentQueryId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,40}$")]
AgentText = Annotated[str, Field(min_length=1, max_length=2000)]


class AgentVisualQuery(BaseModel):
    kind: Literal["visual"]
    id: AgentQueryId
    text: AgentText
    video_ids: list[VideoId] | None = Field(default=None, max_length=50)
    limit: int = Field(default=100, ge=1, le=AGENT_RESULT_LIMIT)


class AgentSimilarQuery(BaseModel):
    kind: Literal["similar"]
    id: AgentQueryId
    video_id: VideoId
    frame_idx: int = Field(ge=0)
    video_ids: list[VideoId] | None = Field(default=None, max_length=50)
    limit: int = Field(default=100, ge=1, le=AGENT_RESULT_LIMIT)


class AgentAudioQuery(BaseModel):
    kind: Literal["audio"]
    id: AgentQueryId
    text: AgentText
    method: Literal["semantic", "keyword"] = "keyword"
    limit: int = Field(default=100, ge=1, le=AGENT_RESULT_LIMIT)


class AgentOcrQuery(BaseModel):
    kind: Literal["ocr"]
    id: AgentQueryId
    text: AgentText
    limit: int = Field(default=100, ge=1, le=AGENT_RESULT_LIMIT)


class AgentTemporalQuery(BaseModel):
    kind: Literal["temporal"]
    id: AgentQueryId
    events: list[AgentText] = Field(min_length=2, max_length=6)
    audio_text: str | None = Field(default=None, max_length=2000)
    max_gap_seconds: int = Field(default=120, ge=1, le=600)
    limit: int = Field(default=20, ge=1, le=AGENT_TEMPORAL_VIDEO_LIMIT)


AgentQuery = Annotated[
    Union[AgentVisualQuery, AgentSimilarQuery, AgentAudioQuery, AgentOcrQuery, AgentTemporalQuery],
    Field(discriminator="kind"),
]


class AgentBatchReq(BaseModel):
    queries: list[AgentQuery] = Field(min_length=1, max_length=AGENT_BATCH_MAX_QUERIES)
    profile: Literal["l2026"] = "l2026"

    @field_validator("queries")
    @classmethod
    def require_unique_ids(cls, queries):
        ids = [query.id for query in queries]
        if len(ids) != len(set(ids)):
            raise ValueError("agent query ids must be unique")
        return queries


class ShotLabelChange(BaseModel):
    shot_index: int = Field(ge=0)
    state: Literal["excluded", "included"] | None
    reason: Literal["intro", "preview", "presenter", "outro", "other"] | None = None

    @model_validator(mode="after")
    def require_reason(self):
        if self.state is not None and self.reason is None:
            raise ValueError("reason is required for a reviewed label")
        return self


class ShotLabelPatch(BaseModel):
    expected_revision: int = Field(ge=0)
    updates: list[ShotLabelChange] = Field(min_length=1, max_length=500)


class SearchSnapshotStore:
    def __init__(
        self,
        page_size=SEARCH_PAGE_SIZE,
        ttl_seconds=SEARCH_SNAPSHOT_TTL_SECONDS,
        result_limit=SEARCH_RESULT_LIMIT,
        clock=time.monotonic,
    ):
        self.page_size = page_size
        self.ttl_seconds = ttl_seconds
        self.result_limit = result_limit
        self.clock = clock
        self._snapshots = {}
        self._cursor_lookup = {}
        self._owner_snapshot = {}

    def create(self, owner, response):
        self._purge()
        previous = self._owner_snapshot.get(owner)
        if previous:
            self._drop(previous)
        snapshot_id = secrets.token_urlsafe(18)
        all_results = response["results"]
        snapshot = {
            "owner": owner,
            "metadata": {key: value for key, value in response.items() if key != "results"},
            "results": all_results[:self.result_limit],
            "truncated": response.get("truncated", False) or len(all_results) >= self.result_limit,
            "expires_at": self.clock() + self.ttl_seconds,
            "cursors": {},
        }
        self._snapshots[snapshot_id] = snapshot
        self._owner_snapshot[owner] = snapshot_id
        return self._page(snapshot_id, 0)

    def next(self, owner, cursor):
        self._purge()
        location = self._cursor_lookup.get(cursor)
        if location is None:
            raise LookupError("search results expired")
        snapshot_id, offset = location
        snapshot = self._snapshots.get(snapshot_id)
        if snapshot is None or snapshot["owner"] != owner:
            raise LookupError("search results expired")
        snapshot["expires_at"] = self.clock() + self.ttl_seconds
        return self._page(snapshot_id, offset)

    def _page(self, snapshot_id, offset):
        snapshot = self._snapshots[snapshot_id]
        end = min(offset + self.page_size, len(snapshot["results"]))
        has_more = end < len(snapshot["results"])
        next_cursor = None
        if has_more:
            next_cursor = snapshot["cursors"].get(end)
            if next_cursor is None:
                next_cursor = secrets.token_urlsafe(24)
                snapshot["cursors"][end] = next_cursor
                self._cursor_lookup[next_cursor] = (snapshot_id, end)
        return {
            **snapshot["metadata"],
            "results": snapshot["results"][offset:end],
            "next_cursor": next_cursor,
            "has_more": has_more,
            "total_results": len(snapshot["results"]),
            "truncated": snapshot["truncated"],
        }

    def _purge(self):
        now = self.clock()
        for snapshot_id, snapshot in list(self._snapshots.items()):
            if snapshot["expires_at"] <= now:
                self._drop(snapshot_id)

    def _drop(self, snapshot_id):
        snapshot = self._snapshots.pop(snapshot_id, None)
        if snapshot is None:
            return
        if self._owner_snapshot.get(snapshot["owner"]) == snapshot_id:
            del self._owner_snapshot[snapshot["owner"]]
        for cursor in snapshot["cursors"].values():
            self._cursor_lookup.pop(cursor, None)


class TemporalSnapshotStore(SearchSnapshotStore):
    def __init__(self, page_size=TEMPORAL_PAGE_SIZE, ttl_seconds=SEARCH_SNAPSHOT_TTL_SECONDS,
                 clock=time.monotonic):
        super().__init__(page_size, ttl_seconds, TEMPORAL_RESULT_LIMIT, clock)

    @staticmethod
    def _rename(page):
        page["videos"] = page.pop("results")
        page["total_videos"] = page.pop("total_results")
        return page

    def create(self, owner, response):
        payload = {**response, "results": response["videos"]}
        del payload["videos"]
        return self._rename(super().create(owner, payload))

    def next(self, owner, cursor):
        return self._rename(super().next(owner, cursor))


def warmup_search():
    mode = os.environ.get("AIC_WARMUP", "metadata").strip().lower()
    total_start = time.perf_counter()

    started = time.perf_counter()
    core.load_meta()
    logger.info("warm-up metadata: %.1f ms", (time.perf_counter() - started) * 1000)

    # Loading PE-Core, the Flat index and Qwen3-Embedding-4B together can consume
    # nearly all RAM on a development workstation. Keep startup cheap; production
    # operators may explicitly opt into eager loading after setting memory limits.
    if mode != "full":
        logger.info("model/index warm-up skipped (AIC_WARMUP=%s)", mode)
        return

    started = time.perf_counter()
    visual_query = core.encode_text("warmup visual search")
    logger.info("warm-up PE-Core: %.1f ms (device=%s)", (time.perf_counter() - started) * 1000, core.DEVICE)

    started = time.perf_counter()
    core.get_index("flat").search(visual_query, 1)
    logger.info("warm-up Flat FAISS: %.1f ms", (time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    audio_search.encode_query("khởi động tìm kiếm lời nói")
    logger.info("warm-up Qwen: %.1f ms", (time.perf_counter() - started) * 1000)
    logger.info("search warm-up complete: %.1f ms", (time.perf_counter() - total_start) * 1000)


@asynccontextmanager
async def lifespan(app):
    app.state.access_config = AccessConfig.from_env()
    app.state.collaboration_store = CollaborationStore(
        os.environ.get("AIC_COLLAB_DB", ROOT / "artifacts" / "collaboration.sqlite3")
    )
    app.state.search_coordinator = LatestWinsCoordinator(
        # One concurrent search per trusted user; only explicit index operations lock.
        max_running=5,
        rate_limit=None,
        window_seconds=10,
        admin_user=app.state.access_config.admin_user,
    )
    app.state.search_snapshots = SearchSnapshotStore()
    app.state.temporal_snapshots = TemporalSnapshotStore()
    core.start_text_batcher()
    audio_search.start_audio_text_batcher()
    try:
        warmup_search()
        yield
    finally:
        core.close_text_batcher()
        audio_search.close_audio_text_batcher()
        _SEARCH_FANOUT_EXECUTOR.shutdown(wait=False, cancel_futures=True)
        _AGENT_BATCH_EXECUTOR.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)
install_access_control(app)
install_collaboration_routes(app)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=FRONTEND_DIST, check_dir=False), name="static")


def resolve(video_id):
    _, _, _, lookup = core.load_meta()
    # Datasets outside the embedded profiles still have videos, frame maps and shots on
    # disk; clip export serves those, so fall back to the media catalogue.
    row = lookup.get(video_id) or shots.media_catalogue().get(video_id)
    if row is None:
        raise HTTPException(404, f"unknown video {video_id}")
    if not os.path.exists(row["path"]):
        raise HTTPException(404, f"video file not found (data disk unmounted?): {row['path']}")
    return row


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIST / "index.html")


def _audio_search_or_503(req, text):
    try:
        return audio_search_response(req, text)
    except Exception as exc:
        raise HTTPException(503, f"audio search unavailable: {exc}") from exc


def _visual_audio_search(req, visual_operation, audio_text):
    if not audio_text:
        return visual_operation()
    executor = _SEARCH_FANOUT_EXECUTOR
    visual_future = executor.submit(visual_operation)
    audio_future = executor.submit(audio_search_response, req, audio_text)
    visual = visual_future.result()
    try:
        audio = audio_future.result()
    except Exception as exc:
        visual["mode"] = "hybrid"
        visual["warning"] = f"audio_unavailable: {exc}"
        visual["latencies"] = {"visual_ms": visual["latency_ms"]}
        return visual
    return {
        "results": fuse_search_results(
            visual["results"], audio["results"], SEARCH_RESULT_LIMIT
        ),
        "latency_ms": round(max(visual["latency_ms"], audio["latency_ms"]), 1),
        "latencies": {
            "visual_ms": visual["latency_ms"],
            "audio_ms": audio["latency_ms"],
        },
        "mode": "hybrid",
        "index": req.index,
        "index_info": visual["index_info"],
        "profile": visual["profile"],
    }


def _with_ocr_filter(req, ocr_text, primary_operation):
    if not ocr_text:
        return primary_operation()
    executor = _SEARCH_FANOUT_EXECUTOR
    primary_future = executor.submit(primary_operation)
    # filter_mode: OCR text is machine-read and often has a wrong character or two, so a
    # verbatim-phrase filter can hard-exclude the right shot over one typo. search_ocr_filter
    # still requires every query term (fuzzy-tolerant) and boosts the literal phrase, so an
    # exact match still outranks a fuzzy one -- it just isn't an all-or-nothing gate anymore.
    ocr_future = executor.submit(ocr_search_response, req, ocr_text, filter_mode=True)
    primary = primary_future.result()
    try:
        ocr = ocr_future.result()
    except Exception as exc:
        raise HTTPException(503, f"ocr search unavailable: {exc}") from exc
    mode = primary["mode"]
    latencies = {**primary.get("latencies", {}), "ocr_ms": ocr["latency_ms"]}
    if mode in {"visual", "audio"}:
        latencies.setdefault(f"{mode}_ms", primary["latency_ms"])
    return {
        **primary,
        "results": filter_search_results(primary["results"], ocr["results"]),
        "latency_ms": round(max(primary["latency_ms"], ocr["latency_ms"]), 1),
        "latencies": latencies,
    }


def search(req: SearchReq):
    legacy_text = (req.text or "").strip()
    visual_text = (req.visual_text or legacy_text).strip()
    audio_text = (req.audio_text or legacy_text).strip()
    ocr_text = (req.ocr_text or "").strip()
    if req.mode == "visual":
        if not visual_text:
            raise HTTPException(422, "visual query is required")
        return _with_ocr_filter(
            req, ocr_text, lambda: visual_search(req, visual_text)
        )
    if req.mode == "audio":
        if not audio_text:
            raise HTTPException(422, "transcript query is required")
        return _with_ocr_filter(
            req, ocr_text, lambda: _audio_search_or_503(req, audio_text)
        )
    if req.mode == "ocr":
        ocr_text = ocr_text or legacy_text
        if not ocr_text:
            raise HTTPException(422, "on-screen text query is required")
        try:
            return ocr_search_response(req, ocr_text)
        except Exception as exc:
            raise HTTPException(503, f"ocr search unavailable: {exc}") from exc

    if not visual_text and not audio_text:
        if ocr_text:
            try:
                return ocr_search_response(req, ocr_text)
            except Exception as exc:
                raise HTTPException(503, f"ocr search unavailable: {exc}") from exc
        raise HTTPException(422, "visual, transcript, or OCR query is required")
    if not visual_text:
        return _with_ocr_filter(
            req, ocr_text, lambda: _audio_search_or_503(req, audio_text)
        )
    return _with_ocr_filter(
        req,
        ocr_text,
        lambda: _visual_audio_search(
            req, lambda: visual_search(req, visual_text), audio_text
        ),
    )


@app.post("/search")
async def search_endpoint(req: SearchReq, request: Request):
    if req.index != "flat" and not request.state.is_admin:
        raise HTTPException(403, "non-flat indexes are restricted to the owner")
    coordinator = request.app.state.search_coordinator
    response = await coordinator.run(
        request.state.login,
        lambda: run_in_threadpool(search, req),
        exclusive=req.index != "flat",
    )
    return request.app.state.search_snapshots.create(request.state.login, response)


def image_conditioned_search(req: ImageSearchReq, image_data=None):
    if (image_data is None) == (req.reference is None):
        raise HTTPException(422, "provide exactly one image or keyframe reference")

    exclude_global_id = None
    if image_data is not None:
        try:
            image_vector = core.encode_image(core.decode_query_image(image_data))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
    else:
        reference = req.reference
        try:
            image_vector, exclude_global_id = core.indexed_frame_embedding(
                req.index, reference.video_id, reference.frame_idx, req.profile
            )
        except (KeyError, RuntimeError):
            try:
                row = resolve(reference.video_id)
                data = thumb.extract(
                    row["path"], reference.frame_idx, frame_maps.load(reference.video_id), w=0
                )
                image_vector = core.encode_image(core.decode_query_image(data))
            except (KeyError, IndexError) as exc:
                raise HTTPException(404, "referenced keyframe is unavailable") from exc
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
                raise HTTPException(503, "referenced keyframe could not be decoded") from exc

    visual_text = (req.visual_text or req.text or "").strip()
    text_vector = (
        core.encode_text(visual_text)
        if visual_text and req.image_weight < 1
        else None
    )
    query_vector = core.fuse_query_vectors(text_vector, image_vector, req.image_weight)

    visual_operation = lambda: visual_vector_search(
        req, query_vector, exclude_global_id
    )
    return _with_ocr_filter(
        req,
        (req.ocr_text or "").strip(),
        lambda: _visual_audio_search(
            req,
            visual_operation,
            (req.audio_text or "").strip() if req.mode == "hybrid" else "",
        ),
    )


@app.post("/search/image")
async def image_search_endpoint(
    request: Request,
    payload: str = Form(max_length=10_000),
    image: UploadFile | None = File(default=None),
):
    try:
        req = ImageSearchReq.model_validate_json(payload)
    except ValidationError as exc:
        raise HTTPException(422, "invalid image search payload") from exc
    if (image is None) == (req.reference is None):
        raise HTTPException(422, "provide exactly one image or keyframe reference")
    image_data = None
    if image is not None:
        image_data = await image.read(MAX_QUERY_IMAGE_BYTES + 1)
        if not image_data:
            raise HTTPException(422, "query image is empty")
        if len(image_data) > MAX_QUERY_IMAGE_BYTES:
            raise HTTPException(413, "query image exceeds 6 MiB")
    if req.index != "flat" and not request.state.is_admin:
        raise HTTPException(403, "non-flat indexes are restricted to the owner")
    response = await request.app.state.search_coordinator.run(
        request.state.login,
        lambda: run_in_threadpool(image_conditioned_search, req, image_data),
        exclusive=req.index != "flat",
    )
    return request.app.state.search_snapshots.create(request.state.login, response)


@app.post("/search/next")
async def next_search_page(req: SearchPageReq, request: Request):
    try:
        return request.app.state.search_snapshots.next(request.state.login, req.cursor)
    except LookupError as exc:
        raise HTTPException(410, str(exc)) from exc


def load_temporal_config():
    path = ROOT / "benchmarks" / "temporal" / "temporal_config.json"
    try:
        config = json.loads(path.read_text())
        candidate_k = int(config["candidate_k"])
        scoring_method = config["scoring_method"]
        min_gap_seconds = float(config.get("min_gap_seconds", 0.0))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(503, "temporal search benchmark config is unavailable") from exc
    if (
        candidate_k not in {500, 1000, 2000, 5000}
        or scoring_method not in SCORING_METHODS
        or not 0 <= min_gap_seconds < 60
    ):
        raise HTTPException(503, "temporal search benchmark config is invalid")
    return candidate_k, scoring_method, min_gap_seconds


@app.post("/search/temporal")
async def temporal_search_endpoint(req: TemporalSearchReq, request: Request):
    candidate_k, scoring_method, min_gap_seconds = load_temporal_config()
    response = await request.app.state.search_coordinator.run(
        request.state.login,
        lambda: run_in_threadpool(
            temporal_search_response, req, candidate_k, scoring_method, min_gap_seconds
        ),
    )
    return request.app.state.temporal_snapshots.create(request.state.login, response)


@app.post("/search/temporal/next")
async def next_temporal_page(req: SearchPageReq, request: Request):
    try:
        return request.app.state.temporal_snapshots.next(request.state.login, req.cursor)
    except LookupError as exc:
        raise HTTPException(410, str(exc)) from exc


def visual_search(req: SearchReq, text: str):
    total_start = time.perf_counter()
    return visual_vector_search(req, core.encode_text(text), started=total_start)


def visual_vector_search(req: SearchReq, q, exclude_global_id=None, started=None):
    profile = req.profile or core.default_profile()
    try:
        indexes = core.available_indexes(profile)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    if req.index not in indexes:
        raise HTTPException(400, f"unknown index {req.index} for profile {profile}")
    total_start = started if started is not None else time.perf_counter()
    idx = core.get_index(req.index, profile)
    cfg = core.index_info(req.index, profile)
    if req.index == "ivf":
        idx.nprobe = req.nprobe if req.nprobe > 0 else cfg.get("nprobe_default", 128)
    if req.index == "hnsw":
        idx.hnsw.efSearch = req.ef if req.ef > 0 else cfg.get("efSearch_default", 256)
    t0 = time.perf_counter()
    result_count = SEARCH_RESULT_LIMIT + (exclude_global_id is not None)
    D, I = idx.search(q, result_count)
    dt = time.perf_counter() - t0

    id_video, id_frame, video_meta, _ = core.load_meta(profile)
    metric = core.index_info(req.index, profile).get("metric", "ip")
    results = []
    for d, gi in zip(D[0], I[0]):
        if gi < 0 or gi == exclude_global_id:
            continue
        row = video_meta[int(id_video[gi])]
        f = int(id_frame[gi])
        mapping = frame_maps.load(row["video_id"])
        score = float(d) if metric == "ip" else float(1.0 - d / 2.0)  # L2 metric returns squared distance
        results.append({
            "rank": len(results),
            "score": round(score, 4),
            "video_id": row["video_id"],
            "frame_idx": f,
            "pts": int(mapping.pts[f]),
            "time": mapping.frame_time(f),
            "thumb": f"/thumb/{row['video_id']}/{f}",
            "frame": f"/frame/{row['video_id']}/{f}",
            "video": f"/video/{row['video_id']}",
            "source": "visual",
            "scores": {
                "visual_score": round(score, 4),
                "visual_rank": len(results) + 1,
                "pe_core_rank": len(results) + 1,
            },
        })
        if len(results) == SEARCH_RESULT_LIMIT:
            break
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - total_start) * 1000, 1),
        "faiss_latency_ms": round(dt * 1000, 1),
        "mode": "visual",
        "index": req.index,
        "index_info": core.index_info(req.index, profile),
        "profile": profile,
    }


def temporal_visual_paths(req: TemporalSearchReq, candidate_k, scoring_method, beam_size=1,
                          candidate_pool_k=None, dedup_seconds=0, min_gap_seconds=0.0):
    total_start = time.perf_counter()
    profile = req.profile or core.default_profile()
    queries = core.encode_texts(req.events)
    try:
        idx = core.get_index("flat", profile)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    search_start = time.perf_counter()
    distances, indexes = idx.search(queries, candidate_pool_k or candidate_k)
    faiss_ms = (time.perf_counter() - search_start) * 1000
    id_video, id_frame, video_meta, _ = core.load_meta(profile)
    id_time = core.load_times(profile)
    candidates = []
    for event_index, (event_distances, event_indexes) in enumerate(zip(distances, indexes)):
        event_hits = []
        for event_rank, (distance, global_index) in enumerate(
            zip(event_distances, event_indexes), 1
        ):
            if global_index < 0:
                continue
            row = video_meta[int(id_video[global_index])]
            frame = int(id_frame[global_index])
            score = float(distance)
            event_hits.append({
                "event_index": event_index,
                "event_rank": event_rank,
                "rank": event_rank - 1,
                "score": score,
                "video_id": row["video_id"],
                "frame_idx": frame,
                "time": float(id_time[global_index]),
                "thumb": f"/thumb/{row['video_id']}/{frame}",
                "frame": f"/frame/{row['video_id']}/{frame}",
                "video": f"/video/{row['video_id']}",
                "source": "visual",
                "scores": {"visual_score": round(score, 4), "visual_rank": event_rank},
            })
        candidates.append(
            deduplicate_candidates(event_hits, dedup_seconds, candidate_k)
            if dedup_seconds else event_hits[:candidate_k]
        )

    paths = build_sequences(
        candidates, req.max_gap_seconds, scoring_method, candidate_k, beam_size,
        min_gap_seconds=min_gap_seconds,
    )
    return {
        "paths": paths,
        "latency_ms": round((time.perf_counter() - total_start) * 1000, 1),
        "faiss_latency_ms": round(faiss_ms, 1),
        "profile": profile,
    }


def finalize_temporal_response(paths, metadata, scoring_method, candidate_k, audio_hits=None):
    if audio_hits is not None:
        paths = fuse_audio(paths, audio_hits, window_seconds=30, rrf_k=60)
    paths = temporal_nms(paths)
    truncated = len(paths) > TEMPORAL_RESULT_LIMIT
    paths = paths[:TEMPORAL_RESULT_LIMIT]
    for rank, path in enumerate(paths):
        path["rank"] = rank

    grouped = {}
    mappings = {}
    for path in paths:
        serialized_events = []
        if path["events"]:
            mapping = mappings.get(path["video_id"])
            if mapping is None:
                mapping = frame_maps.load(path["video_id"])
                mappings[path["video_id"]] = mapping
            serialized_events = [
                {
                    **{
                        key: value for key, value in event.items()
                        if key not in {"normalized_score", "percentile_score", "event_rank"}
                    },
                    "pts": int(mapping.pts[event["frame_idx"]]),
                }
                for event in path["events"]
            ]
        sequence = {
            **path,
            "score": round(path["score"], 6),
            "visual_score": round(path["visual_score"], 6),
            "events": serialized_events,
        }
        group = grouped.setdefault(path["video_id"], {
            "video_id": path["video_id"], "best_score": path["score"], "sequences": []
        })
        group["best_score"] = max(group["best_score"], path["score"])
        group["sequences"].append(sequence)

    videos = sorted(grouped.values(), key=lambda group: (-group["best_score"], group["video_id"]))
    for video in videos:
        video["best_score"] = round(video["best_score"], 6)
        video["sequences"].sort(key=lambda sequence: (
            sequence["start_time"], sequence["end_time"], sequence["rank"]
        ))
    return {
        "videos": videos,
        "total_sequences": len(paths),
        "latency_ms": metadata.get("latency_ms", 0.0),
        "faiss_latency_ms": metadata.get("faiss_latency_ms", 0.0),
        "mode": "temporal",
        "candidate_k": candidate_k,
        "scoring_method": scoring_method,
        "truncated": truncated,
        "profile": metadata.get("profile", core.default_profile()),
    }


def temporal_visual_search(req: TemporalSearchReq, candidate_k, scoring_method,
                           min_gap_seconds=0.0):
    visual = temporal_visual_paths(
        req, candidate_k, scoring_method, min_gap_seconds=min_gap_seconds
    )
    return finalize_temporal_response(
        visual["paths"], visual, scoring_method, candidate_k
    )


def _audio_hits(text, method):
    search_fn = (audio_search.search_audio_semantic if method == "semantic"
                 else audio_search.search_audio_keyword)
    return search_fn(text, SEARCH_RESULT_LIMIT)


def _temporal_audio_hits(text, method):
    return [
        {
            "video_id": hit.video_id,
            "time": hit.anchor_sec,
            "score": hit.score,
            "transcript_snippet": hit.text,
            "transcript_start": hit.start_sec,
            "transcript_end": hit.end_sec,
            "bm25_rank": hit.bm25_rank,
            "vector_rank": hit.vector_rank,
            "keyword_rank": hit.bm25_rank,
            "semantic_rank": hit.vector_rank,
            "audio_method": method,
            "matched_keywords": list(hit.matched_keywords),
        }
        for hit in _audio_hits(text, method)
    ]


def temporal_search_response(req: TemporalSearchReq, candidate_k, scoring_method,
                             min_gap_seconds=0.0):
    audio_text = (req.audio_text or "").strip()
    if not audio_text:
        return temporal_visual_search(req, candidate_k, scoring_method, min_gap_seconds)

    executor = _SEARCH_FANOUT_EXECUTOR
    visual_future = executor.submit(
        temporal_visual_paths, req, candidate_k, scoring_method,
        min_gap_seconds=min_gap_seconds,
    )
    audio_future = executor.submit(_temporal_audio_hits, audio_text, req.audio_method)
    visual = visual_future.result()
    try:
        audio_hits = audio_future.result()
    except Exception as exc:
        response = finalize_temporal_response(
            visual["paths"], visual, scoring_method, candidate_k
        )
        response["warning"] = f"audio_unavailable: {exc}"
        return response
    return finalize_temporal_response(
        visual["paths"], visual, scoring_method, candidate_k, audio_hits
    )


def audio_search_response(req: SearchReq, text: str):
    t0 = time.perf_counter()
    profile = req.profile or core.default_profile()
    hits = _audio_hits(text, req.audio_method)
    meta = core.load_meta(profile)
    lookup = meta[3]
    results = []
    for rank, hit in enumerate(hits, 1):
        row = lookup.get(hit.video_id)
        if row is None:
            continue
        frame = core.nearest_frame(hit.video_id, hit.anchor_sec, profile, meta=meta)
        mapping = frame_maps.load(hit.video_id)
        results.append({
            "rank": len(results),
            "score": round(hit.score, 6),
            "video_id": hit.video_id,
            "frame_idx": frame,
            "pts": int(mapping.pts[frame]),
            "time": mapping.frame_time(frame),
            "thumb": f"/thumb/{hit.video_id}/{frame}",
            "frame": f"/frame/{hit.video_id}/{frame}",
            "video": f"/video/{hit.video_id}",
            "source": "transcript",
            "transcript_snippet": hit.text,
            "transcript_start": hit.start_sec,
            "transcript_end": hit.end_sec,
            "scores": {
                "audio_score": round(hit.score, 6),
                "audio_rank": rank,
                "bm25_rank": hit.bm25_rank,
                "vector_rank": hit.vector_rank,
                "keyword_rank": hit.bm25_rank,
                "semantic_rank": hit.vector_rank,
            },
            "audio_method": req.audio_method,
            "matched_keywords": list(hit.matched_keywords),
        })
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "mode": "audio",
        "index": "elasticsearch",
        "index_info": {"type": "Qwen3 semantic exact FAISS FlatIP" if req.audio_method == "semantic" else "BM25 keyword"},
        "profile": profile,
    }


def ocr_search_response(req: SearchReq, text: str, *, strict=False, filter_mode=False):
    t0 = time.perf_counter()
    profile = req.profile or core.default_profile()
    if filter_mode:
        hits = ocr_search.search_ocr_filter(text, SEARCH_RESULT_LIMIT)
    else:
        hits = ocr_search.search_ocr_keyword(text, SEARCH_RESULT_LIMIT, strict=strict)
        if strict:
            hits = [hit for hit in hits if ocr_search.contains_folded_phrase(text, hit.text)]
    _, _, _, lookup = core.load_meta(profile)
    results = []
    for rank, hit in enumerate(hits, 1):
        row = lookup.get(hit.video_id)
        if row is None:
            continue
        try:
            mapping = frame_maps.load(hit.video_id)
        except (OSError, FileNotFoundError):
            continue
        results.append({
            "rank": len(results),
            "score": round(hit.score, 6),
            "video_id": hit.video_id,
            "frame_idx": hit.frame_idx,
            "pts": int(mapping.pts[hit.frame_idx]),
            "time": mapping.frame_time(hit.frame_idx),
            "thumb": f"/thumb/{hit.video_id}/{hit.frame_idx}",
            "frame": f"/frame/{hit.video_id}/{hit.frame_idx}",
            "video": f"/video/{hit.video_id}",
            "source": "ocr",
            "ocr_snippet": hit.text,
            "ocr_shot_index": hit.shot_idx,
            "ocr_start": hit.start_sec,
            "ocr_end": hit.end_sec,
            "ocr_exact": ocr_search.contains_folded_phrase(text, hit.text),
            "scores": {
                "ocr_score": round(hit.score, 6),
                "ocr_rank": rank,
                "bm25_rank": hit.bm25_rank,
            },
            "matched_keywords": list(hit.matched_keywords),
            "ocr_matched_keywords": list(hit.matched_keywords),
        })
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "mode": "ocr",
        "index": "elasticsearch",
        "index_info": {"type": "BM25 keyword (on-screen text)"},
        "profile": profile,
    }


_SHOT_BOUNDARY_EPSILON = 1e-3


def filter_search_results(results, ocr_results):
    """Keep primary results that fall in the same shot as an on-screen OCR
    match, not merely anywhere in that video.

    Only one representative keyframe per shot is OCR'd (see
    `ocr_search.py`), so `ocr_start`/`ocr_end` are that shot's own
    boundaries -- a primary result only earns the OCR annotation if its own
    frame time lands inside that exact shot.

    The OCR side is a typo-tolerant fuzzy match (see `search_ocr_filter`),
    so a shot can pass on a near-miss alone. Rather than let a fuzzy match
    to an unrelated shot outrank a literal one purely because it happens to
    score higher on the primary (visual/audio) signal, exact OCR matches
    are kept as a first tier ahead of fuzzy-only matches; the primary
    result order is preserved within each tier.
    """
    ocr_by_video = {}
    for hit in ocr_results:
        ocr_by_video.setdefault(hit["video_id"], []).append(hit)

    tiered = []
    for result in results:
        candidates = [
            hit for hit in ocr_by_video.get(result["video_id"], [])
            if hit["ocr_start"] - _SHOT_BOUNDARY_EPSILON
                <= result["time"]
                <= hit["ocr_end"] + _SHOT_BOUNDARY_EPSILON
        ]
        if not candidates:
            continue
        match = min(
            candidates,
            key=lambda hit: (not hit.get("ocr_exact", False), hit.get("scores", {}).get("ocr_rank") or 0),
        )
        item = dict(result)
        item.update({
            key: match[key]
            for key in ("ocr_snippet", "ocr_shot_index", "ocr_start", "ocr_end")
        })
        item["ocr_matched_keywords"] = list(
            match.get("ocr_matched_keywords", match.get("matched_keywords", []))
        )
        item["scores"] = {
            **item.get("scores", {}),
            "ocr_score": match.get("scores", {}).get("ocr_score", match.get("score")),
            "ocr_rank": match.get("scores", {}).get("ocr_rank"),
        }
        tiered.append((not match.get("ocr_exact", False), item))

    filtered = []
    for _, item in sorted(tiered, key=lambda pair: pair[0]):
        item["rank"] = len(filtered)
        filtered.append(item)
    return filtered


def _audio_time_window(audio_hit, *, floor=30.0, ceiling=120.0):
    """How close a visual hit's frame time must be to this transcript
    passage's anchor to count as the same moment.

    Scaled to the passage's own span (`transcript_end - transcript_start`)
    rather than fixed: a short, densely-anchored passage is already precise,
    so a tight window is right, but a long passage's midpoint anchor
    (``time``) can sit tens of seconds from the actual matching moment
    inside it, so it needs more slack to still fuse with the right visual
    hit. Benchmarked on the semantic-chunking hybrid eval (44 KIS/QA
    queries): a fixed 30s window under-credits long passages, a fixed 120s
    window over-merges short/dense ones, and this per-hit adaptive window
    matched or beat the best fixed value for every duration config tried.
    Falls back to `floor` when span info isn't present on the hit.
    """
    start = audio_hit.get("transcript_start")
    end = audio_hit.get("transcript_end")
    if start is None or end is None:
        return floor
    return max(floor, min(ceiling, end - start))


def fuse_search_results(visual, audio, k=100, max_time_delta=30.0, time_delta_ceiling=120.0):
    visual = [dict(hit) for hit in visual]
    audio = [dict(hit) for hit in audio]
    used_audio = set()
    # Grouping by video_id turns the per-visual-hit scan into O(hits for that video)
    # instead of O(len(audio)) -- with both pools up to SEARCH_RESULT_LIMIT (10,000),
    # the original full rescan was up to 10,000 x 10,000 comparisons per hybrid
    # request, measured as the dominant cost behind hybrid mode being several
    # seconds slower than its visual/audio branches individually.
    audio_by_video = defaultdict(list)
    for audio_rank, audio_hit in enumerate(audio, 1):
        audio_by_video[audio_hit["video_id"]].append((audio_rank, audio_hit))
    fused = []
    for visual_rank, visual_hit in enumerate(visual, 1):
        candidates = [
            (abs(visual_hit["time"] - audio_hit["time"]), audio_rank, audio_hit)
            for audio_rank, audio_hit in audio_by_video.get(visual_hit["video_id"], ())
            if audio_rank not in used_audio
            and abs(visual_hit["time"] - audio_hit["time"])
                <= _audio_time_window(audio_hit, floor=max_time_delta, ceiling=time_delta_ceiling)
        ]
        if candidates:
            _, audio_rank, audio_hit = min(candidates, key=lambda item: item[0])
            used_audio.add(audio_rank)
            merged = dict(visual_hit)
            merged.update({
                key: value for key, value in audio_hit.items()
                if key.startswith("transcript_") or key in {"audio_method", "matched_keywords"}
            })
            merged["source"] = "both"
            merged["scores"] = {
                "visual_rank": visual_rank,
                "audio_rank": audio_rank,
                "visual_score": visual_hit.get("score"),
                "audio_score": audio_hit.get("score"),
                "keyword_rank": audio_hit.get("scores", {}).get("keyword_rank"),
                "semantic_rank": audio_hit.get("scores", {}).get("semantic_rank"),
                "pe_core_rank": visual_hit.get("scores", {}).get("pe_core_rank", visual_rank),
                "naflex_rank": visual_hit.get("scores", {}).get("naflex_rank"),
            }
            rrf_score = 1 / (60 + visual_rank) + 1 / (60 + audio_rank)
        else:
            merged = visual_hit
            merged["scores"] = {
                **merged.get("scores", {}), "visual_rank": visual_rank,
                "pe_core_rank": merged.get("scores", {}).get("pe_core_rank", visual_rank),
            }
            rrf_score = 1 / (60 + visual_rank)
        merged["score"] = rrf_score
        fused.append(merged)

    for audio_rank, audio_hit in enumerate(audio, 1):
        if audio_rank in used_audio:
            continue
        hit = dict(audio_hit)
        hit["scores"] = {**hit.get("scores", {}), "audio_rank": audio_rank}
        hit["score"] = 1 / (60 + audio_rank)
        fused.append(hit)

    fused.sort(key=lambda hit: -hit["score"])
    for rank, hit in enumerate(fused[:k]):
        hit["rank"] = rank
    return fused[:k]


def compare(req: CompareReq):
    q = core.encode_text(req.text)
    k = req.k
    index_names = core.available_indexes()
    flat = core.get_index("flat")
    _, If = flat.search(q, k)
    base = set(int(x) for x in If[0])
    del flat

    out = []
    for name in index_names:
        idx = core.get_index(name)
        if name == "ivf":
            idx.nprobe = core.index_info(name).get("nprobe_default", 128)
        elif name == "hnsw":
            idx.hnsw.efSearch = core.index_info(name).get("efSearch_default", 256)
        t0 = time.perf_counter()
        _, I = idx.search(q, k)
        ms = (time.perf_counter() - t0) * 1000
        recall = len(set(int(x) for x in I[0]) & base) / k
        out.append({"index": name, "latency_ms": round(ms, 1),
                    "recall": round(recall, 3), "info": core.index_info(name)})
    return {"results": out}


@app.post("/compare")
async def compare_endpoint(req: CompareReq, request: Request):
    if not request.state.is_admin:
        raise HTTPException(403, "index comparison is restricted to the owner")
    return await request.app.state.search_coordinator.run(
        request.state.login,
        lambda: run_in_threadpool(compare, req),
        exclusive=True,
    )


def _snippet(text):
    text = " ".join((text or "").split())
    return text if len(text) <= AGENT_SNIPPET_CHARS else text[:AGENT_SNIPPET_CHARS - 1] + "…"


def _agent_visual_hits(vector, profile, limit, video_ids=None, exclude_global_id=None):
    """Top keyframes for one query vector, optionally scored only inside `video_ids`.

    Scoping reconstructs the named videos' rows from the Flat index and scores them
    directly, so a within-video search sees every keyframe of those videos instead of
    whatever survives a corpus-wide top-k.
    """
    index = core.get_index("flat", profile)
    id_video, id_frame, video_meta, lookup = core.load_meta(profile)
    times = core.load_times(profile)
    vector = np.asarray(vector, dtype="float32").reshape(1, -1)
    if video_ids:
        wanted = [int(lookup[video_id]["video_idx"]) for video_id in video_ids if video_id in lookup]
        global_ids = np.flatnonzero(np.isin(id_video, wanted))
        if exclude_global_id is not None:
            global_ids = global_ids[global_ids != exclude_global_id]
        if not len(global_ids):
            return []
        scores = index.reconstruct_batch(global_ids) @ vector[0]
        order = np.argsort(-scores, kind="stable")[:limit]
        pairs = [(float(scores[i]), int(global_ids[i])) for i in order]
    else:
        distances, indexes = index.search(vector, limit + (exclude_global_id is not None))
        pairs = [
            (float(distance), int(global_id))
            for distance, global_id in zip(distances[0], indexes[0])
            if global_id >= 0 and global_id != exclude_global_id
        ][:limit]
    return [
        {
            "rank": rank,
            "video_id": video_meta[int(id_video[global_id])]["video_id"],
            "frame_idx": int(id_frame[global_id]),
            "time": round(float(times[global_id]), 3),
            "score": round(score, 4),
        }
        for rank, (score, global_id) in enumerate(pairs, 1)
    ]


def _agent_audio_hits(query, profile):
    search_fn = (audio_search.search_audio_semantic if query.method == "semantic"
                 else audio_search.search_audio_keyword)
    meta = core.load_meta(profile)
    hits = []
    for hit in search_fn(query.text, query.limit):
        if hit.video_id not in meta[3]:
            continue
        hits.append({
            "rank": len(hits) + 1,
            "video_id": hit.video_id,
            "frame_idx": core.nearest_frame(hit.video_id, hit.anchor_sec, profile, meta=meta),
            "time": round(hit.anchor_sec, 3),
            "start": round(hit.start_sec, 3),
            "end": round(hit.end_sec, 3),
            "score": round(hit.score, 6),
            "text": _snippet(hit.text),
            "matched_keywords": list(hit.matched_keywords),
        })
    return hits


def _agent_ocr_hits(query, profile):
    lookup = core.load_meta(profile)[3]
    hits = []
    for hit in ocr_search.search_ocr_keyword(query.text, query.limit):
        if hit.video_id not in lookup:
            continue
        hits.append({
            "rank": len(hits) + 1,
            "video_id": hit.video_id,
            "frame_idx": hit.frame_idx,
            "time": round(hit.anchor_sec, 3),
            "start": round(hit.start_sec, 3),
            "end": round(hit.end_sec, 3),
            "score": round(hit.score, 6),
            "text": _snippet(hit.text),
            "exact": ocr_search.contains_folded_phrase(query.text, hit.text),
            "matched_keywords": list(hit.matched_keywords),
        })
    return hits


def _agent_temporal_hits(query, profile):
    candidate_k, scoring_method, min_gap_seconds = load_temporal_config()
    request = TemporalSearchReq(
        events=query.events, audio_text=query.audio_text, audio_method="keyword",
        max_gap_seconds=query.max_gap_seconds, profile=profile,
    )
    response = temporal_search_response(request, candidate_k, scoring_method, min_gap_seconds)
    videos = []
    for group in response["videos"][:query.limit]:
        sequences = sorted(group["sequences"], key=lambda sequence: sequence["rank"])[:3]
        videos.append({
            "rank": len(videos) + 1,
            "video_id": group["video_id"],
            "score": group["best_score"],
            "sequences": [
                {
                    "score": sequence["score"],
                    "events": [
                        {"frame_idx": event["frame_idx"], "time": round(event["time"], 3),
                         "score": round(event["score"], 4)}
                        for event in sequence["events"]
                    ],
                }
                for sequence in sequences
            ],
        })
    return videos


def _run_agent_query(query, profile):
    started = time.perf_counter()
    if query.kind == "visual":
        hits = _agent_visual_hits(core.encode_text(query.text), profile, query.limit, query.video_ids)
    elif query.kind == "similar":
        vector, global_id = core.indexed_frame_embedding(
            "flat", query.video_id, query.frame_idx, profile
        )
        hits = _agent_visual_hits(vector, profile, query.limit, query.video_ids, global_id)
    elif query.kind == "audio":
        hits = _agent_audio_hits(query, profile)
    elif query.kind == "ocr":
        hits = _agent_ocr_hits(query, profile)
    else:
        hits = _agent_temporal_hits(query, profile)
    return hits, round((time.perf_counter() - started) * 1000, 1)


def agent_search(req: AgentBatchReq):
    """Run every sub-query of one agent step in parallel.

    A failing sub-query reports its own error instead of failing the batch: the
    agent can still act on the modalities that answered.
    """
    started = time.perf_counter()
    futures = [
        (query, _AGENT_BATCH_EXECUTOR.submit(_run_agent_query, query, req.profile))
        for query in req.queries
    ]
    results = {}
    for query, future in futures:
        try:
            hits, latency_ms = future.result()
        except HTTPException as exc:
            results[query.id] = {"kind": query.kind, "error": str(exc.detail)}
            continue
        except Exception as exc:
            results[query.id] = {"kind": query.kind, "error": f"{type(exc).__name__}: {exc}"}
            continue
        results[query.id] = {"kind": query.kind, "hits": hits, "latency_ms": latency_ms}
    return {
        "results": results,
        "profile": req.profile,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
    }


@app.post("/agent/search")
async def agent_search_endpoint(req: AgentBatchReq, request: Request):
    # A separate coordinator identity: the agent runs under a teammate's login, and
    # sharing that login's latest-wins slot would let the agent and the human's own
    # searches supersede each other with 409s.
    return await request.app.state.search_coordinator.run(
        f"{request.state.login}#agent",
        lambda: run_in_threadpool(agent_search, req),
    )


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/session")
async def session(request: Request):
    return {"login": request.state.login, "is_admin": request.state.is_admin}


@app.get("/thumb/{video_id}/{frame_idx}")
def get_thumb(
    video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$"),
    frame_idx: int = Path(ge=0),
    w: int = Query(default=320, ge=32, le=1920),
):
    if w not in (160, 320, 640):
        raise HTTPException(422, "thumbnail width must be 160, 320, or 640")
    row = resolve(video_id)
    try:
        data = thumb.extract(row["path"], frame_idx, frame_maps.load(video_id), w)
    except IndexError:
        raise HTTPException(422, "frame index is outside the video")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        raise HTTPException(500, "ffmpeg failed")
    return Response(data, media_type="image/jpeg")


@app.get("/context/{video_id}/{frame_idx}")
def keyframe_context(
    response: Response,
    video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$"),
    frame_idx: int = Path(ge=0),
    profile: str | None = Query(default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$"),
    radius: int = Query(default=10, ge=0, le=25),
):
    """Return indexed neighbours only; image decoding remains entirely client-side."""
    response.headers["Cache-Control"] = "no-store"
    source = profile or core.default_profile()
    try:
        frames = core.frame_ids(source, video_id)
    except KeyError as exc:
        raise HTTPException(404, "video has no indexed keyframes") from exc
    if len(frames) == 0:
        raise HTTPException(404, "video has no indexed keyframes")
    pos = int(np.searchsorted(frames, frame_idx))
    if pos == len(frames) or (pos and abs(int(frames[pos - 1]) - frame_idx) < abs(int(frames[pos]) - frame_idx)):
        pos -= 1
    mapping = frame_maps.load(video_id)
    neighbours = []
    for frame in frames[max(0, pos - radius):pos + radius + 1]:
        value = int(frame)
        neighbours.append({
            "video_id": video_id, "frame_idx": value, "pts": int(mapping.pts[value]),
            "time": mapping.frame_time(value), "thumb": f"/thumb/{video_id}/{value}",
            "frame": f"/frame/{video_id}/{value}", "video": f"/video/{video_id}",
        })
    return {"video_id": video_id, "frame_idx": int(frames[pos]), "frames": neighbours}


@app.get("/frame/{video_id}/{frame_idx}")
def get_frame(
    video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$"),
    frame_idx: int = Path(ge=0),
):
    row = resolve(video_id)
    try:
        data = thumb.extract(row["path"], frame_idx, frame_maps.load(video_id), w=0)
    except IndexError:
        raise HTTPException(422, "frame index is outside the video")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        raise HTTPException(500, "ffmpeg failed")
    return Response(data, media_type="image/jpeg")


@app.get("/video/{video_id}")
def get_video(video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$")):
    row = resolve(video_id)
    return FileResponse(shots.streaming_path(row), media_type="video/mp4")


@app.get("/frame-map/{video_id}")
def get_frame_map(video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$")):
    try:
        return frame_maps.load(video_id).payload(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"exact frame map unavailable for {video_id}") from exc


@app.get("/videos")
def list_videos(profile: str | None = None):
    """Every video on the data disk, flagged for search index membership and shots."""
    try:
        _, _, _, indexed = core.load_meta(profile)
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    catalogue = {**shots.media_catalogue(), **indexed}
    return [
        {
            "video_id": row["video_id"],
            "group": row["group"],
            "year": str(row["year"]),
            "fps": float(row["fps"]) if row.get("fps") else None,
            "indexed": video_id in indexed,
            "has_shots": shots.has_shots(row),
        }
        for video_id, row in sorted(catalogue.items())
    ]


def _shot_index(video_id):
    """Shot bounds trimmed to the frame map, plus the map itself."""
    row = resolve(video_id)
    try:
        mapping = frame_maps.load(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"exact frame map unavailable for {video_id}") from exc
    try:
        index = shots.load(row)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, f"shot boundaries unavailable for {video_id}") from exc
    except ValueError as exc:
        raise HTTPException(503, f"shot boundaries for {video_id} are malformed") from exc
    return row, mapping, index.clamped(len(mapping.pts))


@app.get("/shots/{video_id}")
def get_shots(
    video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$"),
    keyframe_source: str | None = Query(
        default=None, pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$"
    ),
):
    _, mapping, index = _shot_index(video_id)
    try:
        labels = shot_labels.load(video_id, index.bounds)
    except shot_labels.LabelConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(503, f"shot labels for {video_id} are unavailable") from exc
    sources = core.list_frame_sources(video_id)
    source_names = {source["name"] for source in sources}
    known_sources = {profile["name"] for profile in core.list_profiles()} | {"legacy"}
    if keyframe_source and keyframe_source not in known_sources:
        raise HTTPException(400, f"unknown keyframe source {keyframe_source}")
    selected_source = keyframe_source if keyframe_source in source_names else None
    if selected_source is None and sources:
        preferred = core.default_profile()
        selected_source = preferred if preferred in source_names else (
            "legacy" if "legacy" in source_names else sources[0]["name"]
        )

    by_shot = [[] for _ in range(len(index))]
    if selected_source:
        for frame_id in core.frame_ids(selected_source, video_id):
            frame_id = int(frame_id)
            if frame_id < 0 or frame_id >= len(mapping.pts):
                continue
            position = index.shot_at(frame_id)
            if position is not None:
                by_shot[position].append({
                    "frameId": frame_id,
                    "time": mapping.frame_time(frame_id),
                    "thumb": f"/thumb/{video_id}/{frame_id}",
                    "frame": f"/frame/{video_id}/{frame_id}",
                })
    return {
        "videoId": video_id,
        "fps": index.fps,
        "frameCount": len(mapping.pts),
        "labelRevision": labels["revision"],
        "keyframeSource": selected_source,
        "keyframeSources": sources,
        "shots": [
            {
                "index": position,
                "startFrame": int(start),
                "endFrame": int(end),
                "start": mapping.frame_time(int(start)),
                "end": mapping.frame_time(int(end)) + shots.frame_duration(mapping, int(end)),
                "keyframes": by_shot[position],
                "reviewFrames": sorted({int(start), (int(start) + int(end)) // 2, int(end)}),
                "label": labels["labels"].get(str(position)),
            }
            for position, (start, end) in enumerate(index.bounds.tolist())
        ],
    }


@app.patch("/shots/{video_id}/labels")
def update_shot_labels(
    patch: ShotLabelPatch,
    request: Request,
    video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$"),
):
    if not request.state.is_admin:
        raise HTTPException(403, "shot labels are restricted to the owner")
    _, _, index = _shot_index(video_id)
    try:
        document = shot_labels.update(
            video_id,
            index.bounds,
            patch.expected_revision,
            [change.model_dump() for change in patch.updates],
            request.state.login,
        )
    except shot_labels.LabelConflict as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except OSError as exc:
        raise HTTPException(503, "shot labels could not be saved") from exc
    return {"videoId": video_id, "labelRevision": document["revision"],
            "labels": document["labels"]}


@app.get("/clip/{video_id}/{shot_index}")
def get_clip(
    video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$"),
    shot_index: int = Path(ge=0),
):
    row, mapping, index = _shot_index(video_id)
    if shot_index >= len(index):
        raise HTTPException(422, f"shot {shot_index} is outside {video_id}")
    start_frame, end_frame = (int(value) for value in index.bounds[shot_index])
    start, end = shots.shot_bounds_seconds(mapping, start_frame, end_frame)
    if end - start > shots.MAX_CLIP_SECONDS:
        raise HTTPException(422, f"shot is longer than {shots.MAX_CLIP_SECONDS:g}s")
    handle, output = tempfile.mkstemp(suffix=".mp4", prefix=f"{video_id}_shot{shot_index}_")
    os.close(handle)
    try:
        shots.export(row["path"], start, end, output)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
        os.unlink(output)
        raise HTTPException(500, "ffmpeg failed") from exc
    except BaseException:
        os.unlink(output)
        raise
    return FileResponse(
        output,
        media_type="video/mp4",
        filename=f"{video_id}_shot{shot_index:04d}_f{start_frame:06d}-{end_frame:06d}.mp4",
        background=BackgroundTask(os.unlink, output),
    )


@app.get("/transcript/{video_id}")
def get_transcript(video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$")):
    try:
        cues = audio_search.load_transcript_cues(video_id)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, "transcript data unavailable") from exc
    return {"videoId": video_id, "cues": cues}


@app.get("/indexes")
def list_indexes(profile: str | None = None):
    try:
        return [
            {"name": name, "info": core.index_info(name, profile)}
            for name in core.available_indexes(profile)
        ]
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/profiles")
def list_profiles():
    return core.list_profiles()
