"""Compute-plane endpoints and the data-route proxy for split development.

On the owner's host (`AIC_CORE_URL` unset) the endpoints below expose the heavy
primitives under `/core/*`. On a teammate's machine (`AIC_CORE_URL` set) the same
app instead forwards the media/data routes to the owner's app, so thumbnails,
videos, transcripts and frame maps work without a local `data/` or `indexes/`.

Routes are registered with `add_api_route` to match `collaboration.py` and
`agent_runner.py`: a router object would hide them from `app.routes` iteration.
"""
from __future__ import annotations

import logging
import os
import subprocess
import threading
from contextlib import contextmanager

import httpx
import numpy as np
from fastapi import Body, HTTPException
from pydantic import BaseModel, Field

from . import audio_search, core, frame_maps, ocr_search, shots, thumb
from . import search_core

logger = logging.getLogger("uvicorn.error")

CORE_API_VERSION = "1"

PROXY_PREFIXES = (
    "/thumb/", "/frame/", "/video/", "/clip/", "/shots/",
    "/frame-map/", "/transcript/", "/context/",
)
PROXY_PATHS = ("/videos", "/indexes", "/profiles")

_slots = threading.BoundedSemaphore(max(1, int(os.environ.get("AIC_CORE_SLOTS", "6"))))


@contextmanager
def _compute_slot():
    _slots.acquire()
    try:
        yield
    finally:
        _slots.release()


class EncodeTextRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)


class VisualSearchRequest(BaseModel):
    index: str = "flat"
    profile: str | None = None
    vectors: list[list[float]] = Field(min_length=1, max_length=8)
    k: int = Field(ge=1, le=10000)
    nprobe: int = Field(default=0, ge=0, le=4096)
    ef: int = Field(default=0, ge=0, le=4096)
    video_ids: list[str] | None = Field(default=None, max_length=50)
    exclude_global_id: int | None = None
    with_pts: bool = True


class FrameEmbeddingRequest(BaseModel):
    index: str = "flat"
    profile: str | None = None
    video_id: str
    frame_idx: int = Field(ge=0)


class AnnotatePoint(BaseModel):
    video_id: str
    time: float | None = None
    frame_idx: int | None = None


class AnnotateRequest(BaseModel):
    profile: str | None = None
    points: list[AnnotatePoint] = Field(default_factory=list, max_length=20000)
    with_pts: bool = True


class AudioFlatRequest(BaseModel):
    vector: list[float] = Field(min_length=1)
    k: int = Field(ge=1, le=10000)


class EsSearchRequest(BaseModel):
    index: str
    body: dict


def healthz():
    return {
        "status": "ok",
        "api_version": CORE_API_VERSION,
        "warmup": os.environ.get("AIC_WARMUP", "metadata"),
    }


def catalog(profile: str | None = None):
    name = profile or core.default_profile()
    try:
        lookup = core.load_meta(name)[3]
        indexes = {index: core.index_info(index, name) for index in core.available_indexes(name)}
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "default_profile": core.default_profile(),
        "profile": name,
        "video_ids": sorted(lookup),
        "indexes": indexes,
        "profiles": core.list_profiles(),
    }


def encode_text_endpoint(req: EncodeTextRequest):
    with _compute_slot():
        vectors = core.encode_texts(req.texts)
    return {"vectors": np.asarray(vectors, dtype="float32").tolist()}


def encode_qwen_endpoint(req: EncodeTextRequest):
    with _compute_slot():
        vectors = audio_search.encode_queries(req.texts)
    return {"vectors": np.asarray(vectors, dtype="float32").tolist()}


def encode_image_endpoint(data: bytes = Body(media_type="application/octet-stream")):
    if not data:
        raise HTTPException(422, "query image is empty")
    try:
        with _compute_slot():
            vector = core.encode_image(core.decode_query_image(data))
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"vector": np.asarray(vector, dtype="float32")[0].tolist()}


def visual_search_endpoint(req: VisualSearchRequest):
    try:
        with _compute_slot():
            return search_core.local_visual_search(
                req.index, req.profile, req.vectors, req.k,
                nprobe=req.nprobe, ef=req.ef, video_ids=req.video_ids,
                exclude_global_id=req.exclude_global_id, with_pts=req.with_pts,
            )
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc


def frame_embedding_endpoint(req: FrameEmbeddingRequest):
    try:
        with _compute_slot():
            try:
                vector, global_id = core.indexed_frame_embedding(
                    req.index, req.video_id, req.frame_idx, req.profile
                )
            except (KeyError, RuntimeError):
                try:
                    row = _resolve_media(req.video_id)
                    data = thumb.extract(
                        row["path"], req.frame_idx, frame_maps.load(req.video_id), w=0
                    )
                    vector = core.encode_image(core.decode_query_image(data))
                except (KeyError, IndexError) as exc:
                    raise HTTPException(404, "referenced keyframe is unavailable") from exc
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                        FileNotFoundError) as exc:
                    raise HTTPException(503, "referenced keyframe could not be decoded") from exc
                global_id = None
    except KeyError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"vector": np.asarray(vector, dtype="float32")[0].tolist(), "global_id": global_id}


def annotate_endpoint(req: AnnotateRequest):
    with _compute_slot():
        marks = search_core.local_annotate(
            [point.model_dump(exclude_none=True) for point in req.points],
            req.profile, with_pts=req.with_pts,
        )
    return {"marks": marks}


def audio_flat_endpoint(req: AudioFlatRequest):
    with _compute_slot():
        hits = audio_search.search_audio_flat_local(
            np.asarray(req.vector, dtype="float32"), req.k
        )
    return {"hits": None if hits is None else [_audio_hit_payload(hit) for hit in hits]}


def es_search_endpoint(req: EsSearchRequest):
    if search_core.remote():
        raise HTTPException(503, "elasticsearch is only reachable on the owner host")
    if req.index not in {audio_search.AUDIO_INDEX, ocr_search.OCR_INDEX}:
        raise HTTPException(400, f"unknown search index {req.index}")
    client = (
        audio_search.get_client() if req.index == audio_search.AUDIO_INDEX
        else ocr_search.get_client()
    )
    with _compute_slot():
        try:
            response = client.search(index=req.index, body=req.body)
        except Exception as exc:
            raise HTTPException(503, f"elasticsearch unavailable: {exc}") from exc
    return dict(response)


def _audio_hit_payload(hit):
    return {
        "video_id": hit.video_id,
        "start_sec": hit.start_sec,
        "end_sec": hit.end_sec,
        "anchor_sec": hit.anchor_sec,
        "text": hit.text,
        "score": hit.score,
        "bm25_rank": hit.bm25_rank,
        "vector_rank": hit.vector_rank,
        "passage_id": hit.passage_id,
        "matched_keywords": list(hit.matched_keywords),
    }


def _resolve_media(video_id):
    lookup = core.load_meta()[3]
    row = lookup.get(video_id) or shots.media_catalogue().get(video_id)
    if row is None:
        raise HTTPException(404, f"unknown video {video_id}")
    if not os.path.exists(row["path"]):
        raise HTTPException(404, f"video file not found (data disk unmounted?): {row['path']}")
    return row


CORE_ROUTES = (
    ("/core/healthz", healthz, ["GET"]),
    ("/core/catalog", catalog, ["GET"]),
    ("/core/encode/text", encode_text_endpoint, ["POST"]),
    ("/core/encode/qwen", encode_qwen_endpoint, ["POST"]),
    ("/core/encode/image", encode_image_endpoint, ["POST"]),
    ("/core/visual/search", visual_search_endpoint, ["POST"]),
    ("/core/visual/frame-embedding", frame_embedding_endpoint, ["POST"]),
    ("/core/frame/annotate", annotate_endpoint, ["POST"]),
    ("/core/audio/flat-search", audio_flat_endpoint, ["POST"]),
    ("/core/es/search", es_search_endpoint, ["POST"]),
)


class CoreProxyMiddleware:
    """Forward data-plane routes to the owner's app when `AIC_CORE_URL` is set."""

    def __init__(self, app, core_url, core_user=None):
        self.app = app
        self.core_url = core_url.rstrip("/")
        self.core_user = core_user
        self._client: httpx.AsyncClient | None = None

    def _proxy_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=10.0, read=None),
                limits=httpx.Limits(max_connections=32),
            )
        return self._client

    def _proxied(self, path: str) -> bool:
        return path in PROXY_PATHS or path.startswith(PROXY_PREFIXES)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self._proxied(scope.get("path", "")):
            return await self.app(scope, receive, send)

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
            elif message["type"] == "http.disconnect":
                return

        url = self.core_url + scope["path"]
        query = scope.get("query_string") or b""
        if query:
            url += "?" + query.decode("ascii", "replace")
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
            if key.lower() not in {b"host", b"content-length", b"x-aic-user", b"connection"}
        }
        if self.core_user:
            headers["x-aic-user"] = self.core_user

        started = False
        try:
            async with self._proxy_client().stream(
                scope["method"], url, headers=headers, content=bytes(body) or None,
            ) as response:
                raw_headers = [
                    (key, value)
                    for key, value in response.headers.raw
                    if key.decode("latin-1").lower() not in {
                        "connection", "keep-alive", "transfer-encoding",
                        "proxy-authenticate", "proxy-authorization",
                        "te", "trailers", "upgrade",
                    }
                ]
                started = True
                await send({
                    "type": "http.response.start",
                    "status": response.status_code,
                    "headers": raw_headers,
                })
                async for chunk in response.aiter_raw():
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        except httpx.HTTPError as exc:
            logger.warning("core proxy %s %s failed: %s", scope["method"], scope["path"], exc)
            if not started:
                from fastapi.responses import JSONResponse

                await JSONResponse(
                    {"detail": f"core compute host unreachable: {exc}"}, status_code=502
                )(scope, receive, send)


def install_core_service(app) -> None:
    for path, endpoint, methods in CORE_ROUTES:
        app.add_api_route(path, endpoint, methods=methods)
    url = os.environ.get("AIC_CORE_URL", "").strip()
    if url:
        app.add_middleware(
            CoreProxyMiddleware,
            core_url=url,
            core_user=os.environ.get("AIC_CORE_USER", "").strip() or None,
        )
