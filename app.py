"""FastAPI VBS-style search app."""
import os
import logging
import subprocess
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException, Path, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

import core
import thumb
import audio_search
import frame_maps
from access_control import AccessConfig, LatestWinsCoordinator, install_access_control

ROOT = core.ROOT
FRONTEND_DIST = ROOT / "static" / "dist"
logger = logging.getLogger("uvicorn.error")


class SearchReq(BaseModel):
    text: str | None = Field(default=None, max_length=2000)
    visual_text: str | None = Field(default=None, max_length=2000)
    audio_text: str | None = Field(default=None, max_length=2000)
    mode: Literal["visual", "audio", "hybrid"] = "visual"
    index: str = "flat"
    k: int = Field(default=100, ge=1, le=200)
    nprobe: int = Field(default=0, ge=0, le=4096)
    ef: int = Field(default=0, ge=0, le=4096)


class CompareReq(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=50, ge=1, le=200)


def warmup_search():
    total_start = time.perf_counter()

    started = time.perf_counter()
    core.load_meta()
    logger.info("warm-up metadata: %.1f ms", (time.perf_counter() - started) * 1000)

    started = time.perf_counter()
    visual_query = core.encode_text("warmup visual search")
    logger.info("warm-up PE-Core: %.1f ms", (time.perf_counter() - started) * 1000)

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
    app.state.search_coordinator = LatestWinsCoordinator(
        max_running=5, rate_limit=10, window_seconds=10
    )
    warmup_search()
    yield


app = FastAPI(lifespan=lifespan)
install_access_control(app)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.mount("/static", StaticFiles(directory=FRONTEND_DIST, check_dir=False), name="static")


def resolve(video_id):
    _, _, _, lookup = core.load_meta()
    row = lookup.get(video_id)
    if row is None:
        raise HTTPException(404, f"unknown video {video_id}")
    if not os.path.exists(row["path"]):
        raise HTTPException(404, f"video file not found (data disk unmounted?): {row['path']}")
    return row


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIST / "index.html")


def search(req: SearchReq):
    legacy_text = (req.text or "").strip()
    visual_text = (req.visual_text or legacy_text).strip()
    audio_text = (req.audio_text or legacy_text).strip()
    if req.mode == "visual":
        if not visual_text:
            raise HTTPException(422, "visual query is required")
        return visual_search(req, visual_text)
    if req.mode == "audio":
        if not audio_text:
            raise HTTPException(422, "transcript query is required")
        try:
            return audio_search_response(req, audio_text)
        except Exception as exc:
            raise HTTPException(503, f"audio search unavailable: {exc}") from exc

    if not visual_text and not audio_text:
        raise HTTPException(422, "visual or transcript query is required")
    if not audio_text:
        return visual_search(req, visual_text)
    if not visual_text:
        try:
            return audio_search_response(req, audio_text)
        except Exception as exc:
            raise HTTPException(503, f"audio search unavailable: {exc}") from exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        visual_future = executor.submit(visual_search, req, visual_text)
        audio_future = executor.submit(audio_search_response, req, audio_text)
        visual = visual_future.result()
        try:
            audio = audio_future.result()
        except Exception as exc:
            visual["mode"] = "hybrid"
            visual["warning"] = f"audio_unavailable: {exc}"
            return visual
    return {
        "results": fuse_search_results(visual["results"], audio["results"], req.k),
        "latency_ms": round(max(visual["latency_ms"], audio["latency_ms"]), 1),
        "latencies": {
            "visual_ms": visual["latency_ms"],
            "audio_ms": audio["latency_ms"],
        },
        "mode": "hybrid",
        "index": req.index,
        "index_info": visual["index_info"],
    }


@app.post("/search")
async def search_endpoint(req: SearchReq, request: Request):
    if req.index != "flat" and not request.state.is_admin:
        raise HTTPException(403, "non-flat indexes are restricted to the owner")
    coordinator = request.app.state.search_coordinator
    return await coordinator.run(
        request.state.login,
        lambda: run_in_threadpool(search, req),
        exclusive=req.index != "flat",
    )


def visual_search(req: SearchReq, text: str):
    if req.index not in core.INDEX_NAMES:
        raise HTTPException(400, f"unknown index {req.index}")
    total_start = time.perf_counter()
    q = core.encode_text(text)
    idx = core.get_index(req.index)
    cfg = core.index_info(req.index)
    if req.index == "ivf":
        idx.nprobe = req.nprobe if req.nprobe > 0 else cfg.get("nprobe_default", 128)
    if req.index == "hnsw":
        idx.hnsw.efSearch = req.ef if req.ef > 0 else cfg.get("efSearch_default", 256)
    t0 = time.perf_counter()
    D, I = idx.search(q, req.k)
    dt = time.perf_counter() - t0

    id_video, id_frame, video_meta, _ = core.load_meta()
    metric = core.index_info(req.index).get("metric", "ip")
    results = []
    for d, gi in zip(D[0], I[0]):
        if gi < 0:
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
            "scores": {"visual_score": round(score, 4), "visual_rank": len(results) + 1},
        })
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - total_start) * 1000, 1),
        "faiss_latency_ms": round(dt * 1000, 1),
        "mode": "visual",
        "index": req.index,
        "index_info": core.index_info(req.index),
    }


def audio_search_response(req: SearchReq, text: str):
    t0 = time.perf_counter()
    hits = audio_search.search_audio(text, req.k)
    _, _, _, lookup = core.load_meta()
    results = []
    for rank, hit in enumerate(hits, 1):
        row = lookup.get(hit.video_id)
        if row is None:
            continue
        frame = core.nearest_frame(hit.video_id, hit.anchor_sec)
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
            },
        })
    return {
        "results": results,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "mode": "audio",
        "index": "elasticsearch",
        "index_info": {"type": "BM25 + Qwen3 dense kNN + RRF"},
    }


def fuse_search_results(visual, audio, k=100, max_time_delta=30.0):
    visual = [dict(hit) for hit in visual]
    audio = [dict(hit) for hit in audio]
    used_audio = set()
    fused = []
    for visual_rank, visual_hit in enumerate(visual, 1):
        candidates = [
            (abs(visual_hit["time"] - audio_hit["time"]), audio_rank, audio_hit)
            for audio_rank, audio_hit in enumerate(audio, 1)
            if audio_rank not in used_audio
            and audio_hit["video_id"] == visual_hit["video_id"]
            and abs(visual_hit["time"] - audio_hit["time"]) <= max_time_delta
        ]
        if candidates:
            _, audio_rank, audio_hit = min(candidates, key=lambda item: item[0])
            used_audio.add(audio_rank)
            merged = dict(visual_hit)
            merged.update({
                key: value for key, value in audio_hit.items()
                if key.startswith("transcript_")
            })
            merged["source"] = "both"
            merged["scores"] = {
                "visual_rank": visual_rank,
                "audio_rank": audio_rank,
                "visual_score": visual_hit.get("score"),
                "audio_score": audio_hit.get("score"),
            }
            rrf_score = 1 / (60 + visual_rank) + 1 / (60 + audio_rank)
        else:
            merged = visual_hit
            merged["scores"] = {**merged.get("scores", {}), "visual_rank": visual_rank}
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
    flat = core.get_index("flat")
    _, If = flat.search(q, k)
    base = set(int(x) for x in If[0])
    del flat

    out = []
    for name in core.INDEX_NAMES:
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
    return FileResponse(row["path"], media_type="video/mp4")


@app.get("/frame-map/{video_id}")
def get_frame_map(video_id: str = Path(pattern=r"^[A-Z]\d{2}_V\d{3}$")):
    try:
        return frame_maps.load(video_id).payload(video_id)
    except FileNotFoundError as exc:
        raise HTTPException(503, f"exact frame map unavailable for {video_id}") from exc


@app.get("/indexes")
def list_indexes():
    return [{"name": n, "info": core.index_info(n)} for n in core.INDEX_NAMES]
