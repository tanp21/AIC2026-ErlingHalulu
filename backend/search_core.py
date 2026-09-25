"""Compute-plane access: run heavy primitives locally, or on the owner's host.

`AIC_CORE_URL` selects the mode. Unset (the production container, and the owner's
own workstation) means every primitive runs in-process exactly as before. Set (a
teammate's machine) means PE-Core, Qwen, FAISS and Elasticsearch work is forwarded
to the owner's `/core/*` endpoints, while orchestration, fusion and ranking stay in
this process. The local branches call `core`/`frame_maps` through the module object
so tests that patch `backend.app.core.*` keep working unchanged.
"""
from __future__ import annotations

import os
import threading
import time
from functools import lru_cache

import httpx
import numpy as np
from fastapi import HTTPException

from . import core, frame_maps, object_search

CORE_API_VERSION = "1"
DEFAULT_TIMEOUT = 60.0
RETRIES = 2

_client = None
_client_url = None
_client_lock = threading.Lock()


def core_url() -> str:
    return os.environ.get("AIC_CORE_URL", "").strip().rstrip("/")


def remote() -> bool:
    return bool(core_url())


def _client_for(url: str) -> httpx.Client:
    global _client, _client_url
    with _client_lock:
        if _client is None or _client_url != url:
            if _client is not None:
                _client.close()
            headers = {}
            user = os.environ.get("AIC_CORE_USER", "").strip()
            if user:
                headers["X-AIC-User"] = user
            _client = httpx.Client(
                base_url=url,
                headers=headers,
                timeout=httpx.Timeout(DEFAULT_TIMEOUT, connect=10.0),
                limits=httpx.Limits(max_connections=32),
            )
            _client_url = url
        return _client


def close_client() -> None:
    global _client, _client_url
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None
        _client_url = None


def _detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200]
    return detail if isinstance(detail, str) else str(detail)[:200]


def _request(method, path, *, json=None, params=None, content=None, retries=RETRIES):
    url = core_url()
    if not url:
        raise RuntimeError("remote core is not configured")
    client = _client_for(url)
    attempt = 0
    while True:
        try:
            response = client.request(method, path, json=json, params=params, content=content)
        except httpx.TransportError as exc:
            if attempt >= retries:
                raise HTTPException(503, f"core compute host unreachable: {exc}") from exc
        else:
            if response.status_code < 400:
                return response.json()
            if response.status_code < 500 or attempt >= retries:
                raise HTTPException(response.status_code, _detail(response))
        attempt += 1
        time.sleep(0.25 * attempt)


def healthz() -> dict:
    return _request("GET", "/core/healthz", retries=0)


@lru_cache(maxsize=8)
def _fetch_catalog(profile: str | None) -> dict:
    params = {"profile": profile} if profile else None
    return _request("GET", "/core/catalog", params=params)


def default_profile() -> str:
    if remote():
        return _fetch_catalog(None)["default_profile"]
    return core.default_profile()


def known_videos(profile: str | None) -> set[str]:
    if remote():
        return set(_fetch_catalog(profile)["video_ids"])
    return set(core.load_meta(profile)[3])


def object_catalog():
    if remote():
        return _request("GET", "/core/object/catalog")
    return object_search.catalog()


def object_video_detections(video_id):
    if remote():
        return _request("GET", f"/core/object/{video_id}/detections")
    return object_search.video_detections(video_id)


def search_objects(profile, query, k):
    if remote():
        return _request("POST", "/core/object/search", json={
            "profile": profile, "query": query.model_dump(), "k": k,
        })
    return object_search.search(profile, query, k)


def match_objects(profile, query, points):
    matches = []
    for start in range(0, len(points), 500):
        batch = points[start:start + 500]
        if remote():
            matches.extend(_request("POST", "/core/object/match", json={
                "profile": profile, "query": query.model_dump(), "points": batch,
            })["matches"])
        else:
            matches.extend(object_search.match_frames(profile, query, batch))
    return matches


def encode_text(text: str) -> np.ndarray:
    if remote():
        payload = _request("POST", "/core/encode/text", json={"texts": [str(text)]})
        return np.asarray(payload["vectors"], dtype="float32")
    return core.encode_text(text)


def encode_texts(texts) -> np.ndarray:
    texts = [str(text) for text in texts]
    if remote():
        payload = _request("POST", "/core/encode/text", json={"texts": texts})
        return np.asarray(payload["vectors"], dtype="float32")
    return core.encode_texts(texts)


def encode_image(data: bytes) -> np.ndarray:
    if remote():
        payload = _request("POST", "/core/encode/image", content=data)
        return np.asarray([payload["vector"]], dtype="float32")
    return core.encode_image(core.decode_query_image(data))


def encode_qwen(texts) -> np.ndarray:
    payload = _request("POST", "/core/encode/qwen", json={"texts": [str(t) for t in texts]})
    return np.asarray(payload["vectors"], dtype="float32")


def frame_embedding(index_name: str, video_id: str, frame_idx: int, profile: str | None):
    if remote():
        payload = _request("POST", "/core/visual/frame-embedding", json={
            "index": index_name, "video_id": video_id, "frame_idx": frame_idx,
            "profile": profile,
        })
        vector = np.asarray([payload["vector"]], dtype="float32")
        return vector, payload.get("global_id")
    return core.indexed_frame_embedding(index_name, video_id, frame_idx, profile)


def visual_search(index_name, profile, vectors, k, *, nprobe=0, ef=0,
                  video_ids=None, exclude_global_id=None, with_pts=False):
    if remote():
        array = np.asarray(vectors, dtype="float32")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        payload = _request("POST", "/core/visual/search", json={
            "index": index_name,
            "profile": profile,
            "vectors": array.tolist(),
            "k": int(k),
            "nprobe": int(nprobe),
            "ef": int(ef),
            "video_ids": list(video_ids) if video_ids else None,
            "exclude_global_id": exclude_global_id,
            "with_pts": bool(with_pts),
        })
        return payload
    return local_visual_search(
        index_name, profile, vectors, k, nprobe=nprobe, ef=ef,
        video_ids=video_ids, exclude_global_id=exclude_global_id, with_pts=with_pts,
    )


def local_visual_search(index_name, profile, vectors, k, *, nprobe=0, ef=0,
                        video_ids=None, exclude_global_id=None, with_pts=False):
    profile = profile or core.default_profile()
    if index_name not in core.available_indexes(profile):
        raise KeyError(f"unknown index {index_name} for profile {profile}")
    index = core.get_index(index_name, profile)
    config = core.index_info(index_name, profile)
    if index_name == "ivf":
        index.nprobe = nprobe if nprobe > 0 else config.get("nprobe_default", 128)
    if index_name == "hnsw":
        index.hnsw.efSearch = ef if ef > 0 else config.get("efSearch_default", 256)
    queries = np.asarray(vectors, dtype="float32")
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    metric = config.get("metric", "ip")
    id_video, id_frame, video_meta, lookup = core.load_meta(profile)
    times = core.load_times(profile)
    mappings = {}

    def build_hit(vector_index, global_id, distance, rank):
        video_id = video_meta[int(id_video[global_id])]["video_id"]
        frame = int(id_frame[global_id])
        score = float(distance) if metric == "ip" else float(1.0 - distance / 2.0)
        hit = {
            "vector_index": vector_index,
            "rank": rank,
            "distance": float(distance),
            "score": score,
            "global_id": int(global_id),
            "video_id": video_id,
            "frame_idx": frame,
            "time": float(times[global_id]),
        }
        if with_pts:
            mapping = mappings.get(video_id)
            if mapping is None:
                mapping = mappings[video_id] = frame_maps.load(video_id)
            hit["pts"] = int(mapping.pts[frame])
        return hit

    started = time.perf_counter()
    hits = []
    if video_ids:
        wanted = [int(lookup[video_id]["video_idx"]) for video_id in video_ids if video_id in lookup]
        global_ids = np.flatnonzero(np.isin(id_video, wanted)) if wanted else np.array([], dtype=np.int64)
        if exclude_global_id is not None:
            global_ids = global_ids[global_ids != exclude_global_id]
        if len(global_ids):
            for vector_index, vector in enumerate(queries):
                scores = index.reconstruct_batch(global_ids) @ vector
                order = np.argsort(-scores, kind="stable")[:k]
                for rank, position in enumerate(order, 1):
                    hits.append(build_hit(
                        vector_index, int(global_ids[position]), float(scores[position]), rank,
                    ))
    else:
        result_count = k + (exclude_global_id is not None)
        distances, indexes = index.search(queries, result_count)
        for vector_index in range(len(queries)):
            seen = 0
            for distance, global_id in zip(distances[vector_index], indexes[vector_index]):
                if global_id < 0 or global_id == exclude_global_id:
                    continue
                hits.append(build_hit(vector_index, int(global_id), float(distance), seen + 1))
                seen += 1
                if seen == k:
                    break
    return {
        "hits": hits,
        "index_info": config,
        "faiss_latency_ms": (time.perf_counter() - started) * 1000,
        "profile": profile,
    }


def annotate(points, profile, *, with_pts=True):
    if remote():
        payload = _request("POST", "/core/frame/annotate", json={
            "profile": profile,
            "points": list(points),
            "with_pts": bool(with_pts),
        })
        return payload["marks"]
    return local_annotate(points, profile, with_pts=with_pts)


def local_annotate(points, profile, *, with_pts=True):
    marks = []
    for point in points:
        try:
            frame = point.get("frame_idx")
            if frame is None:
                frame = core.nearest_frame(point["video_id"], float(point["time"]), profile)
            mark = {"frame_idx": int(frame)}
            if with_pts:
                mapping = frame_maps.load(point["video_id"])
                mark["pts"] = int(mapping.pts[frame])
                mark["time"] = mapping.frame_time(frame)
            marks.append(mark)
        except (OSError, FileNotFoundError, KeyError):
            marks.append(None)
    return marks


def audio_flat_search(vector, k):
    return _request("POST", "/core/audio/flat-search", json={
        "vector": np.asarray(vector, dtype="float32").reshape(-1).tolist(),
        "k": int(k),
    })["hits"]


class RemoteElasticsearch:
    """The subset of the Elasticsearch client `audio_search`/`ocr_search` use."""

    def search(self, *, index, body):
        return _request("POST", "/core/es/search", json={"index": index, "body": body})


def remote_elasticsearch() -> RemoteElasticsearch:
    return RemoteElasticsearch()
