"""HTTP client for the search server's agent-facing routes."""
from __future__ import annotations

import logging
import threading
import time

import httpx

from . import compat

logger = logging.getLogger("agent")
RETRYABLE_STATUS = {408, 409, 429, 502, 503, 504}
# Thumbnails are cut by ffmpeg on demand; under a burst the server answers 500
# "ffmpeg failed" for requests that succeed a moment later.
IMAGE_RETRYABLE_STATUS = RETRYABLE_STATUS | {500}
THUMB_WIDTHS = (160, 320, 640)
# Across all queries this process runs; the server bounds ffmpeg work per user too.
MAX_IMAGE_REQUESTS = 6
# A thumbnail normally takes 0.2-1 s, but over the tailnet about one request in 50
# stalls for ~30 s while an immediate retry of the same frame takes 0.3 s; a short
# timeout plus a retry keeps that stall out of the agent's 20 s budget.
IMAGE_TIMEOUT = 5.0
FRAME_TIMEOUT = 10.0
LOOKUP_TIMEOUT = 8.0  # /context, /transcript
# A batch takes 0.3-4 s; a retry supersedes a stalled one under latest-wins.
BATCH_TIMEOUT = 20.0


class SearchError(RuntimeError):
    def __init__(self, message, status=None):
        super().__init__(message)
        self.status = status


class SearchClient:
    """One agent process = one coordinator identity on the server.

    `/agent/search` runs under latest-wins scheduling per login, so two overlapping
    batches from this process would cancel each other (409). Batches are therefore
    serialized here; image and transcript fetches are not coordinated server-side and
    may run concurrently.

    `api="auto"` uses `/agent/search` and falls back, for good, to the public
    search routes (`agent.compat`) the first time a server answers it with 404.

    The caption channel is optional in the same spirit: a server built before it,
    or one whose Elasticsearch has no caption index, makes the first caption
    sub-query fail, and caption queries are then dropped for the rest of the run
    instead of failing a batch per step.
    """

    def __init__(self, base_url, *, user=None, password=None, trusted_user=None,
                 timeout=180.0, retries=3, http=None, sleep=time.sleep, api="auto",
                 captions=True):
        if api not in ("auto", "batch", "compat"):
            raise ValueError(f"unknown search api {api!r}")
        self.api = api
        headers = {"X-AIC-User": trusted_user} if trusted_user else {}
        auth = (user, password) if user and password else None
        self._http = http or httpx.Client(
            base_url=base_url, auth=auth, headers=headers, timeout=timeout,
            limits=httpx.Limits(max_connections=16),
        )
        self._retries = retries
        self._sleep = sleep
        self._batch_lock = threading.Lock()
        # None until a caption query has been tried against this server.
        self.captions_supported = None
        self._image_slots = threading.BoundedSemaphore(MAX_IMAGE_REQUESTS)
        # Set once a server answers `/captions` with 404/405, so an old build is
        # asked exactly once per run instead of once per candidate.
        self._no_captions = False
        # Off entirely under `AIC_AGENT_CAPTIONS=0`.
        self._captions_enabled = captions

    @classmethod
    def from_settings(cls, settings):
        return cls(
            settings.server_url, user=settings.server_user,
            password=settings.server_password, trusted_user=settings.trusted_user,
            api=settings.search_api, captions=getattr(settings, "captions", True),
        )

    def close(self):
        self._http.close()

    def _request(self, method, path, retryable=RETRYABLE_STATUS, **kwargs):
        for attempt in range(self._retries + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise SearchError(f"{method} {path}: {exc}") from exc
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in retryable or attempt == self._retries:
                    raise SearchError(
                        f"{method} {path}: HTTP {response.status_code} {_detail(response)}",
                        response.status_code,
                    )
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(float(retry_after))
                    continue
            self._sleep(min(8.0, 0.5 * 2 ** attempt))
        raise AssertionError("unreachable")

    def search(self, queries: list[dict]) -> dict[str, dict]:
        """Run one batch; returns `{query_id: {"kind", "hits"} | {"kind", "error"}}`."""
        dropped = {}
        if self.captions_supported is False:
            dropped = {query["id"]: {"kind": "caption", "error": "caption channel unavailable"}
                       for query in queries if query.get("kind") == "caption"}
            queries = [query for query in queries if query.get("kind") != "caption"]
            if not queries:
                return dropped
        try:
            results = self._search(queries)
        except SearchError as exc:
            # A server predating the caption channel rejects the whole batch (422).
            kept = [query for query in queries if query.get("kind") != "caption"]
            if exc.status != 422 or len(kept) == len(queries):
                raise
            self.captions_supported = False
            logger.warning("server rejects caption queries (%s); dropping them", exc)
            results = self._search(kept) if kept else {}
        self._note_caption_support(queries, results)
        return {**results, **dropped}

    def _note_caption_support(self, queries, results):
        if self.captions_supported is not None:
            return
        asked = [query["id"] for query in queries if query.get("kind") == "caption"]
        if not asked:
            return
        failed = [qid for qid in asked if "error" in results.get(qid, {})]
        self.captions_supported = len(failed) < len(asked)
        if not self.captions_supported:
            logger.warning("server has no caption channel (%s); dropping caption queries",
                           results.get(failed[0], {}).get("error"))

    def _search(self, queries: list[dict]) -> dict[str, dict]:
        with self._batch_lock:
            if self.api != "compat":
                try:
                    response = self._request("POST", "/agent/search", json={"queries": queries},
                                             timeout=BATCH_TIMEOUT)
                    return response.json()["results"]
                except SearchError as exc:
                    if exc.status != 404 or self.api == "batch":
                        raise
                    logger.warning("server has no /agent/search; using the public search routes")
                    self.api = "compat"
            return compat.run_batch(self._request, queries, SearchError)

    def image(self, video_id: str, frame_idx: int, width: int = 640) -> bytes:
        """JPEG of one frame; `width=0` asks for the full-resolution source frame."""
        with self._image_slots:
            if width == 0:
                return self._request("GET", f"/frame/{video_id}/{frame_idx}",
                                     retryable=IMAGE_RETRYABLE_STATUS, timeout=FRAME_TIMEOUT).content
            width = min(THUMB_WIDTHS, key=lambda allowed: abs(allowed - width))
            return self._request("GET", f"/thumb/{video_id}/{frame_idx}", params={"w": width},
                                 retryable=IMAGE_RETRYABLE_STATUS, timeout=IMAGE_TIMEOUT).content

    def neighbours(self, video_id: str, frame_idx: int, radius: int = 6) -> list[dict]:
        """Indexed keyframes around one frame, in time order."""
        response = self._request(
            "GET", f"/context/{video_id}/{frame_idx}", params={"radius": radius},
            timeout=LOOKUP_TIMEOUT,
        )
        return [
            {"video_id": frame["video_id"], "frame_idx": frame["frame_idx"], "time": frame["time"]}
            for frame in response.json()["frames"]
        ]

    def caption_channel_available(self) -> bool:
        """Whether this server can answer caption queries at all, probed once per run.

        The planner writes two caption queries into every round-1 batch, and a deployment
        without the caption index answers each with 503 -- they then take two of the
        batch's twelve slots and fail. The latch that drops them flips only after the batch
        they cost, which is too late for round 1 to use those slots: measured across the
        saved runs, 28 of 197 planned round-1 queries were caption queries against an index
        that does not exist. Any syntactically valid id probes the deployment; the route
        answers 503 when the index is missing, and 200 with whatever hits exist otherwise.
        """
        if self.captions_supported is None:
            try:
                self._request("GET", "/captions/L26_V001", timeout=LOOKUP_TIMEOUT)
            except SearchError as exc:
                self.captions_supported = exc.status not in (404, 405, 500, 503)
            else:
                self.captions_supported = True
        return bool(self.captions_supported)

    def captions(self, video_id: str) -> list[dict]:
        """`[{"start", "end", "time", "frame_idx", "text"}]` shot captions, in time order.

        Empty when the deployment has no caption index; a caption is what a
        model wrote about the whole shot, so it carries counts and on-screen
        sentences that neither a keyframe nor the transcript shows.
        """
        try:
            return self._request("GET", f"/captions/{video_id}",
                                 timeout=LOOKUP_TIMEOUT).json()["captions"]
        except (SearchError, KeyError, ValueError):
            return []

    def transcript(self, video_id: str) -> list[dict]:
        """`[{"start", "end", "text"}]` cues for the whole video."""
        return self._request("GET", f"/transcript/{video_id}", timeout=LOOKUP_TIMEOUT).json()["cues"]

    def captions_for_frames(self, video_id: str, frames: list[int]) -> dict[int, dict]:
        """CapRL shot captions for `frames`, keyed by frame index.

        The sibling of `captions`, for the verify step: that one asks for a whole
        video in time order, this asks only about the frames being judged and gets
        them back already resolved through the frame map. Both read the same captions.

        Returns `{}` rather than raising when the server has no captions for this
        video — every 2025 `K*` video, and anything whose shot pass has not run. A
        caption is context, not evidence, so losing it must never cost a run; the
        route answers `available: false` rather than 404 for the same reason, and
        the latch is only for a server old enough not to have the route at all.
        """
        if not frames or self._no_captions or not self._captions_enabled:
            return {}
        wanted = ",".join(str(int(frame)) for frame in frames)
        try:
            response = self._request("GET", f"/captions/{video_id}/frames",
                                     params={"frames": wanted}, timeout=LOOKUP_TIMEOUT)
        except SearchError as exc:
            if exc.status in (404, 405):
                self._no_captions = True
                logger.warning("server has no /captions/{id}/frames; "
                               "continuing without shot captions")
            return {}
        found = {}
        for key, value in (response.json().get("captions") or {}).items():
            try:
                found[int(key)] = value
            except (TypeError, ValueError):
                continue
        return found


def _detail(response):
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200]
    return detail if isinstance(detail, str) else str(detail)[:200]
