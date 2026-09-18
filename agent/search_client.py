"""HTTP client for the search server's agent-facing routes."""
from __future__ import annotations

import threading
import time

import httpx

RETRYABLE_STATUS = {408, 409, 429, 502, 503, 504}
THUMB_WIDTHS = (160, 320, 640)


class SearchError(RuntimeError):
    pass


class SearchClient:
    """One agent process = one coordinator identity on the server.

    `/agent/search` runs under latest-wins scheduling per login, so two overlapping
    batches from this process would cancel each other (409). Batches are therefore
    serialized here; image and transcript fetches are not coordinated server-side and
    may run concurrently.
    """

    def __init__(self, base_url, *, user=None, password=None, trusted_user=None,
                 timeout=90.0, retries=3, http=None, sleep=time.sleep):
        headers = {"X-AIC-User": trusted_user} if trusted_user else {}
        auth = (user, password) if user and password else None
        self._http = http or httpx.Client(
            base_url=base_url, auth=auth, headers=headers, timeout=timeout,
            limits=httpx.Limits(max_connections=16),
        )
        self._retries = retries
        self._sleep = sleep
        self._batch_lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings):
        return cls(
            settings.server_url, user=settings.server_user,
            password=settings.server_password, trusted_user=settings.trusted_user,
        )

    def close(self):
        self._http.close()

    def _request(self, method, path, **kwargs):
        for attempt in range(self._retries + 1):
            try:
                response = self._http.request(method, path, **kwargs)
            except httpx.TransportError as exc:
                if attempt == self._retries:
                    raise SearchError(f"{method} {path}: {exc}") from exc
            else:
                if response.status_code < 400:
                    return response
                if response.status_code not in RETRYABLE_STATUS or attempt == self._retries:
                    raise SearchError(f"{method} {path}: HTTP {response.status_code} {_detail(response)}")
                retry_after = response.headers.get("Retry-After", "")
                if retry_after.isdigit():
                    self._sleep(float(retry_after))
                    continue
            self._sleep(min(8.0, 0.5 * 2 ** attempt))
        raise AssertionError("unreachable")

    def search(self, queries: list[dict]) -> dict[str, dict]:
        """Run one batch; returns `{query_id: {"kind", "hits"} | {"kind", "error"}}`."""
        with self._batch_lock:
            response = self._request("POST", "/agent/search", json={"queries": queries})
        return response.json()["results"]

    def image(self, video_id: str, frame_idx: int, width: int = 640) -> bytes:
        """JPEG of one frame; `width=0` asks for the full-resolution source frame."""
        if width == 0:
            return self._request("GET", f"/frame/{video_id}/{frame_idx}").content
        width = min(THUMB_WIDTHS, key=lambda allowed: abs(allowed - width))
        return self._request("GET", f"/thumb/{video_id}/{frame_idx}", params={"w": width}).content

    def neighbours(self, video_id: str, frame_idx: int, radius: int = 6) -> list[dict]:
        """Indexed keyframes around one frame, in time order."""
        response = self._request(
            "GET", f"/context/{video_id}/{frame_idx}", params={"radius": radius}
        )
        return [
            {"video_id": frame["video_id"], "frame_idx": frame["frame_idx"], "time": frame["time"]}
            for frame in response.json()["frames"]
        ]

    def transcript(self, video_id: str) -> list[dict]:
        """`[{"start", "end", "text"}]` cues for the whole video."""
        return self._request("GET", f"/transcript/{video_id}").json()["cues"]


def _detail(response):
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200]
    return detail if isinstance(detail, str) else str(detail)[:200]
