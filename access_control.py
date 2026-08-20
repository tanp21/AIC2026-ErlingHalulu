"""Small, Tailscale-aware access and fair-use controls."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

T = TypeVar("T")
MAX_BODY_BYTES = 16 * 1024
BODY_READ_TIMEOUT_SECONDS = 10
logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class AccessConfig:
    allowed_users: frozenset[str]
    admin_user: str

    @classmethod
    def from_env(cls) -> "AccessConfig":
        raw = os.environ.get("AIC_ALLOWED_USERS", "")
        users = frozenset(part.strip().lower() for part in raw.split(",") if part.strip())
        if not users:
            raise RuntimeError("AIC_ALLOWED_USERS must contain the five permitted logins")
        if len(users) != 5:
            raise RuntimeError("AIC_ALLOWED_USERS must contain exactly five unique logins")
        admin = os.environ.get("AIC_ADMIN_USER", "").strip().lower()
        if not admin:
            raise RuntimeError("AIC_ADMIN_USER is required")
        if admin not in users:
            raise RuntimeError("AIC_ADMIN_USER must be present in the allowlist")
        return cls(users, admin)


def install_access_control(app: FastAPI, config: AccessConfig | None = None) -> None:
    if config is not None:
        app.state.access_config = config
    app.add_middleware(TailscaleAccessMiddleware)


class TailscaleAccessMiddleware:
    def __init__(self, app):
        self.app = app
        self._video_slots = asyncio.Semaphore(5)
        self._user_video_slots = defaultdict(lambda: asyncio.Semaphore(2))
        self._image_slots = asyncio.Semaphore(4)
        self._user_image_slots = defaultdict(lambda: asyncio.Semaphore(2))

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        login = "anonymous"
        status = 500
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        path = scope.get("path", "")

        async def reject(code: int, detail: str):
            nonlocal status
            status = code
            await JSONResponse({"detail": detail}, status_code=code)(scope, receive, send)

        try:
            if path != "/healthz":
                active = getattr(scope["app"].state, "access_config", None)
                if active is None:
                    return await reject(503, "access control is not configured")
                login = headers.get(b"tailscale-user-login", b"").decode("utf-8", "replace").strip().lower()
                if not login:
                    return await reject(401, "Tailscale identity required")
                if login not in active.allowed_users:
                    return await reject(403, "user is not allowed")
                scope.setdefault("state", {})["login"] = login
                scope["state"]["is_admin"] = login == active.admin_user

            content_length = headers.get(b"content-length", b"").decode("ascii", "ignore")
            if content_length:
                try:
                    too_large = int(content_length) > MAX_BODY_BYTES
                except ValueError:
                    return await reject(400, "invalid Content-Length")
                if too_large:
                    return await reject(413, "request body exceeds 16 KiB")

            received = 0
            buffered = []
            try:
                async with asyncio.timeout(BODY_READ_TIMEOUT_SECONDS):
                    while True:
                        message = await receive()
                        if message["type"] == "http.request":
                            received += len(message.get("body", b""))
                            if received > MAX_BODY_BYTES:
                                return await reject(413, "request body exceeds 16 KiB")
                            buffered.append(message)
                            if not message.get("more_body", False):
                                break
                        elif message["type"] == "http.disconnect":
                            return
            except TimeoutError:
                return await reject(408, "request body timed out")

            async def buffered_receive():
                if buffered:
                    return buffered.pop(0)
                return {"type": "http.disconnect"}

            async def audit_send(message):
                nonlocal status
                if message["type"] == "http.response.start":
                    status = message["status"]
                await send(message)

            if path.startswith("/video/"):
                async with self._video_slots, self._user_video_slots[login]:
                    return await self.app(scope, buffered_receive, audit_send)
            if path.startswith(("/thumb/", "/frame/")):
                async with self._image_slots, self._user_image_slots[login]:
                    return await self.app(scope, buffered_receive, audit_send)
            return await self.app(scope, buffered_receive, audit_send)
        finally:
            logger.info(
                "audit user=%s method=%s path=%s status=%s latency_ms=%.1f",
                login,
                scope.get("method", ""),
                path,
                status,
                (time.perf_counter() - started) * 1000,
            )


class LatestWinsCoordinator:
    def __init__(self, max_running: int = 5, rate_limit: int = 10, window_seconds: float = 10):
        self._capacity = asyncio.Semaphore(max_running)
        self._user_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._generations: defaultdict[str, int] = defaultdict(int)
        self._arrivals: defaultdict[str, deque[float]] = defaultdict(deque)
        self._rate_limit = rate_limit
        self._window = window_seconds
        self._gate = _ExclusiveGate()

    async def run(
        self, user: str, operation: Callable[[], Awaitable[T]], *, exclusive: bool = False
    ) -> T:
        now = time.monotonic()
        arrivals = self._arrivals[user]
        while arrivals and now - arrivals[0] >= self._window:
            arrivals.popleft()
        if len(arrivals) >= self._rate_limit:
            retry_after = max(1, int(self._window - (now - arrivals[0]) + 0.999))
            raise HTTPException(
                429, "search rate limit exceeded", headers={"Retry-After": str(retry_after)}
            )
        arrivals.append(now)

        self._generations[user] += 1
        generation = self._generations[user]
        async with self._user_locks[user]:
            if generation != self._generations[user]:
                raise HTTPException(409, "search request superseded by a newer request")
            async with self._gate.hold(exclusive):
                capacity = _NoopAsyncContext() if exclusive else self._capacity
                async with capacity:
                    if generation != self._generations[user]:
                        raise HTTPException(409, "search request superseded by a newer request")
                    task = asyncio.create_task(operation())
                    try:
                        return await asyncio.shield(task)
                    except asyncio.CancelledError:
                        # GPU/thread work cannot be stopped safely. Keep all locks until it
                        # finishes so a disconnected client cannot create hidden overlap.
                        await task
                        raise


class _ExclusiveGate:
    def __init__(self):
        self._condition = asyncio.Condition()
        self._readers = 0
        self._writer = False
        self._waiting_writers = 0

    def hold(self, exclusive):
        return _GateContext(self, exclusive)


class _GateContext:
    def __init__(self, gate, exclusive):
        self.gate = gate
        self.exclusive = exclusive

    async def __aenter__(self):
        async with self.gate._condition:
            if self.exclusive:
                self.gate._waiting_writers += 1
                try:
                    await self.gate._condition.wait_for(
                        lambda: not self.gate._writer and self.gate._readers == 0
                    )
                    self.gate._writer = True
                finally:
                    self.gate._waiting_writers -= 1
            else:
                await self.gate._condition.wait_for(
                    lambda: not self.gate._writer and self.gate._waiting_writers == 0
                )
                self.gate._readers += 1

    async def __aexit__(self, *_):
        async with self.gate._condition:
            if self.exclusive:
                self.gate._writer = False
            else:
                self.gate._readers -= 1
            self.gate._condition.notify_all()


class _NoopAsyncContext:
    async def __aenter__(self):
        pass

    async def __aexit__(self, *_):
        pass
