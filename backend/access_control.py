"""Small, trusted-ingress access and fair-use controls."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

T = TypeVar("T")
MAX_BODY_BYTES = 16 * 1024
MAX_ROUND_UPLOAD_BYTES = 1024 * 1024
MAX_IMAGE_SEARCH_BYTES = 8 * 1024 * 1024
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


class RoleCapacity:
    """Bound total work while keeping part of it unavailable to teammates."""

    def __init__(self, total: int, teammates: int):
        self._total = asyncio.Semaphore(total)
        self._teammates = asyncio.Semaphore(teammates)

    @asynccontextmanager
    async def hold(self, is_admin: bool):
        if not is_admin:
            await self._teammates.acquire()
        try:
            await self._total.acquire()
            try:
                yield
            finally:
                self._total.release()
        finally:
            if not is_admin:
                self._teammates.release()


def install_access_control(app: FastAPI, config: AccessConfig | None = None) -> None:
    if config is not None:
        app.state.access_config = config
    app.add_middleware(TrustedIngressAccessMiddleware)


class TrustedIngressAccessMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        started = time.perf_counter()
        login = "anonymous"
        is_admin = False
        status = 500
        resource = "request"
        admission_wait_ms = 0.0
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        path = scope.get("path", "")
        if path.startswith(("/video/", "/clip/")):
            resource = "video"
        elif path.startswith(("/thumb/", "/frame/")):
            resource = "image"

        async def reject(code: int, detail: str):
            nonlocal status
            status = code
            await JSONResponse({"detail": detail}, status_code=code)(scope, receive, send)

        try:
            if path != "/healthz":
                active = getattr(scope["app"].state, "access_config", None)
                if active is None:
                    return await reject(503, "access control is not configured")
                login = headers.get(b"x-aic-user", b"").decode("utf-8", "replace").strip().lower()
                if not login:
                    return await reject(401, "trusted proxy identity required")
                if login not in active.allowed_users:
                    return await reject(403, "user is not allowed")
                scope.setdefault("state", {})["login"] = login
                is_admin = login == active.admin_user
                scope["state"]["is_admin"] = is_admin
                coordinator = getattr(scope["app"].state, "search_coordinator", None)
                if is_admin and coordinator is not None:
                    coordinator.touch_admin_activity()

            if path == "/search/image" and scope.get("method") == "POST":
                body_limit = MAX_IMAGE_SEARCH_BYTES
            elif (
                path == "/api/collab/rounds"
                and scope.get("method") == "POST"
                and login == active.admin_user
            ):
                body_limit = MAX_ROUND_UPLOAD_BYTES
            else:
                body_limit = MAX_BODY_BYTES
            content_length = headers.get(b"content-length", b"").decode("ascii", "ignore")
            if content_length:
                try:
                    too_large = int(content_length) > body_limit
                except ValueError:
                    return await reject(400, "invalid Content-Length")
                if too_large:
                    return await reject(413, "request body exceeds route limit")

            received = 0
            buffered = []
            try:
                async with asyncio.timeout(BODY_READ_TIMEOUT_SECONDS):
                    while True:
                        message = await receive()
                        if message["type"] == "http.request":
                            received += len(message.get("body", b""))
                            if received > body_limit:
                                return await reject(413, "request body exceeds route limit")
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

            return await self.app(scope, buffered_receive, audit_send)
        finally:
            logger.info(
                "audit user=%s role=%s resource=%s method=%s path=%s status=%s "
                "admission_wait_ms=%.1f latency_ms=%.1f",
                login,
                "admin" if is_admin else "teammate",
                resource,
                scope.get("method", ""),
                path,
                status,
                admission_wait_ms,
                (time.perf_counter() - started) * 1000,
            )


class LatestWinsCoordinator:
    def __init__(
        self,
        max_running: int = 5,
        rate_limit: int | None = 10,
        window_seconds: float = 10,
        admin_user: str | None = None,
        admin_idle_seconds: float = 0,
    ):
        teammate_slots = max_running - 1 if admin_user is not None else max_running
        self._capacity = RoleCapacity(total=max_running, teammates=teammate_slots)
        self._admin_user = admin_user
        self._admin_idle_seconds = admin_idle_seconds
        self._admin_active_until = 0.0
        self._user_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._generations: defaultdict[str, int] = defaultdict(int)
        self._arrivals: defaultdict[str, deque[float]] = defaultdict(deque)
        self._rate_limit = rate_limit
        self._window = window_seconds
        self._gate = _ExclusiveGate()

    def touch_admin_activity(self) -> None:
        self._admin_active_until = max(
            self._admin_active_until,
            time.monotonic() + self._admin_idle_seconds,
        )

    async def _wait_for_admin_idle(self) -> None:
        while delay := self._admin_active_until - time.monotonic():
            if delay <= 0:
                return
            await asyncio.sleep(delay)

    async def run(
        self, user: str, operation: Callable[[], Awaitable[T]], *, exclusive: bool = False
    ) -> T:
        is_admin = user == self._admin_user
        if not is_admin and self._rate_limit is not None:
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

        queued = time.perf_counter()

        async def execute(priority_exclusive: bool) -> T:
            if generation != self._generations[user]:
                raise HTTPException(409, "search request superseded by a newer request")
            logger.info(
                "search_admission user=%s role=%s exclusive=%s queue_wait_ms=%.1f",
                user,
                "admin" if is_admin else "teammate",
                priority_exclusive,
                (time.perf_counter() - queued) * 1000,
            )
            task = asyncio.create_task(operation())
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                # GPU/thread work cannot be stopped safely. Keep all locks until it
                # finishes so a disconnected client cannot create hidden overlap.
                await task
                raise

        if is_admin:
            if exclusive:
                async with self._gate.hold(True):
                    async with self._user_locks[user]:
                        return await execute(True)
            async with self._user_locks[user]:
                async with self._capacity.hold(is_admin=True):
                    return await execute(False)

        async with self._user_locks[user]:
            await self._wait_for_admin_idle()
            priority_exclusive = exclusive
            capacity = (
                _NoopAsyncContext()
                if priority_exclusive
                else self._capacity.hold(is_admin=False)
            )
            async with capacity:
                async with self._gate.hold(priority_exclusive):
                    return await execute(priority_exclusive)


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
