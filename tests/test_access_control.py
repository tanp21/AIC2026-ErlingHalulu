import asyncio
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI, Request

from access_control import AccessConfig, LatestWinsCoordinator, install_access_control
import access_control


class AccessConfigTests(unittest.TestCase):
    def test_requires_allowlist_and_admin_membership(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "AIC_ALLOWED_USERS"):
                AccessConfig.from_env()

        env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "exactly five"):
                AccessConfig.from_env()

        env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "outsider@example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "allowlist"):
                AccessConfig.from_env()


class AccessMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        app = FastAPI()
        config = AccessConfig(
            allowed_users=frozenset({"owner@example.com", "mate@example.com"}),
            admin_user="owner@example.com",
        )
        install_access_control(app, config)

        @app.get("/healthz")
        async def health():
            return {"ok": True}

        @app.get("/private")
        async def private(request: Request):
            return {"login": request.state.login, "admin": request.state.is_admin}

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_is_the_only_anonymous_endpoint(self):
        self.assertEqual((await self.client.get("/healthz")).status_code, 200)
        self.assertEqual((await self.client.get("/private")).status_code, 401)

    async def test_rejects_user_outside_allowlist(self):
        response = await self.client.get("/private", headers={"Tailscale-User-Login": "bad@example.com"})
        self.assertEqual(response.status_code, 403)

    async def test_exposes_normalized_identity_and_admin_flag(self):
        response = await self.client.get(
            "/private", headers={"Tailscale-User-Login": " OWNER@EXAMPLE.COM "}
        )
        self.assertEqual(response.json(), {"login": "owner@example.com", "admin": True})

    async def test_rejects_oversized_request_body(self):
        response = await self.client.post(
            "/private",
            headers={"Tailscale-User-Login": "mate@example.com"},
            content=b"x" * (16 * 1024 + 1),
        )
        self.assertEqual(response.status_code, 413)

    async def test_rejects_oversized_get_body_even_if_endpoint_never_reads_it(self):
        async def chunks():
            yield b"x" * 9000
            yield b"x" * 9000

        response = await self.client.request(
            "GET",
            "/private",
            headers={"Tailscale-User-Login": "mate@example.com"},
            content=chunks(),
        )
        self.assertEqual(response.status_code, 413)

    async def test_times_out_a_slow_request_body(self):
        async def slow_chunks():
            yield b"{"
            await asyncio.sleep(0.02)
            yield b"}"

        with patch.object(access_control, "BODY_READ_TIMEOUT_SECONDS", 0.005):
            response = await self.client.post(
                "/private",
                headers={"Tailscale-User-Login": "mate@example.com"},
                content=slow_chunks(),
            )
        self.assertEqual(response.status_code, 408)


class LatestWinsCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_latest_waiting_request_runs_after_current_finishes(self):
        coordinator = LatestWinsCoordinator(max_running=1, rate_limit=10, window_seconds=10)
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        ran = []

        async def operation(name):
            ran.append(name)
            if name == "first":
                first_started.set()
                await release_first.wait()
            return name

        first = asyncio.create_task(coordinator.run("mate@example.com", lambda: operation("first")))
        await first_started.wait()
        stale = asyncio.create_task(coordinator.run("mate@example.com", lambda: operation("stale")))
        latest = asyncio.create_task(coordinator.run("mate@example.com", lambda: operation("latest")))
        await asyncio.sleep(0)
        release_first.set()

        self.assertEqual(await first, "first")
        with self.assertRaisesRegex(Exception, "superseded"):
            await stale
        self.assertEqual(await latest, "latest")
        self.assertEqual(ran, ["first", "latest"])

    async def test_rate_limit_rejects_eleventh_request_with_retry_after(self):
        coordinator = LatestWinsCoordinator(max_running=5, rate_limit=10, window_seconds=10)
        for _ in range(10):
            await coordinator.run("mate@example.com", lambda: asyncio.sleep(0))
        with self.assertRaisesRegex(Exception, "rate limit") as caught:
            await coordinator.run("mate@example.com", lambda: asyncio.sleep(0))
        self.assertEqual(caught.exception.status_code, 429)
        self.assertIn("Retry-After", caught.exception.headers)

    async def test_exclusive_operation_waits_for_all_regular_operations(self):
        coordinator = LatestWinsCoordinator(max_running=5, rate_limit=10, window_seconds=10)
        release = asyncio.Event()
        both_started = asyncio.Event()
        running = 0
        events = []

        async def regular(name):
            nonlocal running
            running += 1
            events.append(f"start-{name}")
            if running == 2:
                both_started.set()
            await release.wait()
            events.append(f"end-{name}")
            running -= 1

        first = asyncio.create_task(coordinator.run("one", lambda: regular("one")))
        second = asyncio.create_task(coordinator.run("two", lambda: regular("two")))
        await both_started.wait()
        exclusive = asyncio.create_task(
            coordinator.run("owner", lambda: self._record(events, "exclusive"), exclusive=True)
        )
        await asyncio.sleep(0)
        self.assertNotIn("exclusive", events)
        release.set()
        await asyncio.gather(first, second, exclusive)
        self.assertEqual(events[-1], "exclusive")

    @staticmethod
    async def _record(events, value):
        events.append(value)


if __name__ == "__main__":
    unittest.main()
