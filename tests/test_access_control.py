import asyncio
import inspect
import os
import unittest
from unittest.mock import Mock, patch

import httpx
from fastapi import FastAPI, Request

from backend.access_control import AccessConfig, LatestWinsCoordinator, install_access_control
from backend import access_control


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
        app.state.search_coordinator = Mock()
        self.app = app

        @app.get("/healthz")
        async def health():
            return {"ok": True}

        @app.get("/private")
        async def private(request: Request):
            return {"login": request.state.login, "admin": request.state.is_admin}

        @app.post("/api/collab/rounds")
        async def upload_round(request: Request):
            return {"size": len(await request.body())}

        @app.post("/search/image")
        async def search_image(request: Request):
            return {"size": len(await request.body())}

        @app.post("/agent/search")
        async def agent_search(request: Request):
            return {"size": len(await request.body())}

        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_health_is_the_only_anonymous_endpoint(self):
        self.assertEqual((await self.client.get("/healthz")).status_code, 200)
        self.assertEqual((await self.client.get("/private")).status_code, 401)

    async def test_rejects_user_outside_allowlist(self):
        response = await self.client.get("/private", headers={"X-AIC-User": "bad@example.com"})
        self.assertEqual(response.status_code, 403)

    async def test_exposes_normalized_identity_and_admin_flag(self):
        response = await self.client.get(
            "/private", headers={"X-AIC-User": " OWNER@EXAMPLE.COM "}
        )
        self.assertEqual(response.json(), {"login": "owner@example.com", "admin": True})

    async def test_admin_request_refreshes_priority_lease(self):
        await self.client.get("/private", headers={"X-AIC-User": "owner@example.com"})
        await self.client.get("/private", headers={"X-AIC-User": "mate@example.com"})

        self.app.state.search_coordinator.touch_admin_activity.assert_called_once_with()

    async def test_rejects_oversized_request_body(self):
        response = await self.client.post(
            "/private",
            headers={"X-AIC-User": "mate@example.com"},
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
            headers={"X-AIC-User": "mate@example.com"},
            content=chunks(),
        )
        self.assertEqual(response.status_code, 413)

    async def test_allows_larger_round_upload_only_for_owner(self):
        payload = b"x" * (20 * 1024)
        owner = await self.client.post(
            "/api/collab/rounds",
            headers={"X-AIC-User": "owner@example.com"},
            content=payload,
        )
        self.assertEqual(owner.json(), {"size": len(payload)})
        teammate = await self.client.post(
            "/api/collab/rounds",
            headers={"X-AIC-User": "mate@example.com"},
            content=payload,
        )
        self.assertEqual(teammate.status_code, 413)
        too_large = await self.client.post(
            "/api/collab/rounds",
            headers={"X-AIC-User": "owner@example.com"},
            content=b"x" * (1024 * 1024 + 1),
        )
        self.assertEqual(too_large.status_code, 413)

    async def test_allows_bounded_image_search_upload_for_every_teammate(self):
        headers = {"X-AIC-User": "mate@example.com"}
        allowed = await self.client.post(
            "/search/image", headers=headers, content=b"x" * (6 * 1024 * 1024)
        )
        too_large = await self.client.post(
            "/search/image", headers=headers, content=b"x" * (8 * 1024 * 1024 + 1)
        )

        self.assertEqual(allowed.json(), {"size": 6 * 1024 * 1024})
        self.assertEqual(too_large.status_code, 413)

    async def test_allows_a_bounded_agent_batch_for_every_teammate(self):
        headers = {"X-AIC-User": "mate@example.com"}
        allowed = await self.client.post(
            "/agent/search", headers=headers, content=b"x" * (20 * 1024)
        )
        too_large = await self.client.post(
            "/agent/search", headers=headers, content=b"x" * (256 * 1024 + 1)
        )

        self.assertEqual(allowed.json(), {"size": 20 * 1024})
        self.assertEqual(too_large.status_code, 413)

    async def test_times_out_a_slow_request_body(self):
        async def slow_chunks():
            yield b"{"
            await asyncio.sleep(0.02)
            yield b"}"

        with patch.object(access_control, "BODY_READ_TIMEOUT_SECONDS", 0.005):
            response = await self.client.post(
                "/private",
                headers={"X-AIC-User": "mate@example.com"},
                content=slow_chunks(),
            )
        self.assertEqual(response.status_code, 408)

    async def test_media_requests_are_not_queued_by_hardware_limits(self):
        states = {
            kind: {"started": 0, "all_started": asyncio.Event(), "release": asyncio.Event()}
            for kind in ("video", "thumb")
        }

        @self.app.get("/{kind}/{item}")
        async def media(kind: str, item: str):
            state = states[kind]
            state["started"] += 1
            if state["started"] == 6:
                state["all_started"].set()
            await state["release"].wait()
            return {"item": item}

        headers = {"X-AIC-User": "mate@example.com"}
        for kind, state in states.items():
            requests = [
                asyncio.create_task(self.client.get(f"/{kind}/{index}", headers=headers))
                for index in range(6)
            ]
            try:
                await asyncio.wait_for(state["all_started"].wait(), 0.1)
            finally:
                state["release"].set()
                responses = await asyncio.gather(*requests)
            self.assertTrue(all(response.status_code == 200 for response in responses))


class DelegatedIdentityTests(unittest.IsolatedAsyncioTestCase):
    """A media host forwards each teammate's own login; only that host may name someone else.

    Without this the compute host stamps one tailnet identity for the whole team, so every
    teammate shares a coordinator slot and a single agent run.
    """

    MEDIA_HOST = "laptop@example.com"

    def setUp(self):
        app = FastAPI()
        install_access_control(app, AccessConfig(
            allowed_users=frozenset({self.MEDIA_HOST, "owner@example.com", "mate@example.com"}),
            admin_user="owner@example.com",
        ))
        app.state.search_coordinator = Mock()

        @app.get("/private")
        async def private(request: Request):
            return {"login": request.state.login, "admin": request.state.is_admin}

        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def _whoami(self, forwarded_as, delegated=None):
        headers = {"X-AIC-User": forwarded_as}
        if delegated is not None:
            headers["X-AIC-Delegated-User"] = delegated
        response = await self.client.get("/private", headers=headers)
        return response.status_code, response.json().get("login"), response.json().get("admin")

    async def test_media_host_may_name_one_of_its_own_users(self):
        with patch.dict(os.environ, {"AIC_MEDIA_HOST_LOGIN": self.MEDIA_HOST}):
            self.assertEqual(
                await self._whoami(self.MEDIA_HOST, "mate@example.com"),
                (200, "mate@example.com", False),
            )

    async def test_delegated_admin_keeps_admin_rights(self):
        with patch.dict(os.environ, {"AIC_MEDIA_HOST_LOGIN": self.MEDIA_HOST}):
            self.assertEqual(
                await self._whoami(self.MEDIA_HOST, "owner@example.com"),
                (200, "owner@example.com", True),
            )

    async def test_the_delegated_login_still_has_to_be_allowed(self):
        with patch.dict(os.environ, {"AIC_MEDIA_HOST_LOGIN": self.MEDIA_HOST}):
            status, _, _ = await self._whoami(self.MEDIA_HOST, "stranger@example.com")
            self.assertEqual(status, 403)

    async def test_anyone_but_the_media_host_cannot_delegate(self):
        with patch.dict(os.environ, {"AIC_MEDIA_HOST_LOGIN": self.MEDIA_HOST}):
            self.assertEqual(
                await self._whoami("mate@example.com", "owner@example.com"),
                (200, "mate@example.com", False),
            )

    async def test_a_media_host_is_ignored_when_none_is_configured(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIC_MEDIA_HOST_LOGIN", None)
            self.assertEqual(
                await self._whoami(self.MEDIA_HOST, "owner@example.com"),
                (200, self.MEDIA_HOST, False),
            )


class LatestWinsCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_recent_admin_activity_delays_teammate_admission(self):
        self.assertIn(
            "admin_idle_seconds", inspect.signature(LatestWinsCoordinator).parameters
        )
        coordinator = LatestWinsCoordinator(
            max_running=2,
            rate_limit=10,
            window_seconds=10,
            admin_user="owner@example.com",
            admin_idle_seconds=0.05,
        )
        self.assertTrue(hasattr(coordinator, "touch_admin_activity"))
        started = asyncio.Event()
        coordinator.touch_admin_activity()

        teammate = asyncio.create_task(
            coordinator.run(
                "mate@example.com", lambda: self._set_event(started)
            )
        )
        await asyncio.sleep(0.01)
        self.assertFalse(started.is_set())
        await asyncio.wait_for(started.wait(), 0.2)
        await teammate

    async def test_admin_is_not_search_rate_limited(self):
        coordinator = LatestWinsCoordinator(
            max_running=5,
            rate_limit=10,
            window_seconds=10,
            admin_user="owner@example.com",
        )

        try:
            for _ in range(11):
                await coordinator.run("owner@example.com", lambda: asyncio.sleep(0))
        except Exception as exc:
            self.fail(f"admin search was rate limited: {exc}")

    async def test_waiting_exclusive_admin_runs_before_next_teammate(self):
        self.assertIn("admin_user", inspect.signature(LatestWinsCoordinator).parameters)
        coordinator = LatestWinsCoordinator(
            max_running=2,
            rate_limit=10,
            window_seconds=10,
            admin_user="owner@example.com",
        )
        release_first = asyncio.Event()
        release_admin = asyncio.Event()
        first_started = asyncio.Event()
        admin_started = asyncio.Event()
        second_started = asyncio.Event()
        events = []

        async def first_teammate():
            events.append("first")
            first_started.set()
            await release_first.wait()

        async def admin_operation():
            events.append("admin")
            admin_started.set()
            await release_admin.wait()

        async def second_teammate():
            events.append("second")
            second_started.set()

        first = asyncio.create_task(
            coordinator.run("mate1@example.com", first_teammate)
        )
        await asyncio.wait_for(first_started.wait(), 1)
        second = asyncio.create_task(
            coordinator.run("mate2@example.com", second_teammate)
        )
        await asyncio.sleep(0)
        admin = asyncio.create_task(
            coordinator.run("owner@example.com", admin_operation, exclusive=True)
        )

        await asyncio.sleep(0)
        admin_overlapped = admin_started.is_set()
        second_overlapped = second_started.is_set()
        release_first.set()
        release_admin.set()
        await asyncio.gather(first, admin, second)
        self.assertFalse(admin_overlapped)
        self.assertFalse(second_overlapped)
        self.assertEqual(events, ["first", "admin", "second"])

    async def test_second_exclusive_admin_request_keeps_priority_over_waiting_teammate(self):
        coordinator = LatestWinsCoordinator(
            max_running=2,
            rate_limit=10,
            window_seconds=10,
            admin_user="owner@example.com",
        )
        release_first = asyncio.Event()
        first_started = asyncio.Event()
        events = []

        async def first_admin():
            events.append("admin1")
            first_started.set()
            await release_first.wait()

        async def record(name):
            events.append(name)

        first = asyncio.create_task(
            coordinator.run("owner@example.com", first_admin, exclusive=True)
        )
        await asyncio.wait_for(first_started.wait(), 1)
        teammate = asyncio.create_task(
            coordinator.run("mate@example.com", lambda: record("teammate"))
        )
        second = asyncio.create_task(
            coordinator.run(
                "owner@example.com", lambda: record("admin2"), exclusive=True
            )
        )
        await asyncio.sleep(0)
        release_first.set()
        await asyncio.gather(first, second, teammate)

        self.assertEqual(events, ["admin1", "admin2", "teammate"])

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

    @staticmethod
    async def _set_event(event):
        event.set()


class RoleCapacityTests(unittest.IsolatedAsyncioTestCase):
    async def test_reserves_one_total_slot_for_admin(self):
        self.assertTrue(hasattr(access_control, "RoleCapacity"))
        capacity = access_control.RoleCapacity(total=5, teammates=4)
        release = asyncio.Event()
        four_started = asyncio.Event()
        admin_started = asyncio.Event()
        fifth_started = asyncio.Event()
        started_teammates = 0

        async def occupy_teammate(started=None):
            nonlocal started_teammates
            async with capacity.hold(is_admin=False):
                started_teammates += 1
                if started_teammates == 4:
                    four_started.set()
                if started is not None:
                    started.set()
                await release.wait()

        async def occupy_admin():
            async with capacity.hold(is_admin=True):
                admin_started.set()
                await release.wait()

        occupants = [asyncio.create_task(occupy_teammate()) for _ in range(4)]
        await asyncio.wait_for(four_started.wait(), 1)
        fifth = asyncio.create_task(occupy_teammate(fifth_started))
        admin = asyncio.create_task(occupy_admin())

        await asyncio.wait_for(admin_started.wait(), 1)
        self.assertFalse(fifth_started.is_set())
        release.set()
        await asyncio.gather(*occupants, fifth, admin)


if __name__ == "__main__":
    unittest.main()
