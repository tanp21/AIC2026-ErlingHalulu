import unittest
from unittest.mock import patch

import httpx

from backend import app as server
from backend.access_control import AccessConfig, LatestWinsCoordinator


class SecureRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(
            frozenset({"owner@example.com", "mate@example.com"}), "owner@example.com"
        )
        server.app.state.search_coordinator = LatestWinsCoordinator()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_session_reports_trusted_proxy_identity_and_role(self):
        response = await self.client.get(
            "/session", headers={"X-AIC-User": "mate@example.com"}
        )
        body = response.json()
        self.assertEqual(body["login"], "mate@example.com")
        self.assertFalse(body["is_admin"])

    async def test_session_never_carries_the_checker_key_itself(self):
        # The Search tab hides "Check by AI" when the deployment has no OpenRouter key, so it
        # has to be told whether there is one -- and only that. The key stays on the server.
        response = await self.client.get(
            "/session", headers={"X-AIC-User": "mate@example.com"}
        )
        check = response.json()["check"]
        self.assertEqual(sorted(check), ["available", "model", "window_seconds"])
        self.assertIsInstance(check["available"], bool)

    async def test_compare_is_owner_only(self):
        response = await self.client.post(
            "/compare",
            headers={"X-AIC-User": "mate@example.com"},
            json={"text": "astronauts"},
        )
        self.assertEqual(response.status_code, 403)

        async def inline(function, *args):
            return function(*args)

        with patch("backend.app.compare", return_value={"results": []}) as compare, patch(
            "backend.app.run_in_threadpool", side_effect=inline
        ):
            response = await self.client.post(
                "/compare",
                headers={"X-AIC-User": "owner@example.com"},
                json={"text": "astronauts"},
            )
        self.assertEqual(response.status_code, 200)
        compare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
