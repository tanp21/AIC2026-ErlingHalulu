import unittest
from unittest.mock import patch

import httpx

import app as server
from access_control import AccessConfig, LatestWinsCoordinator


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

    async def test_session_reports_tailscale_identity_and_role(self):
        response = await self.client.get(
            "/session", headers={"Tailscale-User-Login": "mate@example.com"}
        )
        self.assertEqual(response.json(), {"login": "mate@example.com", "is_admin": False})

    async def test_compare_is_owner_only(self):
        response = await self.client.post(
            "/compare",
            headers={"Tailscale-User-Login": "mate@example.com"},
            json={"text": "astronauts"},
        )
        self.assertEqual(response.status_code, 403)

        async def inline(function, *args):
            return function(*args)

        with patch("app.compare", return_value={"results": []}) as compare, patch(
            "app.run_in_threadpool", side_effect=inline
        ):
            response = await self.client.post(
                "/compare",
                headers={"Tailscale-User-Login": "owner@example.com"},
                json={"text": "astronauts"},
            )
        self.assertEqual(response.status_code, 200)
        compare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
