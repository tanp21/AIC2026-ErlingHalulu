import io
import tempfile
import unittest
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI

from backend.access_control import AccessConfig, install_access_control
from backend.collaboration import CollaborationStore, install_collaboration_routes


def payload() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("query-1-kis.txt", "Tìm cảnh chiếc xe màu đỏ")
    return buffer.getvalue()


class CollaborationApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        app = FastAPI()
        app.state.collaboration_store = CollaborationStore(Path(self.temp_dir.name) / "db.sqlite3")
        config = AccessConfig(
            frozenset({"owner@example.com", "mate@example.com"}), "owner@example.com"
        )
        install_access_control(app, config)
        install_collaboration_routes(app)
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")
        self.owner = {"X-AIC-User": "owner@example.com"}
        self.mate = {"X-AIC-User": "mate@example.com"}

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_owner_uploads_round_and_teammate_cannot(self):
        denied = await self.client.post(
            "/api/collab/rounds?name=Round%201", content=payload(), headers=self.mate
        )
        self.assertEqual(denied.status_code, 403)
        created = await self.client.post(
            "/api/collab/rounds?name=Round%201",
            content=payload(),
            headers={**self.owner, "Content-Type": "application/zip"},
        )
        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["name"], "Round 1")

    async def test_snapshot_etag_patch_review_and_export(self):
        created = await self.client.post(
            "/api/collab/rounds?name=Round%201", content=payload(), headers=self.owner
        )
        round_id = created.json()["id"]
        snapshot = await self.client.get(f"/api/collab/rounds/{round_id}", headers=self.mate)
        self.assertEqual(snapshot.status_code, 200)
        unchanged = await self.client.get(
            f"/api/collab/rounds/{round_id}",
            headers={**self.mate, "If-None-Match": snapshot.headers["etag"]},
        )
        self.assertEqual(unchanged.status_code, 304)

        query = snapshot.json()["queries"][0]
        patched = await self.client.patch(
            f"/api/collab/queries/{query['id']}", headers=self.mate,
            json={"expected_version": 0, "assignee": "mate@example.com",
                  "answers": [{"video_id": "L00_V001", "frame_id": 12}]},
        )
        self.assertEqual(patched.status_code, 200)
        reviewed = await self.client.put(
            f"/api/collab/queries/{query['id']}/review", headers=self.owner,
            json={"expected_version": patched.json()["version"], "reviewed": True},
        )
        self.assertEqual(reviewed.json()["status"], "reviewed")

        preview = await self.client.get(
            f"/api/collab/rounds/{round_id}/export-preview", headers=self.mate
        )
        self.assertEqual(preview.json(), {"missing": [], "invalid": [], "unreviewed": []})
        exported = await self.client.get(f"/api/collab/rounds/{round_id}/export", headers=self.mate)
        self.assertEqual(exported.status_code, 200)
        self.assertEqual(exported.headers["content-type"], "application/zip")
        self.assertIn("Round1.zip", exported.headers["content-disposition"])

    async def test_stale_patch_returns_current_query(self):
        created = await self.client.post(
            "/api/collab/rounds?name=Round%201", content=payload(), headers=self.owner
        )
        query = created.json()["queries"][0]
        await self.client.patch(
            f"/api/collab/queries/{query['id']}", headers=self.owner,
            json={"expected_version": 0, "assignee": "owner@example.com"},
        )
        stale = await self.client.patch(
            f"/api/collab/queries/{query['id']}", headers=self.mate,
            json={"expected_version": 0, "assignee": "mate@example.com"},
        )
        self.assertEqual(stale.status_code, 409)
        self.assertEqual(stale.json()["current"]["assignee"], "owner@example.com")

    async def test_only_owner_sets_a_trake_event_count(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("query-1-trake.txt", "Tìm các cảnh theo thứ tự.")
        created = await self.client.post(
            "/api/collab/rounds?name=Round%201", content=buffer.getvalue(), headers=self.owner
        )
        query = created.json()["queries"][0]
        self.assertIsNone(query["event_count"])
        denied = await self.client.patch(
            f"/api/collab/queries/{query['id']}", headers=self.mate,
            json={"expected_version": 0, "event_count": 3},
        )
        self.assertEqual(denied.status_code, 403)
        updated = await self.client.patch(
            f"/api/collab/queries/{query['id']}", headers=self.owner,
            json={"expected_version": 0, "event_count": 3},
        )
        self.assertEqual(updated.json()["event_count"], 3)


if __name__ == "__main__":
    unittest.main()
