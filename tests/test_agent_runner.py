import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

from backend import agent_runner
from backend import app as server
from backend.access_control import AccessConfig, LatestWinsCoordinator
from backend.agent_runner import AgentRunReq, AgentRunStore, execute_run

OWNER = "one@example.com"
TEAMMATE = "two@example.com"
KEYS = {"deepseek_api_key": "dsk-test", "gemini_api_key": "gmk-test"}
INT_URL = "http://127.0.0.1:8000"


def fake_result(query_id="q1", kind="kis"):
    return {
        "query_id": query_id, "kind": kind, "query": "a red boat", "answer": None,
        "plan": None, "lookup": [], "lookups": [], "notes": "", "elapsed": 3.0,
        "error": None, "trace": [], "sequences": [],
        "ranking": [{"id": "c0", "video_id": "L01_V001", "frame_idx": 10, "time": 0.4,
                     "fused": 0.9, "verdict": None, "evidence": []}],
    }


class AgentRunStoreTests(unittest.TestCase):
    def store(self, clock=None):
        return AgentRunStore(ttl_seconds=60, clock=clock or (lambda: 0.0))

    def create(self, store, owner=OWNER, run_id="run1"):
        from agent.tasks import task_from_text
        return store.create(run_id, owner, task_from_text(run_id, "a red boat", "kis"),
                            Path("/tmp/out") / run_id, "http://test")

    def test_rejects_a_second_active_run_for_the_same_user(self):
        store = self.store()
        self.create(store, run_id="a")

        with self.assertRaises(agent_runner.AgentRunConflict):
            self.create(store, run_id="b")

        store.finish("a", result=fake_result(), usage=None, error=None)
        self.create(store, run_id="c")  # finished runs do not block

    def test_keeps_runs_private_to_their_owner(self):
        store = self.store()
        self.create(store, run_id="a")

        with self.assertRaises(LookupError):
            store.get(TEAMMATE, "a")

    def test_expires_runs_after_the_ttl(self):
        now = [0.0]
        store = self.store(clock=lambda: now[0])
        self.create(store, run_id="a")
        now[0] = 61.0

        with self.assertRaises(LookupError):
            store.get(OWNER, "a")

    def test_finish_records_result_usage_and_error(self):
        store = self.store()
        self.create(store, run_id="a")

        store.finish("a", result=fake_result(), usage={"deepseek_calls": 2}, error=None)

        run = store.get(OWNER, "a")
        self.assertEqual((run.status, run.usage, run.result["query_id"]),
                         ("done", {"deepseek_calls": 2}, "q1"))

    def test_an_error_without_a_result_marks_the_run_failed(self):
        store = self.store()
        self.create(store, run_id="a")

        store.finish("a", result=None, usage=None, error="RuntimeError: boom")

        self.assertEqual(store.get(OWNER, "a").status, "error")


class ExecuteRunTests(unittest.TestCase):
    def run_in(self, directory):
        store = AgentRunStore()
        from agent.tasks import task_from_text
        run = store.create("run1", OWNER, task_from_text("run1", "a red boat", "kis"),
                           Path(directory) / "run1", "http://test")
        req = AgentRunReq(text="a red boat", kind="kis", **KEYS)
        return store, run, req

    def test_runs_with_the_request_keys_and_writes_the_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, req = self.run_in(directory)
            with (
                patch("backend.agent_runner.run_query",
                      return_value=(fake_result(run.run_id), Mock(as_dict=lambda: {"calls": 1}))) as query,
                patch.dict(os.environ, {"AIC_AGENT_OUT_DIR": directory}),
            ):
                execute_run(store, run, req, INT_URL)

            settings = query.call_args.args[1]
            self.assertEqual(settings.deepseek_api_key, KEYS["deepseek_api_key"])
            self.assertEqual(settings.gemini_api_key, KEYS["gemini_api_key"])
            self.assertEqual(settings.server_url, INT_URL)
            self.assertEqual(settings.trusted_user, OWNER)
            self.assertEqual(settings.search_api, "batch")
            self.assertEqual(query.call_args.args[2:], ("fast", 6, None))

            finished = store.get(OWNER, run.run_id)
            self.assertEqual(finished.status, "done")
            self.assertEqual(finished.usage, {"calls": 1})
            self.assertTrue((run.output_dir / "submission.zip").exists())
            self.assertTrue((run.output_dir / f"{run.run_id}.json").exists())

    def test_forwards_the_mode_and_explicit_time_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, _ = self.run_in(directory)
            req = AgentRunReq(text="a red boat", kind="qa", mode="deep", time_budget=120.0, **KEYS)
            with (
                patch("backend.agent_runner.run_query",
                      return_value=(fake_result(run.run_id), Mock(as_dict=lambda: {}))) as query,
                patch.dict(os.environ, {"AIC_AGENT_OUT_DIR": directory}),
            ):
                execute_run(store, run, req, INT_URL)

            self.assertEqual(query.call_args.args[2:], ("deep", 6, 120.0))

    def test_reports_a_failing_run_instead_of_leaking_it(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, req = self.run_in(directory)
            with patch("backend.agent_runner.run_query", side_effect=RuntimeError("llm down")):
                execute_run(store, run, req, INT_URL)

            failed = store.get(OWNER, run.run_id)
            self.assertEqual(failed.status, "error")
            self.assertIn("llm down", failed.error)


class AgentRunEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(
            frozenset({OWNER, TEAMMATE}), OWNER
        )
        server.app.state.search_coordinator = LatestWinsCoordinator()
        server.app.state.agent_runs = AgentRunStore()
        self.runs = server.app.state.agent_runs
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.env = patch.dict(os.environ, {"AIC_AGENT_OUT_DIR": self.directory.name})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.submit = patch.object(agent_runner._AGENT_RUN_EXECUTOR, "submit")
        self.executor = self.submit.start()
        self.addCleanup(self.submit.stop)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )
        self.headers = {"X-AIC-User": OWNER}

    async def asyncTearDown(self):
        await self.client.aclose()

    async def start(self, headers=None, **overrides):
        payload = {"text": "a red boat", "kind": "kis", **KEYS, **overrides}
        return await self.client.post("/api/agent/runs", headers=headers or self.headers, json=payload)

    async def test_starts_a_run_and_polls_its_status(self):
        response = await self.start()

        self.assertEqual(response.status_code, 202)
        started = response.json()
        self.assertEqual((started["status"], started["kind"]), ("running", "kis"))
        self.executor.assert_called_once()

        polled = await self.client.get(f"/api/agent/runs/{started['run_id']}", headers=self.headers)
        self.assertEqual(polled.json()["status"], "running")

    async def test_rejects_blank_keys_and_unknown_kinds(self):
        self.assertEqual((await self.start(deepseek_api_key=" ")).status_code, 422)
        self.assertEqual((await self.start(kind="caption")).status_code, 422)

    async def test_requires_an_identity(self):
        response = await self.client.post("/api/agent/runs", json={"text": "x", **KEYS})
        self.assertEqual(response.status_code, 401)

    async def test_one_active_run_per_user(self):
        await self.start()

        again = await self.start()

        self.assertEqual(again.status_code, 409)
        self.assertEqual(self.executor.call_count, 1)

    async def test_other_users_cannot_see_or_download_a_run(self):
        run_id = (await self.start()).json()["run_id"]

        polled = await self.client.get(
            f"/api/agent/runs/{run_id}", headers={"X-AIC-User": TEAMMATE}
        )
        downloaded = await self.client.get(
            f"/api/agent/runs/{run_id}/submission.zip", headers={"X-AIC-User": TEAMMATE}
        )
        self.assertEqual((polled.status_code, downloaded.status_code), (404, 404))

    async def test_downloads_the_submission_only_after_the_run_finishes(self):
        run_id = (await self.start()).json()["run_id"]
        run = self.runs.get(OWNER, run_id)
        run.output_dir.mkdir(parents=True, exist_ok=True)
        (run.output_dir / "submission.zip").write_bytes(b"zip-bytes")

        early = await self.client.get(
            f"/api/agent/runs/{run_id}/submission.zip", headers=self.headers
        )
        self.runs.finish(run_id, result=fake_result(run_id), usage=None, error=None)
        done = await self.client.get(
            f"/api/agent/runs/{run_id}/submission.zip", headers=self.headers
        )

        self.assertEqual(early.status_code, 409)
        self.assertEqual(done.status_code, 200)
        self.assertEqual(done.content, b"zip-bytes")


if __name__ == "__main__":
    unittest.main()
