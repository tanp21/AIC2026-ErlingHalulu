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

    def test_active_reports_only_the_owners_live_run(self):
        store = self.store()
        self.create(store, run_id="a")

        self.assertEqual(store.active(OWNER).run_id, "a")
        self.assertIsNone(store.active(TEAMMATE))

        store.finish("a", result=fake_result(), usage=None, error=None)
        self.assertIsNone(store.active(OWNER))

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
            # the run's own review channel and live view reach the agent
            self.assertIs(query.call_args.kwargs["review"], run.review)
            query.call_args.kwargs["progress"]({"stage": "x"})
            self.assertEqual(store.get(OWNER, run.run_id).progress, {"stage": "x"})

            finished = store.get(OWNER, run.run_id)
            self.assertEqual(finished.status, "done")
            self.assertEqual(finished.usage, {"calls": 1})
            self.assertTrue((run.output_dir / "submission.zip").exists())
            self.assertTrue((run.output_dir / f"{run.run_id}.json").exists())

    def test_a_human_judged_run_needs_no_gemini_key_and_builds_no_vision_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, _ = self.run_in(directory)
            req = AgentRunReq(text="a red boat", kind="kis", judge="human", deepseek_api_key="dsk-test")
            with (
                patch("backend.agent_runner.run_query",
                      return_value=(fake_result(run.run_id), Mock(as_dict=lambda: {}))) as query,
                patch.dict(os.environ, {"AIC_AGENT_OUT_DIR": directory, "AIC_AGENT_VISION": "gemini",
                                        "AIC_AGENT_HUMAN_TIMEOUT": "120"}),
            ):
                execute_run(store, run, req, INT_URL)

            self.assertEqual(store.get(OWNER, run.run_id).status, "done")
            self.assertEqual(query.call_args.kwargs["judge"], "human")
            self.assertEqual(query.call_args.kwargs["human_idle_timeout"], 120.0)
            self.assertEqual(query.call_args.args[1].gemini_api_key, "")

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


class DeploymentKeyTests(unittest.TestCase):
    """The keys this deployment holds are the owner's; only the owner's runs spend them.

    They live in the app's environment (compose `env_file`), so the question is not
    just whether a key exists but whether *this* run may use it -- which is why the
    answer is decided once, from the request identity, and carried on the run.
    """

    DEPLOYMENT = {"DEEPSEEK_API_KEY": "dsk-deployment", "GEMINI_API_KEY": "gmk-deployment"}

    def settings_for(self, directory, deployment_keys, **overrides):
        from agent.tasks import task_from_text

        store = AgentRunStore()
        run = store.create("run1", OWNER, task_from_text("run1", "a red boat", "kis"),
                           Path(directory) / "run1", "http://test",
                           deployment_keys=deployment_keys)
        req = AgentRunReq(text="a red boat", kind="kis", **overrides)
        with (
            patch("backend.agent_runner.run_query",
                  return_value=(fake_result(run.run_id), Mock(as_dict=lambda: {}))) as query,
            patch.dict(os.environ, {**self.DEPLOYMENT, "AIC_AGENT_OUT_DIR": directory,
                                    "AIC_AGENT_VISION": "gemini"}),
        ):
            execute_run(store, run, req, INT_URL)
        self.query = query
        return store, run, query

    def test_an_owner_run_with_empty_fields_uses_the_deployment_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, query = self.settings_for(directory, deployment_keys=True)

        settings = query.call_args.args[1]
        self.assertEqual(settings.deepseek_api_key, "dsk-deployment")
        self.assertEqual(settings.gemini_api_key, "gmk-deployment")

    def test_a_key_typed_into_the_panel_still_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, query = self.settings_for(directory, deployment_keys=True,
                                                  deepseek_api_key="dsk-typed")

        settings = query.call_args.args[1]
        self.assertEqual(settings.deepseek_api_key, "dsk-typed")
        self.assertEqual(settings.gemini_api_key, "gmk-deployment")

    def test_the_teams_qwen_key_judges_every_run_as_moments(self):
        # The shared key was handed round as QWEN_API_KEY; with no backend named, having it is
        # what makes a run judge with Qwen on OpenRouter -- anyone's run, not only the owner's.
        from agent.tasks import task_from_text

        store = AgentRunStore()
        with tempfile.TemporaryDirectory() as directory:
            run = store.create("run1", TEAMMATE, task_from_text("run1", "a red boat", "kis"),
                               Path(directory) / "run1", "http://test", deployment_keys=False)
            with (
                patch("backend.agent_runner.run_query",
                      return_value=(fake_result(run.run_id), Mock(as_dict=lambda: {}))) as query,
                patch.dict(os.environ, {"QWEN_API_KEY": "sk-or-team", "AIC_AGENT_VISION": "",
                                        "OPENROUTER_API_KEY": "", "AIC_AGENT_OUT_DIR": directory}),
            ):
                execute_run(store, run, AgentRunReq(text="a red boat", deepseek_api_key="dsk-theirs"), INT_URL)

        settings = query.call_args.args[1]
        self.assertEqual((settings.vision, settings.judge_style, settings.openrouter_api_key),
                         ("openrouter", "moment", "sk-or-team"))
        self.assertEqual(settings.gemini_api_key, "")  # Gemini is only the web search now

    def test_a_run_that_is_not_the_owners_never_reaches_them(self):
        with tempfile.TemporaryDirectory() as directory:
            store, run, query = self.settings_for(directory, deployment_keys=False)

        query.assert_not_called()
        # It failed for want of a key rather than quietly borrowing the deployment's.
        self.assertIn("DEEPSEEK_API_KEY", store.get(OWNER, run.run_id).error)


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
        self.env = patch.dict(os.environ, {
            "AIC_AGENT_OUT_DIR": self.directory.name,
            # The deployment's own model keys live in the process environment, and so
            # does the vision backend. Blank them so a key exported in a developer's
            # shell cannot decide what these assertions mean.
            "DEEPSEEK_API_KEY": "", "GEMINI_API_KEY": "", "AIC_AGENT_VISION": "gemini",
        })
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

    async def test_the_owner_may_leave_both_fields_empty(self):
        # What the deployment's own keys are for: the owner runs without pasting.
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-deployment",
                                     "GEMINI_API_KEY": "gmk-deployment"}):
            response = await self.start(deepseek_api_key="", gemini_api_key="")
        self.assertEqual(response.status_code, 202)
        self.assertTrue(self.runs.get(OWNER, response.json()["run_id"]).deployment_keys)

    async def test_a_teammate_may_not_ride_on_the_deployments_keys(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-deployment",
                                     "GEMINI_API_KEY": "gmk-deployment"}):
            blank = await self.start(headers={"X-AIC-User": TEAMMATE},
                                     deepseek_api_key="", gemini_api_key="")

        self.assertEqual(blank.status_code, 422)
        self.assertIn("DeepSeek", blank.json()["detail"])

    async def test_a_teammate_run_with_their_own_keys_is_not_marked_for_the_deployments(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "dsk-deployment",
                                     "GEMINI_API_KEY": "gmk-deployment"}):
            theirs = await self.start(headers={"X-AIC-User": TEAMMATE})

        self.assertEqual(theirs.status_code, 202)
        self.assertFalse(self.runs.get(TEAMMATE, theirs.json()["run_id"]).deployment_keys)

    async def test_requires_an_identity(self):
        response = await self.client.post("/api/agent/runs", json={"text": "x", **KEYS})
        self.assertEqual(response.status_code, 401)

    async def test_one_active_run_per_user(self):
        await self.start()

        again = await self.start()

        self.assertEqual(again.status_code, 409)
        self.assertEqual(self.executor.call_count, 1)

    async def test_the_active_route_hands_back_the_callers_live_run(self):
        # What a reloaded page adopts: without it the only way back to a run in progress
        # is the 409 above, with nothing to look at or stop.
        empty = await self.client.get("/api/agent/runs/active", headers=self.headers)
        self.assertIsNone(empty.json()["run"])

        started = (await self.start()).json()
        mine = await self.client.get("/api/agent/runs/active", headers=self.headers)

        self.assertEqual(mine.json()["run"]["run_id"], started["run_id"])
        theirs = await self.client.get(
            "/api/agent/runs/active", headers={"X-AIC-User": TEAMMATE}
        )
        self.assertIsNone(theirs.json()["run"])
        self.assertEqual(
            (await self.client.get("/api/agent/runs/active")).status_code, 401
        )

    async def test_other_users_cannot_see_or_download_a_run(self):
        run_id = (await self.start()).json()["run_id"]

        polled = await self.client.get(
            f"/api/agent/runs/{run_id}", headers={"X-AIC-User": TEAMMATE}
        )
        downloaded = await self.client.get(
            f"/api/agent/runs/{run_id}/submission.zip", headers={"X-AIC-User": TEAMMATE}
        )
        self.assertEqual((polled.status_code, downloaded.status_code), (404, 404))

    async def test_the_owner_reviews_a_running_query(self):
        run_id = (await self.start()).json()["run_id"]
        url = f"/api/agent/runs/{run_id}/review"

        accepted = await self.client.post(url, headers=self.headers,
                                          json={"candidate_id": "C3", "decision": "accept"})
        typed = await self.client.post(url, headers=self.headers, json={"answer": "  2.15 "})
        review = self.runs.get(OWNER, run_id).review

        self.assertEqual(accepted.status_code, 200)
        self.assertEqual(typed.json()["review"],
                         {"decisions": {"C3": "accept"}, "answer": "2.15", "stopped": False,
                          "decided": True, "judge": "model", "scores": {}, "sequence": None,
                          "finished": False, "pending_commands": 0})
        self.assertTrue(review.event.is_set())  # the agent stops waiting on Gemini

    async def test_review_is_validated_private_and_only_while_running(self):
        run_id = (await self.start()).json()["run_id"]
        url = f"/api/agent/runs/{run_id}/review"

        no_candidate = await self.client.post(url, headers=self.headers, json={"decision": "accept"})
        bad_id = await self.client.post(url, headers=self.headers,
                                        json={"candidate_id": "../x", "decision": "accept"})
        foreign = await self.client.post(url, headers={"X-AIC-User": TEAMMATE}, json={"stop": True})
        self.runs.finish(run_id, result=fake_result(run_id), usage=None, error=None)
        late = await self.client.post(url, headers=self.headers, json={"stop": True})

        self.assertEqual((no_candidate.status_code, bad_id.status_code), (422, 422))
        self.assertEqual((foreign.status_code, late.status_code), (404, 409))
        self.assertFalse(self.runs.get(OWNER, run_id).review.stopped)

    async def test_a_qwen_judged_run_needs_no_gemini_key(self):
        with patch.dict(os.environ, {"QWEN_API_KEY": "sk-or-team", "AIC_AGENT_VISION": ""}):
            response = await self.start(gemini_api_key="", headers={"X-AIC-User": TEAMMATE})
        self.assertEqual(response.status_code, 202)

    async def test_a_human_judged_run_needs_no_gemini_key(self):
        blank = await self.start(gemini_api_key="")
        human = await self.start(gemini_api_key="", judge="human", headers={"X-AIC-User": TEAMMATE})

        self.assertEqual(blank.status_code, 422)  # a model judge still wants one
        self.assertEqual(human.status_code, 202)
        run = self.runs.get(TEAMMATE, human.json()["run_id"])
        self.assertEqual(run.review.judge, "human")

    async def test_a_human_judged_run_takes_scores_commands_a_sequence_and_finish(self):
        run_id = (await self.start(judge="human")).json()["run_id"]
        url = f"/api/agent/runs/{run_id}/review"
        review = self.runs.get(OWNER, run_id).review

        scored = await self.client.post(url, headers=self.headers, json={"candidate_id": "C2", "score": 5})
        more = await self.client.post(url, headers=self.headers, json={"command": {"type": "more", "hint": "đỏ"}})
        exact = await self.client.post(url, headers=self.headers, json={"command": {
            "type": "add_frame", "video_id": "L21_V001", "frame_idx": 1234, "time": 49.4, "accept": True}})
        path = await self.client.post(url, headers=self.headers, json={"sequence": {
            "video_id": "L21_V001", "frames": [10, 20], "times": [0.4, 0.8]}})

        self.assertEqual([r.status_code for r in (scored, more, exact, path)], [200, 200, 200, 200])
        self.assertEqual(review.verdicts(), {"C2": 5.0})
        self.assertEqual([c["type"] for c in review.take_commands()], ["more", "add_frame"])
        self.assertEqual(review.sequence, {"video_id": "L21_V001", "frames": [10, 20], "times": [0.4, 0.8]})
        self.assertFalse(review.decided())
        finished = await self.client.post(url, headers=self.headers, json={"finish": True})
        self.assertTrue(finished.json()["review"]["finished"] and review.event.is_set())

    async def test_the_reviewers_order_arrives_before_the_finish_that_ends_the_wait(self):
        run_id = (await self.start(judge="human")).json()["run_id"]
        review = self.runs.get(OWNER, run_id).review
        seen = []
        original = review.finish
        review.finish = lambda: (seen.append(review.order), original())[1]

        response = await self.client.post(f"/api/agent/runs/{run_id}/review", headers=self.headers,
                                          json={"finish": True, "order": ["C7", "C3"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(seen, [["C7", "C3"]])     # the run wakes up to an order that is already there

    async def test_an_order_is_for_a_human_judged_run_and_holds_candidate_ids(self):
        model = (await self.start()).json()["run_id"]
        refused = await self.client.post(f"/api/agent/runs/{model}/review", headers=self.headers,
                                         json={"order": ["C1"]})
        self.assertEqual(refused.status_code, 422)
        self.runs.finish(model, result=fake_result(model), usage=None, error=None)

        human = (await self.start(judge="human")).json()["run_id"]
        url = f"/api/agent/runs/{human}/review"
        for body in ({"order": ["../x"]}, {"order": ["C1", 5]}, {"order": [f"C{n}" for n in range(301)]}):
            self.assertEqual((await self.client.post(url, headers=self.headers, json=body)).status_code, 422, body)
        self.assertEqual(self.runs.get(OWNER, human).review.order, [])

    async def test_human_only_actions_are_refused_on_a_model_judged_run(self):
        run_id = (await self.start()).json()["run_id"]
        url = f"/api/agent/runs/{run_id}/review"
        for body in ({"candidate_id": "C1", "score": 5}, {"finish": True},
                     {"command": {"type": "more"}},
                     {"sequence": {"video_id": "L21_V001", "frames": [1]}}):
            response = await self.client.post(url, headers=self.headers, json=body)
            self.assertEqual(response.status_code, 422, body)

    async def test_commands_and_sequences_are_validated(self):
        run_id = (await self.start(judge="human")).json()["run_id"]
        url = f"/api/agent/runs/{run_id}/review"
        for body in (
            {"command": {"type": "scope"}},                                   # needs a video
            {"command": {"type": "scope", "video_id": "../etc"}},              # not a video id
            {"command": {"type": "neighbours", "cid": "x"}},
            {"command": {"type": "add_frame", "video_id": "L21_V001", "frame_idx": -1}},
            {"command": {"type": "more", "hint": "x" * 301}},
            {"command": {"type": "reboot"}},
            {"candidate_id": "C1", "score": 11},
            {"score": 5},                                                      # no candidate
            {"sequence": {"video_id": "L21_V001", "frames": []}},
            {"sequence": {"video_id": "L21_V001", "frames": [1, 2], "times": [1.0]}},
            {"sequence": {"video_id": "L21_V001", "frames": [-3]}},
        ):
            response = await self.client.post(url, headers=self.headers, json=body)
            self.assertEqual(response.status_code, 422, body)
        self.assertEqual(self.runs.get(OWNER, run_id).review.take_commands(), [])

    async def test_polls_carry_the_live_view(self):
        run_id = (await self.start()).json()["run_id"]
        self.runs.update_progress(run_id, {"stage": "judging", "candidates": []})

        polled = (await self.client.get(f"/api/agent/runs/{run_id}", headers=self.headers)).json()

        self.assertEqual(polled["progress"]["stage"], "judging")
        self.assertFalse(polled["review"]["decided"])

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
