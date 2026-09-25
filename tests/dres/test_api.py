import json
import logging
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import numpy as np
from fastapi import FastAPI

from backend.dres import (
    Candidate, ContextPatch, DresClient, DresConfig, DresError,
    DresService, DresSubmissionStore, SubmitRequest, install_dres_routes,
)
from backend.frame_maps import FrameMap


class MockDres:
    def __init__(self):
        self.requests = []
        self.evaluations = [{"id": "eval-1", "name": "Final", "status": "ACTIVE", "type": "SYNCHRONOUS"}]
        self.eval_response = None
        self.submit_response = httpx.Response(200, json={"status": "OK"})
        self.submit_exception = None
        self.login_response = httpx.Response(200, json={
            "id": "team", "username": "team", "role": "PARTICIPANT", "sessionId": "session-secret",
        })

    def __call__(self, request):
        self.requests.append(request)
        if request.url.path == "/api/v2/login":
            return self.login_response
        if request.url.path == "/api/v2/client/evaluation/list":
            return self.eval_response or httpx.Response(200, json=self.evaluations)
        if request.url.path.startswith("/api/v2/submit/"):
            if self.submit_exception:
                raise self.submit_exception
            return self.submit_response
        return httpx.Response(404)

    def submits(self):
        return [request for request in self.requests if request.url.path.startswith("/api/v2/submit/")]


class DresHttpTests(unittest.TestCase):
    def setUp(self):
        self.fake = MockDres()
        self.config = DresConfig("http://localhost", "team", "password")
        self.client = DresClient(self.config, httpx.Client(
            transport=httpx.MockTransport(self.fake), base_url="http://localhost",
            follow_redirects=False,
        ))

    def tearDown(self):
        self.client.close()

    def test_login_and_evaluation_contract(self):
        session = self.client.login()
        self.assertEqual(session, "session-secret")
        self.assertEqual(json.loads(self.fake.requests[0].content), {"username": "team", "password": "password"})
        self.assertEqual(self.client.evaluations(session), [{"id": "eval-1", "name": "Final"}])
        self.assertEqual(self.fake.requests[1].url.params["session"], session)
        self.assertNotIn("authorization", self.fake.requests[1].headers)

    def test_bad_credentials_and_malformed_login(self):
        self.fake.login_response = httpx.Response(401, json={"message": "bad credentials"})
        with self.assertRaisesRegex(DresError, "HTTP 401"):
            self.client.login()
        self.fake.login_response = httpx.Response(200, text="not json")
        with self.assertRaisesRegex(DresError, "malformed JSON"):
            self.client.login()
        self.fake.login_response = httpx.Response(200, json={"role": "USER", "sessionId": "x"})
        with self.assertRaisesRegex(DresError, "participant sessionId"):
            self.client.login()

    def test_filter_active_synchronous_and_validate_schema(self):
        self.fake.evaluations = [
            {"id": "old", "status": "INACTIVE", "type": "SYNCHRONOUS"},
            {"id": "async", "status": "ACTIVE", "type": "ASYNCHRONOUS"},
            {"id": "right", "status": "ACTIVE", "type": "SYNCHRONOUS"},
        ]
        self.assertEqual(self.client.evaluations("session-secret"), [{"id": "right", "name": ""}])
        self.fake.evaluations = [{"status": "ACTIVE", "type": "SYNCHRONOUS"}]
        with self.assertRaisesRegex(DresError, "no ID"):
            self.client.evaluations("session-secret")
        self.fake.eval_response = httpx.Response(200, text="not json")
        with self.assertRaisesRegex(DresError, "malformed JSON"):
            self.client.evaluations("session-secret")

    def test_base_url_and_session_log_safety(self):
        for bad in ("http://eventretrieval.one", "https://user:password@host", "https://host/api"):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                DresConfig(bad)
        self.assertGreaterEqual(logging.getLogger("httpx").level, logging.WARNING)
        self.assertGreaterEqual(logging.getLogger("httpcore").level, logging.WARNING)


class DresServiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.video = root / "data" / "2026" / "videos" / "L01" / "L01_V001.MP4"
        self.video.parent.mkdir(parents=True)
        self.video.write_bytes(b"fake original video")
        keyframe_dir = root / "data" / "2026" / "map-keyframes"
        keyframe_dir.mkdir(parents=True)
        (keyframe_dir / "L01_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.100,30,1\n2,123.456,30,2\n3,3.000,30,3\n",
            encoding="utf-8",
        )
        map_dir = root / "frame_maps"
        map_dir.mkdir()
        stat = self.video.stat()
        (map_dir / "manifest.json").write_text(json.dumps({"version": 2, "videos": {
            "L01_V001": {"path": str(self.video), "size": stat.st_size, "mtimeNs": stat.st_mtime_ns},
        }}))
        self.mapping = FrameMap(np.array([90000, 210000, 213456, 216000]), 1, 1000)
        self.fake = MockDres()
        self.config = DresConfig("http://localhost", "team", "password")
        client = DresClient(self.config, httpx.Client(
            transport=httpx.MockTransport(self.fake), base_url="http://localhost",
        ))
        self.store = DresSubmissionStore(root / "ledger.sqlite3")
        self.service = DresService(self.config, self.store, lambda _id: {"path": str(self.video)},
                                   load_map=lambda _id: self.mapping, index_dir=root, client=client)
        self.addCleanup(self.service.close)
        self.service.connect()
        self.service.set_context(ContextPatch(expected_revision=0, kind="kis", label="Hall task 1"), "member")

    def request(self, prepared, *, retry=False):
        return SubmitRequest(candidate=Candidate(video_id="L01_V001", frame_idx=2),
                             fingerprint=prepared["fingerprint"], expected_evaluation_id=prepared["evaluation_id"],
                             expected_context_revision=prepared["context_revision"], retry_uncertain=retry)

    def test_kis_submission_exact_payload_and_duplicate_guard(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.assertEqual(prepared["timestamp_ms"], 123456)
        evaluation_gets = sum(request.url.path == "/api/v2/client/evaluation/list" for request in self.fake.requests)
        with self.assertLogs("uvicorn.error", level="INFO") as logs:
            attempt = self.service.submit(self.request(prepared), "member")
        self.assertEqual(sum(request.url.path == "/api/v2/client/evaluation/list" for request in self.fake.requests), evaluation_gets + 1)
        self.assertEqual(attempt["status"], "accepted")
        self.assertEqual(len(self.fake.submits()), 1)
        request = self.fake.submits()[0]
        self.assertEqual(request.url.path, "/api/v2/submit/eval-1")
        self.assertEqual(request.url.params["session"], "session-secret")
        self.assertEqual(json.loads(request.content), {
            "answerSets": [{"answers": [{"mediaItemName": "L01_V001", "start": "123456", "end": "123456"}]}],
        })
        with self.assertRaisesRegex(DresError, "already attempted"):
            self.service.submit(self.request(prepared), "member")
        self.assertEqual(len(self.fake.submits()), 1)
        self.service.set_context(ContextPatch(expected_revision=1, kind="kis", label="Next hall task"), "member")
        same_payload_new_label = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        with self.assertRaisesRegex(DresError, "already attempted"):
            self.service.submit(self.request(same_payload_new_label), "member")
        self.assertEqual(len(self.fake.submits()), 1)
        self.assertNotIn("session-secret", json.dumps(self.store.history()))
        self.assertNotIn("password", json.dumps(self.store.history()))
        self.assertIn("time_ms=123456", logs.output[0])
        self.assertNotIn("session-secret", logs.output[0])
        self.assertNotIn("password", logs.output[0])

    def test_evaluation_is_selected_from_unique_task_kind_name(self):
        self.fake.evaluations = []
        self.service.refresh_evaluations()
        self.assertIsNone(self.service.evaluation_id)
        with self.assertRaisesRegex(DresError, "active evaluation"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.fake.evaluations = [
            {"id": "QA-eval", "name": "AIC2026 - QA Test 1", "status": "ACTIVE", "type": "SYNCHRONOUS"},
            {"id": "KIS-eval", "name": "Textual KIS Test 1", "status": "ACTIVE", "type": "SYNCHRONOUS"},
        ]
        self.service.refresh_evaluations()
        self.assertEqual(self.service.evaluation_id, "KIS-eval")
        self.assertFalse(self.service.status()["context_valid"])
        with self.assertRaisesRegex(DresError, "current task for this evaluation"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.service.set_context(ContextPatch(expected_revision=1, kind="kis", label="KIS task"), "member")
        self.assertEqual(self.service.evaluation_id, "KIS-eval")
        self.assertTrue(self.service.status()["context_valid"])
        self.service.set_context(ContextPatch(expected_revision=2, kind="qa", label="QA task"), "member")
        self.assertEqual(self.service.evaluation_id, "QA-eval")
        self.assertTrue(self.service.status()["context_valid"])

    def test_ambiguous_evaluation_name_match_fails_closed(self):
        self.fake.evaluations = [
            {"id": "KIS-A", "name": "KIS final 1", "status": "ACTIVE", "type": "SYNCHRONOUS"},
            {"id": "KIS-B", "name": "KIS final 2", "status": "ACTIVE", "type": "SYNCHRONOUS"},
        ]
        self.service.refresh_evaluations()
        with self.assertRaisesRegex(DresError, "Could not uniquely match task type KIS"):
            self.service.set_context(ContextPatch(expected_revision=1, kind="kis", label="KIS task"), "member")
        self.assertIsNone(self.service.evaluation_id)
        selected = self.service.select_evaluation("KIS-B")
        self.assertTrue(selected["evaluation_manual"])
        self.service.set_context(ContextPatch(expected_revision=1, kind="kis", label="KIS task"), "member")
        self.assertTrue(self.service.status()["context_valid"])
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        attempt = self.service.submit(self.request(prepared), "member")
        self.assertEqual(attempt["status"], "accepted")
        self.assertEqual(self.fake.submits()[0].url.path, "/api/v2/submit/KIS-B")

    def test_submit_uses_fresh_evaluation_id_after_preview(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.fake.evaluations = [
            {"id": "QA-new", "name": "QA Test", "status": "ACTIVE", "type": "SYNCHRONOUS"},
            {"id": "KIS-new", "name": "Textual KIS Test", "status": "ACTIVE", "type": "SYNCHRONOUS"},
        ]
        result = self.service.submit(self.request(prepared), "member")
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(self.fake.submits()[0].url.path, "/api/v2/submit/KIS-new")

    def test_unique_but_wrong_task_name_requires_fallback(self):
        self.fake.evaluations = [{"id": "QA-only", "name": "QA Test", "status": "ACTIVE", "type": "SYNCHRONOUS"}]
        self.service.refresh_evaluations()
        self.assertIsNone(self.service.evaluation_id)
        selected = self.service.select_evaluation("QA-only")
        self.assertTrue(selected["evaluation_manual"])
        self.service.set_context(ContextPatch(expected_revision=1, kind="kis", label="KIS task"), "member")
        self.assertTrue(self.service.status()["context_valid"])

    def test_process_restart_requires_task_context_to_be_set_again(self):
        self.assertTrue(self.service.status()["context_valid"])
        self.service.context_evaluation_id = None
        self.assertFalse(self.service.status()["context_valid"])
        with self.assertRaisesRegex(DresError, "current task for this evaluation"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))

    def test_original_video_signature_must_match(self):
        self.video.write_bytes(b"changed")
        with self.assertRaisesRegex(DresError, "stale"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))

    def test_signature_accepts_different_bind_mount_root(self):
        root = Path(self.tmp.name)
        build_path = root / "build" / "2026" / "videos" / "L01" / "L01_V001.MP4"
        mount_path = root / "mount" / "2026" / "videos" / "L01" / "L01_V001.MP4"
        build_path.parent.mkdir(parents=True)
        mount_path.parent.mkdir(parents=True)
        os.link(self.video, build_path)
        os.link(self.video, mount_path)
        sig = build_path.stat()
        self.service.manifest_path.write_text(json.dumps({"videos": {"L01_V001": {
            "path": str(build_path), "size": sig.st_size, "mtimeNs": sig.st_mtime_ns,
        }}}))
        self.service.resolve_video = lambda _id: {"path": str(mount_path)}
        self.assertEqual(self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))["timestamp_ms"], 123456)

    def test_rejection_uncertain_and_no_automatic_retry(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.fake.submit_response = httpx.Response(422, json={"message": "wrong answer"})
        rejected = self.service.submit(self.request(prepared), "member")
        self.assertEqual((rejected["status"], rejected["http_status"]), ("rejected", 422))
        self.assertIn("wrong answer", rejected["server_message"])
        self.assertEqual(len(self.fake.submits()), 1)

        self.service.set_context(ContextPatch(expected_revision=1, kind="kis", label="Hall task 2"), "member")
        second = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=1))
        self.fake.submit_exception = httpx.ReadTimeout("lost response")
        uncertain = self.service.submit(self.request_for(second, 1), "member")
        self.assertEqual(uncertain["status"], "uncertain")
        self.assertEqual(len(self.fake.submits()), 2)
        with self.assertRaisesRegex(DresError, "already attempted"):
            self.service.submit(self.request_for(second, 1), "member")
        self.assertEqual(len(self.fake.submits()), 2)
        self.fake.submit_exception = None
        self.fake.submit_response = httpx.Response(200, json={"status": "OK"})
        retry = self.service.submit(self.request_for(second, 1, retry=True), "member")
        self.assertEqual(retry["status"], "accepted")
        self.assertEqual(len(self.fake.submits()), 3)

    def request_for(self, prepared, frame, retry=False):
        return SubmitRequest(candidate=Candidate(video_id="L01_V001", frame_idx=frame),
                             fingerprint=prepared["fingerprint"], expected_evaluation_id=prepared["evaluation_id"],
                             expected_context_revision=prepared["context_revision"], retry_uncertain=retry)

    def test_connection_failure_is_not_sent_and_retry_allowed(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.fake.submit_exception = httpx.ConnectError("DNS failure")
        first = self.service.submit(self.request(prepared), "member")
        self.assertEqual(first["status"], "not_sent")
        self.fake.submit_exception = None
        second = self.service.submit(self.request(prepared), "member")
        self.assertEqual(second["status"], "accepted")

    def test_qa_and_trake_serialization_from_selected_frames(self):
        self.service.set_context(ContextPatch(expected_revision=1, kind="qa", label="QA task"), "member")
        qa = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2, answer="  A-B  "))
        self.assertEqual(qa["payload"], {"answerSets": [{"answers": [{"text": "QA-A-B-L01_V001-123456"}]}]})
        self.assertEqual(qa["timestamp_ms"], 123456)
        with self.assertRaisesRegex(DresError, "answer is required"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2, answer=""))

        self.service.set_context(ContextPatch(expected_revision=2, kind="trake", label="TR task"), "member")
        trake = self.service.prepare(Candidate(video_id="L01_V001", frame_ids=[3, 1, 2]))
        self.assertEqual(trake["payload"], {"answerSets": [{"answers": [{"text": "TR-L01_V001-3,1,2"}]}]})
        self.assertIsNone(trake["timestamp_ms"])
        shorter = self.service.prepare(Candidate(video_id="L01_V001", frame_ids=[1, 2]))
        self.assertEqual(shorter["preview"], "TR-L01_V001-1,2")
        with self.assertRaisesRegex(DresError, "1–100"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_ids=[]))
        with self.assertRaisesRegex(DresError, "unique"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_ids=[1, 1, 2]))
        # The event is the source frame of the moment, so a frame the organizer map does not list is
        # a valid answer; only the video's own bounds are checked.
        off_map = self.service.prepare(Candidate(video_id="L01_V001", frame_ids=[0, 2]))
        self.assertEqual(off_map["preview"], "TR-L01_V001-0,2")
        with self.assertRaisesRegex(DresError, "outside the original video"):
            self.service.prepare(Candidate(video_id="L01_V001", frame_ids=[1, 2, 99]))

    def test_official_keyframes_keep_btc_id_and_resolve_source_preview_frame(self):
        keyframe_dir = self.video.parents[2] / "map-keyframes"
        (keyframe_dir / "L01_V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,123.456,30,250\n", encoding="utf-8")
        self.assertEqual(self.service.official_keyframes("L01_V001"), [{
            "frame_id": 250, "source_frame_idx": 2, "timestamp_ms": 123456,
        }])

    def test_vfr_organizer_pts_and_frame_ids_are_read_without_fps_math(self):
        keyframe_dir = self.video.parents[2] / "map-keyframes"
        (keyframe_dir / "N001-V001.csv").write_text(
            "n,pts_time,fps,frame_idx\n1,0.000,25,0\n2,2.1254,25,50\n",
            encoding="utf-8",
        )
        metadata = self.service._organizer_keyframes(self.video, "N001_V001")
        self.assertEqual(metadata[50], [(2125, 2)])
        self.assertEqual(metadata[0], [(0, 1)])
        self.assertEqual(self.service._selected_timestamp(self.mapping, metadata, "N001_V001", 50), 2125)
        with self.assertRaisesRegex(ValueError, "absent from the organizer PTS map"):
            self.service._selected_timestamp(self.mapping, metadata, "N001_V001", 51)

    def test_player_timestamp_is_original_frame_pts_and_trake_mapping_is_exact(self):
        metadata = {2: [(120000, 1)]}
        self.assertEqual(self.service._selected_timestamp(
            self.mapping, metadata, "L01_V001", 2, "original_frame"), 123456)
        self.assertEqual(self.service._selected_timestamp(
            self.mapping, metadata, "L01_V001", 2), 120000)

        self.mapping = FrameMap(np.array([0, 123456]), 1, 1000)
        self.assertEqual(self.service.official_frame_id("L01_V001", 1), 2)
        self.mapping = FrameMap(np.array([0, 123457]), 1, 1000)
        with self.assertRaisesRegex(DresError, "not a unique official TRAKE keyframe"):
            self.service.official_frame_id("L01_V001", 1)

    def test_malformed_success_and_server_error_are_uncertain(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=1))
        self.fake.submit_response = httpx.Response(200, text="unexpected receipt")
        result = self.service.submit(self.request_for(prepared, 1), "member")
        self.assertEqual(result["status"], "uncertain")
        self.assertEqual(len(self.fake.submits()), 1)
        other = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.fake.submit_response = httpx.Response(503, json={"message": "backend unavailable"})
        result = self.service.submit(self.request(other), "member")
        self.assertEqual((result["status"], result["http_status"]), ("uncertain", 503))
        self.assertEqual(len(self.fake.submits()), 2)

    def test_expired_session_resets_runtime(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        self.fake.submit_response = httpx.Response(401, json={"message": "expired session-secret"})
        attempt = self.service.submit(self.request(prepared), "member")
        self.assertEqual(attempt["status"], "rejected")
        self.assertFalse(self.service.status()["connected"])
        self.assertNotIn("session-secret", attempt["server_message"])

    def test_relogin_uses_credentials_held_only_in_runtime(self):
        self.assertEqual(self.service.session, "session-secret")
        self.service.connect()
        self.assertEqual(self.service.session, "session-secret")
        self.assertTrue(self.service.status()["context_valid"])
        self.assertEqual(len([request for request in self.fake.requests if request.url.path == "/api/v2/login"]), 2)
        self.assertEqual(self.fake.requests[-1].url.params["session"], "session-secret")

    def test_rejected_evaluation_session_retains_credentials_for_relogin(self):
        self.fake.eval_response = httpx.Response(401, json={"message": "session expired"})
        with self.assertRaisesRegex(DresError, "session was rejected"):
            self.service.refresh_evaluations()
        self.assertFalse(self.service.status()["connected"])
        self.assertIsNone(self.service.status()["evaluation_id"])
        self.assertEqual(len([request for request in self.fake.requests if request.url.path == "/api/v2/login"]), 1)

    def test_pending_lock_and_restart_uncertain(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        attempt_id = self.store.begin(prepared, "member")
        with self.assertRaisesRegex(DresError, "pending"):
            self.store.begin(prepared, "other")
        reopened = DresSubmissionStore(self.store.path)
        self.assertEqual(reopened.get(attempt_id)["status"], "uncertain")

    def test_local_ledger_failures_have_safe_send_classification(self):
        prepared = self.service.prepare(Candidate(video_id="L01_V001", frame_idx=2))
        with patch.object(self.store, "begin", side_effect=sqlite3.OperationalError("locked")):
            with self.assertRaises(DresError) as error:
                self.service.submit(self.request(prepared), "member")
        self.assertEqual(error.exception.category, "not_sent")
        self.assertEqual(len(self.fake.submits()), 0)
        with patch.object(self.store, "finish", side_effect=sqlite3.OperationalError("locked")):
            with self.assertRaises(DresError) as error:
                self.service.submit(self.request(prepared), "member")
        self.assertEqual(error.exception.category, "uncertain")
        self.assertEqual(len(self.fake.submits()), 1)


class DresRouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        async def inline(func, *args, **kwargs):
            return func(*args, **kwargs)

        self.threadpool_patch = patch("backend.dres.run_in_threadpool", inline)
        self.threadpool_patch.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.video = self.root / "L01_V001.mp4"
        self.video.write_bytes(b"fake")
        map_dir = self.root / "frame_maps"
        map_dir.mkdir()
        stat = self.video.stat()
        (map_dir / "manifest.json").write_text(json.dumps({"videos": {"L01_V001": {
            "path": str(self.video), "size": stat.st_size, "mtimeNs": stat.st_mtime_ns,
        }}}))
        self.fake = MockDres()
        config = DresConfig("http://localhost", "team", "password")
        client = DresClient(config, httpx.Client(transport=httpx.MockTransport(self.fake), base_url="http://localhost"))
        self.service = DresService(config, DresSubmissionStore(self.root / "ledger.sqlite3"),
                                   lambda _id: {"path": str(self.video)},
                                   load_map=lambda _id: FrameMap(np.array([0, 123456]), 1, 1000),
                                   index_dir=self.root, client=client)
        app = FastAPI()
        app.state.dres_service = self.service

        class Identity:
            def __init__(self, inner):
                self.inner = inner

            async def __call__(self, scope, receive, send):
                if scope["type"] == "http":
                    headers = dict(scope["headers"])
                    login = headers.get(b"x-aic-user", b"member").decode()
                    scope.setdefault("state", {}).update(login=login, is_admin=login == "owner")
                await self.inner(scope, receive, send)

        app.add_middleware(Identity)

        install_dres_routes(app)
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

    async def asyncTearDown(self):
        self.threadpool_patch.stop()
        await self.http.aclose()
        self.service.close()
        self.tmp.cleanup()

    async def test_owner_auth_and_member_submission(self):
        self.assertEqual((await self.http.post("/api/dres/connect")).status_code, 403)
        connected = await self.http.post("/api/dres/connect", headers={"X-AIC-User": "owner"})
        self.assertEqual(connected.status_code, 200)
        self.assertNotIn("session-secret", connected.text)
        self.assertNotIn('"password"', connected.text)
        self.assertEqual(json.loads(self.fake.requests[0].content), {"username": "team", "password": "password"})
        context = await self.http.post("/api/dres/context", json={
            "expected_revision": 0, "kind": "kis", "label": "Task on hall screen", "event_count": None,
        })
        self.assertEqual(context.status_code, 200)
        preview = await self.http.post("/api/dres/prepare", json={"candidate": {"video_id": "L01_V001", "frame_idx": 1}})
        self.assertEqual(preview.status_code, 200)
        data = preview.json()
        self.assertEqual(data["timestamp_ms"], 123456)
        submission = await self.http.post("/api/dres/submit", json={
            "candidate": {"video_id": "L01_V001", "frame_idx": 1},
            "fingerprint": data["fingerprint"], "expected_evaluation_id": data["evaluation_id"],
            "expected_context_revision": data["context_revision"],
        })
        self.assertEqual(submission.status_code, 200)
        self.assertEqual(submission.json()["status"], "accepted")
        self.assertEqual(len(self.fake.submits()), 1)

    async def test_api_rejects_browser_interval_and_invalid_frame(self):
        await self.http.post("/api/dres/connect", headers={"X-AIC-User": "owner"})
        await self.http.post("/api/dres/context", json={"expected_revision": 0, "kind": "kis", "label": "task"})
        extra = await self.http.post("/api/dres/prepare", json={
            "candidate": {"video_id": "L01_V001", "frame_idx": 1, "start": 120000, "end": 126000},
        })
        self.assertEqual(extra.status_code, 422)
        correct = await self.http.post("/api/dres/prepare", json={
            "candidate": {"video_id": "L01_V001", "frame_idx": 1},
        })
        answer = correct.json()["payload"]["answerSets"][0]["answers"][0]
        self.assertEqual((answer["start"], answer["end"]), ("123456", "123456"))
        for frame in (-1, 1.5, True, 5):
            response = await self.http.post("/api/dres/prepare", json={"candidate": {"video_id": "L01_V001", "frame_idx": frame}})
            self.assertIn(response.status_code, (422,))
