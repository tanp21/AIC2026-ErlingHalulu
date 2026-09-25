"""`POST /api/check` and `GET /api/check/lookup`.

The behaviour that matters most here is the cache: the whole point of keying a check by its
SHOT is that stepping along the filmstrip, or scrubbing the player, must not pay for the same
judgement again.
"""
import unittest
from unittest.mock import patch

import httpx

from backend import app as server
from backend import checker
from backend.access_control import AccessConfig, LatestWinsCoordinator

OWNER = "one@example.com"
MATE = "two@example.com"
USERS = "one@example.com,two@example.com,three@example.com,four@example.com,five@example.com"
FPS = 25.0
#: One shot of 20 s from 480 s: frames 12000-12499. The next starts at 12500.
SHOTS = [
    {"index": 0, "start": 460.0, "end": 480.0, "start_frame": 11500, "end_frame": 11999,
     "keyframes": [{"video_id": "L22_V011", "frame_idx": 11500, "time": 460.0}]},
    {"index": 1, "start": 480.0, "end": 500.0, "start_frame": 12000, "end_frame": 12499,
     "keyframes": [{"video_id": "L22_V011", "frame_idx": 12000, "time": 480.0},
                   {"video_id": "L22_V011", "frame_idx": 12300, "time": 492.0}]},
    {"index": 2, "start": 500.0, "end": 520.0, "start_frame": 12500, "end_frame": 12999,
     "keyframes": [{"video_id": "L22_V011", "frame_idx": 12500, "time": 500.0}]},
]
QUERY = {"visual_text": "Một người cầm điện thoại", "kind": "kis", "video_id": "L22_V011"}


class FakeSearch:
    def shots(self, video_id):
        return SHOTS

    def neighbours(self, video_id, frame_idx, radius=6):
        return [key for shot in SHOTS for key in shot["keyframes"]]

    def image(self, video_id, frame_idx, width=640):
        return b"jpeg"

    def transcript(self, video_id):
        return []

    def ocr(self, video_id):
        return []

    def caption_windows(self, video_id):
        return []


class FakeJudge:
    def __init__(self):
        self.judged = 0
        self.details_calls = 0

    def judge(self, images, prompt, schema, **kwargs):
        self.judged += 1
        return {"details": [{"detail": 0, "why": "a phone is held up", "verdict": "exact",
                             "note_vi": "có người cầm điện thoại"}],
                "overall": {"score": 9, "reason": "đúng cảnh"}}

    def chat_json(self, messages, **kwargs):
        self.details_calls += 1
        return {"details": [{"vi": "một người cầm điện thoại", "terms_en": ["phone"],
                             "terms_vi": ["điện thoại"], "meaning_en": "a person holding a phone",
                             "distinctiveness": 0.8}]}


def _annotate(points, profile, **kwargs):
    marks = []
    for point in points:
        frame = point.get("frame_idx")
        if frame is None:
            # nearest indexed keyframe to the requested time, as the real one does
            keys = [key for shot in SHOTS for key in shot["keyframes"]]
            nearest = min(keys, key=lambda key: abs(key["time"] - point["time"]))
            marks.append({"frame_idx": nearest["frame_idx"], "time": nearest["time"]})
        else:
            marks.append({"frame_idx": int(frame), "time": int(frame) / FPS})
    return marks


class _CheckerApp:
    """The app with a fake judge and a fake search behind the checker."""

    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(frozenset(USERS.split(",")), OWNER)
        server.app.state.search_coordinator = LatestWinsCoordinator()
        server.app.state.check_results = checker.CheckStore()
        server.app.state.check_details = checker.DetailsStore()
        self.judge = FakeJudge()
        self.search = FakeSearch()
        self.patches = [
            patch("backend.checker.checker_key", return_value="sk-or-test"),
            patch("backend.checker.judge_client", return_value=self.judge),
            patch("backend.checker.search_client", return_value=self.search),
            patch("backend.checker.search_core.annotate", side_effect=_annotate),
            patch("backend.checker.search_core.default_profile", return_value="l2026"),
        ]
        for started in self.patches:
            started.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test")

    async def asyncTearDown(self):
        for started in self.patches:
            started.stop()
        await self.client.aclose()

    async def check(self, user=MATE, **body):
        return await self.client.post("/api/check", json={**QUERY, **body},
                                      headers={"X-AIC-User": user})


class CheckRouteTests(_CheckerApp, unittest.IsolatedAsyncioTestCase):
    async def test_a_check_comes_back_with_its_checklist_verdict_and_where_it_applies(self):
        response = await self.check(frame_idx=12000)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], {"score": 9.0, "reason": "đúng cảnh"})
        self.assertEqual([check["status"] for check in body["checks"]], ["supported"])
        self.assertEqual(body["shot_index"], 1)
        self.assertEqual(body["covers"], {"start_frame": 12000, "end_frame": 12499,
                                          "start": 480.0, "end": 500.0})
        self.assertFalse(body["cached"])
        self.assertEqual(body["by"], MATE)
        self.assertEqual(body["window_seconds"], 300.0)

    async def test_another_frame_of_the_same_shot_is_answered_from_the_cache(self):
        # The point of keying by shot: stepping along the filmstrip inside one shot must not
        # pay for the same judgement again.
        first = await self.check(frame_idx=12000)
        second = await self.check(frame_idx=12300)

        self.assertEqual(self.judge.judged, 1)
        self.assertFalse(first.json()["cached"])
        self.assertTrue(second.json()["cached"])
        self.assertEqual(first.json()["key"], second.json()["key"])

    async def test_a_frame_in_the_next_shot_is_a_new_check(self):
        await self.check(frame_idx=12000)
        await self.check(frame_idx=12500)

        self.assertEqual(self.judge.judged, 2)

    async def test_the_checklist_is_built_once_however_many_candidates_are_checked(self):
        await self.check(frame_idx=12000)
        await self.check(frame_idx=12500)
        await self.check(frame_idx=11500)

        self.assertEqual(self.judge.judged, 3)
        self.assertEqual(self.judge.details_calls, 1)

    async def test_a_different_query_does_not_read_another_query_s_answer(self):
        await self.check(frame_idx=12000)
        await self.check(frame_idx=12000, visual_text="Một con thuyền đỏ trên sông")

        self.assertEqual(self.judge.judged, 2)

    async def test_the_answer_is_shared_between_teammates(self):
        # A check is a fact about the video and the query, not about who asked, so the second
        # person to look at a candidate gets the first one's answer for free.
        await self.check(user=MATE, frame_idx=12000)
        second = await self.check(user=OWNER, frame_idx=12000)

        self.assertEqual(self.judge.judged, 1)
        self.assertTrue(second.json()["cached"])
        self.assertEqual(second.json()["by"], MATE)

    async def test_force_re_reads_one_exact_frame_without_disturbing_the_shot_s_answer(self):
        shot = await self.check(frame_idx=12000)
        forced = await self.check(frame_idx=12300, force=True)
        again = await self.check(frame_idx=12300)

        self.assertEqual(self.judge.judged, 2)
        self.assertNotEqual(shot.json()["key"], forced.json()["key"])
        self.assertIsNone(forced.json()["covers"])
        self.assertTrue(again.json()["cached"])          # the shot's answer is still there
        self.assertEqual(again.json()["key"], shot.json()["key"])

    async def test_a_time_is_snapped_to_the_nearest_indexed_keyframe(self):
        # The player cannot always resolve an exact frame while playing, so it sends the time.
        response = await self.check(time=491.0)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["frame_idx"], 12300)
        self.assertEqual(response.json()["shot_index"], 1)

    async def test_a_time_and_a_frame_of_the_same_shot_share_one_answer(self):
        await self.check(frame_idx=12000)
        scrubbed = await self.check(time=491.0)

        self.assertEqual(self.judge.judged, 1)
        self.assertTrue(scrubbed.json()["cached"])

    async def test_a_request_naming_no_moment_at_all_is_rejected(self):
        response = await self.check()

        self.assertEqual(response.status_code, 422)

    async def test_a_request_with_no_search_channel_at_all_is_rejected(self):
        response = await self.client.post(
            "/api/check", json={"kind": "kis", "video_id": "L22_V011", "frame_idx": 12000},
            headers={"X-AIC-User": MATE})

        self.assertEqual(response.status_code, 422)

    async def test_an_audio_or_ocr_only_search_can_still_be_checked(self):
        # A spoken-word or on-screen-text search has no visual description at all; the request
        # must not require one.
        response = await self.client.post(
            "/api/check", json={"audio_text": "cá nhám", "kind": "kis",
                                "video_id": "L22_V011", "frame_idx": 12000},
            headers={"X-AIC-User": MATE})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.judge.judged, 1)

    async def test_visual_and_ocr_channels_of_the_same_words_are_different_checks(self):
        # The same string means something different as a scene description than as text to find
        # on screen, so they must not share a cache entry.
        await self.check(frame_idx=12000)
        await self.client.post(
            "/api/check", json={"ocr_text": QUERY["visual_text"], "kind": "kis",
                                "video_id": "L22_V011", "frame_idx": 12000},
            headers={"X-AIC-User": MATE})

        self.assertEqual(self.judge.judged, 2)

    async def test_a_window_outside_the_allowed_range_is_rejected(self):
        self.assertEqual((await self.check(frame_idx=12000, window_seconds=9000)).status_code, 422)

    async def test_a_narrower_window_is_a_separate_answer(self):
        await self.check(frame_idx=12000)
        await self.check(frame_idx=12000, window_seconds=60)

        self.assertEqual(self.judge.judged, 2)

    async def test_the_route_needs_a_trusted_identity(self):
        response = await self.client.post("/api/check", json={**QUERY, "frame_idx": 12000})

        self.assertEqual(response.status_code, 401)

    async def test_a_judge_that_fails_is_reported_as_a_bad_gateway_not_a_bug(self):
        with patch.object(self.judge, "judge", side_effect=RuntimeError("upstream is down")):
            response = await self.check(frame_idx=12000)

        self.assertEqual(response.status_code, 502)
        # the failure must not be cached as if it were an answer
        self.assertIsNone(server.app.state.check_results.get(response.json().get("key", "")))


class CheckUnavailableTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(frozenset(USERS.split(",")), OWNER)
        server.app.state.check_results = checker.CheckStore()
        server.app.state.check_details = checker.DetailsStore()
        self.patch = patch("backend.checker.checker_key", return_value="")
        self.patch.start()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test")

    async def asyncTearDown(self):
        self.patch.stop()
        await self.client.aclose()

    async def test_a_deployment_with_no_key_says_so_instead_of_failing_obscurely(self):
        response = await self.client.post(
            "/api/check", json={**QUERY, "frame_idx": 12000}, headers={"X-AIC-User": MATE})

        self.assertEqual(response.status_code, 503)
        self.assertIn("OPENROUTER_API_KEY", response.json()["detail"])

    async def test_lookup_says_unavailable_rather_than_pretending_nothing_is_cached(self):
        response = await self.client.get(
            "/api/check/lookup", params={**QUERY, "frame_idx": 12000},
            headers={"X-AIC-User": MATE})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "unavailable", "result": None})


class LookupTests(_CheckerApp, unittest.IsolatedAsyncioTestCase):
    async def test_lookup_answers_miss_then_hit_without_ever_calling_the_model(self):
        params = {**QUERY, "frame_idx": 12000}
        miss = await self.client.get("/api/check/lookup", params=params,
                                     headers={"X-AIC-User": MATE})
        self.assertEqual(miss.json()["status"], "miss")
        self.assertEqual(self.judge.judged, 0)

        await self.check(frame_idx=12000)
        hit = await self.client.get("/api/check/lookup", params=params,
                                    headers={"X-AIC-User": MATE})

        self.assertEqual(hit.json()["status"], "hit")
        self.assertEqual(hit.json()["result"]["verdict"]["score"], 9.0)
        self.assertEqual(self.judge.judged, 1)

    async def test_lookup_finds_a_neighbouring_frame_s_answer_through_its_shot(self):
        # What lets the filmstrip badge every frame a result already covers.
        await self.check(frame_idx=12000)
        hit = await self.client.get("/api/check/lookup", params={**QUERY, "frame_idx": 12300},
                                    headers={"X-AIC-User": MATE})

        self.assertEqual(hit.json()["status"], "hit")

    async def test_lookup_needs_a_moment_too(self):
        response = await self.client.get("/api/check/lookup", params=QUERY,
                                         headers={"X-AIC-User": MATE})
        self.assertEqual(response.status_code, 422)


class CacheStoreTests(unittest.TestCase):
    def test_an_entry_expires_and_the_clock_is_injectable(self):
        now = [0.0]
        store = checker.CheckStore(ttl_seconds=10, clock=lambda: now[0])
        store.put("k", {"verdict": 1})

        self.assertIsNotNone(store.get("k"))
        now[0] = 11.0
        self.assertIsNone(store.get("k"))

    def test_the_store_is_bounded(self):
        store = checker.CheckStore(maxsize=3)
        for index in range(6):
            store.put(f"k{index}", {"n": index})

        kept = [name for name in (f"k{index}" for index in range(6)) if store.get(name)]
        self.assertEqual(len(kept), 3)

    def test_only_one_caller_per_key_runs_the_work(self):
        store = checker.CheckStore()
        first, mine = store.claim("k")
        second, theirs = store.claim("k")

        self.assertTrue(mine)
        self.assertFalse(theirs)
        self.assertIs(first, second)

        store.settle("k", first, {"done": True}, None)
        self.assertEqual(second.result(), {"done": True})
        # settled: the next caller runs it again rather than waiting on a finished future
        _, again = store.claim("k")
        self.assertTrue(again)

    def test_a_failure_reaches_everyone_who_was_waiting_on_it(self):
        store = checker.CheckStore()
        future, _ = store.claim("k")
        store.settle("k", future, None, RuntimeError("boom"))

        with self.assertRaises(RuntimeError):
            future.result()
        self.assertFalse(store.running("k"))


class InFlightLimitTests(unittest.TestCase):
    def test_one_person_may_only_have_so_many_checks_running(self):
        # About the wait, not the money: a second press while the first is still running is
        # almost always the same person changing their mind, and the client aborts the first.
        held = [checker._Slot(MATE).__enter__() for _ in range(checker.PER_USER_LIMIT)]
        try:
            with self.assertRaises(Exception) as caught:
                with checker._Slot(MATE):
                    pass
            self.assertEqual(caught.exception.status_code, 429)
            with checker._Slot(OWNER):           # someone else is unaffected
                pass
        finally:
            for slot in held:
                slot.__exit__(None, None, None)

    def test_a_slot_is_released_however_the_request_ends(self):
        with self.assertRaises(RuntimeError):
            with checker._Slot(MATE):
                raise RuntimeError("boom")

        self.assertNotIn(MATE, checker._INFLIGHT)
        with checker._Slot(MATE):
            pass
        self.assertNotIn(MATE, checker._INFLIGHT)


class PlacementTests(unittest.TestCase):
    def test_a_frame_is_placed_in_the_shot_that_holds_it(self):
        placement = checker.place(FakeSearch(), "L22_V011", 12300, 492.0)

        self.assertEqual(placement.bucket, "s1")
        self.assertEqual(placement.shot_index, 1)
        self.assertEqual(placement.covers["end_frame"], 12499)

    def test_a_video_with_no_shots_falls_back_to_a_time_bucket(self):
        class NoShots(FakeSearch):
            def shots(self, video_id):
                return []

        near = checker.place(NoShots(), "K01_V001", 100, 42.0)
        same = checker.place(NoShots(), "K01_V001", 140, 46.0)
        later = checker.place(NoShots(), "K01_V001", 400, 58.0)

        self.assertIsNone(near.shot_index)
        self.assertEqual(near.bucket, same.bucket)      # scrubbing still reuses
        self.assertNotEqual(near.bucket, later.bucket)

    def test_a_server_that_cannot_answer_for_shots_still_places_the_frame(self):
        class Broken(FakeSearch):
            def shots(self, video_id):
                raise RuntimeError("503")

        placement = checker.place(Broken(), "L22_V011", 12300, 492.0)
        self.assertIsNone(placement.shot_index)
        self.assertTrue(placement.bucket.startswith("t"))


if __name__ == "__main__":
    unittest.main()
