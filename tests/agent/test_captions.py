import unittest

import httpx

from agent.plan import initial_queries, parse_plan
from agent.pool import CandidatePool
from agent.search_client import SearchClient
from agent.tasks import task_from_text
from agent.vision import candidate_label, caption_at
from tests.agent.test_orchestrator import PLAN

CAPTIONS = [
    {"start": 0.0, "end": 12.0, "time": 6.0, "frame_idx": 150, "text": "A man walks in."},
    {"start": 12.0, "end": 30.0, "time": 21.0, "frame_idx": 520,
     "text": "Three women seal a cardboard box with tape."},
]


class CaptionPlanTests(unittest.TestCase):
    def test_caption_queries_join_the_round_one_batch(self):
        task = task_from_text("q-kis", "ba người phụ nữ dán thùng carton")
        plan = parse_plan({**PLAN, "caption_en": ["Three women seal a cardboard box", ""]}, task)
        queries = initial_queries(plan, task)

        caption = [query for query in queries if query["kind"] == "caption"]
        self.assertEqual([query["text"] for query in caption],
                         ["Three women seal a cardboard box"])
        # the target scene's visual query still leads the batch
        self.assertEqual(queries[0]["kind"], "visual")

    def test_a_plan_without_captions_asks_for_none(self):
        task = task_from_text("q-kis", "x")
        queries = initial_queries(parse_plan(PLAN, task), task)
        self.assertEqual([query for query in queries if query["kind"] == "caption"], [])


class CaptionEvidenceTests(unittest.TestCase):
    def test_a_caption_backs_every_moment_in_its_shot_and_reaches_gemini(self):
        pool = CandidatePool()
        pool.add_results({
            "v": {"kind": "visual", "hits": [
                {"rank": 1, "video_id": "L01_V001", "frame_idx": 500, "time": 20.0},
                {"rank": 2, "video_id": "L01_V001", "frame_idx": 900, "time": 36.0}]},
            "c": {"kind": "caption", "hits": [{
                "rank": 1, "video_id": "L01_V001", "frame_idx": 520, "time": 21.0,
                "start": 12.0, "end": 30.0, "score": 7.0,
                "text": "Three women seal a cardboard box with tape."}]},
        })
        inside, outside = pool.ordered()[0], pool.get("C2")
        self.assertEqual(inside.frame_idx, 500)  # the span attached to the moment it covers
        self.assertIn("c", inside.evidence)
        self.assertNotIn("c", outside.evidence)
        self.assertIn("shot caption: \"Three women seal", candidate_label(inside))

    def test_caption_at_picks_the_window_around_a_moment(self):
        self.assertEqual(caption_at(CAPTIONS, 20.0), "Three women seal a cardboard box with tape.")
        self.assertEqual(caption_at(CAPTIONS, 3.0), "A man walks in.")
        self.assertEqual(caption_at(CAPTIONS, 900.0), "Three women seal a cardboard box with tape.")
        self.assertEqual(caption_at([], 1.0), "(none)")


class CaptionCapabilityTests(unittest.TestCase):
    """A server without the caption channel must cost one failed sub-query, not a run."""

    def client(self, handler):
        transport = httpx.MockTransport(handler)
        http = httpx.Client(transport=transport, base_url="http://server")
        return SearchClient("http://server", http=http, sleep=lambda _: None, api="batch")

    def test_an_unsupported_kind_retries_the_batch_without_captions(self):
        seen = []

        def handler(request):
            body = request.read().decode()
            seen.append(body)
            if "caption" in body:
                return httpx.Response(422, json={"detail": "union_tag_invalid"})
            return httpx.Response(200, json={"results": {"v": {"kind": "visual", "hits": []}}})

        client = self.client(handler)
        results = client.search([{"kind": "visual", "id": "v", "text": "x"},
                                 {"kind": "caption", "id": "c", "text": "y"}])

        self.assertEqual(len(seen), 2)  # rejected, then retried without the caption query
        self.assertFalse(client.captions_supported)
        self.assertIn("v", results)
        # the next batch never asks again
        client.search([{"kind": "caption", "id": "c2", "text": "z"}])
        self.assertEqual(len(seen), 2)

    def test_a_missing_caption_index_disables_the_channel_after_one_failure(self):
        def handler(request):
            return httpx.Response(200, json={"results": {
                "c": {"kind": "caption", "error": "RuntimeError: index_not_found_exception"}}})

        client = self.client(handler)
        client.search([{"kind": "caption", "id": "c", "text": "y"}])
        self.assertFalse(client.captions_supported)
        results = client.search([{"kind": "caption", "id": "c3", "text": "y"}])
        self.assertEqual(results["c3"]["error"], "caption channel unavailable")

    def test_a_working_caption_channel_stays_enabled(self):
        def handler(request):
            return httpx.Response(200, json={"results": {"c": {"kind": "caption", "hits": []}}})

        client = self.client(handler)
        client.search([{"kind": "caption", "id": "c", "text": "y"}])
        self.assertTrue(client.captions_supported)


if __name__ == "__main__":
    unittest.main()
