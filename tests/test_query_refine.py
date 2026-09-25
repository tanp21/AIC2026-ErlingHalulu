import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from agent.llm import LLMError
from agent.tasks import Task
from backend import query_refine
from backend.app import RefineReq, refine_endpoint

PLAN = {
    "summary_vi": "tìm cảnh thuyền đỏ",
    "scenes": [
        {"description_vi": "bờ sông", "visual_en": ["a riverbank at dusk"]},
        {"description_vi": "thuyền", "visual_en": ["a red boat on a river", "a red wooden boat"]},
    ],
    "target_scene": 1,
    "readings_en": ["a red ferry crossing a river"],
    "transcript_vi": ["thuyền đỏ"],
    "caption_en": ["a boat moves across the frame"],
    "ocr": ["bến phà"],
    "lookups": ["tên bến phà sông Hậu"],
}


class FakeDeepSeek:
    """Mirrors tests/agent/test_orchestrator.py's stub: chat_json returns the plan."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def chat_json(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        reply = self.replies.pop(0) if self.replies else PLAN
        if isinstance(reply, Exception):
            raise reply
        return reply


def refine(*replies, task=Task("search", "kis", "thuyền đỏ")):
    with patch.object(query_refine, "_tokens", lambda text: len(text.split())):
        return query_refine.refine(task, FakeDeepSeek(*replies))


class RefineTests(unittest.TestCase):
    def test_orders_visual_candidates_from_the_target_scene_outwards(self):
        result = refine()
        self.assertEqual([candidate.text for candidate in result.visual], [
            "a red boat on a river", "a red wooden boat",   # target scene, every variant
            "a riverbank at dusk",                          # other scenes, first variant only
            "a red ferry crossing a river",                 # alternate readings last
        ])
        self.assertEqual(result.summary_vi, "tìm cảnh thuyền đỏ")
        self.assertEqual(result.transcript, ["thuyền đỏ"])
        self.assertEqual(result.ocr, ["bến phà"])
        self.assertEqual(result.lookups, ["tên bến phà sông Hậu"])

    def test_flags_only_the_target_scenes_variants_as_primary(self):
        """A secondary caption describes a less distinctive moment of the same clip and
        measurably ranks the gold video far worse alone -- the UI must tell them apart."""
        result = refine()
        primary = {candidate.text: candidate.primary for candidate in result.visual}
        self.assertEqual(primary, {
            "a red boat on a river": True, "a red wooden boat": True,
            "a riverbank at dusk": False, "a red ferry crossing a river": False,
        })

    def test_drops_channels_the_search_tab_cannot_query(self):
        """caption_en has no input box (the caption channel is agent-only)."""
        result = refine()
        self.assertNotIn("a boat moves across the frame",
                         [candidate.text for candidate in result.visual])

    def test_deduplicates_candidates_across_scenes_and_readings(self):
        plan = {**PLAN, "readings_en": ["A Red Boat On A River"], "target_scene": 1}
        result = refine(plan)
        self.assertEqual([candidate.text for candidate in result.visual],
                         ["a red boat on a river", "a red wooden boat", "a riverbank at dusk"])

    def test_a_secondary_caption_duplicating_the_primary_is_dropped_not_relabeled(self):
        """A duplicate must not survive as a second, lower-confidence copy of itself."""
        plan = {**PLAN, "readings_en": ["a red boat on a river"], "target_scene": 1}
        result = refine(plan)
        self.assertEqual([c.text for c in result.visual if not c.primary], ["a riverbank at dusk"])

    def test_carries_token_counts_so_the_ui_can_warn_on_truncation(self):
        self.assertEqual(refine().visual[0].tokens, 6)  # "a red boat on a river"

    def test_token_count_is_none_when_the_tokenizer_is_unavailable(self):
        with patch.object(query_refine, "count_tokens", create=True, side_effect=OSError("no cache")):
            result = query_refine.refine(Task("search", "kis", "x"), FakeDeepSeek())
        self.assertIsNone(result.visual[0].tokens)

    def test_trake_events_come_back_for_the_temporal_composer(self):
        plan = {**PLAN, "events_en": ["an athlete runs up", "the athlete jumps"]}
        result = refine(plan, task=Task("search", "trake", "E1: chạy đà\nE2: bật nhảy"))
        self.assertEqual([candidate.text for candidate in result.events],
                         ["an athlete runs up", "the athlete jumps"])

    def test_retries_once_before_giving_up(self):
        client = FakeDeepSeek(LLMError("down"), PLAN)
        result = query_refine.refine(Task("search", "kis", "x"), client)
        self.assertEqual(len(client.calls), 2)
        self.assertTrue(result.visual)

    def test_raises_after_the_last_attempt_fails(self):
        client = FakeDeepSeek(LLMError("down"), {"scenes": []})
        with self.assertRaises(LLMError):
            query_refine.refine(Task("search", "kis", "x"), client)
        self.assertEqual(len(client.calls), 2)

    def test_plans_without_thinking(self):
        """Planning is 5-10x faster without it and a person is waiting on this call."""
        client = FakeDeepSeek()
        query_refine.refine(Task("search", "kis", "x"), client)
        self.assertEqual(client.calls[0][1]["thinking"], False)


class RefineEndpointTests(unittest.TestCase):
    def call(self, **kwargs):
        return asyncio.run(refine_endpoint(RefineReq(**kwargs)))

    def test_returns_the_refinement_as_plain_json(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k"}), \
             patch.object(query_refine, "refine", return_value=query_refine.Refinement(
                 summary_vi="s", visual=[query_refine.Candidate("a red boat", 4)])), \
             patch("backend.app._refiner_client"):
            body = self.call(text="thuyền đỏ")
        self.assertEqual(body["visual"], [{"text": "a red boat", "tokens": 4, "primary": True}])
        self.assertEqual(body["summary_vi"], "s")
        self.assertEqual(body["transcript"], [])

    def test_rejects_a_blank_query(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k"}), \
             self.assertRaises(HTTPException) as caught:
            self.call(text="   ")
        self.assertEqual(caught.exception.status_code, 422)

    def test_reports_a_deployment_without_a_key_as_unavailable(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaises(HTTPException) as caught:
            self.call(text="thuyền đỏ")
        self.assertEqual(caught.exception.status_code, 503)

    def test_a_failed_plan_is_a_503_not_a_500(self):
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "k"}), \
             patch.object(query_refine, "refine", side_effect=LLMError("down")), \
             patch("backend.app._refiner_client"), \
             self.assertRaises(HTTPException) as caught:
            self.call(text="thuyền đỏ")
        self.assertEqual(caught.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
