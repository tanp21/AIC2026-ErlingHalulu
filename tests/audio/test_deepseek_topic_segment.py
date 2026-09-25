from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.audio.deepseek_topic_segment import (
    DeepSeekTopicSegmenter,
    _cache_key,
    segment_batch,
)


class FakeClient:
    """Queues one behavior (a return value, or a callable that raises) per
    expected ``chat_json`` call -- mirrors ``tests.audio.test_audio_gemini_clean``'s
    ``FakeClient``/``FakeModels`` pattern for the DeepSeek-shaped client."""

    def __init__(self, behaviors, model="fake-model"):
        self.model = model
        self._behaviors = list(behaviors)
        self.calls = 0

    def chat_json(self, messages, *, salvage=False, **kwargs):
        self.calls += 1
        behavior = self._behaviors.pop(0)
        return behavior() if callable(behavior) else behavior


class SegmentBatchTests(unittest.TestCase):
    def test_valid_response_returns_boundaries(self):
        client = FakeClient([{"boundaries": [1, 3]}])

        self.assertEqual(segment_batch(client, ["a", "b", "c", "d", "e"]), [1, 3])

    def test_empty_boundaries_is_a_valid_response(self):
        # "this whole excerpt is one topic" is a legitimate answer, not a
        # rejection -- it must not be confused with a failed call.
        client = FakeClient([{"boundaries": []}])

        self.assertEqual(segment_batch(client, ["a", "b", "c"]), [])

    def test_out_of_range_index_is_rejected_then_gives_up(self):
        client = FakeClient([{"boundaries": [99]}, {"boundaries": [99]}])

        self.assertIsNone(segment_batch(client, ["a", "b", "c"], max_retries=1))
        self.assertEqual(client.calls, 2)

    def test_non_increasing_indices_are_rejected(self):
        client = FakeClient([{"boundaries": [2, 1]}] * 2)

        self.assertIsNone(segment_batch(client, ["a", "b", "c", "d"], max_retries=1))

    def test_missing_key_is_rejected(self):
        client = FakeClient([{}] * 2)

        self.assertIsNone(segment_batch(client, ["a", "b", "c"], max_retries=1))

    def test_non_json_response_is_rejected_after_retries_exhausted(self):
        def fail():
            raise ValueError("not json")

        client = FakeClient([fail, fail])

        self.assertIsNone(segment_batch(client, ["a", "b", "c"], max_retries=1))

    def test_retry_succeeds_after_one_bad_response(self):
        client = FakeClient([{"boundaries": [9]}, {"boundaries": [0]}])

        self.assertEqual(segment_batch(client, ["a", "b", "c"], max_retries=1), [0])

    def test_fewer_than_two_sentences_never_calls_the_client(self):
        client = FakeClient([])

        self.assertEqual(segment_batch(client, ["only"]), [])
        self.assertEqual(client.calls, 0)


class DeepSeekTopicSegmenterCacheTests(unittest.TestCase):
    def test_identical_input_is_served_from_cache_on_the_second_call(self):
        client = FakeClient([{"boundaries": [0]}])
        with tempfile.TemporaryDirectory() as cache_dir:
            segmenter = DeepSeekTopicSegmenter(client, cache_dir=Path(cache_dir))

            first = segmenter(["a", "b", "c"])
            second = segmenter(["a", "b", "c"])

        self.assertEqual(first, [0])
        self.assertEqual(second, [0])
        self.assertEqual(client.calls, 1)

    def test_a_rejected_response_is_never_cached_and_is_retried_next_call(self):
        client = FakeClient([
            {"boundaries": [99]}, {"boundaries": [99]},  # 1st segmenter() call: both rejected
            {"boundaries": [0]},                          # 2nd segmenter() call: succeeds
        ])
        with tempfile.TemporaryDirectory() as cache_dir:
            segmenter = DeepSeekTopicSegmenter(client, cache_dir=Path(cache_dir))

            first = segmenter(["a", "b", "c"])
            second = segmenter(["a", "b", "c"])

        self.assertEqual(first, [])
        self.assertEqual(second, [0])
        self.assertEqual(client.calls, 3)

    def test_a_genuinely_empty_result_is_cached(self):
        client = FakeClient([{"boundaries": []}])
        with tempfile.TemporaryDirectory() as cache_dir:
            segmenter = DeepSeekTopicSegmenter(client, cache_dir=Path(cache_dir))

            first = segmenter(["a", "b", "c"])
            second = segmenter(["a", "b", "c"])

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual(client.calls, 1)


class CacheKeyTests(unittest.TestCase):
    def test_same_model_and_sentences_produce_the_same_key(self):
        self.assertEqual(
            _cache_key("model-a", ["x", "y"]), _cache_key("model-a", ["x", "y"])
        )

    def test_different_sentences_produce_different_keys(self):
        self.assertNotEqual(
            _cache_key("model-a", ["x", "y"]), _cache_key("model-a", ["x", "z"])
        )

    def test_different_models_produce_different_keys(self):
        self.assertNotEqual(
            _cache_key("model-a", ["x", "y"]), _cache_key("model-b", ["x", "y"])
        )

    def test_prompt_change_invalidates_the_cache_key(self):
        original = _cache_key("model-a", ["x", "y"])

        with patch(
            "scripts.audio.deepseek_topic_segment._PROMPT_HEADER",
            "a different prompt {max_index}{numbered}",
        ):
            changed = _cache_key("model-a", ["x", "y"])

        self.assertNotEqual(original, changed)


if __name__ == "__main__":
    unittest.main()
