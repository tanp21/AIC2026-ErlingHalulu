from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from google.genai import errors as genai_errors

from concurrent.futures import ThreadPoolExecutor

from scripts.audio.gemini_clean import (
    DailyQuotaExhausted,
    GeminiCleaner,
    _RateLimiter,
    _align_key,
    _is_daily_quota_error,
    _retry_delay_seconds,
    _tokenize_cleaned_text,
    realign_words,
    restore_words_gemini,
)
from scripts.audio.postprocess import Word


class FakeResponse:
    def __init__(self, text):
        self.text = text


class FakeModels:
    """Queues one callable per expected call to ``generate_content``."""

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)

    def generate_content(self, *, model, contents, config):
        return self._behaviors.pop(0)()


class FakeClient:
    def __init__(self, behaviors):
        self.models = FakeModels(behaviors)


def cleaner_with(behaviors, **kwargs) -> GeminiCleaner:
    cleaner = GeminiCleaner("fake-model", max_retries=3, **kwargs)
    cleaner._client = FakeClient(behaviors)
    return cleaner


class GeminiCleanerRestoreTests(unittest.TestCase):
    def test_restore_returns_gemini_output_on_success(self):
        cleaner = cleaner_with([lambda: FakeResponse("Xin chào.")])
        self.assertEqual(cleaner.restore_or_none("xin chao"), "Xin chào.")

    def test_restore_retries_on_a_retryable_error_then_succeeds(self):
        def fail():
            raise genai_errors.APIError(429, {})

        cleaner = cleaner_with([fail, lambda: FakeResponse("Xin chào.")])
        with patch("time.sleep", return_value=None):
            self.assertEqual(cleaner.restore_or_none("xin chao"), "Xin chào.")

    def test_restore_falls_back_to_none_after_exhausting_retries(self):
        def fail():
            raise genai_errors.APIError(503, {})

        cleaner = cleaner_with([fail, fail, fail])
        with patch("time.sleep", return_value=None):
            self.assertIsNone(cleaner.restore_or_none("xin chao"))

    def test_restore_does_not_retry_a_non_retryable_client_error(self):
        def fail():
            raise genai_errors.APIError(400, {})

        # Only one behaviour queued: a retry would raise IndexError from the
        # empty queue and fail the test, proving the retry never happens.
        cleaner = cleaner_with([fail])
        self.assertIsNone(cleaner.restore_or_none("xin chao"))

    def test_restore_falls_back_to_input_text_when_the_response_is_empty(self):
        cleaner = cleaner_with([lambda: FakeResponse("")])
        self.assertEqual(cleaner.restore("xin chao"), "xin chao")

    def test_empty_text_is_a_no_op_without_calling_the_client(self):
        cleaner = cleaner_with([])  # any call would raise IndexError
        self.assertEqual(cleaner.restore_or_none(""), "")
        self.assertEqual(cleaner.restore_or_none("   "), "   ")

    def test_restore_falls_back_to_the_input_when_restore_or_none_gives_up(self):
        def fail():
            raise genai_errors.APIError(400, {})

        cleaner = cleaner_with([fail])
        self.assertEqual(cleaner.restore("xin chao"), "xin chao")


def _quota_error(quota_id: str) -> genai_errors.APIError:
    return genai_errors.APIError(429, {
        "error": {
            "code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED",
            "details": [
                {
                    "@type": "type.googleapis.com/google.rpc.QuotaFailure",
                    "violations": [{
                        "quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                        "quotaId": quota_id,
                        "quotaDimensions": {"location": "global", "model": "gemini-3.5-flash-lite"},
                        "quotaValue": "500",
                    }],
                },
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "59s"},
            ],
        }
    })


class RequestTimeoutTests(unittest.TestCase):
    """A stalled connection must not hang the run — see REQUEST_TIMEOUT_SECONDS."""

    def test_load_sets_the_timeout_on_the_client_in_milliseconds(self):
        # The SDK builds its httpx client with timeout=None, so if this is not
        # set a request that never returns blocks forever and max_retries never
        # fires, because a call that never returns never raises.
        with patch("google.genai.Client") as client:
            GeminiCleaner("m", api_key="k", timeout=90.0).load()
        self.assertEqual(client.call_args.kwargs["http_options"], {"timeout": 90_000})

    def test_a_transport_timeout_is_retried_rather_than_treated_as_permanent(self):
        # httpx.ReadTimeout is neither an APIError nor a builtin TimeoutError;
        # unlisted, it would fall through to the bare `except Exception` and be
        # misread as permanent, leaving the chunk raw for the whole run.
        def stall():
            raise httpx.ReadTimeout("read timed out")

        cleaner = cleaner_with([stall, lambda: FakeResponse("Xin chào.")])
        with patch("time.sleep", return_value=None):
            self.assertEqual(cleaner.restore_or_none("xin chao"), "Xin chào.")

    def test_a_stall_that_never_clears_gives_up_after_the_retry_budget(self):
        def stall():
            raise httpx.ReadTimeout("read timed out")

        cleaner = cleaner_with([stall, stall, stall])
        with patch("time.sleep", return_value=None):
            self.assertIsNone(cleaner.restore_or_none("xin chao"))


class DailyQuotaErrorDetectionTests(unittest.TestCase):
    def test_a_per_day_quota_violation_is_detected(self):
        error = _quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        self.assertTrue(_is_daily_quota_error(error))

    def test_a_per_minute_quota_violation_is_not_flagged_as_daily(self):
        error = _quota_error("GenerateRequestsPerMinutePerProjectPerModel-FreeTier")
        self.assertFalse(_is_daily_quota_error(error))

    def test_an_error_with_no_quota_violation_at_all_is_not_flagged(self):
        self.assertFalse(_is_daily_quota_error(TimeoutError("boom")))

    def test_restore_or_none_raises_instead_of_retrying_a_daily_quota_error(self):
        def fail():
            raise _quota_error("GenerateRequestsPerDayPerProjectPerModel-FreeTier")

        # Only one behaviour queued: a retry would raise IndexError from the
        # empty queue and fail the test, proving no retry is attempted.
        cleaner = cleaner_with([fail])
        with self.assertRaises(DailyQuotaExhausted):
            cleaner.restore_or_none("xin chao")

    def test_restore_words_gemini_propagates_daily_quota_exhaustion(self):
        class FakeCleaner:
            model_id = "fake-model"
            concurrency = 1

            def __init__(self, cache_dir):
                self.cache_dir = cache_dir

            def restore_or_none(self, text):
                raise DailyQuotaExhausted("out of quota")

        raw = words("xin chao")
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(DailyQuotaExhausted):
                restore_words_gemini(raw, FakeCleaner(Path(directory)))


class RetryDelaySecondsTests(unittest.TestCase):
    def test_parses_the_retry_info_hint_from_a_real_quota_error(self):
        error = genai_errors.APIError(429, {
            "error": {
                "code": 429, "message": "quota exceeded", "status": "RESOURCE_EXHAUSTED",
                "details": [
                    {"@type": "type.googleapis.com/google.rpc.Help", "links": []},
                    {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "29s"},
                ],
            }
        })
        self.assertEqual(_retry_delay_seconds(error), 29.0)

    def test_returns_none_when_there_is_no_retry_info(self):
        error = genai_errors.APIError(503, {"error": {"code": 503, "details": []}})
        self.assertIsNone(_retry_delay_seconds(error))

    def test_returns_none_for_an_error_with_no_details_at_all(self):
        self.assertIsNone(_retry_delay_seconds(TimeoutError("boom")))


class RateLimiterTests(unittest.TestCase):
    def test_no_limit_never_sleeps(self):
        limiter = _RateLimiter(None)
        with patch("time.sleep") as sleep:
            for _ in range(5):
                limiter.wait()
        sleep.assert_not_called()

    def test_consecutive_calls_are_spaced_by_the_target_interval(self):
        limiter = _RateLimiter(requests_per_minute=600)  # 0.1s apart
        slept = []
        with patch("time.sleep", side_effect=slept.append), \
             patch("time.monotonic", side_effect=[0.0, 0.0, 0.0, 0.0]):
            limiter.wait()
            limiter.wait()
        self.assertAlmostEqual(slept[0], 0.1, places=6)

    def test_the_combined_rate_across_threads_is_capped(self):
        limiter = _RateLimiter(requests_per_minute=6000)  # 0.01s apart
        with patch("time.sleep", return_value=None):
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(lambda _: limiter.wait(), range(40)))
        # Every dispatch claimed a distinct, monotonically later slot even
        # though 8 threads raced for them concurrently.
        self.assertGreater(limiter._next_allowed, 39 * 0.01 - 1e-6)


class AlignKeyTests(unittest.TestCase):
    def test_diacritics_and_case_are_folded_away(self):
        self.assertEqual(_align_key("Trực"), _align_key("truc"))

    def test_punctuation_is_stripped(self):
        self.assertEqual(_align_key("Trực,"), _align_key("trực"))
        self.assertEqual(_align_key("giang."), _align_key("Giang"))

    def test_a_lone_punctuation_token_folds_to_an_empty_key(self):
        self.assertEqual(_align_key("--"), "")
        self.assertEqual(_align_key("*"), "")


class TokenizeCleanedTextTests(unittest.TestCase):
    def test_stray_markdown_only_tokens_are_dropped(self):
        self.assertEqual(_tokenize_cleaned_text("Xin * chào - bạn"), ["Xin", "chào", "bạn"])

    def test_a_word_with_attached_punctuation_is_kept_whole(self):
        self.assertEqual(_tokenize_cleaned_text('Xin chào, "bạn."'), ["Xin", "chào,", '"bạn."'])


def words(text: str, *, nll: float = 0.1) -> list[Word]:
    return [Word(t, float(i), nll) for i, t in enumerate(text.split())]


class RealignWordsTests(unittest.TestCase):
    def test_empty_raw_words_return_empty(self):
        self.assertEqual(realign_words([], "anything"), [])

    def test_empty_cleaned_text_falls_back_to_raw_words(self):
        raw = words("xin chao cac ban")
        self.assertEqual(realign_words(raw, ""), raw)
        self.assertEqual(realign_words(raw, "   "), raw)

    def test_matching_words_keep_their_original_offset_and_confidence(self):
        raw = words("xin chao cac ban", nll=0.2)
        aligned = realign_words(raw, "Xin chào các bạn.")
        self.assertEqual([w.text for w in aligned], ["Xin", "chào", "các", "bạn."])
        self.assertEqual([w.offset for w in aligned], [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(all(w.nll == 0.2 for w in aligned))

    def test_a_replaced_word_gets_an_offset_between_its_neighbours(self):
        # "hai" (offset 1.0) is replaced by "hai," + an inserted word "ba,"
        # before the next raw-matching word "bon" (offset 2.0).
        raw = [Word("mot", 0.0, 0.1), Word("hai", 1.0, 0.1), Word("bon", 2.0, 0.1)]
        aligned = realign_words(raw, "Một hai, ba, bốn.")
        texts = [w.text for w in aligned]
        self.assertEqual(texts, ["Một", "hai,", "ba,", "bốn."])
        offsets = [w.offset for w in aligned]
        self.assertEqual(offsets[0], 0.0)
        self.assertEqual(offsets[-1], 2.0)
        # The two words standing in for "hai" sit strictly between its
        # neighbours, in order.
        self.assertTrue(0.0 < offsets[1] < offsets[2] < 2.0)

    def test_an_inserted_word_at_the_start_inherits_the_first_raw_offset(self):
        raw = [Word("chao", 5.0, 0.1), Word("ban", 6.0, 0.1)]
        aligned = realign_words(raw, "Xin chào bạn.")
        self.assertEqual([w.text for w in aligned], ["Xin", "chào", "bạn."])
        self.assertEqual(aligned[0].offset, 5.0)

    def test_an_inserted_word_at_the_end_inherits_the_last_raw_offset(self):
        raw = [Word("xin", 0.0, 0.1), Word("chao", 1.0, 0.1)]
        aligned = realign_words(raw, "Xin chào bạn.")
        self.assertEqual([w.text for w in aligned], ["Xin", "chào", "bạn."])
        self.assertEqual(aligned[-1].offset, 1.0)

    def test_a_dropped_raw_word_leaves_no_trace_in_the_output(self):
        # Within the allowance (1 word per 1000), a dropped word is applied and
        # simply has no counterpart in the result.
        raw = [Word(f"tu{i}", float(i), 0.1) for i in range(1000)]
        raw[500] = Word("um", 500.0, 0.1)
        cleaned = " ".join(
            f"Tu{i}." for i in range(1000) if i != 500
        )
        aligned = realign_words(raw, cleaned)
        self.assertEqual(len(aligned), 999)
        self.assertNotIn("um", [w.text for w in aligned])

    def test_a_replaced_span_carries_the_worst_confidence_of_what_it_replaced(self):
        raw = [Word("mot", 0.0, 0.1), Word("hai", 1.0, 9.0), Word("ba", 2.0, 0.1)]
        aligned = realign_words(raw, "Một XX ba.")
        # "hai" (nll=9.0) was replaced by "XX" -> that word inherits 9.0.
        replaced = [w for w in aligned if w.text == "XX"]
        self.assertEqual(len(replaced), 1)
        self.assertEqual(replaced[0].nll, 9.0)

    def test_a_response_too_different_from_the_input_falls_back_to_raw(self):
        raw = words("xin chao cac ban hom nay troi dep qua")
        garbled = "hoàn toàn không liên quan gì đến nội dung gốc cả đâu bạn nhé"
        self.assertEqual(realign_words(raw, garbled), raw)


class RemovalGuardTests(unittest.TestCase):
    """Cleaning repairs mistakes; it never removes content."""

    def test_a_response_that_drops_content_is_rejected(self):
        # The dangerous case: the response keeps the first half and silently
        # loses the rest. Everything present still matches, so similarity stays
        # ~0.67 -- above MIN_SIMILARITY_RATIO -- and the old similarity-only
        # guard let the diff drop the missing half outright.
        raw = words(" ".join(f"tu{i}" for i in range(1000)))
        truncated = " ".join(f"tu{i}" for i in range(500))
        self.assertEqual(realign_words(raw, truncated), raw)

    def test_a_localized_removal_is_rejected_however_short_the_video_is(self):
        # A long video can lose a whole stretch in the middle and still come
        # back at 98% overall, which is why the guard counts deletions rather
        # than comparing total lengths.
        raw = words(" ".join(f"tu{i}" for i in range(1000)))
        with_gap = " ".join(
            f"Tu{i}." for i in range(1000) if not 400 <= i < 600
        )
        self.assertEqual(realign_words(raw, with_gap), raw)

    def test_a_response_within_the_allowance_is_still_applied(self):
        # The guard must not be so tight that ordinary correction is rejected.
        # 99.5% of the words survive here, well inside MAX_DELETED_PER_1000.
        raw = words(" ".join(f"tu{i}" for i in range(1000)))
        trimmed = " ".join(f"Tu{i}." for i in range(995))
        result = realign_words(raw, trimmed)
        self.assertEqual(len(result), 995)
        self.assertEqual(result[0].text, "Tu0.")

    def test_the_prompt_forbids_removal_explicitly(self):
        # The guard is the backstop; the instruction is the primary lever, and
        # an earlier looser wording was demonstrably ignored.
        from scripts.audio.gemini_clean import _PROMPT

        for phrase in ("Do not delete, drop, skip, shorten or merge",
                       "advertisements, sponsor reads, song lyrics",
                       "Do not convert spoken words into symbols",
                       "Every word of the input must appear in your output"):
            self.assertIn(phrase, _PROMPT)

    def test_a_content_removing_response_is_not_cached_and_is_retried(self):
        calls = []

        class TruncatingCleaner:
            model_id = "fake-model"
            concurrency = 1

            def __init__(self, cache_dir):
                self.cache_dir = cache_dir

            def restore_or_none(self, text):
                calls.append(text)
                # Half the input back, as a cut-off generation returns.
                return " ".join(text.split()[:50])

        raw = words(" ".join(f"tu{i}" for i in range(100)))
        with tempfile.TemporaryDirectory() as directory:
            cleaner = TruncatingCleaner(Path(directory))
            full = len(raw)
            first = restore_words_gemini(raw, cleaner, chunk_words=full)
            second = restore_words_gemini(raw, cleaner, chunk_words=full)
        # Raw kept both times (never the truncated text) and never cached, so
        # the second run re-asks instead of replaying a bad answer forever.
        self.assertEqual([w.text for w in first], [w.text for w in raw])
        self.assertEqual([w.text for w in second], [w.text for w in raw])
        self.assertEqual(len(calls), 2)


class RestoreWordsGeminiTests(unittest.TestCase):
    def test_a_second_call_is_served_from_the_on_disk_cache(self):
        calls = []

        class FakeCleaner:
            model_id = "fake-model"
            concurrency = 1

            def __init__(self, cache_dir):
                self.cache_dir = cache_dir

            def restore_or_none(self, text):
                calls.append(text)
                return "Xin chào."

        raw = words("xin chao")
        with tempfile.TemporaryDirectory() as directory:
            cleaner = FakeCleaner(Path(directory))
            restore_words_gemini(raw, cleaner, chunk_words=10)
            restore_words_gemini(raw, cleaner, chunk_words=10)
        self.assertEqual(len(calls), 1)

    def test_a_gave_up_chunk_is_not_cached_and_is_retried_next_run(self):
        calls = []

        class FlakyThenGoodCleaner:
            model_id = "fake-model"
            concurrency = 1

            def __init__(self, cache_dir):
                self.cache_dir = cache_dir
                self.attempts = 0

            def restore_or_none(self, text):
                calls.append(text)
                self.attempts += 1
                return None if self.attempts == 1 else "Xin chào."

        raw = words("xin chao")
        with tempfile.TemporaryDirectory() as directory:
            cleaner = FlakyThenGoodCleaner(Path(directory))
            first = restore_words_gemini(raw, cleaner, chunk_words=10)
            second = restore_words_gemini(raw, cleaner, chunk_words=10)
        self.assertEqual([w.text for w in first], ["xin", "chao"])
        self.assertEqual([w.text for w in second], ["Xin", "chào."])
        self.assertEqual(len(calls), 2)

    def test_empty_words_return_empty_without_touching_the_cache_dir(self):
        class FakeCleaner:
            model_id = "fake-model"
            concurrency = 1
            cache_dir = Path("/nonexistent/should-not-be-created")

            def restore_or_none(self, text):
                raise AssertionError("should not be called for empty input")

        self.assertEqual(restore_words_gemini([], FakeCleaner()), [])

    def test_a_long_video_is_split_into_multiple_chunk_requests(self):
        calls = []

        class FakeCleaner:
            model_id = "fake-model"
            concurrency = 1

            def __init__(self, cache_dir):
                self.cache_dir = cache_dir

            def restore_or_none(self, text):
                calls.append(text)
                return text[0].upper() + text[1:] + "."

        raw = words("mot hai ba bon nam sau bay tam")
        with tempfile.TemporaryDirectory() as directory:
            cleaner = FakeCleaner(Path(directory))
            restore_words_gemini(raw, cleaner, chunk_words=3)
        self.assertEqual(len(calls), 3)  # 8 words / 3 per chunk -> 3 requests


if __name__ == "__main__":
    unittest.main()
