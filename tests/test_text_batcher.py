import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from backend import core


class TextEmbeddingBatcherTests(unittest.TestCase):
    def tearDown(self):
        close = getattr(core, "close_text_batcher", None)
        if close is not None:
            close()
        start = getattr(core, "start_text_batcher", None)
        if start is not None:
            start()

    def test_five_requests_share_one_call_and_keep_their_rows(self):
        requests = [
            ["one"],
            ["two"],
            ["event-a", "event-b"],
            ["four"],
            ["five"],
        ]
        values = {text: index for index, text in enumerate(sum(requests, []), 1)}
        calls = []
        barrier = threading.Barrier(len(requests))

        def encoder(texts):
            calls.append(list(texts))
            return np.array([[values[text], -values[text]] for text in texts], dtype="float32")

        batcher = core.TextEmbeddingBatcher(encoder, window_seconds=0.05)

        def encode(texts):
            barrier.wait()
            return batcher.encode(texts)

        try:
            with ThreadPoolExecutor(max_workers=len(requests)) as executor:
                results = list(executor.map(encode, requests))
        finally:
            batcher.close()

        self.assertEqual(len(calls), 1)
        self.assertCountEqual(calls[0], sum(requests, []))
        event_start = calls[0].index("event-a")
        self.assertEqual(calls[0][event_start:event_start + 2], ["event-a", "event-b"])
        for texts, result in zip(requests, results):
            expected = np.array(
                [[values[text], -values[text]] for text in texts], dtype="float32"
            )
            np.testing.assert_array_equal(result, expected)

    def test_failed_batch_does_not_poison_the_next_batch(self):
        calls = 0
        lock = threading.Lock()
        barrier = threading.Barrier(5)

        def encoder(texts):
            nonlocal calls
            with lock:
                calls += 1
                call_number = calls
            if call_number == 1:
                raise RuntimeError("GPU inference failed")
            return np.ones((len(texts), 2), dtype="float32")

        batcher = core.TextEmbeddingBatcher(encoder, window_seconds=0.05)

        def encode_failed(index):
            barrier.wait()
            return batcher.encode([str(index)])

        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(encode_failed, index) for index in range(5)]
                for future in futures:
                    with self.assertRaisesRegex(RuntimeError, "GPU inference failed"):
                        future.result()

            recovered = batcher.encode(["recovered"])
        finally:
            batcher.close()

        np.testing.assert_array_equal(recovered, np.ones((1, 2), dtype="float32"))
        self.assertEqual(calls, 2)

    def test_close_then_start_recreates_the_lazy_global_batcher(self):
        def encoder(texts):
            return np.ones((len(texts), 2), dtype="float32")

        core.close_text_batcher()
        core.start_text_batcher()
        with patch.object(core, "_encode_text_batch", side_effect=encoder) as raw_encode:
            first = core.encode_texts(["first"])
            core.close_text_batcher()
            core.start_text_batcher()
            second = core.encode_texts(["second"])

        np.testing.assert_array_equal(first, np.ones((1, 2), dtype="float32"))
        np.testing.assert_array_equal(second, np.ones((1, 2), dtype="float32"))
        self.assertEqual(raw_encode.call_count, 2)

    def test_close_blocks_a_concurrent_replacement_until_restart(self):
        entered_encoder = threading.Event()
        release_encoder = threading.Event()

        def encoder(texts):
            if texts == ["active"]:
                entered_encoder.set()
                release_encoder.wait(1)
            return np.ones((len(texts), 2), dtype="float32")

        core.close_text_batcher()
        core.start_text_batcher()
        with (
            patch.object(core, "_encode_text_batch", side_effect=encoder),
            ThreadPoolExecutor(max_workers=3) as executor,
        ):
            active = executor.submit(core.encode_texts, ["active"])
            self.assertTrue(entered_encoder.wait(1))
            old_batcher = core._text_batcher
            close_entered = threading.Event()
            original_close = old_batcher.close

            def close_old_batcher():
                close_entered.set()
                original_close()

            old_batcher.close = close_old_batcher
            closing = executor.submit(core.close_text_batcher)
            self.assertTrue(close_entered.wait(1))
            late = executor.submit(core.encode_texts, ["late"])
            release_encoder.set()

            np.testing.assert_array_equal(active.result(1), np.ones((1, 2), dtype="float32"))
            closing.result(1)
            with self.assertRaisesRegex(RuntimeError, "not running"):
                late.result(1)

            core.start_text_batcher()
            recovered = core.encode_texts(["recovered"])

        np.testing.assert_array_equal(recovered, np.ones((1, 2), dtype="float32"))


if __name__ == "__main__":
    unittest.main()
