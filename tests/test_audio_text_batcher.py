import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from backend import audio_search


class AudioTextBatcherTests(unittest.TestCase):
    def tearDown(self):
        audio_search.close_audio_text_batcher()
        audio_search.start_audio_text_batcher()

    def test_five_concurrent_queries_share_one_encode_call(self):
        calls = []
        barrier = threading.Barrier(5)

        def encoder(texts):
            calls.append(list(texts))
            return np.ones((len(texts), 4), dtype="float32")

        with patch.object(audio_search, "_encode_audio_batch", side_effect=encoder):
            def query(index):
                barrier.wait()
                return audio_search.encode_query(f"query-{index}")

            with ThreadPoolExecutor(max_workers=5) as executor:
                results = list(executor.map(query, range(5)))

        self.assertEqual(len(calls), 1)
        self.assertCountEqual(calls[0], [f"query-{i}" for i in range(5)])
        for vector in results:
            self.assertEqual(vector, [1.0, 1.0, 1.0, 1.0])

    def test_close_then_start_recreates_the_lazy_global_batcher(self):
        def encoder(texts):
            return np.ones((len(texts), 2), dtype="float32")

        audio_search.close_audio_text_batcher()
        audio_search.start_audio_text_batcher()
        with patch.object(audio_search, "_encode_audio_batch", side_effect=encoder) as raw_encode:
            first = audio_search.encode_query("first")
            audio_search.close_audio_text_batcher()
            audio_search.start_audio_text_batcher()
            second = audio_search.encode_query("second")

        self.assertEqual(first, [1.0, 1.0])
        self.assertEqual(second, [1.0, 1.0])
        self.assertEqual(raw_encode.call_count, 2)

    def test_encode_query_after_close_raises(self):
        audio_search.close_audio_text_batcher()
        with self.assertRaisesRegex(RuntimeError, "not running"):
            audio_search.encode_query("late")


if __name__ == "__main__":
    unittest.main()
