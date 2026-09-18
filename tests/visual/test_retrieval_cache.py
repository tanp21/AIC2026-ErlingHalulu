import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.benchmarks.bench_query_expansion import retrieve_cached, text_hash


class FakeIndex:
    """Deterministic 'search': rows and descending scores derived from the vector's first value."""

    def __init__(self):
        self.calls = []

    def search(self, vectors, k):
        self.calls.append(len(vectors))
        base = vectors[:, :1].astype(np.int64)
        rows = base + np.arange(k)[None, :]
        scores = (1.0 - np.arange(k)[None, :] * 0.01).repeat(len(vectors), axis=0).astype(np.float32)
        return scores, rows


def vec(value):
    return np.array([value, 0.0], dtype=np.float32)


class RetrievalCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self.tmp.name) / "cache.npz"

    def tearDown(self):
        self.tmp.cleanup()

    def test_second_call_keeps_first_calls_entries_and_scores(self):
        index = FakeIndex()
        retrieve_cached(["a"], {"a": vec(10)}, index, k=5, cache_path=self.cache)
        retrieve_cached(["b"], {"b": vec(20)}, index, k=5, cache_path=self.cache)
        with np.load(self.cache) as cached:
            keys = set(cached.files)
        for text in ("a", "b"):
            self.assertIn(f"{text_hash(text)}_rows", keys)
            self.assertIn(f"{text_hash(text)}_scores", keys)
        rows, scores = retrieve_cached(["a", "b"], {}, index, k=5, cache_path=self.cache)
        self.assertEqual(index.calls, [1, 1])  # third call served fully from cache
        np.testing.assert_array_equal(rows["a"], np.arange(10, 15))
        self.assertEqual(len(scores["b"]), 5)

    def test_rows_only_legacy_entry_is_researched_for_scores(self):
        with open(self.cache, "wb") as handle:
            np.savez(handle, **{f"{text_hash('old')}_rows": np.arange(5)})
        index = FakeIndex()
        rows, scores = retrieve_cached(["old"], {"old": vec(3)}, index, k=5, cache_path=self.cache)
        self.assertEqual(index.calls, [1])
        np.testing.assert_array_equal(rows["old"], np.arange(3, 8))
        self.assertEqual(len(scores["old"]), 5)

    def test_duplicate_texts_are_searched_once(self):
        index = FakeIndex()
        retrieve_cached(["x", "x"], {"x": vec(1)}, index, k=3, cache_path=self.cache)
        self.assertEqual(index.calls, [1])


if __name__ == "__main__":
    unittest.main()
