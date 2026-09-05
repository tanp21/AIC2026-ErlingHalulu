import unittest

from backend.app import SearchSnapshotStore


def response(count):
    return {
        "results": [{"rank": rank} for rank in range(count)],
        "latency_ms": 12.5,
        "mode": "visual",
        "index": "flat",
        "index_info": {"type": "FlatIP"},
    }


class SearchSnapshotStoreTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.store = SearchSnapshotStore(
            page_size=2,
            ttl_seconds=10,
            result_limit=5,
            clock=lambda: self.now,
        )

    def test_create_returns_first_page_and_cursor_for_remaining_results(self):
        page = self.store.create("one@example.com", response(5))

        self.assertEqual([item["rank"] for item in page["results"]], [0, 1])
        self.assertEqual(page["total_results"], 5)
        self.assertTrue(page["has_more"])
        self.assertTrue(page["truncated"])
        self.assertIsInstance(page["next_cursor"], str)

    def test_cursor_is_idempotent_and_returns_global_rank_slice(self):
        first = self.store.create("one@example.com", response(5))

        second = self.store.next("one@example.com", first["next_cursor"])
        repeated = self.store.next("one@example.com", first["next_cursor"])

        self.assertEqual([item["rank"] for item in second["results"]], [2, 3])
        self.assertEqual(repeated, second)
        self.assertNotEqual(second["next_cursor"], first["next_cursor"])

    def test_short_snapshot_has_no_cursor_and_is_not_truncated(self):
        page = self.store.create("one@example.com", response(1))

        self.assertEqual(page["results"], [{"rank": 0}])
        self.assertFalse(page["has_more"])
        self.assertFalse(page["truncated"])
        self.assertIsNone(page["next_cursor"])

    def test_cursor_is_rejected_for_another_owner_or_after_idle_ttl(self):
        page = self.store.create("one@example.com", response(5))

        with self.assertRaisesRegex(LookupError, "expired"):
            self.store.next("two@example.com", page["next_cursor"])

        self.now = 111.0
        with self.assertRaisesRegex(LookupError, "expired"):
            self.store.next("one@example.com", page["next_cursor"])

    def test_successful_new_snapshot_invalidates_previous_cursor(self):
        old = self.store.create("one@example.com", response(5))
        self.store.create("one@example.com", response(3))

        with self.assertRaisesRegex(LookupError, "expired"):
            self.store.next("one@example.com", old["next_cursor"])


if __name__ == "__main__":
    unittest.main()
