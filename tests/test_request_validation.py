import unittest

from pydantic import ValidationError

from backend.app import CompareReq, SearchReq


class RequestValidationTests(unittest.TestCase):
    def test_search_query_is_limited_to_2000_characters(self):
        with self.assertRaises(ValidationError):
            SearchReq(visual_text="x" * 2001)

    def test_search_request_no_longer_exposes_a_result_count(self):
        self.assertNotIn("k", SearchReq.model_fields)

    def test_search_request_only_accepts_the_2026_profile(self):
        self.assertEqual(SearchReq(profile="l2026").profile, "l2026")
        with self.assertRaises(ValidationError):
            SearchReq(profile="l2025")

    def test_compare_uses_the_same_query_and_result_limits(self):
        with self.assertRaises(ValidationError):
            CompareReq(text="x" * 2001)
        with self.assertRaises(ValidationError):
            CompareReq(text="query", k=201)


if __name__ == "__main__":
    unittest.main()
