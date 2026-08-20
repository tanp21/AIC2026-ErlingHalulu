import unittest

from pydantic import ValidationError

from app import CompareReq, SearchReq


class RequestValidationTests(unittest.TestCase):
    def test_search_query_is_limited_to_2000_characters(self):
        with self.assertRaises(ValidationError):
            SearchReq(visual_text="x" * 2001)

    def test_search_result_count_is_bounded(self):
        for invalid in (0, 201):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                SearchReq(text="query", k=invalid)

    def test_compare_uses_the_same_query_and_result_limits(self):
        with self.assertRaises(ValidationError):
            CompareReq(text="x" * 2001)
        with self.assertRaises(ValidationError):
            CompareReq(text="query", k=201)


if __name__ == "__main__":
    unittest.main()
