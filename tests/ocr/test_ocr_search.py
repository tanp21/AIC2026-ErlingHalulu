import unittest

from backend.ocr_search import (
    build_bm25_request,
    build_filter_request,
    build_index_definition,
    contains_folded_phrase,
    parse_search_response,
    search_ocr_filter,
    search_ocr_keyword,
)


class ElasticsearchContractTests(unittest.TestCase):
    def test_index_mapping_supports_accented_text_and_shot_fields(self):
        definition = build_index_definition()

        folded = definition["settings"]["analysis"]["analyzer"]["vi_folded"]
        text = definition["mappings"]["properties"]["text"]
        properties = definition["mappings"]["properties"]
        self.assertEqual(folded["filter"], ["lowercase", "asciifolding"])
        self.assertEqual(text["fields"]["folded"]["analyzer"], "vi_folded")
        self.assertEqual(properties["shot_idx"]["type"], "integer")
        self.assertEqual(properties["frame_idx"]["type"], "integer")
        self.assertEqual(properties["video_id"]["type"], "keyword")
        self.assertNotIn("embedding", properties)

    def test_bm25_request_matches_audio_search_shape(self):
        request = build_bm25_request("bien so xe", k=20)

        should = request["query"]["bool"]["should"]
        best_fields_clauses = [c["multi_match"] for c in should if c["multi_match"]["type"] == "best_fields"]
        phrase = next(c["multi_match"] for c in should if c["multi_match"]["type"] == "phrase")
        best_fields = next(c for c in best_fields_clauses if "fuzziness" not in c)
        fuzzy = next(c for c in best_fields_clauses if "fuzziness" in c)
        self.assertEqual(best_fields["fields"], ["text^2", "text.folded"])
        self.assertEqual(phrase["fields"], ["text^4", "text.folded^2"])
        self.assertEqual(fuzzy["fuzziness"], "AUTO")
        self.assertEqual(request["query"]["bool"]["minimum_should_match"], 1)
        self.assertEqual(request["size"], 100)

    def test_strict_request_requires_the_literal_folded_phrase(self):
        request = build_bm25_request("Quảng Nam", k=20, strict=True)

        match = request["query"]["multi_match"]
        self.assertEqual(match["type"], "phrase")
        self.assertEqual(match["slop"], 0)
        self.assertEqual(match["fields"], ["text^2", "text.folded"])

    def test_folded_phrase_rejects_fuzzy_partial_hits(self):
        self.assertTrue(contains_folded_phrase("Quảng Nam", "TIN TỈNH QUANG NAM"))
        self.assertTrue(contains_folded_phrase("Quảng Nam", "tỉnh Quảng Năm hôm nay"))
        self.assertFalse(contains_folded_phrase("Quảng Nam", "Quán vợt Alcaraz không tin Sinner"))

    def test_filter_request_requires_every_term_near_each_other_but_tolerates_a_typo(self):
        request = build_filter_request("Quảng Nam", k=20)

        should = request["query"]["bool"]["should"]
        phrase = next(c["multi_match"] for c in should if "multi_match" in c)
        span_near = next(c["span_near"] for c in should if "span_near" in c)
        self.assertEqual(phrase["fields"], ["text^4", "text.folded^2"])
        clause_terms = [
            clause["span_multi"]["match"]["fuzzy"]["text.folded"]["value"]
            for clause in span_near["clauses"]
        ]
        self.assertEqual(clause_terms, ["quang", "nam"])
        for clause in span_near["clauses"]:
            fuzzy_term = clause["span_multi"]["match"]["fuzzy"]["text.folded"]
            self.assertEqual(fuzzy_term["fuzziness"], "AUTO")
        self.assertEqual(span_near["in_order"], True)
        self.assertGreaterEqual(span_near["slop"], 1)
        self.assertEqual(request["query"]["bool"]["minimum_should_match"], 1)
        self.assertEqual(request["size"], 100)

    def test_filter_request_with_no_extractable_terms_only_keeps_the_phrase_clause(self):
        request = build_filter_request("!!!", k=20)

        should = request["query"]["bool"]["should"]
        self.assertEqual(len(should), 1)
        self.assertIn("multi_match", should[0])


class ParseSearchResponseTests(unittest.TestCase):
    def test_parses_hit_with_highlight_into_matched_keywords(self):
        response = {
            "hits": {
                "hits": [
                    {
                        "_score": 3.5,
                        "_source": {
                            "video_id": "L21_V001",
                            "shot_idx": 12,
                            "frame_idx": 4567,
                            "start_sec": 100.0,
                            "end_sec": 104.5,
                            "anchor_sec": 102.0,
                            "text": "CHANG 1 06:30:11",
                        },
                        "highlight": {"text": ["<aic-hit>CHANG 1</aic-hit> 06:30:11"]},
                    }
                ]
            }
        }

        hits = parse_search_response(response)

        self.assertEqual(len(hits), 1)
        hit = hits[0]
        self.assertEqual(hit.video_id, "L21_V001")
        self.assertEqual(hit.shot_idx, 12)
        self.assertEqual(hit.frame_idx, 4567)
        self.assertEqual((hit.start_sec, hit.end_sec), (100.0, 104.5))
        self.assertEqual(hit.matched_keywords, ("CHANG 1",))

    def test_malformed_hit_missing_required_fields_is_ignored(self):
        response = {"hits": {"hits": [{"_score": 1.0, "_source": {"video_id": "L21_V001"}}]}}

        self.assertEqual(parse_search_response(response), [])


class SearchOcrKeywordTests(unittest.TestCase):
    def test_assigns_bm25_rank_in_response_order(self):
        class Client:
            def search(self, **_kwargs):
                return {
                    "hits": {
                        "hits": [
                            {
                                "_score": 5.0,
                                "_source": {
                                    "video_id": "L21_V001", "shot_idx": 1, "frame_idx": 100,
                                    "start_sec": 10.0, "end_sec": 12.0, "anchor_sec": 11.0,
                                    "text": "first hit",
                                },
                            },
                            {
                                "_score": 2.0,
                                "_source": {
                                    "video_id": "L21_V002", "shot_idx": 2, "frame_idx": 200,
                                    "start_sec": 20.0, "end_sec": 22.0, "anchor_sec": 21.0,
                                    "text": "second hit",
                                },
                            },
                        ]
                    }
                }

        hits = search_ocr_keyword("hit", k=10, client=Client())

        self.assertEqual([h.bm25_rank for h in hits], [1, 2])
        self.assertEqual([h.video_id for h in hits], ["L21_V001", "L21_V002"])


class SearchOcrFilterTests(unittest.TestCase):
    def test_uses_the_and_fuzzy_query_and_ranks_the_response_order(self):
        seen_body = {}

        class Client:
            def search(self, *, index, body):
                seen_body["body"] = body
                return {
                    "hits": {
                        "hits": [
                            {
                                "_score": 5.0,
                                "_source": {
                                    "video_id": "L21_V001", "shot_idx": 1, "frame_idx": 100,
                                    "start_sec": 10.0, "end_sec": 12.0, "anchor_sec": 11.0,
                                    "text": "first hit",
                                },
                            },
                        ]
                    }
                }

        hits = search_ocr_filter("Quảng Nam", k=10, client=Client())

        self.assertEqual([h.bm25_rank for h in hits], [1])
        self.assertEqual([h.video_id for h in hits], ["L21_V001"])
        self.assertEqual(seen_body["body"], build_filter_request("Quảng Nam", 10))


if __name__ == "__main__":
    unittest.main()
