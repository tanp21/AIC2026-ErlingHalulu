import unittest

from audio_search import (
    AudioHit,
    build_bm25_request,
    build_index_definition,
    build_knn_request,
    make_passages,
    merge_temporal_hits,
    parse_search_response,
    reciprocal_rank_fusion,
    search_audio,
)


class PassageTests(unittest.TestCase):
    def test_sixty_second_passage_keeps_spread_out_query_evidence(self):
        rows = [
            {"offset": 848.56, "token": " chuyến bay tư nhân"},
            {"offset": 857.16, "token": " phi hành đoàn bốn người"},
            {"offset": 889.52, "token": " ánh sáng cực quang ở vùng cực"},
        ]

        passages = make_passages("L21_V015", rows, window_seconds=60, stride_seconds=15)

        passage = next(p for p in passages if p.start_sec == 840)
        self.assertEqual(passage.end_sec, 900)
        self.assertEqual(passage.anchor_sec, 870)
        self.assertIn("chuyến bay tư nhân", passage.text)
        self.assertIn("ánh sáng cực quang", passage.text)

    def test_passages_cover_tokens_on_window_boundary(self):
        rows = [
            {"offset": 0.0, "token": " first"},
            {"offset": 59.999, "token": " inside"},
            {"offset": 60.0, "token": " boundary"},
        ]

        passages = make_passages("L21_V001", rows, window_seconds=60, stride_seconds=15)

        self.assertIn("inside", passages[0].text)
        self.assertNotIn("boundary", passages[0].text)
        self.assertIn("boundary", passages[1].text)


class FusionTests(unittest.TestCase):
    def test_rrf_uses_rank_not_incompatible_source_scores(self):
        fused = reciprocal_rank_fusion(
            [
                [("A", 0.01), ("B", 0.001)],
                [("B", 99_000), ("A", 98_000)],
            ],
            rank_constant=0,
        )

        self.assertEqual([key for key, _ in fused], ["A", "B"])
        self.assertAlmostEqual(dict(fused)["A"], 1.5)
        self.assertAlmostEqual(dict(fused)["B"], 1.5)

    def test_overlapping_hits_merge_but_distant_hits_remain(self):
        hits = [
            AudioHit("L21_V015", 840, 900, 870, "first", 1.0),
            AudioHit("L21_V015", 855, 915, 885, "better", 2.0),
            AudioHit("L21_V015", 1100, 1160, 1130, "later", 0.5),
        ]

        merged = merge_temporal_hits(hits)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].text, "better")
        self.assertEqual(merged[0].start_sec, 840)
        self.assertEqual(merged[0].end_sec, 915)


class ElasticsearchContractTests(unittest.TestCase):
    def test_index_mapping_supports_accented_text_and_1024_dimension_vectors(self):
        definition = build_index_definition()

        folded = definition["settings"]["analysis"]["analyzer"]["vi_folded"]
        embedding = definition["mappings"]["properties"]["embedding"]
        text = definition["mappings"]["properties"]["text"]
        self.assertEqual(folded["filter"], ["lowercase", "asciifolding"])
        self.assertEqual(text["fields"]["folded"]["analyzer"], "vi_folded")
        self.assertEqual(embedding["type"], "dense_vector")
        self.assertEqual(embedding["dims"], 1024)
        self.assertEqual(embedding["similarity"], "cosine")

    def test_basic_license_requests_bm25_and_knn_separately(self):
        bm25 = build_bm25_request("nghiên cứu cực quang", k=20)
        knn = build_knn_request([0.0] * 1024, k=20)

        fields = bm25["query"]["multi_match"]["fields"]
        self.assertEqual(fields, ["text^2", "text.folded"])
        self.assertEqual(bm25["size"], 100)
        self.assertEqual(knn["knn"]["field"], "embedding")
        self.assertEqual(knn["knn"]["k"], 100)
        self.assertEqual(knn["size"], 100)
        self.assertNotIn("retriever", bm25)
        self.assertNotIn("retriever", knn)

    def test_search_parses_and_merges_overlapping_elasticsearch_hits(self):
        class Client:
            calls = 0

            def search(self, **_kwargs):
                self.calls += 1
                return {
                    "hits": {
                        "hits": [
                            {
                                "_score": 2.0,
                                "_source": {
                                    "video_id": "L21_V015",
                                    "start_sec": 840,
                                    "end_sec": 900,
                                    "anchor_sec": 870,
                                    "text": "chuyến bay tư nhân",
                                },
                            },
                            {
                                "_score": 1.0,
                                "_source": {
                                    "video_id": "L21_V015",
                                    "start_sec": 855,
                                    "end_sec": 915,
                                    "anchor_sec": 885,
                                    "text": "ánh sáng cực quang",
                                },
                            },
                        ]
                    }
                }

        hits = search_audio(
            "private spacecraft aurora",
            k=10,
            client=Client(),
            embed=lambda _text: [0.0] * 1024,
        )

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].bm25_rank, 1)
        self.assertEqual(hits[0].vector_rank, 1)
        self.assertEqual(hits[0].video_id, "L21_V015")
        self.assertEqual((hits[0].start_sec, hits[0].end_sec), (840, 915))

    def test_malformed_search_hit_is_ignored(self):
        response = {"hits": {"hits": [{"_score": 1.0, "_source": {"video_id": "L21"}}]}}

        self.assertEqual(parse_search_response(response), [])


if __name__ == "__main__":
    unittest.main()
