import unittest
from unittest.mock import Mock

from backend import caption_search
from scripts.indexing import build_caption_index


def response(*sources):
    return {"hits": {"hits": [
        {"_score": 10.0 - index, "_source": source,
         "highlight": {"text": ["a <aic-hit>golden lion</aic-hit> dances"]}}
        for index, source in enumerate(sources)
    ]}}


def source(video_id="L01_V001", shot=2, window=0, frame=500, start=10.0, end=25.0, text="x"):
    return {"unit_id": f"{video_id}:{shot}:{window}", "video_id": video_id, "group": "L01",
            "year": "2026", "shot_idx": shot, "window_idx": window, "frame_idx": frame,
            "start_sec": start, "end_sec": end, "anchor_sec": (start + end) / 2, "text": text}


class IndexDefinitionTests(unittest.TestCase):
    def test_captions_are_searchable_stemmed_and_folded(self):
        mapping = caption_search.build_index_definition()["mappings"]["properties"]
        self.assertEqual(set(mapping["text"]["fields"]), {"english", "folded"})
        self.assertNotIn("embedding", mapping)

    def test_dense_adds_the_vector_field_only_when_asked(self):
        mapping = caption_search.build_index_definition(dense=True)["mappings"]["properties"]
        self.assertEqual(mapping["embedding"]["dims"], caption_search.EMBEDDING_DIM)

    def test_the_bm25_request_boosts_a_phrase_over_scattered_words(self):
        body = caption_search.build_bm25_request("golden lion", k=20)
        clauses = body["query"]["bool"]["should"]
        def boost(clause, field="text"):
            spec = next(f for f in clause["multi_match"]["fields"] if f.startswith(field + "^"))
            return float(spec.split("^")[1])

        phrase = next(c for c in clauses if c["multi_match"]["type"] == "phrase")
        loose = next(c for c in clauses if c["multi_match"]["type"] == "best_fields")
        self.assertGreater(boost(phrase), boost(loose))
        self.assertEqual(body["_source"]["excludes"], ["embedding"])


class ParseAndFuseTests(unittest.TestCase):
    def test_parsing_keeps_the_window_and_the_matched_words(self):
        hit = caption_search.parse_search_response(response(source()))[0]
        self.assertEqual((hit.video_id, hit.shot_idx, hit.frame_idx), ("L01_V001", 2, 500))
        self.assertEqual((hit.start_sec, hit.end_sec, hit.anchor_sec), (10.0, 25.0, 17.5))
        self.assertEqual(hit.matched_keywords, ("golden lion",))

    def test_a_row_missing_its_span_is_skipped(self):
        broken = {key: value for key, value in source().items() if key != "start_sec"}
        self.assertEqual(caption_search.parse_search_response(response(broken)), [])

    def test_rrf_rewards_a_window_both_rankings_found(self):
        both = caption_search.parse_search_response(response(source(shot=1)))[0]
        keyword_only = caption_search.parse_search_response(response(source(shot=9)))[0]
        merged = caption_search.reciprocal_rank_fusion([[keyword_only, both], [both]])
        self.assertEqual([hit.shot_idx for hit in merged], [1, 9])
        self.assertEqual([hit.rank for hit in merged], [1, 2])

    def test_keyword_search_ranks_and_truncates(self):
        client = Mock()
        client.search.return_value = response(source(shot=1), source(shot=2), source(shot=3))
        hits = caption_search.search_captions_keyword("x", 2, client=client, index="idx")
        self.assertEqual([hit.rank for hit in hits], [1, 2])
        self.assertEqual(client.search.call_args.kwargs["index"], "idx")

    def test_captions_for_video_asks_in_time_order(self):
        client = Mock()
        client.search.return_value = response(source())
        caption_search.captions_for_video("L01_V001", client=client, index="idx")
        body = client.search.call_args.kwargs["body"]
        self.assertEqual(body["query"], {"term": {"video_id": "L01_V001"}})
        self.assertEqual(body["sort"], [{"start_sec": "asc"}])

    def test_index_exists_is_false_when_elasticsearch_is_unreachable(self):
        client = Mock()
        client.indices.exists.side_effect = RuntimeError("connection refused")
        self.assertFalse(caption_search.index_exists(client=client))


class BuildIndexTests(unittest.TestCase):
    class FakeMap:
        def frame_at_media_time(self, seconds):
            return int(seconds * 25)

    def documents(self, rows, mapping=None):
        loader = build_caption_index.frame_maps.load
        build_caption_index.frame_maps.load = lambda video_id: (
            mapping if mapping is not None else self.FakeMap())
        try:
            return list(build_caption_index.iter_documents(rows, log=lambda *a, **k: None))
        finally:
            build_caption_index.frame_maps.load = loader

    def test_a_split_shot_uses_its_own_window_and_middle_frame(self):
        rows = [{"video_id": "L01_V001", "group": "L01", "year": "2026", "shot_index": 7,
                 "window_index": 2, "start_sec": 0.0, "end_sec": 100.0,
                 "window_start_sec": 40.0, "window_end_sec": 60.0, "caption": "  a  lion  "}]
        (doc_id, document), = self.documents(rows)
        self.assertEqual(doc_id, "L01_V001:7:2")
        self.assertEqual((document["start_sec"], document["end_sec"]), (40.0, 60.0))
        self.assertEqual((document["anchor_sec"], document["frame_idx"]), (50.0, 1250))
        self.assertEqual(document["text"], "a lion")  # whitespace normalised

    def test_the_older_per_shot_schema_reads_as_window_zero(self):
        rows = [{"video_id": "L01_V002", "shot_index": 3, "start_sec": 2.0, "end_sec": 6.0,
                 "caption": "a boat"}]
        (doc_id, document), = self.documents(rows)
        self.assertEqual((doc_id, document["window_idx"], document["group"]),
                         ("L01_V002:3:0", 0, "L01"))

    def test_empty_captions_and_unmappable_videos_are_skipped(self):
        rows = [{"video_id": "L01_V001", "shot_index": 1, "start_sec": 0.0, "end_sec": 1.0,
                 "caption": "   "},
                {"video_id": "L01_V001", "shot_index": 2, "start_sec": 0.0, "end_sec": 1.0,
                 "caption": "real"}]

        class Missing:
            def frame_at_media_time(self, seconds):
                raise ValueError("no frame map")

        self.assertEqual(self.documents(rows, Missing()), [])
        self.assertEqual(len(self.documents(rows)), 1)


if __name__ == "__main__":
    unittest.main()
