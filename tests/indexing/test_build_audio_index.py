import unittest

from backend.audio_search import SemanticChunkConfig, TopicChunkConfig
from scripts.indexing.build_audio_index import iter_passages, validate_coverage


class AudioIndexBuildTests(unittest.TestCase):
    def test_iter_passages_groups_videos_and_adds_search_metadata(self):
        rows = [
            {"audio_key": "L21_V002", "offset": 0.0, "token": " second"},
            {"audio_key": "L21_V001", "offset": 0.0, "token": " first"},
            {"audio_key": "L21_V001", "offset": 30.0, "token": " passage"},
        ]

        passages = list(iter_passages(rows))

        first = next(p for p in passages if p["video_id"] == "L21_V001")
        self.assertEqual(first["group"], "L21")
        self.assertEqual(first["year"], "2026")
        self.assertEqual(first["passage_id"], "L21_V001:0")
        self.assertEqual(first["text"], "first passage")
        self.assertNotIn("embedding", first)

    def test_iter_passages_semantic_strategy_delegates_to_make_semantic_passages(self):
        rows = [
            {"audio_key": "L21_V001", "offset": 0.0, "token": " một."},
            {"audio_key": "L21_V001", "offset": 20.0, "token": " hai."},
        ]

        def embed_cues(texts):
            return [[1.0, 0.0], [0.0, 1.0]]  # orthogonal -> two passages

        passages = list(iter_passages(rows, strategy="semantic", embed_cues=embed_cues,
                                      similarity_threshold=0.5))

        self.assertEqual(len(passages), 2)
        for passage in passages:
            self.assertEqual(passage["video_id"], "L21_V001")
            self.assertEqual(passage["group"], "L21")
            self.assertEqual(passage["year"], "2026")

    def test_iter_passages_semantic_duration_strategy_delegates_to_bounded_assembler(self):
        rows = [
            {"audio_key": "L21_V001", "offset": 0.0, "token": " một."},
            {"audio_key": "L21_V001", "offset": 20.0, "token": " hai."},
        ]

        def embed_cues(texts):
            return [[1.0, 0.0], [0.0, 1.0]]

        passages = list(iter_passages(
            rows, strategy="semantic_duration", embed_cues=embed_cues,
            chunk_config=SemanticChunkConfig(min_scores_for_percentile=1),
        ))

        # Only one boundary score exists (two sentences); orthogonal vectors
        # would trigger an immediate split at the default min_scores=4 too,
        # but min_scores_for_percentile=1 here just confirms the config is
        # actually threaded through, not the specific split decision.
        self.assertTrue(passages)
        for passage in passages:
            self.assertEqual(passage["video_id"], "L21_V001")
            self.assertEqual(passage["group"], "L21")
            self.assertEqual(passage["year"], "2026")

    def test_iter_passages_topic_strategy_delegates_to_make_topic_passages(self):
        rows = [
            {"audio_key": "L21_V001", "offset": 0.0, "token": " một."},
            {"audio_key": "L21_V001", "offset": 10.0, "token": " hai."},
        ]

        def llm_segment_fn(texts):
            self.assertEqual(texts, ["một.", "hai."])
            return [0]

        passages = list(iter_passages(
            rows, strategy="topic", llm_segment_fn=llm_segment_fn,
            topic_config=TopicChunkConfig(min_duration_seconds=1),
        ))

        self.assertEqual(len(passages), 2)
        for passage in passages:
            self.assertEqual(passage["video_id"], "L21_V001")
            self.assertEqual(passage["group"], "L21")
            self.assertEqual(passage["year"], "2026")
        self.assertEqual(passages[0]["text"], "một.")
        self.assertEqual(passages[1]["text"], "hai.")

    def test_coverage_reports_transcript_with_only_placeholders_as_unsearchable(self):
        source = {"L21_V001", "L24_V022"}
        indexed = {"L21_V001"}

        skipped = validate_coverage(source, indexed, expected_total=2)

        self.assertEqual(skipped, {"L24_V022"})

    def test_coverage_rejects_missing_source_transcripts(self):
        with self.assertRaisesRegex(RuntimeError, "expected 2 source transcripts"):
            validate_coverage({"L21_V001"}, {"L21_V001"}, expected_total=2)


if __name__ == "__main__":
    unittest.main()
