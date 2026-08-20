import unittest

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
