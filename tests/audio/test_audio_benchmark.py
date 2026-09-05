import json
import unittest
from pathlib import Path

from backend.audio_search import Passage
from scripts.benchmarks.bench_audio_search import (
    LexicalIndex,
    QueryResult,
    Report,
    evaluate,
    load_ground_truth,
    tokenize,
)

ROOT = Path(__file__).resolve().parents[2]
GROUND_TRUTH = ROOT / "benchmarks" / "audio" / "audio_ground_truth.json"


class GroundTruthTests(unittest.TestCase):
    """The answer keys are the only real labels we have; guard their shape."""

    @classmethod
    def setUpClass(cls):
        cls.queries = load_ground_truth(GROUND_TRUTH)

    def test_covers_both_competition_rounds(self):
        years = {query["id"].split(":")[0] for query in self.queries}
        self.assertEqual(years, {"2025", "2026"})

    def test_every_query_has_text_and_a_gold_video(self):
        for query in self.queries:
            with self.subTest(query["id"]):
                self.assertTrue(query["text"].strip())
                self.assertRegex(query["video_id"], r"^L\d{2}_V\d{3}$")
                self.assertIn(query["kind"], {"kis", "qa", "trake"})

    def test_query_ids_are_unique(self):
        ids = [query["id"] for query in self.queries]
        self.assertEqual(len(ids), len(set(ids)))

    def test_gold_videos_are_in_the_transcribed_range(self):
        # Round 1 is judged on L21-L30, which is exactly what the 2026
        # transcripts cover. A gold video outside it would be unscoreable.
        for query in self.queries:
            group = int(query["video_id"][1:3])
            self.assertTrue(21 <= group <= 30, query["video_id"])

    def test_answer_strings_are_not_stored(self):
        # The Q&A answers are read off the screen, not spoken, so they measure
        # OCR rather than transcript retrieval. Storing them would invite
        # scoring the audio branch against text it can never contain.
        for query in self.queries:
            self.assertNotIn("answer", query)


class TokenizeTests(unittest.TestCase):
    def test_diacritics_are_kept_when_requested(self):
        self.assertEqual(tokenize("Đèo Tà Pứa", diacritics=True), ["đèo", "tà", "pứa"])

    def test_folding_strips_tone_marks_and_stroked_d(self):
        self.assertEqual(tokenize("Đèo Tà Pứa", diacritics=False), ["deo", "ta", "pua"])

    def test_punctuation_and_underscores_are_dropped(self):
        self.assertEqual(tokenize("xin chào, các bạn!", diacritics=True),
                         ["xin", "chào", "các", "bạn"])

    def test_digits_survive_so_normalised_numbers_are_searchable(self):
        self.assertEqual(tokenize("năm 2024", diacritics=True), ["năm", "2024"])


class LexicalIndexTests(unittest.TestCase):
    def passage(self, video_id: str, start: float, text: str) -> Passage:
        return Passage(
            passage_id=f"{video_id}:{int(start * 1000)}",
            video_id=video_id,
            start_sec=start,
            end_sec=start + 60,
            anchor_sec=start + 30,
            text=text,
        )

    def test_ranks_the_matching_video_first(self):
        index = LexicalIndex([
            self.passage("L21_V001", 0, "đàn cá heo bơi ngoài khơi"),
            self.passage("L22_V002", 0, "công nhân lắp ráp khung xe trong xưởng"),
        ])
        self.assertEqual(index.rank_videos("lắp ráp khung xe")[0], "L22_V002")

    def test_a_video_appears_once_however_many_passages_match(self):
        index = LexicalIndex([
            self.passage("L21_V001", 0, "đàn cá heo bơi ngoài khơi"),
            self.passage("L21_V001", 15, "đàn cá heo bơi rất nhanh"),
            self.passage("L22_V002", 0, "công nhân lắp ráp khung xe"),
        ])
        ranking = index.rank_videos("đàn cá heo")
        self.assertEqual(ranking.count("L21_V001"), 1)
        self.assertEqual(ranking[0], "L21_V001")

    def test_matches_without_diacritics_through_the_folded_field(self):
        index = LexicalIndex([
            self.passage("L21_V001", 0, "đèo tà pứa bị sạt lở"),
            self.passage("L22_V002", 0, "món ngon mỗi ngày"),
        ])
        self.assertEqual(index.rank_videos("deo ta pua")[0], "L21_V001")

    def test_a_query_matching_nothing_ranks_nothing(self):
        index = LexicalIndex([self.passage("L21_V001", 0, "đàn cá heo")])
        self.assertEqual(index.rank_videos("khủng long bạo chúa"), [])

    def test_exact_contiguous_phrase_outranks_scattered_terms_at_equal_bm25(self):
        # Same 8-word bag (identical term frequency and length in both, so
        # plain BM25 scores them equal) — only word order differs, so only
        # the phrase bonus can tell them apart.
        index = LexicalIndex([
            self.passage("L21_V001", 0, "công nhân lắp ráp khung xe trong xưởng"),
            self.passage("L22_V002", 0, "khung xe trong xưởng công nhân lắp ráp"),
        ])
        self.assertEqual(index.rank_videos("lắp ráp khung xe")[0], "L21_V001")

    def test_phrase_bonus_is_a_no_op_for_single_term_queries(self):
        index = LexicalIndex([self.passage("L21_V001", 0, "đàn cá heo bơi ngoài khơi")])
        field = index._fields[0][2]
        self.assertEqual(index._phrase_bonus(field, ["heo"]), {})


class ReportTests(unittest.TestCase):
    REPORT = Report("t", [
        QueryResult("q1", "kis", "L21_V001", 1),
        QueryResult("q2", "kis", "L21_V002", 7),
        QueryResult("q3", "kis", "L21_V003", None),
        QueryResult("q4", "kis", "L21_V004", 3),
    ])

    def test_recall_counts_only_ranks_within_the_cutoff(self):
        self.assertEqual(self.REPORT.recall_at(1), 0.25)
        self.assertEqual(self.REPORT.recall_at(5), 0.5)
        self.assertEqual(self.REPORT.recall_at(10), 0.75)

    def test_unranked_gold_videos_never_count_as_hits(self):
        self.assertEqual(self.REPORT.recall_at(10_000), 0.75)

    def test_median_rank_treats_a_miss_as_infinitely_deep(self):
        self.assertEqual(self.REPORT.median_rank, 5.0)

    def test_empty_report_has_zero_recall(self):
        self.assertEqual(Report("empty", []).recall_at(10), 0.0)


class EvaluateTests(unittest.TestCase):
    QUERIES = [
        {"id": "q1", "kind": "kis", "text": "cá heo", "video_id": "L21_V001"},
        {"id": "q2", "kind": "qa", "text": "khung xe", "video_id": "L21_V009"},
    ]

    def test_records_the_rank_of_each_gold_video(self):
        report = evaluate("t", self.QUERIES, lambda text: ["L21_V002", "L21_V001"])
        self.assertEqual(report.ranks_by_query(), {"q1": 2, "q2": None})


if __name__ == "__main__":
    unittest.main()
