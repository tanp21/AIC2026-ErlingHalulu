import unittest
from pathlib import Path

from scripts.audio.postprocess import Word, merge_tokens_to_words, words_to_tokens

ROOT = Path(__file__).resolve().parents[2]
STEP_DATA = ROOT / "data" / "2026" / "audio" / "step_data.parquet"


def rows(*triples):
    return [
        {"offset": offset, "token": token, "log_prob": nll}
        for offset, token, nll in triples
    ]


class MergeTokensToWordsTests(unittest.TestCase):
    """The ASR emits BPE pieces; everything downstream reasons about words."""

    def test_leading_space_starts_a_new_word(self):
        words = merge_tokens_to_words(rows(
            (0.0, " tham", 0.1), (0.12, " dự", 0.2), (0.72, " với", 0.3)
        ))
        self.assertEqual([w.text for w in words], ["tham", "dự", "với"])

    def test_pieces_without_a_leading_space_extend_the_current_word(self):
        words = merge_tokens_to_words(rows(
            (0.0, " bằ", 0.1), (0.1, "ng", 0.4), (0.5, " chứng", 0.2)
        ))
        self.assertEqual([w.text for w in words], ["bằng", "chứng"])

    def test_a_word_takes_the_offset_of_the_piece_that_started_it(self):
        words = merge_tokens_to_words(rows(
            (0.0, " bằ", 0.1), (0.9, "ng", 0.4), (1.5, " chứng", 0.2)
        ))
        self.assertEqual([w.offset for w in words], [0.0, 1.5])

    def test_a_word_takes_the_worst_confidence_of_its_pieces(self):
        # Merging must not launder a garbage piece into a confident word, so the
        # word carries the highest (worst) NLL of everything that formed it.
        words = merge_tokens_to_words(rows((0.0, " bằ", 0.1), (0.9, "ng", 3.4)))
        self.assertEqual([w.nll for w in words], [3.4])

    def test_a_token_may_hold_more_than_one_word(self):
        words = merge_tokens_to_words(rows((0.0, " xin chào", 0.1), (1.0, " bạn", 0.2)))
        self.assertEqual([w.text for w in words], ["xin", "chào", "bạn"])
        self.assertEqual([w.offset for w in words], [0.0, 0.0, 1.0])

    def test_whitespace_only_and_empty_tokens_produce_no_words(self):
        self.assertEqual(merge_tokens_to_words(rows((0.0, " ", 0.1), (1.0, "", 0.2))), [])

    def test_rows_are_sorted_by_offset_before_merging(self):
        words = merge_tokens_to_words(rows(
            (2.0, " sau", 0.1), (1.0, " trước", 0.2)
        ))
        self.assertEqual([w.text for w in words], ["trước", "sau"])

    def test_unusable_rows_are_skipped(self):
        words = merge_tokens_to_words([
            {"offset": None, "token": " lỗi", "log_prob": 0.1},
            {"offset": "bad", "token": " lỗi", "log_prob": 0.1},
            {"offset": 1.0, "token": None, "log_prob": 0.1},
            {"offset": -1.0, "token": " âm", "log_prob": 0.1},
            {"offset": 2.0, "token": " hợp lệ", "log_prob": 0.1},
        ])
        self.assertEqual([w.text for w in words], ["hợp", "lệ"])

    def test_a_missing_confidence_is_treated_as_worst_case(self):
        words = merge_tokens_to_words([{"offset": 0.0, "token": " a"}])
        self.assertEqual(words[0].nll, float("inf"))

    def test_no_rows_produce_no_words(self):
        self.assertEqual(merge_tokens_to_words([]), [])


class WordsToTokensTests(unittest.TestCase):
    def test_emits_one_row_per_word_carrying_its_offset(self):
        tokens = words_to_tokens("L21_V001", [Word("xin", 0.0, 0.1), Word("chào", 1.0, 0.2)])
        self.assertEqual(tokens, [
            {"audio_key": "L21_V001", "offset": 0.0, "token": " xin", "log_prob": 0.1},
            {"audio_key": "L21_V001", "offset": 1.0, "token": " chào", "log_prob": 0.2},
        ])

    def test_an_infinite_confidence_is_written_as_a_finite_number(self):
        # Parquet round-trips inf, but a NaN/inf in a float column is a trap for
        # anything that later aggregates the column. Clamp on the way out.
        tokens = words_to_tokens("L21_V001", [Word("a", 0.0, float("inf"))])
        self.assertTrue(0.0 <= tokens[0]["log_prob"] < float("inf"))

    def test_no_words_produce_no_rows(self):
        self.assertEqual(words_to_tokens("L21_V001", []), [])


class RoundTripTests(unittest.TestCase):
    """Tokens -> words -> tokens must preserve the word sequence and timing."""

    def test_round_trip_preserves_words_and_offsets(self):
        original = rows(
            (0.0, " tham", 0.1), (0.12, " dự", 0.2), (0.72, " bằ", 0.3),
            (0.9, "ng", 0.4), (1.5, " chứng", 0.2),
        )
        words = merge_tokens_to_words(original)
        again = merge_tokens_to_words(words_to_tokens("L21_V001", words))
        self.assertEqual([w.text for w in again], [w.text for w in words])
        self.assertEqual([w.offset for w in again], [w.offset for w in words])

    @unittest.skipUnless(STEP_DATA.exists(), "needs the 2026 transcript parquet")
    def test_round_trip_on_the_real_corpus(self):
        import pyarrow.parquet as pq

        table = pq.read_table(
            STEP_DATA,
            columns=["audio_key", "offset", "token", "log_prob"],
            filters=[("audio_key", "=", "L21_V001")],
        ).to_pylist()
        self.assertTrue(table)

        words = merge_tokens_to_words(table)
        # The word stream is exactly what a reader of the raw tokens would see.
        self.assertEqual(
            [w.text for w in words],
            "".join(str(row["token"]) for row in table).split(),
        )
        # Timing never runs backwards, or passages and subtitles would drift.
        offsets = [w.offset for w in words]
        self.assertEqual(offsets, sorted(offsets))

        again = merge_tokens_to_words(words_to_tokens("L21_V001", words))
        self.assertEqual([w.text for w in again], [w.text for w in words])
        self.assertEqual([w.offset for w in again], [w.offset for w in words])


if __name__ == "__main__":
    unittest.main()
