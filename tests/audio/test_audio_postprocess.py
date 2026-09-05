import unittest
from pathlib import Path

from scripts.audio.postprocess import (
    Word,
    apply_capu_output,
    drop_low_confidence,
    find_boilerplate,
    merge_tokens_to_words,
    normalize_numbers,
    strip_boilerplate,
    words_to_tokens,
)

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


def text_to_words(text: str, *, start: float = 0.0, nll: float = 0.1) -> list[Word]:
    return [
        Word(token, start + index, nll) for index, token in enumerate(text.split())
    ]


class FindBoilerplateTests(unittest.TestCase):
    """Sponsor reads recur across videos; ordinary speech recurs within one."""

    SPONSOR = "chương trình được tài trợ bởi công ty cổ phần"

    def test_a_phrase_spanning_enough_videos_is_boilerplate(self):
        corpus = {
            f"L2{i}_V001": text_to_words(f"mở đầu {self.SPONSOR} rồi nội dung riêng {i}")
            for i in range(5)
        }
        phrases = find_boilerplate(corpus, phrase_words=8, min_videos=5)
        self.assertTrue(phrases)

    def test_a_phrase_in_too_few_videos_is_left_alone(self):
        corpus = {
            f"L2{i}_V001": text_to_words(f"mở đầu {self.SPONSOR} rồi nội dung riêng {i}")
            for i in range(4)
        }
        self.assertEqual(find_boilerplate(corpus, phrase_words=8, min_videos=5), frozenset())

    def test_repetition_inside_one_video_is_not_boilerplate(self):
        # A cooking show saying the same line ten times is still that show's
        # content. Only cross-video recurrence marks a sponsor read.
        corpus = {"L21_V001": text_to_words(" ".join([self.SPONSOR] * 10))}
        self.assertEqual(find_boilerplate(corpus, phrase_words=8, min_videos=5), frozenset())

    def test_detection_ignores_diacritics(self):
        corpus = {
            "L21_V001": text_to_words(self.SPONSOR),
            "L21_V002": text_to_words(self.SPONSOR.upper()),
            "L21_V003": text_to_words("chuong trinh duoc tai tro boi cong ty co phan"),
        }
        self.assertTrue(find_boilerplate(corpus, phrase_words=8, min_videos=3))

    def test_a_video_shorter_than_the_phrase_contributes_nothing(self):
        corpus = {"L21_V001": text_to_words("quá ngắn")}
        self.assertEqual(find_boilerplate(corpus, phrase_words=8, min_videos=1), frozenset())


class StripBoilerplateTests(unittest.TestCase):
    SPONSOR = "chương trình được tài trợ bởi công ty cổ phần"

    def phrases(self, *texts, phrase_words=8):
        corpus = {f"v{i}": text_to_words(t) for i, t in enumerate(texts)}
        return find_boilerplate(corpus, phrase_words=phrase_words, min_videos=len(texts))

    def test_removes_the_matching_span_and_keeps_the_rest(self):
        phrases = self.phrases(self.SPONSOR, self.SPONSOR)
        words = text_to_words(f"đàn cá heo {self.SPONSOR} bơi ngoài khơi")
        kept = " ".join(w.text for w in strip_boilerplate(words, phrases, phrase_words=8))
        self.assertEqual(kept, "đàn cá heo bơi ngoài khơi")

    def test_overlapping_matches_chain_to_clear_a_long_read(self):
        # The detector only ever knows fixed-width phrases, but consecutive
        # windows overlap, so a 20-word sponsor read is removed in full.
        long_read = self.SPONSOR + " sữa tươi tiệt trùng số một việt nam"
        phrases = self.phrases(long_read, long_read)
        words = text_to_words(f"trước {long_read} sau")
        kept = [w.text for w in strip_boilerplate(words, phrases, phrase_words=8)]
        self.assertEqual(kept, ["trước", "sau"])

    def test_surviving_words_keep_their_original_timing(self):
        phrases = self.phrases(self.SPONSOR, self.SPONSOR)
        words = text_to_words(f"{self.SPONSOR} đàn cá heo")
        kept = strip_boilerplate(words, phrases, phrase_words=8)
        self.assertEqual([w.text for w in kept], ["đàn", "cá", "heo"])
        # The sponsor read is 10 words, so the survivors keep offsets 10-12
        # rather than being renumbered from zero.
        self.assertEqual([w.offset for w in kept], [10.0, 11.0, 12.0])

    def test_content_is_untouched_when_nothing_matches(self):
        words = text_to_words("đàn cá heo bơi ngoài khơi rất là nhanh")
        self.assertEqual(strip_boilerplate(words, frozenset(), phrase_words=8), words)

    def test_stripping_can_empty_a_video_that_is_only_boilerplate(self):
        phrases = self.phrases(self.SPONSOR, self.SPONSOR)
        self.assertEqual(strip_boilerplate(text_to_words(self.SPONSOR), phrases, phrase_words=8), [])


class DropLowConfidenceTests(unittest.TestCase):
    """NLL is a loss: lower is better, so the filter is an upper bound."""

    WORDS = [Word("chắc", 0.0, 0.01), Word("chắn", 1.0, 0.5), Word("<", 2.0, 3.0)]

    def test_keeps_words_at_or_below_the_threshold(self):
        kept = drop_low_confidence(self.WORDS, max_nll=0.5)
        self.assertEqual([w.text for w in kept], ["chắc", "chắn"])

    def test_an_infinite_threshold_keeps_everything(self):
        self.assertEqual(drop_low_confidence(self.WORDS, max_nll=float("inf")), self.WORDS)

    def test_words_with_no_recorded_confidence_are_dropped_by_any_finite_bound(self):
        words = [Word("a", 0.0, float("inf"))]
        self.assertEqual(drop_low_confidence(words, max_nll=1000.0), [])


class NormalizeNumbersTests(unittest.TestCase):
    def normalize(self, text: str) -> str:
        return " ".join(w.text for w in normalize_numbers(text_to_words(text)))

    def test_years_become_digits(self):
        self.assertEqual(self.normalize("vào năm hai ngàn không trăm hai mươi tư"),
                         "vào năm 2024")
        self.assertEqual(self.normalize("từ năm một ngàn chín trăm bốn mươi lăm"),
                         "từ năm 1945")

    def test_a_full_date_is_converted_piece_by_piece(self):
        self.assertEqual(
            self.normalize("ngày ba mốt tháng mười năm hai ngàn không trăm hai mươi tư"),
            "ngày 31 tháng 10 năm 2024",
        )

    def test_percentages_become_a_sign(self):
        self.assertEqual(self.normalize("tăng hai mươi phần trăm"), "tăng 20%")

    def test_round_millions_keep_their_scale_word(self):
        self.assertEqual(self.normalize("trị giá một trăm triệu đồng"),
                         "trị giá 100 triệu đồng")

    def test_the_indefinite_article_is_left_alone(self):
        # "một" is "a/an" far more often than "1"; converting it would break
        # every query phrased "một người đàn ông".
        self.assertEqual(self.normalize("một người đàn ông"), "một người đàn ông")
        self.assertEqual(self.normalize("có một con cá"), "có một con cá")

    def test_a_lone_number_after_a_date_trigger_is_converted(self):
        self.assertEqual(self.normalize("ngày một tháng tám"), "ngày 1 tháng 8")

    def test_nam_meaning_year_is_not_read_as_five(self):
        self.assertEqual(self.normalize("trong năm qua"), "trong năm qua")
        self.assertEqual(self.normalize("hai mươi năm sau"), "20 năm sau")

    def test_khong_meaning_not_is_not_read_as_zero(self):
        self.assertEqual(self.normalize("tôi không biết"), "tôi không biết")
        self.assertEqual(self.normalize("không có gì"), "không có gì")

    def test_nam_as_five_is_read_when_a_scale_word_follows(self):
        self.assertEqual(self.normalize("năm mươi lăm người"), "55 người")

    def test_filler_zero_inside_a_hundred(self):
        self.assertEqual(self.normalize("một trăm linh năm"), "105")

    def test_text_with_no_numbers_is_left_untouched(self):
        self.assertEqual(self.normalize("con cá heo bơi ngoài khơi"),
                         "con cá heo bơi ngoài khơi")

    def test_unit_variants_need_a_tens_word_before_them(self):
        # "tư" is four only after a tens word. Standing anywhere else it is far
        # more likely part of an ordinary word such as "tư duy", so the guard
        # gives up the "thứ tư" -> "thứ 4" conversion to stay safe.
        self.assertEqual(self.normalize("thứ tư duy nhất"), "thứ tư duy nhất")
        self.assertEqual(self.normalize("hai mươi tư"), "24")

    def test_the_converted_word_keeps_the_start_of_the_run(self):
        words = text_to_words("năm hai ngàn không trăm hai mươi tư")
        converted = normalize_numbers(words)
        self.assertEqual([w.text for w in converted], ["năm", "2024"])
        self.assertEqual(converted[1].offset, 1.0)

    def test_the_converted_word_carries_the_worst_confidence_of_the_run(self):
        words = [Word("hai", 0.0, 0.1), Word("mươi", 1.0, 2.5), Word("mốt", 2.0, 0.2)]
        self.assertEqual(normalize_numbers(words)[0].nll, 2.5)

    def test_no_words_produce_no_words(self):
        self.assertEqual(normalize_numbers([]), [])


class ApplyCapuOutputTests(unittest.TestCase):
    """Restoration may change letters and add punctuation, never word count."""

    WORDS = [Word("xin", 0.0, 0.1), Word("chào", 1.0, 0.2), Word("bạn", 2.0, 0.3)]

    def test_overlays_case_and_punctuation_onto_the_timed_words(self):
        result = apply_capu_output(self.WORDS, "Xin chào, bạn.")
        self.assertEqual([w.text for w in result], ["Xin", "chào,", "bạn."])

    def test_timing_and_confidence_survive_the_overlay(self):
        result = apply_capu_output(self.WORDS, "Xin chào, bạn.")
        self.assertEqual([w.offset for w in result], [0.0, 1.0, 2.0])
        self.assertEqual([w.nll for w in result], [0.1, 0.2, 0.3])

    def test_a_model_that_drops_words_is_rejected_wholesale(self):
        self.assertEqual(apply_capu_output(self.WORDS, "Xin chào."), self.WORDS)

    def test_a_model_that_invents_words_is_rejected_wholesale(self):
        self.assertEqual(
            apply_capu_output(self.WORDS, "Xin chào các bạn thân mến."), self.WORDS
        )

    def test_empty_output_is_rejected_rather_than_erasing_the_chunk(self):
        self.assertEqual(apply_capu_output(self.WORDS, ""), self.WORDS)

    def test_no_words_produce_no_words(self):
        self.assertEqual(apply_capu_output([], ""), [])


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
