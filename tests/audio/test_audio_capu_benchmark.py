import unittest

from scripts.audio.capu import CHUNK_WORDS, restore_words
from scripts.audio.postprocess import Word
from scripts.benchmarks.bench_audio_capu import (
    CapuReport,
    Token,
    build_samples,
    load_reference_text,
    score_sample,
    to_input,
    tokenize_reference,
)


class TokenizeReferenceTests(unittest.TestCase):
    def test_separates_a_word_from_its_trailing_punctuation(self):
        self.assertEqual(
            tokenize_reference("Xin chào, bạn."),
            [Token("xin", True, ""), Token("chào", False, ","), Token("bạn", False, ".")],
        )

    def test_capitalisation_is_recorded_and_the_word_lowercased(self):
        token = tokenize_reference("Việt")[0]
        self.assertTrue(token.capitalised)
        self.assertEqual(token.word, "việt")

    def test_unsupported_marks_fold_onto_the_supported_ones(self):
        # The models restore only . , : ? so the reference must not demand more.
        self.assertEqual(tokenize_reference("Ôi!")[0].punctuation, ".")
        self.assertEqual(tokenize_reference("a; b")[0].punctuation, ",")

    def test_diacritics_survive_tokenisation(self):
        self.assertEqual(tokenize_reference("Đèo Tà Pứa")[2].word, "pứa")

    def test_quotes_and_brackets_are_not_mistaken_for_words(self):
        self.assertEqual([t.word for t in tokenize_reference('“a” (b)')], ["a", "b"])

    def test_digits_are_kept_as_words(self):
        self.assertEqual([t.word for t in tokenize_reference("hình ảnh 4 phi")][2], "4")


class BuildSamplesTests(unittest.TestCase):
    def test_samples_never_span_two_sources(self):
        samples = build_samples(["một hai ba bốn năm sáu", "bảy tám chín mười mười một"],
                                words_per_sample=40)
        self.assertEqual(len(samples), 2)

    def test_fragments_shorter_than_five_words_are_dropped(self):
        self.assertEqual(build_samples(["một hai"]), [])

    def test_long_sources_are_cut_into_model_sized_pieces(self):
        samples = build_samples([" ".join(["từ"] * 100)], words_per_sample=40)
        self.assertEqual([len(s) for s in samples], [40, 40, 20])

    def test_the_model_input_is_lowercase_and_unpunctuated(self):
        sample = tokenize_reference("Xin chào, các bạn thân mến.")
        self.assertEqual(to_input(sample), "xin chào các bạn thân mến")


class ScoreSampleTests(unittest.TestCase):
    REFERENCE = tokenize_reference("Xin chào, bạn hiền.")

    def report(self, restored: str) -> CapuReport:
        report = CapuReport("test")
        score_sample(report, self.REFERENCE, restored)
        return report

    def test_a_perfect_restoration_scores_one(self):
        report = self.report("Xin chào, bạn hiền.")
        self.assertEqual(report.preservation, 1.0)
        self.assertEqual(report.capital_accuracy, 1.0)
        self.assertEqual(report.punctuation_f1(","), 1.0)
        self.assertEqual(report.punctuation_f1("."), 1.0)

    def test_a_word_count_mismatch_earns_no_credit_at_all(self):
        report = self.report("Xin chào.")
        self.assertEqual(report.preservation, 0.0)
        self.assertEqual(report.words, 0)
        self.assertEqual(report.macro_f1, 0.0)

    def test_missing_punctuation_costs_recall(self):
        report = self.report("Xin chào bạn hiền")
        self.assertEqual(report.preservation, 1.0)
        self.assertEqual(report.punctuation_f1(","), 0.0)

    def test_wrong_capitalisation_is_counted(self):
        report = self.report("xin chào, bạn hiền.")
        self.assertEqual(report.capital_accuracy, 0.75)
        self.assertEqual(report.capital_f1, 0.0)

    def test_a_mark_absent_from_both_sides_scores_zero_not_one(self):
        # "?" never appears here; an empty class must not inflate the macro.
        self.assertEqual(self.report("Xin chào, bạn hiền.").punctuation_f1("?"), 0.0)


class RestoreWordsTests(unittest.TestCase):
    class FakeModel:
        """Title-cases and adds a full stop, preserving the word count."""

        def __init__(self):
            self.calls = []

        def restore(self, text: str) -> str:
            self.calls.append(text)
            return " ".join(w.capitalize() for w in text.split()) + "."

    class BatchingModel:
        """Exposes restore_many, the path a real CapuModel takes."""

        def __init__(self):
            self.batches = []

        def restore_many(self, texts):
            texts = list(texts)
            self.batches.append(texts)
            return [" ".join(w.capitalize() for w in t.split()) + "." for t in texts]

    class BrokenModel:
        def restore(self, text: str) -> str:
            return "nope"

    def words(self, count: int):
        return [Word(f"từ{i}", float(i), 0.1) for i in range(count)]

    def test_restores_across_chunk_boundaries(self):
        model = self.FakeModel()
        result = restore_words(self.words(5), model, chunk_words=2)
        self.assertEqual(len(result), 5)
        self.assertEqual(model.calls, ["từ0 từ1", "từ2 từ3", "từ4"])

    def test_offsets_are_preserved_across_chunks(self):
        result = restore_words(self.words(5), self.FakeModel(), chunk_words=2)
        self.assertEqual([w.offset for w in result], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_a_chunk_the_model_mangles_falls_back_to_the_raw_words(self):
        original = self.words(4)
        result = restore_words(original, self.BrokenModel(), chunk_words=2)
        self.assertEqual(result, original)

    def test_chunks_go_out_together_rather_than_one_at_a_time(self):
        model = self.BatchingModel()
        result = restore_words(self.words(5), model, chunk_words=2, batch_chunks=16)
        self.assertEqual(len(model.batches), 1)
        self.assertEqual(model.batches[0], ["từ0 từ1", "từ2 từ3", "từ4"])
        self.assertEqual([w.text for w in result],
                         ["Từ0", "Từ1.", "Từ2", "Từ3.", "Từ4."])

    def test_a_long_video_is_split_into_bounded_batches(self):
        # Unbounded, the longest video in the corpus would arrive at the GPU as
        # several hundred sequences at once.
        model = self.BatchingModel()
        restore_words(self.words(20), model, chunk_words=2, batch_chunks=3)
        self.assertEqual([len(batch) for batch in model.batches], [3, 3, 3, 1])

    def test_batching_does_not_change_the_result(self):
        words = self.words(20)
        self.assertEqual(
            restore_words(words, self.BatchingModel(), chunk_words=2, batch_chunks=3),
            restore_words(words, self.BatchingModel(), chunk_words=2, batch_chunks=100),
        )

    def test_the_batched_and_single_paths_agree(self):
        words = self.words(5)
        self.assertEqual(
            restore_words(words, self.BatchingModel(), chunk_words=2),
            restore_words(words, self.FakeModel(), chunk_words=2),
        )

    def test_no_words_produce_no_words(self):
        self.assertEqual(restore_words([], self.FakeModel()), [])
        self.assertEqual(restore_words([], self.BatchingModel()), [])

    def test_the_default_chunk_fits_inside_the_encoder(self):
        self.assertLessEqual(CHUNK_WORDS, 512)


class ReferenceCorpusTests(unittest.TestCase):
    """The eval set is built from repo files; guard that it still exists."""

    def test_the_repo_still_supplies_punctuated_vietnamese(self):
        samples = build_samples(load_reference_text())
        self.assertGreater(len(samples), 100)
        words = sum(len(s) for s in samples)
        self.assertGreater(words, 5000)
        self.assertGreater(sum(t.capitalised for s in samples for t in s), 200)
        self.assertGreater(sum(bool(t.punctuation) for s in samples for t in s), 200)


if __name__ == "__main__":
    unittest.main()
