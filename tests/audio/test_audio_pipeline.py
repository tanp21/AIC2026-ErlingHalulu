import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts.audio.build_clean_transcripts import load_config, write
from scripts.audio.pipeline import RAW, CleanConfig, clean_corpus, sweep_configs
from scripts.audio.postprocess import Word, boilerplate_document_frequency


def words(text: str, *, nll: float = 0.1) -> list[Word]:
    return [Word(t, float(i), nll) for i, t in enumerate(text.split())]


class CleanConfigTests(unittest.TestCase):
    def test_the_default_config_changes_nothing(self):
        corpus = {"L21_V001": words("đàn cá heo bơi ngoài khơi")}
        self.assertEqual(clean_corpus(corpus, RAW), corpus)

    def test_round_trips_through_json(self):
        config = CleanConfig(max_nll=2.0, boilerplate_min_videos=5, numbers=True,
                             capu_model="dragonSwing/xlm-roberta-capu")
        self.assertEqual(CleanConfig.from_json(config.to_json()), config)

    def test_an_absent_threshold_round_trips_as_infinity(self):
        self.assertTrue(math.isinf(CleanConfig.from_json(RAW.to_json()).max_nll))

    def test_the_label_names_every_enabled_step(self):
        label = CleanConfig(max_nll=2.0, boilerplate_min_videos=5, numbers=True,
                            capu_model="dragonSwing/xlm-roberta-capu").label
        for expected in ("nll<=2", "boiler>=5v", "numbers", "capu:xlm-roberta"):
            self.assertIn(expected, label)

    def test_the_raw_config_labels_itself(self):
        self.assertEqual(RAW.label, "raw")

    def test_gemini_model_round_trips_through_json(self):
        config = CleanConfig(gemini_model="gemini-flash-lite-latest")
        self.assertEqual(CleanConfig.from_json(config.to_json()), config)

    def test_gemini_model_appears_in_the_label(self):
        label = CleanConfig(gemini_model="gemini-flash-lite-latest").label
        self.assertIn("gemini:gemini-flash-lite-latest", label)

    def test_capu_and_gemini_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            CleanConfig(capu_model="dragonSwing/xlm-roberta-capu",
                       gemini_model="gemini-flash-lite-latest")


class CleanCorpusTests(unittest.TestCase):
    SPONSOR = "chương trình được tài trợ bởi công ty cổ phần abc"

    def corpus(self):
        return {
            f"L2{i}_V001": words(f"nội dung riêng số {i} {self.SPONSOR} rồi kết thúc")
            for i in range(6)
        }

    def test_confidence_filtering_drops_only_what_exceeds_the_bound(self):
        corpus = {"L21_V001": [Word("tốt", 0.0, 0.1), Word("xấu", 1.0, 9.0)]}
        cleaned = clean_corpus(corpus, CleanConfig(max_nll=1.0))
        self.assertEqual([w.text for w in cleaned["L21_V001"]], ["tốt"])

    def test_boilerplate_removal_spares_the_unique_content(self):
        cleaned = clean_corpus(self.corpus(), CleanConfig(boilerplate_min_videos=5))
        for video_id, kept in cleaned.items():
            text = " ".join(w.text for w in kept)
            self.assertIn("nội dung riêng", text)
            self.assertNotIn("tài trợ", text)

    def test_a_precomputed_frequency_gives_the_same_answer(self):
        corpus = self.corpus()
        config = CleanConfig(boilerplate_min_videos=5)
        frequency = boilerplate_document_frequency(
            corpus, phrase_words=config.boilerplate_phrase_words
        )
        self.assertEqual(
            clean_corpus(corpus, config, document_frequency=frequency),
            clean_corpus(corpus, config),
        )

    def test_numbers_run_before_restoration_so_they_see_bare_words(self):
        # Restoration appends punctuation; "tư." would no longer parse as four.
        # The ordering in clean_corpus is what keeps this working.
        class Restorer:
            def restore(self, text):
                return " ".join(w.capitalize() for w in text.split()) + "."

        corpus = {"L21_V001": words("vào năm hai ngàn không trăm hai mươi tư")}
        cleaned = clean_corpus(
            corpus,
            CleanConfig(numbers=True, capu_model="fake/model"),
            capu=Restorer(),
        )
        self.assertEqual(" ".join(w.text for w in cleaned["L21_V001"]),
                         "Vào Năm 2024.")

    def test_gemini_restoration_runs_through_the_same_pipeline_slot_as_capu(self):
        class FakeGemini:
            model_id = "fake-gemini"
            concurrency = 1

            def __init__(self, cache_dir):
                self.cache_dir = cache_dir

            def restore_or_none(self, text):
                return " ".join(w.capitalize() for w in text.split()) + "."

        corpus = {"L21_V001": words("xin chào các bạn")}
        with tempfile.TemporaryDirectory() as directory:
            gemini = FakeGemini(Path(directory))
            cleaned = clean_corpus(
                corpus, CleanConfig(gemini_model="fake-gemini"), gemini=gemini,
            )
        self.assertEqual(" ".join(w.text for w in cleaned["L21_V001"]),
                         "Xin Chào Các Bạn.")

    def test_every_step_preserves_the_offset_of_what_survives(self):
        original = words("giữ nguyên mốc thời gian của từng từ")
        cleaned = clean_corpus({"L21_V001": original}, CleanConfig(max_nll=1.0))
        self.assertEqual([w.offset for w in cleaned["L21_V001"]],
                         [w.offset for w in original])


class SweepConfigTests(unittest.TestCase):
    def test_the_sweep_starts_from_the_untouched_baseline(self):
        self.assertEqual(sweep_configs()[0], RAW)

    def test_the_sweep_varies_one_axis_at_a_time(self):
        for config in sweep_configs()[1:]:
            enabled = sum([
                math.isfinite(config.max_nll),
                config.boilerplate_min_videos is not None,
                config.numbers,
                config.capu_model is not None,
            ])
            self.assertEqual(enabled, 1, config.label)


class LoadConfigTests(unittest.TestCase):
    def test_a_missing_file_yields_the_inert_config(self):
        self.assertEqual(load_config(Path("/nonexistent/audio_config.json")), RAW)

    def test_the_capu_winner_is_picked_up_from_its_own_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio_config.json"
            path.write_text(json.dumps({
                "clean": {"max_nll": None, "boilerplate_min_videos": None,
                          "boilerplate_phrase_words": 8, "numbers": True},
                "capu": {"model_id": "dragonSwing/xlm-roberta-capu"},
            }), encoding="utf-8")
            config = load_config(path)
        self.assertTrue(config.numbers)
        self.assertEqual(config.capu_model, "dragonSwing/xlm-roberta-capu")


class WriteTests(unittest.TestCase):
    def test_written_parquets_reload_with_the_source_schema(self):
        import pyarrow.parquet as pq

        corpus = {
            "L21_V001": words("xin chào các bạn"),
            "L21_V002": words("hôm nay trời đẹp"),
        }
        with tempfile.TemporaryDirectory() as directory:
            step = Path(directory) / "step_data_clean.parquet"
            transcripts = Path(directory) / "transcription_clean.parquet"
            write(corpus, step, transcripts)

            step_table = pq.read_table(step)
            self.assertEqual(step_table.column_names,
                             ["audio_key", "offset", "token", "log_prob"])
            self.assertEqual(step_table.num_rows, 8)

            transcript_table = pq.read_table(transcripts).to_pydict()
            self.assertEqual(transcript_table["audio_key"], ["L21_V001", "L21_V002"])
            self.assertEqual(transcript_table["text"][0], "xin chào các bạn")

    def test_the_written_tokens_rebuild_the_transcript(self):
        import pyarrow.parquet as pq

        from scripts.audio.postprocess import merge_tokens_to_words

        corpus = {"L21_V001": words("một hai ba bốn năm")}
        with tempfile.TemporaryDirectory() as directory:
            step = Path(directory) / "s.parquet"
            write(corpus, step, Path(directory) / "t.parquet")
            rows = pq.read_table(step).to_pylist()
        self.assertEqual(merge_tokens_to_words(rows), corpus["L21_V001"])


if __name__ == "__main__":
    unittest.main()
