import json
import tempfile
import unittest
from pathlib import Path

from scripts.audio.build_clean_transcripts import load_config, write
from scripts.audio.pipeline import RAW, CleanConfig, clean_corpus
from scripts.audio.postprocess import Word


def words(text: str, *, nll: float = 0.1) -> list[Word]:
    return [Word(t, float(i), nll) for i, t in enumerate(text.split())]


class CleanConfigTests(unittest.TestCase):
    def test_the_default_config_changes_nothing(self):
        corpus = {"L21_V001": words("đàn cá heo bơi ngoài khơi")}
        self.assertEqual(clean_corpus(corpus, RAW), corpus)

    def test_round_trips_through_json(self):
        config = CleanConfig(gemini_model="gemini-flash-lite-latest")
        self.assertEqual(CleanConfig.from_json(config.to_json()), config)

    def test_the_raw_config_labels_itself(self):
        self.assertEqual(RAW.label, "raw (no transform enabled)")

    def test_gemini_model_appears_in_the_label(self):
        label = CleanConfig(gemini_model="gemini-flash-lite-latest").label
        self.assertIn("gemini:gemini-flash-lite-latest", label)


class CleanCorpusTests(unittest.TestCase):
    def test_gemini_restoration_runs_through_the_pipeline(self):
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
        cleaned = clean_corpus({"L21_V001": original}, RAW)
        self.assertEqual([w.offset for w in cleaned["L21_V001"]],
                         [w.offset for w in original])


class LoadConfigTests(unittest.TestCase):
    def test_a_missing_file_yields_the_inert_config(self):
        self.assertEqual(load_config(Path("/nonexistent/audio_config.json")), RAW)

    def test_the_gemini_model_is_picked_up_from_its_own_section(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audio_config.json"
            path.write_text(json.dumps({
                "clean": {"gemini_model": "gemini-flash-lite-latest"},
            }), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.gemini_model, "gemini-flash-lite-latest")


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
