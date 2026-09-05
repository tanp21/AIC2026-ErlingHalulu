import importlib.util
import json
from unittest.mock import patch
import unittest
from pathlib import Path

from scripts.ocr.scoring import (
    ConfigReport,
    FrameScore,
    character_error_rate,
    contains,
    fold,
    levenshtein,
    normalize,
    rank_configs,
    score_frame,
    strip_spans,
    word_recall,
)


class UnreadableExclusionTests(unittest.TestCase):
    """Text proven unreadable must cost nothing, in either direction."""

    ENTRY = {
        "id": "k17",
        "full_text": "KHU DU LỊCH QUỐC GIA Mộc Châu HOÀNG SA TRƯỜNG SA",
        "unreadable": ["HOÀNG SA", "TRƯỜNG SA"],
        "required": ["KHU DU LỊCH QUỐC GIA"],
    }

    def test_strip_spans_removes_and_normalises(self):
        self.assertEqual(strip_spans("a HOÀNG SA b", ["HOÀNG SA"]), "a b")

    def test_strip_spans_without_spans_is_just_normalise(self):
        self.assertEqual(strip_spans("  A   b "), "a b")

    def test_missing_the_excluded_text_is_not_penalised(self):
        perfect = "KHU DU LỊCH QUỐC GIA Mộc Châu"
        self.assertEqual(score_frame(self.ENTRY, perfect).cer, 0.0)

    def test_reading_the_excluded_text_is_not_penalised_either(self):
        # The asymmetric implementation — stripping the reference only — would
        # score this worse than the line above, penalising a correct read.
        also_perfect = "KHU DU LỊCH QUỐC GIA Mộc Châu HOÀNG SA TRƯỜNG SA"
        self.assertEqual(score_frame(self.ENTRY, also_perfect).cer, 0.0)

    def test_real_errors_outside_the_exclusion_still_count(self):
        wrong = "KHU DU LỊCH QUỐC GIA Mọc Chau"
        self.assertGreater(score_frame(self.ENTRY, wrong).cer, 0.0)

    def test_word_recall_ignores_excluded_text(self):
        self.assertEqual(
            score_frame(self.ENTRY, "KHU DU LỊCH QUỐC GIA Mộc Châu").word_recall, 1.0)

    def test_entry_without_the_field_is_unaffected(self):
        entry = {k: v for k, v in self.ENTRY.items() if k != "unreadable"}
        self.assertGreater(score_frame(entry, "KHU DU LỊCH QUỐC GIA Mộc Châu").cer, 0.0)

GROUND_TRUTH = Path("benchmarks/ocr/ocr_ground_truth.json")


class NormalizationTests(unittest.TestCase):
    def test_normalize_collapses_whitespace_and_case(self):
        self.assertEqual(normalize("  HỎA\n\tHỒNG  "), "hỏa hồng")

    def test_normalize_preserves_diacritics(self):
        self.assertNotEqual(normalize("TẢO"), normalize("TAO"))

    def test_normalize_unifies_composed_and_decomposed_forms(self):
        # NFC vs NFD spellings of the same word must compare equal, otherwise an
        # engine would be penalised purely for its Unicode output convention.
        self.assertEqual(normalize("Ả"), normalize("Ả"))

    def test_fold_strips_diacritics(self):
        self.assertEqual(fold("TẢO"), fold("TAO"))

    def test_fold_maps_vietnamese_d_stroke(self):
        # Đ has no combining-mark decomposition; without an explicit mapping it
        # would survive diacritic stripping and never match D.
        self.assertEqual(fold("ĐỊA"), "dia")


class DistanceTests(unittest.TestCase):
    def test_levenshtein_identical(self):
        self.assertEqual(levenshtein("thần", "thần"), 0)

    def test_levenshtein_counts_edits(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_levenshtein_empty_operands(self):
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)

    def test_cer_zero_for_exact_match(self):
        self.assertEqual(character_error_rate("Nhân bánh cuốn", "nhân  bánh cuốn"), 0.0)

    def test_cer_empty_reference_is_zero(self):
        self.assertEqual(character_error_rate("", "spurious output"), 0.0)

    def test_cer_can_exceed_one_when_hallucinating(self):
        self.assertGreater(character_error_rate("ab", "ab" + "x" * 20), 1.0)


class WordRecallTests(unittest.TestCase):
    def test_order_independent(self):
        # The whole point: reordered blocks must not be punished here, so that
        # CER and word recall together separate ordering from recognition.
        self.assertEqual(word_recall("hỏa hồng nhật tảo", "tảo nhật hồng hỏa"), 1.0)

    def test_counts_multiplicity(self):
        self.assertAlmostEqual(word_recall("a a b", "a b"), 2 / 3)

    def test_empty_reference_is_full_recall(self):
        self.assertEqual(word_recall("", "anything"), 1.0)


class ContainsTests(unittest.TestCase):
    def test_diacritic_sensitive_by_default(self):
        self.assertFalse(contains("HOA HONG", "HỎA"))

    def test_diacritic_blind_opt_in(self):
        self.assertTrue(contains("HOA HONG", "HỎA", diacritic_blind=True))


class ScoreFrameTests(unittest.TestCase):
    def _entry(self):
        return {
            "id": "pillar",
            "acceptance": True,
            "tags": ["two-column"],
            "required": ["HỎA", "HỒNG", "THẦN"],
            "required_phrases": ["HỎA HỒNG NHẬT TẢO"],
            "full_text": "HỎA HỒNG NHẬT TẢO THẦN",
        }

    def test_perfect_output_passes(self):
        score = score_frame(self._entry(), "HỎA HỒNG NHẬT TẢO THẦN", seconds=0.2)
        self.assertTrue(score.passed)
        self.assertEqual(score.required_recall, 1.0)
        self.assertEqual(score.phrase_recall, 1.0)
        self.assertEqual(score.cer, 0.0)

    def test_missing_required_string_fails(self):
        score = score_frame(self._entry(), "HỎA HỒNG NHẬT TẢO")
        self.assertFalse(score.passed)
        self.assertEqual(score.required_missing, ("THẦN",))

    def test_interleaved_columns_pass_required_but_fail_phrase(self):
        # This is the frame-2 regression in miniature: every word is detected,
        # so required recall is perfect, but reading order is wrong so the
        # phrase check must catch it.
        score = score_frame(self._entry(), "HỎA THẦN HỒNG NHẬT TẢO")
        self.assertEqual(score.required_recall, 1.0)
        self.assertEqual(score.phrase_missing, ("HỎA HỒNG NHẬT TẢO",))
        self.assertFalse(score.passed)

    def test_folded_count_exposes_diacritic_only_failures(self):
        score = score_frame(self._entry(), "HOA HONG NHAT TAO THAN")
        self.assertEqual(score.required_found, 0)
        self.assertEqual(score.required_found_folded, 3)

    def test_no_requirements_is_vacuously_full_recall(self):
        score = score_frame({"id": "blank"}, "")
        self.assertEqual(score.required_recall, 1.0)
        self.assertEqual(score.phrase_recall, 1.0)
        self.assertTrue(score.passed)


def _score(frame_id, *, acceptance=True, passed=True, seconds=0.1, cer=0.0):
    missing = () if passed else ("X",)
    return FrameScore(
        frame_id=frame_id,
        acceptance=acceptance,
        required_total=1,
        required_found=1 if passed else 0,
        required_missing=missing,
        required_found_folded=1,
        phrase_total=0,
        phrase_found=0,
        phrase_missing=(),
        phrase_found_folded=0,
        cer=cer,
        word_recall=1.0,
        seconds=seconds,
        tags=(),
    )


class ConfigReportTests(unittest.TestCase):
    def test_acceptance_gate_requires_every_acceptance_frame(self):
        report = ConfigReport("a", [_score("f1"), _score("f2", passed=False)])
        self.assertFalse(report.acceptance_passed)

    def test_acceptance_gate_ignores_non_acceptance_failures(self):
        report = ConfigReport(
            "a", [_score("f1"), _score("f2", acceptance=False, passed=False)]
        )
        self.assertTrue(report.acceptance_passed)

    def test_no_acceptance_frames_does_not_silently_pass(self):
        report = ConfigReport("a", [_score("f1", acceptance=False)])
        self.assertFalse(report.acceptance_passed)

    def test_frames_per_second(self):
        report = ConfigReport("a", [_score("f1", seconds=0.25), _score("f2", seconds=0.25)])
        self.assertAlmostEqual(report.frames_per_second, 4.0)

    def test_summary_lists_failures(self):
        summary = ConfigReport("a", [_score("f1", passed=False)]).summary()
        self.assertEqual(summary["failures"]["f1"]["required"], ["X"])


class AcceptanceGateTests(unittest.TestCase):
    """The gate must distinguish a wrong tone mark from text that was never read."""

    # Ten required strings, so a single error costs exactly 10% and lands on the
    # 0.90 threshold — the same shape as the real acceptance set, where 35
    # required strings across three frames yield 0.905.
    PERFECT = "HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA KIẾM QUỶ THẦN"

    def _report(self, text):
        entry = {
            "id": "pillar",
            "acceptance": True,
            "required": self.PERFECT.split(),
            "required_phrases": [self.PERFECT],
            "full_text": self.PERFECT,
        }
        return ConfigReport("c", [score_frame(entry, text, 0.1)])

    def test_perfect_output_passes(self):
        self.assertTrue(self._report(self.PERFECT).acceptance_passed)

    def test_single_diacritic_failure_passes(self):
        # QUỶ read as QUỲ: found and correctly placed, so the asciifolding
        # sub-field still retrieves it at query time.
        report = self._report(self.PERFECT.replace("QUỶ", "QUỲ"))
        summary = report.summary()
        self.assertAlmostEqual(summary["acceptance_required_recall"], 0.9)
        self.assertEqual(summary["acceptance_folded_recall"], 1.0)
        self.assertEqual(summary["folded_phrase_recall"], 1.0)
        self.assertTrue(report.acceptance_passed)

    def test_missing_text_fails_even_though_the_rest_is_perfect(self):
        # Not a diacritic problem: the word is absent, so folding cannot save it
        # and no query-time analyzer can recover it.
        report = self._report(self.PERFECT.replace("QUỶ ", ""))
        self.assertLess(report.summary()["acceptance_folded_recall"], 1.0)
        self.assertFalse(report.acceptance_passed)

    def test_scrambled_reading_order_fails(self):
        # Every word present and spelled correctly, but the phrase is broken up —
        # the two-column interleaving failure mode.
        scrambled = " ".join(reversed(self.PERFECT.split()))
        report = self._report(scrambled)
        self.assertEqual(report.summary()["acceptance_required_recall"], 1.0)
        self.assertEqual(report.summary()["acceptance_folded_recall"], 1.0)
        self.assertEqual(report.summary()["folded_phrase_recall"], 0.0)
        self.assertFalse(report.acceptance_passed)

    def test_two_tone_errors_fails(self):
        report = self._report(
            self.PERFECT.replace("QUỶ", "QUỲ").replace("TẢO", "TÁO")
        )
        self.assertEqual(report.summary()["acceptance_folded_recall"], 1.0)
        self.assertFalse(report.acceptance_passed)

    def test_thresholds_are_overridable(self):
        report = self._report(self.PERFECT.replace("QUỶ", "QUỲ"))
        self.assertTrue(report.acceptance_passed)
        self.assertFalse(report.gate(min_required=1.0))


class RankConfigsTests(unittest.TestCase):
    def test_correctness_beats_speed(self):
        slow_correct = ConfigReport("slow", [_score("f1", seconds=1.0)])
        fast_wrong = ConfigReport("fast", [_score("f1", passed=False, seconds=0.01)])
        self.assertEqual(
            [r.config_id for r in rank_configs([fast_wrong, slow_correct])],
            ["slow", "fast"],
        )

    def test_min_fps_demotes_correct_but_too_slow(self):
        fast = ConfigReport("fast", [_score("f1", seconds=0.05)])   # 20 fps
        slow = ConfigReport("slow", [_score("f1", seconds=10.0)])   # 0.1 fps
        self.assertEqual(
            [r.config_id for r in rank_configs([slow, fast], min_fps=5.0)],
            ["fast", "slow"],
        )

    def test_ties_broken_by_throughput(self):
        a = ConfigReport("a", [_score("f1", seconds=0.5)])
        b = ConfigReport("b", [_score("f1", seconds=0.1)])
        self.assertEqual([r.config_id for r in rank_configs([a, b])], ["b", "a"])


class GroundTruthFileTests(unittest.TestCase):
    """The eval set is a checked-in artifact; guard its shape and its promises."""

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))

    def test_has_the_acceptance_frames(self):
        # The original three hand-labelled frames, plus the 2026-09 shot audit's
        # holdout extension (recipe-card motion-blur, charity banner, finish arch).
        acceptance = {f["id"] for f in self.data["frames"] if f.get("acceptance")}
        self.assertEqual(
            acceptance,
            {
                "gauge-digits", "pillar-couplet", "recipe-card",
                "recipe-card-blur-1", "recipe-card-blur-2", "recipe-card-blur-3",
                "charity-banner-1", "charity-banner-2", "charity-banner-3",
                "finish-arch-partial", "finish-arch-clear-1", "finish-arch-clear-2",
            },
        )

    def test_entries_are_well_formed(self):
        for frame in self.data["frames"]:
            with self.subTest(frame=frame["id"]):
                for key in ("path", "video_id", "frame_idx", "required", "full_text"):
                    self.assertIn(key, frame)
                self.assertIsInstance(frame["frame_idx"], int)
                # A path must agree with the video_id and frame_idx it claims,
                # or the bench would score the wrong image.
                self.assertTrue(
                    frame["path"].endswith(f"{frame['frame_idx']:06d}.png"),
                    frame["path"],
                )
                self.assertIn(f"/{frame['video_id']}/", frame["path"])

    def test_frame_ids_unique(self):
        ids = [f["id"] for f in self.data["frames"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_required_phrases_appear_in_full_text(self):
        # Guards against a typo in the eval set making a config unpassable.
        for frame in self.data["frames"]:
            for phrase in frame.get("required_phrases", []):
                with self.subTest(frame=frame["id"], phrase=phrase):
                    self.assertTrue(contains(frame["full_text"], phrase))

    def test_required_strings_appear_in_full_text(self):
        for frame in self.data["frames"]:
            for required in frame["required"]:
                with self.subTest(frame=frame["id"], required=required):
                    self.assertTrue(contains(frame["full_text"], required))

    def test_gauge_frame_requires_every_visible_depth_marking(self):
        gauge = next(f for f in self.data["frames"] if f["id"] == "gauge-digits")
        self.assertEqual(
            sorted(gauge["required"], key=int), ["8", "10", "12", "14", "16", "18", "20"]
        )

    def test_pillar_frame_requires_both_couplet_lines_in_order(self):
        pillar = next(f for f in self.data["frames"] if f["id"] == "pillar-couplet")
        self.assertEqual(
            pillar["required_phrases"],
            [
                "HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA",
                "KIẾM BẠT KIÊN GIANG KHẤP QUỶ THẦN",
            ],
        )


if __name__ == "__main__":
    unittest.main()


class CandidateGridTests(unittest.TestCase):
    """The 2026-09 shot audit added bare-callable engines alongside OcrConfig."""

    def test_sweep_grid_includes_the_alt_engines(self):
        from scripts.benchmarks.bench_ocr import candidate_configs
        from scripts.ocr.engines import OcrConfig

        grid = candidate_configs(sweep=True)
        for name in ("gemini", "paddleocr", "easyocr", "tesseract"):
            self.assertIn(name, grid)
            self.assertNotIsInstance(grid[name], OcrConfig)
        self.assertIsInstance(grid["baseline"], OcrConfig)

    def test_non_sweep_grid_is_just_the_baseline(self):
        from scripts.benchmarks.bench_ocr import candidate_configs

        self.assertEqual(list(candidate_configs(sweep=False)), ["baseline"])

    def test_evaluate_calls_a_bare_engine_directly(self):
        from scripts.benchmarks.bench_ocr import evaluate

        calls = []

        def fake_engine(image, *, ignore_regions=None):
            calls.append((image.shape, ignore_regions))
            return "hello"

        import numpy as np

        frame = {
            "id": "f1", "path": "does/not/matter.png", "video_id": "V1",
            "frame_idx": 0, "required": ["hello"], "full_text": "hello",
            "ignore_regions": [[0.0, 0.0, 0.1, 0.1]],
        }
        # evaluate() reads the frame from disk via DATA / entry["path"], so this
        # test stands up a real (tiny) file rather than mocking cv2.imread.
        import tempfile
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            import scripts.benchmarks.bench_ocr as bench_ocr

            image_path = Path(tmp) / "does" / "not"
            image_path.mkdir(parents=True)
            cv2.imwrite(str(image_path / "matter.png"), np.zeros((4, 4, 3), dtype="uint8"))
            original_data = bench_ocr.DATA
            bench_ocr.DATA = Path(tmp)
            try:
                report = evaluate(fake_engine, [frame], verbose=False)
            finally:
                bench_ocr.DATA = original_data

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], [[0.0, 0.0, 0.1, 0.1]])
        self.assertTrue(report.scores[0].passed)


@unittest.skipUnless(
    importlib.util.find_spec("vietocr") is not None,
    "needs the vietocr package (only installed in .venv-ocr)",
)
class VietocrConfigTests(unittest.TestCase):
    """The vendored config fix for vocr.vn's expired TLS certificate (2026-09)."""

    def test_merges_base_and_model_config_without_touching_the_network(self):
        from scripts.ocr.engines import _load_vietocr_config

        with patch(
            "vietocr.tool.utils.download_config",
            side_effect=AssertionError("should not hit the network"),
        ):
            cfg = _load_vietocr_config("vgg_seq2seq")

        self.assertIn("vocab", cfg)          # comes from base.yml
        self.assertEqual(cfg["backbone"], "vgg19_bn")  # comes from vgg-seq2seq.yml

    def test_falls_back_to_the_network_if_vendored_files_are_missing(self):
        from scripts.ocr import engines

        with patch.object(engines, "VIETOCR_CONFIG_DIR", Path("/does/not/exist")), \
                patch("vietocr.tool.config.Cfg.load_config_from_name") as fallback:
            fallback.return_value = {"sentinel": True}
            result = engines._load_vietocr_config("vgg_seq2seq")

        fallback.assert_called_once_with("vgg_seq2seq")
        self.assertEqual(result, {"sentinel": True})


class DegenerateOutputTests(unittest.TestCase):
    """Repetition loops are decoder failures, not readings."""

    def test_catches_a_real_collapse(self):
        from scripts.ocr.engines import is_degenerate
        # Verbatim from the signboard tagline at 0.26 confidence.
        self.assertTrue(is_degenerate(
            "Viêm lới Hiện nhườ, khư tri như thế giai thuy nhau thu thu thu thu thu thu"))

    def test_leaves_normal_text_alone(self):
        from scripts.ocr.engines import is_degenerate
        for text in ("KHU DU LỊCH QUỐC GIA Mộc Châu",
                     "20 18 16 14 12 10 8",
                     "HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA"):
            self.assertFalse(is_degenerate(text), text)

    def test_a_word_repeated_twice_is_not_a_collapse(self):
        from scripts.ocr.engines import is_degenerate
        # 'nước nêm nếm' misread as 'nêm nêm' is an ordinary error, not a loop.
        self.assertFalse(is_degenerate("ráo nước nêm nêm vừa ăn"))

    def test_short_input(self):
        from scripts.ocr.engines import is_degenerate
        self.assertFalse(is_degenerate(""))
        self.assertFalse(is_degenerate("thu thu"))

    def test_threshold_is_configurable(self):
        from scripts.ocr.engines import is_degenerate
        self.assertTrue(is_degenerate("a a a", min_repeats=3))
        self.assertFalse(is_degenerate("a a a", min_repeats=4))
