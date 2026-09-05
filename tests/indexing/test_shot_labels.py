import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend import shot_labels
from scripts.indexing.suggest_shot_exclusions import (
    load_preview_anchors,
    load_preview_overlay_matches,
    preview_overlay_runs,
    suggest_opening_intro,
    suggest_video,
)


class ShotLabelTests(unittest.TestCase):
    def test_suggestions_are_inert_until_reviewed(self):
        bounds = np.array([[0, 9], [10, 19]])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shot_labels.replace_suggestions(
                "L21_V001", bounds,
                {0: {"reason": "intro", "confidence": 0.95}}, root,
            )
            self.assertEqual(shot_labels.excluded_shots("L21_V001", bounds, root), set())
            document = shot_labels.load("L21_V001", bounds, root)
            document = shot_labels.update(
                "L21_V001", bounds, document["revision"],
                [{"shot_index": 0, "state": "excluded", "reason": "intro"}],
                "owner@example.com", root,
            )
            self.assertEqual(shot_labels.excluded_shots("L21_V001", bounds, root), {0})
            self.assertEqual(document["labels"]["0"]["reviewed_by"], "owner@example.com")

    def test_changed_boundaries_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shot_labels.replace_suggestions(
                "L21_V001", [[0, 9]],
                {0: {"reason": "intro", "confidence": 1}}, root,
            )
            with self.assertRaises(shot_labels.LabelConflict):
                shot_labels.load("L21_V001", [[0, 10]], root)

    def test_structure_marks_preview_through_presenter_after_second_bumper(self):
        bounds = np.array([[0, 99], [100, 199], [200, 299], [300, 399], [400, 499]])
        vectors = {
            ("L21_V001", 0): np.array([0.0, 1.0]),
            ("L21_V001", 1): np.array([0.0, 1.0]),
            ("L21_V001", 2): np.array([0.0, 1.0]),
            ("L21_V001", 3): np.array([1.0, 0.0]),
            ("L21_V001", 4): np.array([0.5, 0.5]),
        }
        suggestions = suggest_video(
            "L21_V001", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.0, 1.0]),
        )
        self.assertEqual(set(suggestions), {0, 1, 2, 3})
        self.assertEqual(suggestions[3]["reason"], "presenter")

    def test_transcript_anchor_marks_short_montage_before_presenter(self):
        bounds = np.array([[index * 20, index * 20 + 19] for index in range(12)])
        vectors = {
            ("L21_V001", index): np.array([1.0, 0.0]) if index == 11
            else np.array([0.0, 1.0])
            for index in range(12)
        }
        suggestions = suggest_video(
            "L21_V001", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.0, -1.0]),
            preview_anchors=[1.0],
        )
        self.assertEqual(
            {index for index, label in suggestions.items() if label["reason"] == "preview"},
            set(range(1, 11)),
        )
        self.assertEqual(suggestions[11]["reason"], "presenter")

    def test_transcript_anchor_rejects_slow_sequence(self):
        bounds = np.array([[index * 50, index * 50 + 49] for index in range(6)])
        vectors = {
            ("L21_V001", index): np.array([1.0, 0.0]) if index == 5
            else np.array([0.0, 1.0])
            for index in range(6)
        }
        suggestions = suggest_video(
            "L21_V001", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.0, -1.0]),
            preview_anchors=[1.0],
        )
        self.assertEqual(set(suggestions), {5})

    def test_presenter_variants_use_best_prototype_and_drop_weak_isolated_match(self):
        bounds = np.array([[0, 99], [100, 199]])
        vectors = {
            ("L21_V002", 0): np.array([0.8, 0.6]),
            ("L21_V002", 1): np.array([0.0, 1.0]),
        }
        suggestions = suggest_video(
            "L21_V002", bounds, 10, vectors,
            presenter=np.array([[1.0, 0.0], [0.0, 1.0]]), intro=None,
        )
        self.assertEqual(set(suggestions), {1})
        self.assertEqual(suggestions[1]["reason"], "presenter")

    def test_weak_presenter_requires_stronger_layout_match(self):
        bounds = np.array([[0, 99], [100, 199], [200, 299]])
        weak_vector = np.array([0.775, np.sqrt(1 - 0.775 ** 2)])
        suggestions = suggest_video(
            "L21_V008", bounds, 10,
            vectors={
                ("L21_V008", 0): np.array([1.0, 0.0]),
                ("L21_V008", 1): weak_vector,
                ("L21_V008", 2): weak_vector,
            },
            presenter=np.array([1.0, 0.0]), intro=None,
            layouts={
                ("L21_V008", 0): np.array([1.0, 0.0]),
                ("L21_V008", 1): np.array([0.7, np.sqrt(1 - 0.7 ** 2)]),
                ("L21_V008", 2): np.array([0.62, np.sqrt(1 - 0.62 ** 2)]),
            },
        )
        self.assertEqual(set(suggestions), {0, 1})
        self.assertEqual(suggestions[1]["reason"], "presenter")

    def test_opening_intro_uses_fixed_prefix(self):
        bounds = np.array([[index * 10, index * 10 + 9] for index in range(4)])
        suggestions = suggest_opening_intro(
            "L26_V001", bounds, 10, {}, last_shot=2,
        )
        self.assertEqual(set(suggestions), {0, 1, 2})
        self.assertTrue(all(label["reason"] == "intro" for label in suggestions.values()))

    def test_opening_intro_anchor_fails_closed_below_threshold(self):
        bounds = np.array([[0, 9], [10, 19], [20, 29]])
        vectors = {
            ("L30_V001", 0): np.array([0.5, np.sqrt(1 - 0.5 ** 2)]),
            ("L30_V001", 1): np.array([0.96, np.sqrt(1 - 0.96 ** 2)]),
            ("L30_V001", 2): np.array([0.7, np.sqrt(1 - 0.7 ** 2)]),
        }
        suggestions = suggest_opening_intro(
            "L30_V001", bounds, 10, vectors,
            prototypes=np.array([[1.0, 0.0]]), max_sec=2, threshold=0.9,
        )
        self.assertEqual(set(suggestions), {0, 1})
        self.assertEqual(
            set(suggest_opening_intro(
                "L30_V001", bounds, 10, vectors,
                prototypes=np.array([[1.0, 0.0]]), max_sec=2, threshold=0.9,
                include_anchor=False,
            )),
            {0},
        )
        self.assertEqual(
            suggest_opening_intro(
                "L30_V001", bounds, 10, vectors,
                prototypes=np.array([[0.0, 1.0]]), max_sec=2, threshold=0.99,
            ),
            {},
        )

    def test_preview_structure_promotes_lower_confidence_presenter_boundary(self):
        bounds = np.array([[index * 20, index * 20 + 19] for index in range(7)])
        vectors = {
            ("L21_V001", index): np.array([0.8, 0.6]) if index == 5
            else np.array([1.0, 0.0]) if index == 6
            else np.array([0.0, 1.0])
            for index in range(7)
        }
        suggestions = suggest_video(
            "L21_V001", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=None,
            preview_anchors=[1.0],
            layouts={
                ("L21_V001", 5): np.array([1.0, 0.0]),
                ("L21_V001", 6): np.array([1.0, 0.0]),
            },
        )
        self.assertEqual(suggestions[5]["reason"], "presenter")
        self.assertEqual(
            {index for index, label in suggestions.items() if label["reason"] == "preview"},
            {1, 2, 3, 4},
        )

    def test_intro_contrast_does_not_call_bumper_presenter(self):
        bounds = np.array([[0, 99], [100, 199]])
        vectors = {
            ("L21_V001", 0): np.array([0.8, 0.6]),
            ("L21_V001", 1): np.array([1.0, 0.0]),
        }
        suggestions = suggest_video(
            "L21_V001", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.8, 0.6]),
        )
        self.assertEqual(suggestions[0]["reason"], "intro")
        self.assertEqual(suggestions[1]["reason"], "presenter")

    def test_high_confidence_bumper_is_suggested_outside_opening(self):
        bounds = np.array([[0, 99], [100, 199], [1000, 1099]])
        vectors = {
            ("L21_V002", 0): np.array([1.0, 0.0]),
            ("L21_V002", 1): np.array([1.0, 0.0]),
            ("L21_V002", 2): np.array([0.0, 1.0]),
        }
        suggestions = suggest_video(
            "L21_V002", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.0, 1.0]),
        )
        self.assertEqual(suggestions[2]["reason"], "intro")

    def test_bumper_marks_following_short_montage_before_presenter(self):
        bounds = np.array([[index * 20, index * 20 + 19] for index in range(6)])
        vectors = {
            ("L21_V002", 0): np.array([0.0, 1.0]),
            **{
                ("L21_V002", index): np.array([0.0, -1.0])
                for index in range(1, 5)
            },
            ("L21_V002", 5): np.array([1.0, 0.0]),
        }
        suggestions = suggest_video(
            "L21_V002", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.0, 1.0]),
        )
        self.assertEqual(suggestions[0]["reason"], "intro")
        self.assertEqual(
            {index for index, label in suggestions.items() if label["reason"] == "preview"},
            {1, 2, 3, 4},
        )

    def test_visual_overlay_joins_one_gap_and_adjacent_bumper(self):
        bounds = np.array([[index * 20, index * 20 + 19] for index in range(9)])
        vectors = {
            ("L21_V003", index): np.array([0.0, -1.0]) for index in range(9)
        }
        vectors[("L21_V003", 7)] = np.array([np.sqrt(1 - 0.85 ** 2), 0.85])
        vectors[("L21_V003", 8)] = np.array([1.0, 0.0])
        suggestions = suggest_video(
            "L21_V003", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.0, 1.0]),
            preview_max_sec=0,
            preview_overlay_matches={
                ("L21_V003", index) for index in (2, 3, 5, 6)
            },
        )
        self.assertEqual(
            {index for index, label in suggestions.items() if label["reason"] == "preview"},
            {2, 3, 4, 5, 6},
        )
        self.assertEqual(suggestions[7]["reason"], "intro")
        self.assertEqual(suggestions[8]["reason"], "presenter")

    def test_visual_overlay_rejects_short_or_slow_runs(self):
        fast = np.array([[index * 20, index * 20 + 19] for index in range(4)])
        slow = np.array([[index * 40, index * 40 + 39] for index in range(4)])
        self.assertEqual(preview_overlay_runs({0, 1, 2}, fast, 10), [])
        self.assertEqual(preview_overlay_runs({0, 1, 2, 3}, slow, 10), [])

    def test_adjacent_bumper_does_not_override_presenter(self):
        bounds = np.array([[index * 20, index * 20 + 19] for index in range(5)])
        vectors = {
            ("L21_V002", index): np.array([0.0, -1.0]) for index in range(4)
        }
        vectors[("L21_V002", 4)] = np.array([1.0, 0.0])
        suggestions = suggest_video(
            "L21_V002", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=np.array([0.8, 0.6]),
            preview_max_sec=0,
            preview_overlay_matches={
                ("L21_V002", index) for index in range(4)
            },
        )
        self.assertEqual(suggestions[4]["reason"], "presenter")

    def test_preview_overlay_requires_both_fixed_red_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "L21" / "L21_V003"
            target.mkdir(parents=True)
            complete = np.zeros((720, 1280, 3), dtype=np.uint8)
            complete[43:172, :281] = (0, 0, 255)
            complete[518:662] = (0, 0, 255)
            top_only = complete.copy()
            top_only[518:662] = 0
            cv2.imwrite(str(target / "000001.png"), complete)
            cv2.imwrite(str(target / "000002.png"), top_only)
            self.assertEqual(
                load_preview_overlay_matches(
                    root,
                    {("L21_V003", 0): 1, ("L21_V003", 1): 2},
                ),
                {("L21_V003", 0)},
            )

    def test_short_final_outro_uses_its_own_reason(self):
        bounds = np.array([[0, 19], [20, 39], [40, 79]])
        vectors = {
            ("L21_V006", 0): np.array([0.0, 1.0]),
            ("L21_V006", 1): np.array([1.0, 0.0]),
            ("L21_V006", 2): np.array([0.0, 1.0]),
        }
        suggestions = suggest_video(
            "L21_V006", bounds, 10, vectors,
            presenter=np.array([1.0, 0.0]), intro=None,
            outro=np.array([0.0, 1.0]),
        )
        self.assertEqual(suggestions[2]["reason"], "outro")
        self.assertNotEqual(suggestions.get(0, {}).get("reason"), "outro")

        slow_bounds = np.array([[0, 19], [20, 119]])
        slow_vectors = {
            ("L21_V006", 0): np.array([1.0, 0.0]),
            ("L21_V006", 1): np.array([0.0, 1.0]),
        }
        slow = suggest_video(
            "L21_V006", slow_bounds, 10, slow_vectors,
            presenter=np.array([1.0, 0.0]), intro=None,
            outro=np.array([0.0, 1.0]),
        )
        self.assertNotIn(1, slow)

    def test_load_preview_anchors_and_missing_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "transcript.parquet"
            pq.write_table(pa.Table.from_pylist([
                {"audio_key": "L21_V001", "offset": 738.0, "token": " trong"},
                {"audio_key": "L21_V001", "offset": 738.2, "token": " phần"},
                {"audio_key": "L21_V001", "offset": 738.5, "token": " sau"},
                {"audio_key": "L22_V001", "offset": 10.0, "token": " phần"},
                {"audio_key": "L22_V001", "offset": 10.2, "token": " sau"},
            ]), path)
            self.assertEqual(
                load_preview_anchors(path, {"L21_V001"}),
                {"L21_V001": [738.2]},
            )
            self.assertEqual(load_preview_anchors(path.with_name("missing.parquet"), {"L21_V001"}), {})


if __name__ == "__main__":
    unittest.main()
