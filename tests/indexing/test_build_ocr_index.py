import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from backend import frame_maps
from backend import shots
from scripts.indexing.build_ocr_index import _group_by_video, _pick_anchor, iter_documents


def raw(text, x0, y0, x1, y1, conf=0.95):
    return {"text": text, "box": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], "conf": conf}


class GroupByVideoTests(unittest.TestCase):
    def test_groups_and_sorts_by_frame_idx(self):
        source = {
            "L21/L21_V001/50": {"lines": [raw("b", 0, 0, 10, 10)]},
            "L21/L21_V001/10": {"lines": [raw("a", 0, 0, 10, 10)]},
            "L21/L21_V002/0": {"lines": []},
        }
        grouped = _group_by_video(source)
        self.assertEqual(
            [r["frame_idx"] for r in grouped[("L21", "L21_V001")]], [10, 50]
        )
        self.assertEqual(len(grouped[("L21", "L21_V002")]), 1)


class PickAnchorTests(unittest.TestCase):
    def test_picks_the_keyframe_with_most_weighted_text(self):
        records = [
            {"frame_idx": 0, "lines": [raw("hi", 0, 0, 10, 10, conf=0.9)]},
            {"frame_idx": 1, "lines": [raw("a much longer line of text", 0, 0, 10, 10, conf=0.9)]},
        ]
        self.assertEqual(_pick_anchor(records)["frame_idx"], 1)

    def test_returns_none_when_every_keyframe_is_empty(self):
        records = [{"frame_idx": 0, "lines": []}, {"frame_idx": 1, "lines": []}]
        self.assertIsNone(_pick_anchor(records))


class IterDocumentsTests(unittest.TestCase):
    """A synthetic newscast: fixed logo, ticking clock, a scrolling ticker that
    should never reach the index, and a real on-screen banner tied to the shot's
    own story that should survive suppression and get indexed."""

    def _write_source(self, tmp_path: Path) -> Path:
        # The banner changes every 3-4 shots, as it would across real story
        # segments -- no single value appears in >50% of the video's frames, so
        # it doesn't trip the constant-logo rule, and its low novelty (3
        # distinct values over 10 occupied frames) keeps it out of the
        # scrolling-ticker rule too. The ticker's text is unique every frame,
        # which is what should get it dropped.
        banners = [
            "AN GIANG: TRIET PHA DUONG DAY TIEN GIA",
            "AN GIANG: TRIET PHA DUONG DAY TIEN GIA",
            "AN GIANG: TRIET PHA DUONG DAY TIEN GIA",
            "CAN THO: LU LUT GAY THIET HAI",
            "CAN THO: LU LUT GAY THIET HAI",
            "CAN THO: LU LUT GAY THIET HAI",
            "HA NOI: GIAO THONG UN TAC",
            "HA NOI: GIAO THONG UN TAC",
            "HA NOI: GIAO THONG UN TAC",
            "HA NOI: GIAO THONG UN TAC",
        ]
        source = {}
        for shot_idx in range(10):
            frame_idx = shot_idx * 25
            lines = [
                raw("HTV7HD", 1180, 10, 1260, 40),                       # station logo
                raw(f"06:39:{shot_idx:02d}", 1150, 45, 1260, 75),        # clock
                raw(f"tin thời sự số {shot_idx}", 100, 650, 900, 690),   # scrolling ticker
                raw(banners[shot_idx], 50, 590, 950, 630),               # story banner
            ]
            source[f"L99/L99_V001/{frame_idx}"] = {"lines": lines}
        path = tmp_path / "ocr_source.json"
        path.write_text(json.dumps(source))
        return path

    def _patched(self, tmp_path: Path):
        shot_index = shots.ShotIndex(
            fps=25.0,
            bounds=np.array([[i * 25, i * 25 + 24] for i in range(10)]),
        )
        mapping = frame_maps.FrameMap(
            pts=np.arange(0, 300, dtype=np.int64), time_base_num=1, time_base_den=25,
        )
        return (
            patch("scripts.indexing.build_ocr_index.shots.load", return_value=shot_index),
            patch("scripts.indexing.build_ocr_index.frame_maps.load", return_value=mapping),
        )

    def test_ticker_and_logo_are_dropped_but_the_story_banner_survives(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = self._write_source(Path(tmp))
            load_shots, load_frame_map = self._patched(Path(tmp))
            with load_shots, load_frame_map:
                docs = dict(iter_documents(source_path))

        texts = " ".join(doc["text"] for doc in docs.values())
        self.assertNotIn("HTV7HD", texts)
        self.assertNotIn("06:39", texts)
        self.assertNotIn("tin thời sự", texts)
        self.assertIn("AN GIANG", texts)

    def test_indexed_documents_keep_the_existing_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            source_path = self._write_source(Path(tmp))
            load_shots, load_frame_map = self._patched(Path(tmp))
            with load_shots, load_frame_map:
                doc_id, doc = next(iter(iter_documents(source_path)))

        self.assertEqual(
            set(doc),
            {"video_id", "group", "shot_idx", "frame_idx", "start_sec", "end_sec",
             "anchor_sec", "text"},
        )
        self.assertEqual(doc["video_id"], "L99_V001")
        self.assertEqual(doc["group"], "L99")

    def test_shot_with_only_ticker_and_logo_is_skipped(self):
        source = {"L99/L99_V001/0": {"lines": [
            raw("HTV7HD", 1180, 10, 1260, 40),
            raw("tin thời sự số 0", 100, 650, 900, 690),
        ]}}
        # Pad the rest of the video so suppression's frequency stats kick in.
        # The real content's box moves between shots (like scene text does),
        # so no single cell builds up the occupancy the ticker rule looks for.
        for i in range(1, 10):
            x = 50 + (i % 4) * 200
            y = 300 + (i % 3) * 100
            source[f"L99/L99_V001/{i * 25}"] = {"lines": [
                raw("HTV7HD", 1180, 10, 1260, 40),
                raw(f"tin thời sự số {i}", 100, 650, 900, 690),
                raw(f"nội dung riêng {i}", x, y, x + 200, y + 40),
            ]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source))
            load_shots, load_frame_map = self._patched(Path(tmp))
            with load_shots, load_frame_map:
                docs = dict(iter_documents(path))

        self.assertNotIn("L99_V001:0", docs)
        self.assertIn("L99_V001:1", docs)


class GroupIgnoreRegionTests(unittest.TestCase):
    """The fixed ticker-band mask for L21/L22 (GROUP_IGNORE_REGIONS), independent
    of suppress_static_overlays's frequency statistics -- it should catch a ticker
    even in a two-keyframe video, where min_frames=8 leaves the statistical rule
    with nothing to do."""

    def _patched(self, bounds):
        shot_index = shots.ShotIndex(fps=25.0, bounds=np.array(bounds))
        mapping = frame_maps.FrameMap(
            pts=np.arange(0, max(e for _, e in bounds) + 10, dtype=np.int64),
            time_base_num=1, time_base_den=25,
        )
        return (
            patch("scripts.indexing.build_ocr_index.shots.load", return_value=shot_index),
            patch("scripts.indexing.build_ocr_index.frame_maps.load", return_value=mapping),
        )

    def test_l22_drops_the_ticker_band_even_on_a_short_video(self):
        source = {
            "L22/L22_V999/0": {"lines": [raw("Quảng Nam năm 2024", 7, 653, 749, 692)]},
            "L22/L22_V999/25": {"lines": [raw("AN GIANG: TRIET PHA", 50, 590, 950, 626)]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source))
            load_shots, load_frame_map = self._patched([[0, 49]])
            with load_shots, load_frame_map:
                docs = dict(iter_documents(path))

        texts = " ".join(doc["text"] for doc in docs.values())
        self.assertNotIn("Quảng Nam", texts)
        self.assertIn("AN GIANG", texts)

    def test_other_groups_are_not_masked(self):
        source = {"L23/L23_V999/0": {"lines": [raw("Quảng Nam năm 2024", 7, 653, 749, 692)]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source))
            load_shots, load_frame_map = self._patched([[0, 49]])
            with load_shots, load_frame_map:
                docs = dict(iter_documents(path))

        texts = " ".join(doc["text"] for doc in docs.values())
        self.assertIn("Quảng Nam", texts)

    def test_l23_drops_the_top_right_sports_bug(self):
        source = {
            "L23/L23_V999/0": {"lines": [
                raw("HTV", 1108, 32, 1228, 82),
                raw("chặng đua thứ nhất", 200, 300, 700, 340),
            ]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source))
            load_shots, load_frame_map = self._patched([[0, 49]])
            with load_shots, load_frame_map:
                docs = dict(iter_documents(path))

        texts = " ".join(doc["text"] for doc in docs.values())
        self.assertNotIn("HTV", texts)
        self.assertIn("chặng đua thứ nhất", texts)

    def test_l26_drops_the_top_left_online_bug(self):
        source = {
            "L26/L26_V999/0": {"lines": [
                raw("HTV", 9, 2, 129, 50),
                raw("Online", 3, 49, 117, 87),
                raw("nước cốt hành tỏi", 700, 300, 1000, 340),
            ]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source))
            load_shots, load_frame_map = self._patched([[0, 49]])
            with load_shots, load_frame_map:
                docs = dict(iter_documents(path))

        texts = " ".join(doc["text"] for doc in docs.values())
        self.assertNotIn("HTV", texts)
        self.assertNotIn("Online", texts)
        self.assertIn("nước cốt hành tỏi", texts)

    def test_l25_is_not_masked(self):
        # The watermark's footprint overlaps the real article text almost
        # entirely on this video -- a region mask would risk deleting the
        # content it exists to preserve, so L25 gets no fixed region at all.
        source = {"L25/L25_V999/0": {"lines": [
            raw("THANHNIEN", 406, 299, 882, 419),
            raw("một đoạn văn thật của bài báo", 413, 315, 1134, 345),
        ]}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(source))
            load_shots, load_frame_map = self._patched([[0, 49]])
            with load_shots, load_frame_map:
                docs = dict(iter_documents(path))

        texts = " ".join(doc["text"] for doc in docs.values())
        self.assertIn("THANHNIEN", texts)
        self.assertIn("một đoạn văn thật của bài báo", texts)


if __name__ == "__main__":
    unittest.main()
