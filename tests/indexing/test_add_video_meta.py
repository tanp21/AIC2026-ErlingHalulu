import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.indexing import add_video_meta as addmeta


class DiscoverCandidatesTests(unittest.TestCase):
    def test_finds_mp4s_under_requested_groups_only(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for group, ids in (("M01", ["M01_V001", "M01_V002"]), ("S01", ["S01_V001"])):
                d = data_dir / "2026" / "videos" / group
                d.mkdir(parents=True)
                for vid in ids:
                    (d / f"{vid}.mp4").touch()

            candidates = addmeta.discover_candidates(data_dir, "2026", {"M01"})

            self.assertEqual(sorted(candidates), ["M01_V001", "M01_V002"])


class PlanNewRowsTests(unittest.TestCase):
    def test_skips_existing_and_continues_video_idx(self):
        existing = [
            {"video_idx": 0, "video_id": "L21_V001", "group": "L21", "year": "2026",
             "path": "/x/L21_V001.mp4", "fps": 30.0},
            {"video_idx": 1, "video_id": "L21_V002", "group": "L21", "year": "2026",
             "path": "/x/L21_V002.mp4", "fps": 30.0},
        ]
        candidates = {
            "M01_V001": Path("/data/2026/videos/M01/M01_V001.mp4"),
            "M01_V002": Path("/data/2026/videos/M01/M01_V002.mp4"),
        }
        fps_by_id = {"M01_V001": 25.0, "M01_V002": 30.0}

        new_rows = addmeta.plan_new_rows(existing, candidates, fps_by_id, "2026")

        self.assertEqual([r["video_idx"] for r in new_rows], [2, 3])
        self.assertEqual(new_rows[0]["video_id"], "M01_V001")
        self.assertEqual(new_rows[0]["group"], "M01")
        self.assertEqual(new_rows[0]["year"], "2026")
        self.assertEqual(new_rows[0]["fps"], 25.0)

    def test_already_present_video_id_is_skipped(self):
        existing = [{"video_idx": 5, "video_id": "M01_V001", "group": "M01", "year": "2026",
                    "path": "/x", "fps": 25.0}]
        candidates = {"M01_V001": Path("/x/M01_V001.mp4")}

        new_rows = addmeta.plan_new_rows(existing, candidates, {"M01_V001": 25.0}, "2026")

        self.assertEqual(new_rows, [])

    def test_starts_at_zero_when_no_existing_rows(self):
        candidates = {"S01_V001": Path("/x/S01_V001.mp4")}

        new_rows = addmeta.plan_new_rows([], candidates, {"S01_V001": 30.0}, "2026")

        self.assertEqual(new_rows[0]["video_idx"], 0)


class WriteAtomicRoundTripTests(unittest.TestCase):
    def test_write_then_load_existing_round_trips(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_meta.parquet"
            rows = [{"video_idx": 0, "video_id": "M01_V001", "group": "M01", "year": "2026",
                    "path": "/x", "fps": 25.0}]

            addmeta.write_atomic(path, rows)
            loaded_rows, max_idx = addmeta.load_existing(path)

            self.assertEqual(loaded_rows, rows)
            self.assertEqual(max_idx, 0)

    def test_load_existing_on_missing_file_returns_empty(self):
        rows, max_idx = addmeta.load_existing(Path("/nonexistent/video_meta.parquet"))
        self.assertEqual(rows, [])
        self.assertEqual(max_idx, -1)


if __name__ == "__main__":
    unittest.main()
