import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.media import generate_batch2_thumbnails as gen


def write_meta(path, rows):
    pq.write_table(pa.table({
        "video_id": [r[0] for r in rows],
        "frame_idx": pa.array([r[1] for r in rows], type=pa.int32()),
    }), path)


class FramesByVideoForGroupsTests(unittest.TestCase):
    def test_filters_by_group_prefix_and_groups_by_video(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.parquet"
            write_meta(path, [
                ("M01_V001", 3), ("M01_V001", 40), ("S01_V001", 7), ("M02_V001", 1),
            ])

            by_video = gen.frames_by_video_for_groups(path, {"M01", "S01"})

            self.assertEqual(by_video, {
                "M01_V001": [3, 40],
                "S01_V001": [7],
            })

    def test_empty_when_no_group_matches(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.parquet"
            write_meta(path, [("M01_V001", 3)])

            by_video = gen.frames_by_video_for_groups(path, {"S01"})

            self.assertEqual(by_video, {})


if __name__ == "__main__":
    unittest.main()
