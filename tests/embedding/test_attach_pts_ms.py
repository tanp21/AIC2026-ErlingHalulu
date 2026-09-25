import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.embedding import attach_pts_ms as attach


def write_meta(path, rows):
    pq.write_table(pa.table({
        "shard_idx": pa.array([row[0] for row in rows], type=pa.int32()),
        "row_idx": pa.array([row[1] for row in rows], type=pa.int32()),
        "video_id": pa.array([row[2] for row in rows], type=pa.string()),
        "frame_idx": pa.array([row[3] for row in rows], type=pa.int32()),
    }), path)


def write_points(path, rows):
    pq.write_table(pa.table({
        "video_id": pa.array([row[0] for row in rows], type=pa.string()),
        "group": pa.array([row[1] for row in rows], type=pa.string()),
        "keyframe_n": pa.array([row[2] for row in rows], type=pa.int32()),
        "frame_idx": pa.array([row[3] for row in rows], type=pa.int32()),
        "pts_ms": pa.array([row[4] for row in rows], type=pa.int64()),
    }), path)


class AttachTests(unittest.TestCase):
    def test_joins_pts_ms_and_keyframe_n_by_video_and_frame(self):
        with TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.parquet"
            write_meta(meta, [(0, 0, "N001_V001", 0), (0, 1, "N001_V001", 50),
                              (0, 2, "N001_V002", 0)])
            points = {("N001_V001", 0): {"pts_ms": 0, "keyframe_n": 1},
                      ("N001_V001", 50): {"pts_ms": 2000, "keyframe_n": 2},
                      ("N001_V002", 0): {"pts_ms": 0, "keyframe_n": 1}}
            attach.attach(meta, points)

            table = pq.read_table(meta)
            self.assertEqual(table.column_names,
                             ["shard_idx", "row_idx", "video_id", "frame_idx", "pts_ms",
                              "keyframe_n"])
            self.assertEqual(table["pts_ms"].to_pylist(), [0, 2000, 0])
            self.assertEqual(table["keyframe_n"].to_pylist(), [1, 2, 1])

    def test_row_order_and_source_columns_are_untouched(self):
        with TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.parquet"
            write_meta(meta, [(1, 7, "N002_V001", 900), (0, 0, "N001_V001", 0)])
            attach.attach(meta, {("N002_V001", 900): {"pts_ms": 36000, "keyframe_n": 19},
                                 ("N001_V001", 0): {"pts_ms": 0, "keyframe_n": 1}})
            table = pq.read_table(meta)
            self.assertEqual(table["row_idx"].to_pylist(), [7, 0])
            self.assertEqual(table["shard_idx"].to_pylist(), [1, 0])

    def test_a_row_with_no_point_fails_rather_than_going_without_a_time(self):
        with TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.parquet"
            write_meta(meta, [(0, 0, "N001_V001", 0), (0, 1, "N001_V001", 50)])
            with self.assertRaises(ValueError) as caught:
                attach.attach(meta, {("N001_V001", 0): {"pts_ms": 0, "keyframe_n": 1}})
            self.assertIn("N001_V001", str(caught.exception))

    def test_rejects_a_file_that_is_not_a_shard_writer_output(self):
        with TemporaryDirectory() as tmp:
            meta = Path(tmp) / "meta.parquet"
            pq.write_table(pa.table({"video_id": pa.array(["N001_V001"], type=pa.string())}),
                           meta)
            with self.assertRaises(ValueError):
                attach.attach(meta, {})

    def test_missing_meta_is_reported(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                attach.attach(Path(tmp) / "meta.parquet", {})


class LoadPointsTests(unittest.TestCase):
    def test_reads_the_ingest_side_table(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "points.parquet"
            write_points(path, [("N001_V001", "N001", 1, 0, 0),
                                ("N001_V001", "N001", 2, 50, 2000)])
            points = attach.load_points(path)
            # Only the columns the join needs, keyed by what it joins on.
            self.assertEqual(points[("N001_V001", 50)], {"video_id": "N001_V001",
                                                         "keyframe_n": 2,
                                                         "frame_idx": 50, "pts_ms": 2000})


if __name__ == "__main__":
    unittest.main()
