import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.extraction import build_batch2_kaggle_manifest as builder


def write_meta(path, rows):
    pq.write_table(pa.table({
        "video_id": [v for v, _ in rows],
        "frame_idx": pa.array([f for _, f in rows], type=pa.int32()),
    }), path)


def write_video_meta(path, rows):
    pq.write_table(pa.table({
        "video_id": [r[0] for r in rows],
        "group": [r[1] for r in rows],
        "fps": pa.array([r[2] for r in rows], type=pa.float64()),
    }), path)


class LoadFramesTests(unittest.TestCase):
    def test_sorts_and_deduplicates_per_video(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.parquet"
            write_meta(path, [("M01_V001", 31), ("M01_V001", 4), ("M01_V001", 4),
                              ("S01_V001", 7)])

            frames = builder.load_frames(path)

            self.assertEqual(frames, {"M01_V001": [4, 31], "S01_V001": [7]})


class LoadVideoMetaTests(unittest.TestCase):
    def test_reads_group_and_fps(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_meta.parquet"
            write_video_meta(path, [("M01_V001", "M01", 25.0)])

            meta = builder.load_video_meta(path)

            self.assertEqual(meta, {"M01_V001": ("M01", 25.0)})

    def test_raises_on_missing_fps(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "video_meta.parquet"
            write_video_meta(path, [("M01_V001", "M01", None)])

            with self.assertRaisesRegex(ValueError, "M01_V001"):
                builder.load_video_meta(path)


class BuildManifestTests(unittest.TestCase):
    def test_manifest_shape_matches_batch1_schema(self):
        frames_by_video = {"M01_V001": [1, 4], "S01_V001": [7]}
        video_meta = {"M01_V001": ("M01", 25.0), "S01_V001": ("S01", 30.0)}

        manifest = builder.build_manifest(
            "batch2", "2026", "data/2026/embeddings_batch2/pe-core/meta.parquet",
            "/kaggle/input/datasets/x/aic-batch2", frames_by_video, video_meta)

        self.assertEqual(manifest["batch"], "batch2")
        self.assertEqual(manifest["year"], "2026")
        self.assertEqual(manifest["video_count"], 2)
        self.assertEqual(manifest["keyframe_count"], 3)
        self.assertEqual(manifest["videos"]["M01_V001"], {
            "rel": "M01/M01_V001.mp4", "fps": 25.0, "frames": [1, 4],
        })
        # No image data anywhere in the manifest.
        self.assertNotIn("png", str(manifest).lower())
        self.assertNotIn("zip", str(manifest["videos"]).lower())

    def test_video_missing_from_video_meta_raises(self):
        with self.assertRaisesRegex(KeyError, "M01_V009"):
            builder.build_manifest("batch2", "2026", "src", "root",
                                   {"M01_V009": [1]}, {})


if __name__ == "__main__":
    unittest.main()
