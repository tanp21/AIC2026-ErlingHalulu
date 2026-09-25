import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.indexing.build_flat_profile import build_profile
from scripts.indexing.extend_flat_profile import extend_profile


class ExtendFlatProfileTests(unittest.TestCase):
    """The N groups are placed by the organisers' pts_time, and appended to a built profile."""

    #: 25 fps would put 2 s 50 frames apart; these videos are VFR, so the frame carrying a
    #: keyframe's time is not its label -- which is exactly what the N groups look like.
    FPS_TIMES = [0, 2, 6, 10, 14, 18, 22, 26, 30, 34]

    def _base_index(self, root):
        base = root / "indexes"
        maps = base / "frame_maps"
        maps.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([
            {"video_idx": 0, "video_id": "L21_V001", "group": "L21", "year": "2026",
             "path": "/data/L1.mp4", "fps": 25.0},
            {"video_idx": 1, "video_id": "N001_V001", "group": "N001", "year": "2026",
             "path": "/data/N1.mov", "fps": 25.0},
        ]), base / "video_meta.parquet")
        for video_id in ("L21_V001", "N001_V001"):
            pts = np.asarray(self.FPS_TIMES, dtype=np.int64) * 1000
            np.savez_compressed(maps / f"{video_id}.npz", pts=pts,
                                time_base_num=1, time_base_den=1000)
        return base

    def _source(self, root, name, rows, vectors):
        source = root / name
        source.mkdir(parents=True, exist_ok=True)
        np.save(source / "emb_000000.npy", np.asarray(vectors, dtype=np.float32))
        pq.write_table(pa.Table.from_pylist([
            {"shard_idx": 0, "row_idx": index, **row} for index, row in enumerate(rows)
        ]), source / "meta.parquet")
        return source

    def _base_profile(self, root, base_index):
        source = self._source(root, "lms", [{"video_id": "L21_V001", "frame_idx": 1}],
                              [[1.0, 0.0]])
        profile = root / "profiles" / "l2026-v2"
        build_profile(source, profile, base_index, prefix="")
        return profile

    def test_appends_n_by_pts_time_and_leaves_the_base_alone(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_index = self._base_index(root)
            profile = self._base_profile(root, base_index)
            # Labels that no frame index would give the right answer for: 2 s and 14 s.
            source = self._source(root, "n", [
                {"video_id": "N001_V001", "frame_idx": 50, "pts_ms": 2000},
                {"video_id": "N001_V001", "frame_idx": 350, "pts_ms": 14000},
            ], [[0.0, 1.0], [0.0, 2.0]])
            output = root / "profiles" / "l2026-v3"

            result = extend_profile(profile, output, source, base_index, frame_from="pts-ms")

            self.assertEqual(result["count"], 3)
            self.assertEqual(result["appended"], 2)
            self.assertEqual(result["video_count"], 2)
            self.assertEqual(result["frame_time_off_seconds"], 0.0)
            self.assertEqual(faiss.read_index(str(output / "flat.ip")).ntotal, 3)
            np.testing.assert_array_equal(np.load(output / "id_video.npy"), [0, 1, 1])
            np.testing.assert_array_equal(np.load(output / "id_frame.npy"), [1, 1, 4])
            np.testing.assert_allclose(np.load(output / "id_time.npy"), [2.0, 2.0, 14.0])
            rows = pq.read_table(output / "video_meta.parquet").to_pylist()
            self.assertEqual([row["video_id"] for row in rows], ["L21_V001", "N001_V001"])
            self.assertEqual([row["video_idx"] for row in rows], [0, 1])
            info = json.loads((output / "info.json").read_text())
            self.assertEqual(info["extended_from"], "l2026-v2")
            self.assertEqual(info["count"], 3)
            # the base profile still describes what it did before
            self.assertEqual(faiss.read_index(str(profile / "flat.ip")).ntotal, 1)

    def test_matches_a_standalone_build_of_the_same_source(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_index = self._base_index(root)
            profile = self._base_profile(root, base_index)
            rows = [{"video_id": "N001_V001", "frame_idx": 50, "pts_ms": 2000},
                    {"video_id": "N001_V001", "frame_idx": 350, "pts_ms": 14000}]
            vectors = [[0.0, 1.0], [0.0, 2.0]]
            source = self._source(root, "n", rows, vectors)
            output = root / "profiles" / "l2026-v3"
            reference = root / "profiles" / "n-reference"

            extend_profile(profile, output, source, base_index, frame_from="pts-ms")
            build_profile(source, reference, base_index, prefix="N", frame_from="pts-ms")

            # Same order in, same frames and times out: appending is what a rebuild would do.
            np.testing.assert_array_equal(np.load(output / "id_frame.npy")[1:],
                                          np.load(reference / "id_frame.npy"))
            np.testing.assert_allclose(np.load(output / "id_time.npy")[1:],
                                       np.load(reference / "id_time.npy"))
            extended = faiss.read_index(str(output / "flat.ip"))
            standalone = faiss.read_index(str(reference / "flat.ip"))
            np.testing.assert_array_equal(extended.reconstruct_n(1, 2),
                                          standalone.reconstruct_n(0, 2))

    def test_refuses_a_source_without_pts_ms_in_pts_ms_mode(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_index = self._base_index(root)
            profile = self._base_profile(root, base_index)
            source = self._source(root, "n", [{"video_id": "N001_V001", "frame_idx": 1}],
                                  [[0.0, 1.0]])

            with self.assertRaisesRegex(ValueError, "no pts_ms"):
                extend_profile(profile, root / "profiles" / "v3", source, base_index,
                               frame_from="pts-ms")

    def test_refuses_an_unknown_video_and_an_existing_target(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_index = self._base_index(root)
            profile = self._base_profile(root, base_index)
            unknown = self._source(root, "unknown", [{"video_id": "N999_V001", "frame_idx": 0}],
                                   [[0.0, 1.0]])
            with self.assertRaisesRegex(ValueError, "metadata unavailable"):
                extend_profile(profile, root / "profiles" / "v3", unknown, base_index)
            with self.assertRaises(FileExistsError):
                extend_profile(profile, profile, unknown, base_index)


if __name__ == "__main__":
    unittest.main()
