import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend import shot_labels
from scripts.indexing.build_flat_profile import build_profile, register_profile


class FlatProfileBuildTests(unittest.TestCase):
    def _base_index(self, root):
        base = root / "indexes"
        maps = base / "frame_maps"
        maps.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([
            {"video_idx": 0, "video_id": "K01_V001", "group": "K01", "year": "2025", "path": "/data/K.mp4", "fps": 25.0},
            {"video_idx": 1, "video_id": "L21_V001", "group": "L21", "year": "2026", "path": "/data/L1.mp4", "fps": 25.0},
            {"video_idx": 2, "video_id": "L21_V002", "group": "L21", "year": "2026", "path": "/data/L2.mp4", "fps": 30.0},
        ]), base / "video_meta.parquet")
        for video_id in ("L21_V001", "L21_V002"):
            np.savez_compressed(maps / f"{video_id}.npz", pts=np.array([100, 110, 130]), time_base_num=1, time_base_den=10)
        return base

    def test_builds_l_only_profile_from_raw_tree(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            for video_id, frame, value in (("K01_V001", 0, 9), ("L21_V001", 1, 1), ("L21_V002", 2, 2)):
                path = source / video_id[:3] / video_id / f"{frame:06d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.full(4, value, dtype=np.float32))
            output = root / "profiles" / "l2025"

            result = build_profile(source, output, self._base_index(root), prefix="L")

            self.assertEqual(result, {"count": 2, "video_count": 2, "dim": 4,
                                      "excluded_count": 0, "excluded_by_reason": {}})
            self.assertEqual(faiss.read_index(str(output / "flat.ip")).ntotal, 2)
            np.testing.assert_array_equal(np.load(output / "id_video.npy"), [0, 1])
            np.testing.assert_array_equal(np.load(output / "id_frame.npy"), [1, 2])
            np.testing.assert_allclose(np.load(output / "id_time.npy"), [1, 3])
            rows = pq.read_table(output / "video_meta.parquet").to_pylist()
            self.assertEqual([row["video_id"] for row in rows], ["L21_V001", "L21_V002"])

    def test_builds_profile_from_shards_and_registers_manifest(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shards"
            source.mkdir()
            np.save(source / "emb_000000.npy", np.array([[9, 0], [1, 0], [0, 1]], dtype=np.float32))
            pq.write_table(pa.Table.from_pylist([
                {"shard_idx": 0, "row_idx": 0, "video_id": "K01_V001", "frame_idx": 0},
                {"shard_idx": 0, "row_idx": 1, "video_id": "L21_V001", "frame_idx": 1},
                {"shard_idx": 0, "row_idx": 2, "video_id": "L21_V002", "frame_idx": 2},
            ]), source / "meta.parquet")
            output = root / "profiles" / "l2026"

            result = build_profile(source, output, self._base_index(root), prefix="L")
            register_profile(root / "profiles", "l2026", "L · embeddings 2026", result, default=False)

            self.assertEqual(result["count"], 2)
            manifest = json.loads((root / "profiles" / "manifest.json").read_text())
            self.assertEqual(manifest["profiles"]["l2026"]["directory"], "l2026")
            self.assertEqual(manifest["profiles"]["l2026"]["count"], 2)

    def test_places_a_vfr_keyframe_by_its_pts_time(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = self._base_index(root)
            # 2 s apart at a nominal 25 fps is 50 frames, but these videos are VFR: the frame
            # that carries 2 s is 1, and 3.9 s sits between frames 1 and 2.
            np.savez_compressed(base / "frame_maps" / "L21_V002.npz",
                                pts=np.array([0, 2000, 6000, 10000]), time_base_num=1,
                                time_base_den=1000)
            source = root / "shards"
            source.mkdir()
            np.save(source / "emb_000000.npy", np.array([[1, 0], [0, 1]], dtype=np.float32))
            pq.write_table(pa.Table.from_pylist([
                {"shard_idx": 0, "row_idx": 0, "video_id": "L21_V002", "frame_idx": 50, "pts_ms": 2000},
                {"shard_idx": 0, "row_idx": 1, "video_id": "L21_V002", "frame_idx": 97, "pts_ms": 3900},
            ]), source / "meta.parquet")

            result = build_profile(source, root / "profile", base, prefix="", frame_from="pts-ms")

            self.assertEqual(result["count"], 2)
            self.assertEqual(result["frame_from"], "pts-ms")
            self.assertAlmostEqual(result["frame_time_off_seconds"], 1.9)
            np.testing.assert_array_equal(np.load(root / "profile" / "id_frame.npy"), [1, 1])
            np.testing.assert_allclose(np.load(root / "profile" / "id_time.npy"), [2.0, 2.0])

    def test_filters_only_reviewed_excluded_shots(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "raw"
            for video_id, frame in (("L21_V001", 1), ("L21_V002", 2)):
                path = source / "L21" / video_id / f"{frame:06d}.npy"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, np.array([1, frame], dtype=np.float32))
            data = root / "data"
            shot_dir = data / "2026" / "shots" / "L21"
            shot_dir.mkdir(parents=True)
            for video_id in ("L21_V001", "L21_V002"):
                (shot_dir / f"{video_id}.json").write_text(json.dumps({
                    "fps": 30, "shots": [[0, 1], [2, 2]],
                }))
            labels = root / "labels"
            bounds = [[0, 1], [2, 2]]
            suggested = shot_labels.replace_suggestions(
                "L21_V001", bounds,
                {0: {"reason": "outro", "confidence": .9}}, labels,
            )
            shot_labels.update(
                "L21_V001", bounds, suggested["revision"],
                [{"shot_index": 0, "state": "excluded", "reason": "outro"}],
                "owner@example.com", labels,
            )

            result = build_profile(
                source, root / "profile", self._base_index(root), prefix="L",
                shot_label_dir=labels, data_dir=data,
            )

            self.assertEqual(result["count"], 1)
            self.assertEqual(result["excluded_by_reason"], {"outro": 1})
            np.testing.assert_array_equal(np.load(root / "profile" / "id_frame.npy"), [2])


if __name__ == "__main__":
    unittest.main()
