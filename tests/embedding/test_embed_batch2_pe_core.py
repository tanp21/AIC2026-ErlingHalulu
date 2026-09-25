import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from scripts.embedding import embed_batch2_pe_core as embedder


class CollectItemsTests(unittest.TestCase):
    def _make_tree(self, root):
        for group, video_id, frame in (
            ("M01", "M01_V001", 3), ("M01", "M01_V001", 40),
            ("S01", "S01_V001", 7),
        ):
            d = root / group / video_id
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{frame:06d}.png").touch()
        (root / "M01" / "M01_V001" / "000099.tmp.png").parent.mkdir(
            parents=True, exist_ok=True)
        (root / "M01" / "M01_V001" / "000099.tmp.png").touch()

    def test_skips_tmp_files_and_derives_ids_from_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)

            items = embedder.collect_items(root, groups=set(), only=set())

            self.assertEqual(len(items), 3)
            self.assertTrue(all(not p.name.endswith(".tmp.png") for p, _, _ in items))

    def test_group_filter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)

            items = embedder.collect_items(root, groups={"S01"}, only=set())

            self.assertEqual([vid for _, vid, _ in items], ["S01_V001"])

    def test_only_filter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_tree(root)

            items = embedder.collect_items(root, groups=set(), only={"S01_V001"})

            self.assertEqual([vid for _, vid, _ in items], ["S01_V001"])


class ShardWriterTests(unittest.TestCase):
    def test_resume_continues_shard_numbering_and_fill(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            writer = embedder.ShardWriter(out, dim=4, shard_size=3)
            for i in range(7):
                writer.add(np.full(4, i, dtype=np.float32), f"v{i // 3}", i % 3)
            writer.close()
            self.assertEqual(writer.total, 7)
            self.assertEqual(writer.fill, 0)

            resumed = embedder.ShardWriter(out, dim=4, shard_size=3)
            self.assertEqual(resumed.total, 7)
            self.assertEqual(resumed.shard_idx, 2)
            self.assertEqual(resumed.fill, 1)
            self.assertEqual(len(resumed.done), 7)

            for i in range(7, 9):
                resumed.add(np.full(4, i, dtype=np.float32), f"v{i // 3}", i % 3)
            resumed.close()
            self.assertEqual(resumed.total, 9)

            shards = sorted(out.glob("emb_*.npy"))
            self.assertEqual(len(shards), 3)
            self.assertEqual([np.load(s).shape[0] for s in shards], [3, 3, 3])
            meta = pq.read_table(out / "meta.parquet")
            self.assertEqual(meta.num_rows, 9)

    def test_never_restarts_from_scratch_when_data_already_committed(self):
        # This is exactly the failure mode embed_keyframes.py has: pointing at an existing
        # production directory must not silently overwrite emb_000000.npy.
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            first = embedder.ShardWriter(out, dim=2, shard_size=2)
            first.add(np.array([1.0, 1.0], dtype=np.float32), "batch1_V001", 0)
            first.add(np.array([2.0, 2.0], dtype=np.float32), "batch1_V001", 1)
            first.close()
            original_shard = np.load(out / "emb_000000.npy").copy()

            second = embedder.ShardWriter(out, dim=2, shard_size=2)
            second.add(np.array([3.0, 3.0], dtype=np.float32), "batch2_V001", 0)
            second.close()

            np.testing.assert_array_equal(np.load(out / "emb_000000.npy"), original_shard)
            self.assertTrue((out / "emb_000001.npy").exists())


if __name__ == "__main__":
    unittest.main()
