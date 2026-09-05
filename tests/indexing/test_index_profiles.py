import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import Mock, patch

from backend import app
from backend.core import ProfileStore


class IndexProfileTests(unittest.TestCase):
    def _profile(self, root, name, video_id, vector):
        directory = root / name
        directory.mkdir(parents=True)
        index = faiss.IndexFlatIP(2)
        index.add(np.asarray([vector], dtype=np.float32))
        faiss.write_index(index, str(directory / "flat.ip"))
        np.save(directory / "id_video.npy", np.array([0], dtype=np.int32))
        np.save(directory / "id_frame.npy", np.array([7], dtype=np.int32))
        np.save(directory / "id_time.npy", np.array([1.5]))
        pq.write_table(pa.Table.from_pylist([{
            "video_idx": 0, "video_id": video_id, "group": video_id[:3],
            "year": "2026", "path": "/ignored", "fps": 25.0,
        }]), directory / "video_meta.parquet")
        (directory / "info.json").write_text(json.dumps({
            "dim": 2, "count": 1,
            "indexes": {"flat": {"type": "FlatIP", "metric": "ip", "exact": True}},
        }))

    def test_manifest_scopes_indexes_and_metadata_by_profile(self):
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "indexes"
            profiles = index_dir / "profiles"
            self._profile(profiles, "l2025", "L21_V001", [1, 0])
            self._profile(profiles, "l2026", "L21_V002", [0, 1])
            (profiles / "manifest.json").write_text(json.dumps({
                "default": "l2025", "cache_size": 2,
                "profiles": {
                    "l2025": {"label": "L 2025", "directory": "l2025", "count": 1, "video_count": 1},
                    "l2026": {"label": "L 2026", "directory": "l2026", "count": 1, "video_count": 1},
                },
            }))

            store = ProfileStore(index_dir)

            self.assertEqual(store.default, "l2025")
            self.assertEqual([p["name"] for p in store.list_profiles()], ["l2025", "l2026"])
            self.assertEqual(store.load_meta("l2025")[3]["L21_V001"]["video_idx"], 0)
            self.assertEqual(store.load_meta("l2026")[3]["L21_V002"]["video_idx"], 0)
            self.assertIsNot(store.get_index("flat", "l2025"), store.get_index("flat", "l2026"))

    def test_frame_sources_reuse_profile_and_original_index_metadata(self):
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "indexes"
            profiles = index_dir / "profiles"
            self._profile(profiles, "l2026", "L21_V001", [1, 0])
            self._profile(index_dir, "staging", "K01_V001", [0, 1])
            for name in ("flat.ip", "id_video.npy", "id_frame.npy", "id_time.npy", "video_meta.parquet", "info.json"):
                (index_dir / "staging" / name).replace(index_dir / name)
            (index_dir / "staging").rmdir()
            (profiles / "manifest.json").write_text(json.dumps({
                "default": "l2026", "profiles": {
                    "l2026": {"label": "L 2026", "directory": "l2026"},
                },
            }))

            store = ProfileStore(index_dir)

            self.assertEqual(store.list_frame_sources("L21_V001"), [
                {"name": "l2026", "label": "L 2026"},
            ])
            self.assertEqual(store.list_frame_sources("K01_V001"), [
                {"name": "legacy", "label": "Index gốc"},
            ])
            np.testing.assert_array_equal(store.frame_ids("legacy", "K01_V001"), [7])

    def test_rejects_unknown_profile_and_index(self):
        with TemporaryDirectory() as tmp:
            index_dir = Path(tmp) / "indexes"
            index_dir.mkdir()
            self._profile(index_dir, "unused", "L21_V001", [1, 0])
            for name in ("flat.ip", "id_video.npy", "id_frame.npy", "id_time.npy", "video_meta.parquet", "info.json"):
                (index_dir / "unused" / name).replace(index_dir / name)
            store = ProfileStore(index_dir)

            with self.assertRaisesRegex(KeyError, "unknown profile"):
                store.get_index("flat", "missing")
            with self.assertRaisesRegex(KeyError, "unknown index"):
                store.get_index("ivf")

    def test_visual_search_uses_the_requested_profile(self):
        index = Mock()
        index.search.return_value = (np.array([[0.5]], dtype=np.float32), np.array([[0]]))
        meta = (
            np.array([0], dtype=np.int32), np.array([1], dtype=np.int32),
            {0: {"video_id": "L21_V001"}}, {},
        )
        mapping = Mock(pts=np.array([0, 10]))
        mapping.frame_time.return_value = 1.0
        request = app.SearchReq(mode="visual", profile="l2026")
        with (
            patch.object(app.core, "available_indexes", return_value=["flat"]),
            patch.object(app.core, "get_index", return_value=index) as get_index,
            patch.object(app.core, "index_info", return_value={"metric": "ip", "type": "FlatIP"}),
            patch.object(app.core, "load_meta", return_value=meta) as load_meta,
            patch.object(app.frame_maps, "load", return_value=mapping),
        ):
            response = app.visual_vector_search(request, np.array([[1, 0]], dtype=np.float32))

        get_index.assert_called_once_with("flat", "l2026")
        load_meta.assert_called_once_with("l2026")
        self.assertEqual(response["profile"], "l2026")

    def test_compare_only_uses_indexes_available_in_the_default_profile(self):
        index = Mock()
        index.search.return_value = (np.array([[1.0]], dtype=np.float32), np.array([[0]]))
        with (
            patch.object(app.core, "encode_text", return_value=np.array([[1, 0]], dtype=np.float32)),
            patch.object(app.core, "available_indexes", return_value=["flat"]),
            patch.object(app.core, "get_index", return_value=index),
            patch.object(app.core, "index_info", return_value={"type": "FlatIP"}),
        ):
            result = app.compare(app.CompareReq(text="query", k=1))

        self.assertEqual([row["index"] for row in result["results"]], ["flat"])


if __name__ == "__main__":
    unittest.main()
