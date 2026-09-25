"""Which keyframes the object store lets a query match, and which stay playback-only."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.indexing import build_object_index


class ObjectIndexBuildTests(unittest.TestCase):
    def _profile(self, root, name, videos):
        directory = root / "profiles" / name
        directory.mkdir(parents=True)
        np.save(directory / "id_video.npy", np.arange(len(videos), dtype=np.int32))
        np.save(directory / "id_frame.npy", np.asarray([v[1] for v in videos], dtype=np.int32))
        pq.write_table(pa.Table.from_pylist([
            {"video_idx": index, "video_id": video_id, "group": video_id[:3],
             "year": "2026", "path": "/data/x.mp4", "fps": 25.0}
            for index, (video_id, _) in enumerate(videos)
        ]), directory / "video_meta.parquet")
        for video_id, _ in videos:
            np.savez_compressed(root / "frame_maps" / f"{video_id}.npz",
                                pts=np.arange(0, 40000, 40), time_base_num=1, time_base_den=1000)

    def _source(self, root, prefix, video_id, frame_idx, label):
        path = root / f"{prefix}.parquet"
        pq.write_table(pa.Table.from_pylist([
            {"video_id": video_id, "frame_idx": frame_idx, "time": frame_idx / 25,
             "labels": [label], "scores": [.9], "boxes": [[.4, .1, .6, .3]]},
        ]), path)
        return (prefix, path)

    def _manifests(self, root, profiles):
        (root / "profiles").mkdir(parents=True, exist_ok=True)
        (root / "frame_maps").mkdir(parents=True, exist_ok=True)
        (root / "profiles" / "manifest.json").write_text(json.dumps({
            "default": sorted(profiles)[0],
            "profiles": {name: {"label": name, "directory": name} for name in profiles},
        }))

    def test_matches_keyframes_of_a_searchable_group_and_keeps_the_rest_for_playback(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifests(root, ["l2025", "l2026-v3"])
            self._profile(root, "l2025", [("K01_V001", 10)])
            self._profile(root, "l2026-v3", [("L21_V001", 20), ("N001_V001", 30)])
            sources = [self._source(root, "L", "L21_V001", 20, "car"),
                       self._source(root, "N", "N001_V001", 30, "truck")]

            result = build_object_index.build(sources, root / "profiles", root / "frame_maps",
                                              root / "objects")

            self.assertEqual(result["keyframes"], 2)
            keyframes = pq.read_table(root / "objects" / "current" / "keyframes.parquet").to_pylist()
            self.assertEqual(sorted(row["video_id"] for row in keyframes),
                             ["L21_V001", "N001_V001"])
            masks = json.loads((root / "objects" / "current" / "manifest.json").read_text())["profiles"]
            self.assertEqual(masks["l2026-v3"]["keyframes"], 2)
            self.assertEqual(masks["l2025"]["keyframes"], 0)
            labels = json.loads((root / "objects" / "current" / "manifest.json").read_text())["labels"]
            self.assertEqual(labels, ["car", "truck"])

    def test_a_group_outside_the_search_prefixes_is_playback_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._manifests(root, ["l2026-v3"])
            self._profile(root, "l2026-v3", [("N001_V001", 30)])
            sources = [self._source(root, "N", "N001_V001", 30, "truck")]

            result = build_object_index.build(sources, root / "profiles", root / "frame_maps",
                                              root / "objects", prefixes=("L", "M", "S"))

            self.assertEqual(result["keyframes"], 0)
            playback = list((root / "objects" / "current" / "playback").rglob("*.parquet"))
            self.assertEqual(len(playback), 1)
            self.assertEqual(pq.read_table(playback[0]).num_rows, 1)


if __name__ == "__main__":
    unittest.main()
