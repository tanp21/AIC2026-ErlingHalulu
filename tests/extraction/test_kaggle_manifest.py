import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.extraction import build_kaggle_manifest as builder
from scripts.extraction import kaggle_extract_keyframes as extractor


LAYOUT = {
    "Videos_L21_a": ["L21_V001", "L21_V002"],
    "Videos_L26_a": ["L26_V001"],
    "Videos_L26_b": ["L26_V100"],
}

VIDEO_META = {
    "L21_V001": ("L21", 30.0),
    "L21_V002": ("L21", 25.0),
    "L26_V100": ("L26", 30.0),
    "K01_V001": ("K01", 25.0),
    "K09_V001": ("K09", 30.0),
    "K20_V001": ("K20", 30.0),
}

SPEC_L = {"batch": "batch1", "year": "2026", "prefix": "L",
          "source": "data/2026/embeddings/meta.parquet"}
SPEC_K = {"batch": "batch2", "year": "2025", "prefix": "K",
          "source": "data/2025/embeddings/meta.parquet"}


def write_meta(path, rows):
    pq.write_table(pa.table({
        "video_id": [v for v, _ in rows],
        "frame_idx": pa.array([f for _, f in rows], type=pa.int32()),
    }), path)


class DatasetSlugTests(unittest.TestCase):
    def test_batch1_is_a_single_dataset(self):
        self.assertEqual(builder.dataset_slug("batch1", "L26"), "aic-batch1-videos")

    def test_batch2_groups_map_to_their_part(self):
        cases = {
            "K01": "part1", "K07": "part1",
            "K08": "part2", "K14": "part2",
            "K15": "part3", "K20": "part3",
        }
        for group, part in cases.items():
            with self.subTest(group=group):
                self.assertEqual(builder.dataset_slug("batch2", group),
                                 f"aic-batch2-videos-{part}")


class FolderResolutionTests(unittest.TestCase):
    def test_explicit_layout_entry_wins_over_the_group_default(self):
        index = builder.folder_index(LAYOUT)

        self.assertEqual(builder.zip_folder("L26_V100", "L26", index), "Videos_L26_b")

    def test_k_videos_fall_back_to_their_group_folder(self):
        self.assertEqual(builder.zip_folder("K09_V001", "K09", {}), "Videos_K09")

    def test_unlisted_l_video_raises(self):
        with self.assertRaisesRegex(KeyError, "L30_V001"):
            builder.zip_folder("L30_V001", "L30", {})

    def test_video_listed_under_two_folders_raises(self):
        overlapping = {"Videos_L25_a": ["L25_V050"], "Videos_L25_b": ["L25_V050"]}

        with self.assertRaisesRegex(ValueError, "L25_V050"):
            builder.folder_index(overlapping)

    def test_relative_path_joins_dataset_folder_and_video(self):
        index = builder.folder_index(LAYOUT)

        self.assertEqual(
            builder.relative_path("L26_V100", "L26", "batch1", index),
            "aic-batch1-videos/Videos_L26_b/video/L26_V100.mp4")
        self.assertEqual(
            builder.relative_path("K09_V001", "K09", "batch2", index),
            "aic-batch2-videos-part2/Videos_K09/video/K09_V001.mp4")


class LoadFramesTests(unittest.TestCase):
    def test_filters_by_prefix_and_sorts_deduplicated_frames(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "meta.parquet"
            write_meta(path, [
                ("L21_V001", 31), ("L21_V001", 4), ("L21_V001", 4),
                ("K01_V001", 7),
            ])

            self.assertEqual(builder.load_frames(path, "L"), {"L21_V001": [4, 31]})
            self.assertEqual(builder.load_frames(path, "K"), {"K01_V001": [7]})


class BuildManifestTests(unittest.TestCase):
    def setUp(self):
        self.index = builder.folder_index(LAYOUT)

    def test_manifest_shape_and_counts(self):
        manifest = builder.build_manifest(
            SPEC_L, {"L21_V001": [1, 4], "L26_V100": [7]}, VIDEO_META, self.index)

        self.assertEqual(manifest["batch"], "batch1")
        self.assertEqual(manifest["year"], "2026")
        self.assertEqual(manifest["root"], builder.BATCH1_ROOT)
        self.assertEqual(manifest["source"], SPEC_L["source"])
        self.assertEqual(manifest["video_count"], 2)
        self.assertEqual(manifest["keyframe_count"], 3)
        self.assertEqual(
            manifest["keyframe_count"],
            sum(len(v["frames"]) for v in manifest["videos"].values()))
        self.assertEqual(manifest["videos"]["L21_V001"], {
            "rel": "aic-batch1-videos/Videos_L21_a/video/L21_V001.mp4",
            "fps": 30.0,
            "frames": [1, 4],
        })

    def test_batch2_uses_the_other_kaggle_root(self):
        manifest = builder.build_manifest(
            SPEC_K, {"K01_V001": [0], "K20_V001": [3]}, VIDEO_META, self.index)

        self.assertEqual(manifest["root"], builder.BATCH2_ROOT)
        self.assertEqual(manifest["videos"]["K20_V001"]["rel"],
                         "aic-batch2-videos-part3/Videos_K20/video/K20_V001.mp4")

    def test_video_missing_from_video_meta_raises(self):
        with self.assertRaisesRegex(KeyError, "L21_V009"):
            builder.build_manifest(SPEC_L, {"L21_V009": [1]}, VIDEO_META, self.index)


class CentralDirectoryTests(unittest.TestCase):
    def test_reads_entry_names_from_a_real_zip_tail(self):
        with TemporaryDirectory() as tmp:
            archive = Path(tmp) / "Videos_T01.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("video/T01_V001.mp4", b"a")
                zf.writestr("video/T01_V002.mp4", b"b")
                zf.writestr("notes.txt", b"c")

            names = builder.parse_central_directory(archive.read_bytes())

        self.assertEqual(
            names, ["video/T01_V001.mp4", "video/T01_V002.mp4", "notes.txt"])


class CommittedLayoutTests(unittest.TestCase):
    """The layout ships with the repo, so guard the invariants it must hold."""

    def setUp(self):
        self.layout = builder.load_layout()

    def test_covers_every_l_video_exactly_once(self):
        index = builder.folder_index(self.layout)

        self.assertEqual(len(index), 873)
        self.assertTrue(all(v.startswith("L") for v in index))

    def test_excludes_the_duplicate_l25_zip(self):
        self.assertIn("Videos_L25_b", builder.SKIP_ZIPS)
        self.assertNotIn("Videos_L25_b", self.layout)
        self.assertEqual(len(self.layout["Videos_L25_a"]), 88)


class SelectVideosTests(unittest.TestCase):
    manifest = {"videos": {
        "L21_V001": {"frames": [1]},
        "L21_V002": {"frames": [2]},
        "L26_V100": {"frames": [3]},
    }}

    def test_no_filter_returns_everything(self):
        self.assertEqual(len(extractor.select_videos(self.manifest, None)), 3)

    def test_filters_by_group_and_by_video_id(self):
        self.assertEqual(
            sorted(extractor.select_videos(self.manifest, "L21")),
            ["L21_V001", "L21_V002"])
        self.assertEqual(
            sorted(extractor.select_videos(self.manifest, "L21_V002,L26_V100")),
            ["L21_V002", "L26_V100"])

    def test_unmatched_token_is_an_error(self):
        with self.assertRaisesRegex(SystemExit, "L99"):
            extractor.select_videos(self.manifest, "L99")


class ResolveTests(unittest.TestCase):
    videos = {
        "L21_V001": {"rel": "ds/Videos_L21_a/video/L21_V001.mp4", "frames": [1]},
        "L21_V002": {"rel": "ds/Videos_L21_a/video/L21_V002.mp4", "frames": [2]},
    }

    def make_root(self, tmp, present):
        root = Path(tmp)
        for video_id in present:
            path = root / self.videos[video_id]["rel"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return root

    def test_missing_videos_are_skipped_and_reported(self):
        with TemporaryDirectory() as tmp:
            root = self.make_root(tmp, ["L21_V001"])

            found, missing = extractor.resolve(root, self.videos, "kaggle", False)

        self.assertEqual(list(found), ["L21_V001"])
        self.assertEqual(missing, ["L21_V002"])

    def test_strict_turns_a_missing_video_into_an_error(self):
        with TemporaryDirectory() as tmp:
            root = self.make_root(tmp, ["L21_V001"])

            with self.assertRaisesRegex(SystemExit, "L21_V002"):
                extractor.resolve(root, self.videos, "kaggle", True)

    def test_flat_layout_looks_under_group_directories(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "L21").mkdir()
            (root / "L21" / "L21_V002.mp4").touch()

            found, missing = extractor.resolve(root, self.videos, "flat", False)

        self.assertEqual(list(found), ["L21_V002"])
        self.assertEqual(missing, ["L21_V001"])


class ManifestFilesTests(unittest.TestCase):
    """The generated manifests, when they exist locally (data/ is gitignored)."""

    EXPECTED = {
        "batch1_2025": (873, 293239),
        "batch2_2025": (605, 562393),
        "batch1_2026": (873, 289875),
    }

    def test_counts_match_the_embedding_meta(self):
        directory = Path(builder.ROOT) / "data" / "kaggle_manifests"
        if not directory.is_dir():
            self.skipTest("manifests not built in this checkout")
        for name, (videos, keyframes) in self.EXPECTED.items():
            with self.subTest(manifest=name):
                manifest = json.loads((directory / f"{name}.json").read_text())
                self.assertEqual(manifest["video_count"], videos)
                self.assertEqual(manifest["keyframe_count"], keyframes)
                self.assertEqual(
                    sum(len(v["frames"]) for v in manifest["videos"].values()),
                    keyframes)


if __name__ == "__main__":
    unittest.main()
