import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.ocr import build_ocr_kaggle_manifest as builder


def write_shots(path, fps, shots):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fps": fps, "shots": shots}))


class LoadShotsTests(unittest.TestCase):
    def test_computes_keyframes_per_video_filtered_by_group(self):
        with TemporaryDirectory() as tmp:
            shots_dir = Path(tmp)
            write_shots(shots_dir / "M01" / "M01_V001.json", 25.0, [[0, 20]])
            write_shots(shots_dir / "M02" / "M02_V001.json", 30.0, [[0, 10]])

            videos = builder.load_shots(shots_dir, {"M01"})

            self.assertEqual(set(videos), {"M01_V001"})
            self.assertEqual(videos["M01_V001"], {
                "group": "M01", "fps": 25.0, "frames": [3, 10, 17],
            })

    def test_no_group_filter_loads_everything(self):
        with TemporaryDirectory() as tmp:
            shots_dir = Path(tmp)
            write_shots(shots_dir / "M01" / "M01_V001.json", 25.0, [[0, 20]])
            write_shots(shots_dir / "M02" / "M02_V001.json", 30.0, [[0, 10]])

            videos = builder.load_shots(shots_dir, set())

            self.assertEqual(set(videos), {"M01_V001", "M02_V001"})


class BuildManifestTests(unittest.TestCase):
    def test_manifest_shape_uses_nested_kaggle_layout(self):
        videos_by_id = {
            "M01_V001": {"group": "M01", "fps": 25.0, "frames": [3, 10, 17]},
            "M02_V001": {"group": "M02", "fps": 30.0, "frames": [5]},
        }

        manifest = builder.build_manifest(
            "batch2", "2026", "data/2026/shots", "/kaggle/input/x/aic-batch2", videos_by_id)

        self.assertEqual(manifest["batch"], "batch2")
        self.assertEqual(manifest["year"], "2026")
        self.assertEqual(manifest["video_count"], 2)
        self.assertEqual(manifest["keyframe_count"], 4)
        self.assertEqual(manifest["videos"]["M01_V001"], {
            "rel": "Videos_M01/videos/M01_V001.mp4", "fps": 25.0, "frames": [3, 10, 17],
        })
        # No image data anywhere in the manifest.
        self.assertNotIn("png", str(manifest).lower())
        self.assertNotIn(".zip", str(manifest["videos"]).lower())

    def test_empty_input_yields_empty_manifest(self):
        manifest = builder.build_manifest("batch2", "2026", "src", "root", {})

        self.assertEqual(manifest["video_count"], 0)
        self.assertEqual(manifest["keyframe_count"], 0)


if __name__ == "__main__":
    unittest.main()
