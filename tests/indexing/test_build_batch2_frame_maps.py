import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.indexing import build_batch2_frame_maps as bfm


class MainTests(unittest.TestCase):
    def test_probes_only_videos_in_requested_groups(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            for group, ids in (("M01", ["M01_V001"]), ("S01", ["S01_V001"])):
                d = data_dir / "2026" / "videos" / group
                d.mkdir(parents=True)
                for vid in ids:
                    (d / f"{vid}.mp4").touch()

            probed = []
            saved = []
            with patch.object(bfm, "probe", side_effect=lambda p: probed.append(p) or "MAP"), \
                 patch.object(bfm, "save_map", side_effect=lambda vid, m: saved.append((vid, m))):
                bfm.main(["--year", "2026", "--groups", "M01", "--data-dir", str(data_dir)])

            self.assertEqual(len(probed), 1)
            self.assertEqual(saved, [("M01_V001", "MAP")])

    def test_one_failure_does_not_stop_the_batch_but_raises_at_the_end(self):
        with TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            d = data_dir / "2026" / "videos" / "M01"
            d.mkdir(parents=True)
            (d / "M01_V001.mp4").touch()
            (d / "M01_V002.mp4").touch()

            def flaky_probe(path):
                if "V001" in path.name:
                    raise RuntimeError("boom")
                return "MAP"

            saved = []
            with patch.object(bfm, "probe", side_effect=flaky_probe), \
                 patch.object(bfm, "save_map", side_effect=lambda vid, m: saved.append(vid)):
                with self.assertRaisesRegex(RuntimeError, "M01_V001"):
                    bfm.main(["--year", "2026", "--groups", "M01", "--data-dir", str(data_dir)])

            self.assertEqual(saved, ["M01_V002"])


if __name__ == "__main__":
    unittest.main()
