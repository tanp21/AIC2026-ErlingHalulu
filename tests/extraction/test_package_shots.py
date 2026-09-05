import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from scripts.extraction import package_shots


def write_shot_json(root, year, group, video_id, payload):
    directory = root / year / "shots" / group
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{video_id}.json").write_text(json.dumps(payload))


class PackageShotsTests(unittest.TestCase):
    def test_packages_each_batch_into_its_own_zip_with_the_shots_layout(self):
        with TemporaryDirectory() as data_dir, TemporaryDirectory() as out_dir:
            data_dir, out_dir = Path(data_dir), Path(out_dir)
            write_shot_json(data_dir, "2026", "L21", "L21_V001", {"fps": 30.0, "shots": [[0, 9]]})
            write_shot_json(data_dir, "2026", "L22", "L22_V001", {"fps": 25.0, "shots": [[0, 5]]})
            write_shot_json(data_dir, "2025", "K01", "K01_V001", {"fps": 30.0, "shots": [[0, 3]]})

            written = [package_shots.package_batch(batch, data_dir, out_dir)
                       for batch in package_shots.BATCHES]

            self.assertEqual([path.name for path in written],
                              ["batch1_2026_shots.zip", "batch2_2025_shots.zip"])

            with ZipFile(out_dir / "batch1_2026_shots.zip") as archive:
                self.assertEqual(sorted(archive.namelist()), [
                    "shots/L21/L21_V001.json", "shots/L22/L22_V001.json",
                ])
                payload = json.loads(archive.read("shots/L21/L21_V001.json"))
                self.assertEqual(payload, {"fps": 30.0, "shots": [[0, 9]]})

            with ZipFile(out_dir / "batch2_2025_shots.zip") as archive:
                self.assertEqual(archive.namelist(), ["shots/K01/K01_V001.json"])

    def test_skips_a_batch_whose_dataset_has_no_shots_yet(self):
        with TemporaryDirectory() as data_dir, TemporaryDirectory() as out_dir:
            data_dir, out_dir = Path(data_dir), Path(out_dir)
            write_shot_json(data_dir, "2026", "L21", "L21_V001", {"fps": 30.0, "shots": []})

            batch1, batch2 = (
                package_shots.package_batch(batch, data_dir, out_dir)
                for batch in package_shots.BATCHES
            )

            self.assertIsNotNone(batch1)
            self.assertIsNone(batch2)
            self.assertFalse((out_dir / "batch2_2025_shots.zip").exists())

    def test_main_fails_loudly_when_no_shots_exist_anywhere(self):
        with TemporaryDirectory() as data_dir, TemporaryDirectory() as out_dir:
            with self.assertRaises(SystemExit):
                package_shots.main(["--data-dir", data_dir, "--out", out_dir])


if __name__ == "__main__":
    unittest.main()
