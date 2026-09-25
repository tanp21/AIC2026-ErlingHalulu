import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.extraction import ingest_zip_videos as ingest


class GroupFromZipNameTests(unittest.TestCase):
    def test_video_singular_prefix(self):
        self.assertEqual(ingest.group_from_zip_name("Video_S01.zip"), "S01")

    def test_videos_plural_prefix(self):
        self.assertEqual(ingest.group_from_zip_name("Videos_M01.zip"), "M01")

    def test_rejects_unexpected_name(self):
        with self.assertRaises(ValueError):
            ingest.group_from_zip_name("random.zip")


class NormalizedVideoIdTests(unittest.TestCase):
    def test_hyphen_is_normalized_to_underscore(self):
        self.assertEqual(ingest.normalized_video_id("videos/S01-V001.mp4", "S01"), "S01_V001")

    def test_underscore_passes_through(self):
        self.assertEqual(ingest.normalized_video_id("videos/M01_V001.mp4", "M01"), "M01_V001")

    def test_rejects_unexpected_member(self):
        with self.assertRaises(ValueError):
            ingest.normalized_video_id("videos/notes.txt", "S01")


class ExtractZipTests(unittest.TestCase):
    def _make_zip(self, path, names):
        with zipfile.ZipFile(path, "w") as archive:
            for name in names:
                archive.writestr(name, b"fake-mp4-bytes")

    def test_extracts_and_renames_hyphenated_members(self):
        with TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "Video_S01.zip"
            self._make_zip(zip_path, ["videos/S01-V001.mp4", "videos/S01-V002.mp4"])
            out_dir = Path(tmp) / "out"

            extracted, skipped, total = ingest.extract_zip(zip_path, "S01", out_dir, force=False)

            self.assertEqual((extracted, skipped, total), (2, 0, 2))
            self.assertTrue((out_dir / "S01_V001.mp4").exists())
            self.assertTrue((out_dir / "S01_V002.mp4").exists())
            self.assertEqual((out_dir / "S01_V001.mp4").read_bytes(), b"fake-mp4-bytes")

    def test_skips_existing_unless_forced(self):
        with TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "Videos_M01.zip"
            self._make_zip(zip_path, ["videos/M01_V001.mp4"])
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            (out_dir / "M01_V001.mp4").write_bytes(b"already-here")

            extracted, skipped, total = ingest.extract_zip(zip_path, "M01", out_dir, force=False)
            self.assertEqual((extracted, skipped, total), (0, 1, 1))
            self.assertEqual((out_dir / "M01_V001.mp4").read_bytes(), b"already-here")

            extracted, skipped, total = ingest.extract_zip(zip_path, "M01", out_dir, force=True)
            self.assertEqual((extracted, skipped, total), (1, 0, 1))
            self.assertEqual((out_dir / "M01_V001.mp4").read_bytes(), b"fake-mp4-bytes")


class CollectZipsTests(unittest.TestCase):
    def test_filters_by_only(self):
        with TemporaryDirectory() as tmp:
            for name in ("Video_S01.zip", "Videos_M01.zip", "Videos_M02.zip", "readme.txt"):
                (Path(tmp) / name).touch()

            zips = ingest.collect_zips(tmp, only={"M01"})

            self.assertEqual([group for _, group in zips], ["M01"])

    def test_no_filter_returns_all_matching_zips(self):
        with TemporaryDirectory() as tmp:
            for name in ("Video_S01.zip", "Videos_M01.zip", "other.zip"):
                (Path(tmp) / name).touch()

            zips = ingest.collect_zips(tmp, only=set())

            self.assertEqual(sorted(group for _, group in zips), ["M01", "S01"])


if __name__ == "__main__":
    unittest.main()
