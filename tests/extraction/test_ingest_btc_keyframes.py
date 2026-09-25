import io
import json
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq
from PIL import Image

from scripts.extraction import ingest_btc_keyframes as ingest


def make_jpeg(color=(10, 20, 30), size=(32, 24)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_map(path, rows, header="n,pts_time,fps,frame_idx"):
    lines = [header] + [",".join(str(value) for value in row) for row in rows]
    Path(path).write_text("\n".join(lines) + "\n")


class VideoIdTests(unittest.TestCase):
    def test_member_video_id_is_read_off_the_path(self):
        self.assertEqual(ingest.btc_video_id("keyframes/N001-V001/079.jpg"), "N001-V001")

    def test_non_keyframe_members_are_ignored(self):
        self.assertIsNone(ingest.btc_video_id("keyframes/N001-V001/"))
        self.assertIsNone(ingest.btc_video_id("readme.txt"))

    def test_hyphenated_id_is_normalized_to_the_repo_form(self):
        self.assertEqual(ingest.repo_video_id("N001-V001"), "N001_V001")

    def test_underscore_id_passes_through(self):
        self.assertEqual(ingest.repo_video_id("M05_V031"), "M05_V031")

    def test_group_is_the_prefix_before_the_first_underscore(self):
        self.assertEqual(ingest.group_of("N001_V001"), "N001")


class ReadMapKeyframesTests(unittest.TestCase):
    def test_reads_ordinals_into_frame_idx_and_milliseconds(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "N001-V001.csv"
            write_map(path, [(1, 0.0, 25.0, 0), (2, 2.0, 25.0, 50), (3, 4.5, 25.0, 112)])
            rows, derived = ingest.read_map_keyframes(path)
        self.assertFalse(derived)
        self.assertEqual(rows, {1: (0, 0), 2: (50, 2000), 3: (112, 4500)})

    def test_derives_frame_idx_from_pts_when_the_column_is_blank(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "N001-V001.csv"
            write_map(path, [(1, 0.0, 25.0, ""), (2, 3.0, 25.0, "")])
            rows, derived = ingest.read_map_keyframes(path)
        self.assertTrue(derived)
        self.assertEqual(rows, {1: (0, 0), 2: (75, 3000)})

    def test_derives_frame_idx_when_the_column_is_absent(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "N001-V001.csv"
            write_map(path, [(1, 2.0, 25.0), (2, 4.0, 25.0)], header="n,pts_time,fps")
            rows, derived = ingest.read_map_keyframes(path)
        self.assertTrue(derived)
        self.assertEqual(rows, {1: (50, 2000), 2: (100, 4000)})

    def test_rejects_a_map_without_the_required_columns(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "N001-V001.csv"
            write_map(path, [(1, 0.0)], header="n,pts_time")
            with self.assertRaises(ValueError):
                ingest.read_map_keyframes(path)

    def test_rejects_a_repeated_ordinal(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "N001-V001.csv"
            write_map(path, [(1, 0.0, 25.0, 0), (1, 2.0, 25.0, 50)])
            with self.assertRaises(ValueError):
                ingest.read_map_keyframes(path)

    def test_rejects_an_empty_map(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "N001-V001.csv"
            write_map(path, [])
            with self.assertRaises(ValueError):
                ingest.read_map_keyframes(path)


class CheckVideoRowsTests(unittest.TestCase):
    def test_accepts_a_monotonic_map(self):
        rows = {1: (0, 0), 2: (50, 2000), 3: (100, 4000)}
        self.assertEqual(ingest.check_video_rows("N001_V001", rows),
                         [(1, (0, 0)), (2, (50, 2000)), (3, (100, 4000))])

    def test_rejects_two_ordinals_sharing_a_frame_idx(self):
        # Both would be written as the same file name, so one keyframe would vanish.
        rows = {1: (0, 0), 2: (0, 2000)}
        with self.assertRaises(ValueError):
            ingest.check_video_rows("N001_V001", rows)

    def test_rejects_pts_that_do_not_increase(self):
        rows = {1: (0, 4000), 2: (50, 2000)}
        with self.assertRaises(ValueError):
            ingest.check_video_rows("N001_V001", rows)


class CheckImagesMatchTests(unittest.TestCase):
    def test_accepts_images_that_cover_the_map_exactly(self):
        ingest.check_images_match("N001_V001", {1: "a", 2: "b"}, {1: (0, 0), 2: (50, 2000)})

    def test_rejects_a_missing_image(self):
        with self.assertRaises(ValueError):
            ingest.check_images_match("N001_V001", {1: "a"}, {1: (0, 0), 2: (50, 2000)})

    def test_rejects_an_image_with_no_map_row(self):
        with self.assertRaises(ValueError):
            ingest.check_images_match("N001_V001", {1: "a", 2: "b", 3: "c"},
                                      {1: (0, 0), 2: (50, 2000)})


class MainTests(unittest.TestCase):
    def build(self, tmp):
        """One zip with two videos, and a map-keyframes dir covering both."""
        zips = Path(tmp) / "zips"
        zips.mkdir()
        maps = Path(tmp) / "map-keyframes"
        maps.mkdir()
        with zipfile.ZipFile(zips / "Keyframes_N001-N010.zip", "w") as archive:
            for ordinal in (1, 2, 3):
                archive.writestr(f"keyframes/N001-V001/{ordinal:03d}.jpg", make_jpeg())
            for ordinal in (1, 2):
                archive.writestr(f"keyframes/N001-V002/{ordinal:03d}.jpg",
                                 make_jpeg((1, 2, 3)))
        # N001-V001 strides unevenly, so frame_idx is not simply ordinal * step.
        write_map(maps / "N001-V001.csv",
                  [(1, 0.0, 25.0, 0), (2, 2.0, 25.0, 50), (3, 8.5, 25.0, 212)])
        write_map(maps / "N001-V002.csv", [(1, 0.0, 25.0, 0), (2, 2.0, 25.0, 50)])
        return zips, maps

    def test_writes_pngs_named_by_frame_idx_and_a_points_table(self):
        with TemporaryDirectory() as tmp:
            zips, maps = self.build(tmp)
            keyframes = Path(tmp) / "keyframes"
            points = Path(tmp) / "points.parquet"
            ingest.main([
                "--zips-dir", str(zips), "--map-keyframes-dir", str(maps),
                "--keyframes-dir", str(keyframes), "--points", str(points),
            ])

            written = sorted(path.name for path in
                             (keyframes / "N001" / "N001_V001").glob("*.png"))
            self.assertEqual(written, ["000000.png", "000050.png", "000212.png"])
            with Image.open(keyframes / "N001" / "N001_V001" / "000212.png") as image:
                self.assertEqual(image.size, (32, 24))

            rows = pq.read_table(points).to_pylist()
            self.assertEqual([row["video_id"] for row in rows],
                             ["N001_V001"] * 3 + ["N001_V002"] * 2)
            self.assertEqual(rows[2], {"video_id": "N001_V001", "group": "N001",
                                       "keyframe_n": 3, "frame_idx": 212, "pts_ms": 8500})

    def test_groups_filter_selects_videos(self):
        with TemporaryDirectory() as tmp:
            zips, maps = self.build(tmp)
            keyframes = Path(tmp) / "keyframes"
            ingest.main([
                "--zips-dir", str(zips), "--map-keyframes-dir", str(maps),
                "--keyframes-dir", str(keyframes), "--points", str(Path(tmp) / "p.parquet"),
                "--groups", "N099",
            ])
            self.assertFalse(keyframes.exists())

    def test_second_run_is_a_no_op(self):
        with TemporaryDirectory() as tmp:
            zips, maps = self.build(tmp)
            keyframes = Path(tmp) / "keyframes"
            points = Path(tmp) / "points.parquet"
            argv = ["--zips-dir", str(zips), "--map-keyframes-dir", str(maps),
                    "--keyframes-dir", str(keyframes), "--points", str(points)]
            ingest.main(argv)
            first = sorted(path.stat().st_mtime_ns for path in keyframes.rglob("*.png"))

            ingest.main(argv)

            self.assertEqual(sorted(path.stat().st_mtime_ns
                                    for path in keyframes.rglob("*.png")), first)
            self.assertEqual(pq.read_table(points).num_rows, 5)

    def test_dry_run_writes_nothing(self):
        with TemporaryDirectory() as tmp:
            zips, maps = self.build(tmp)
            keyframes = Path(tmp) / "keyframes"
            ingest.main(["--zips-dir", str(zips), "--map-keyframes-dir", str(maps),
                         "--keyframes-dir", str(keyframes), "--dry-run"])
            self.assertFalse(keyframes.exists())


if __name__ == "__main__":
    unittest.main()
