import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.embedding import embed_batch2_stream as stream


class FakeWriter:
    def __init__(self, done):
        self.done = set(done)


class GroupKeyframeIdsTests(unittest.TestCase):
    def test_collects_every_shot_derived_frame_for_the_group(self):
        with TemporaryDirectory() as tmp:
            shots_dir = Path(tmp) / "shots"
            (shots_dir / "M01").mkdir(parents=True)
            (shots_dir / "M01" / "M01_V001.json").write_text(json.dumps({
                "fps": 25.0, "shots": [[0, 20]],
            }))
            (shots_dir / "S01").mkdir(parents=True)
            (shots_dir / "S01" / "S01_V001.json").write_text(json.dumps({
                "fps": 25.0, "shots": [[0, 20]],
            }))

            expected = stream.group_keyframe_ids(shots_dir, "M01")

            self.assertEqual(expected, {("M01_V001", f) for f in
                                        stream.compute_keyframes([(0, 20)])})


class GroupFullyCommittedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.shots_dir = Path(self.tmp.name) / "shots"
        (self.shots_dir / "M01").mkdir(parents=True)
        (self.shots_dir / "M01" / "M01_V001.json").write_text(json.dumps({
            "fps": 25.0, "shots": [[0, 20]],
        }))
        self.expected = stream.group_keyframe_ids(self.shots_dir, "M01")

    def tearDown(self):
        self.tmp.cleanup()

    def test_true_when_every_frame_is_in_both_writers(self):
        pe_core = FakeWriter(self.expected)
        naflex = FakeWriter(self.expected)

        self.assertTrue(stream.group_fully_committed(self.shots_dir, "M01", pe_core, naflex))

    def test_false_when_naflex_is_missing_a_frame(self):
        pe_core = FakeWriter(self.expected)
        naflex = FakeWriter(set(list(self.expected)[:-1]))

        self.assertFalse(stream.group_fully_committed(self.shots_dir, "M01", pe_core, naflex))

    def test_false_when_group_has_no_shots(self):
        pe_core = FakeWriter(set())
        naflex = FakeWriter(set())

        self.assertFalse(stream.group_fully_committed(self.shots_dir, "S99", pe_core, naflex))


class ExtractGroupKeyframesTests(unittest.TestCase):
    def test_raises_when_extraction_fails_for_any_video(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shots_dir, videos_dir, out_dir = root / "shots", root / "videos", root / "kf"
            (shots_dir / "M01").mkdir(parents=True)
            (videos_dir / "M01").mkdir(parents=True)
            (videos_dir / "M01" / "M01_V001.mp4").touch()
            (shots_dir / "M01" / "M01_V001.json").write_text(json.dumps({
                "fps": 25.0, "shots": [[0, 20]],
            }))

            with patch.object(stream, "extract_video_batch", return_value=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    stream.extract_group_keyframes(
                        "M01", shots_dir, videos_dir, out_dir, workers=2, threads=1)

    def test_skips_videos_with_no_matching_shot_file_silently(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shots_dir, videos_dir, out_dir = root / "shots", root / "videos", root / "kf"
            (shots_dir / "M01").mkdir(parents=True)
            (videos_dir / "M01").mkdir(parents=True)
            # No .json files at all under shots/M01 -> zero jobs, not an error.

            with patch.object(stream, "extract_video_batch") as mock_extract:
                total = stream.extract_group_keyframes(
                    "M01", shots_dir, videos_dir, out_dir, workers=2, threads=1)

            mock_extract.assert_not_called()
            self.assertEqual(total, 0)


if __name__ == "__main__":
    unittest.main()
