import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.extraction import extract_batch2_keyframes as extractor


class ComputeKeyframesTests(unittest.TestCase):
    def test_three_positions_per_shot_clamped_to_bounds(self):
        frames = extractor.compute_keyframes([(0, 20)])
        self.assertEqual(frames, [3, 10, 17])

    def test_short_shot_still_produces_frames_within_bounds(self):
        frames = extractor.compute_keyframes([(5, 6)])
        for f in frames:
            self.assertTrue(5 <= f <= 6)

    def test_multiple_shots_are_sorted_and_deduplicated(self):
        frames = extractor.compute_keyframes([(0, 10), (10, 10)])
        self.assertEqual(frames, sorted(set(frames)))
        self.assertEqual(frames[-1], 10)


class IterChunksTests(unittest.TestCase):
    def test_splits_purely_sparse_frames_by_time_span_not_just_count(self):
        # 30fps, one frame every 60s for 200 frames -> way under the 100-count limit but
        # each successive frame is 2s apart, so 120s/2s = 60 frames per chunk max.
        fps = 30.0
        frames = [i * 60 for i in range(200)]  # 0, 60, 120, ... every 2s of video

        chunks = list(extractor.iter_chunks(frames, fps))

        self.assertGreater(len(chunks), 2)  # far more than ceil(200/100)=2
        for chunk in chunks:
            span_seconds = (chunk[-1] - chunk[0]) / fps
            self.assertLessEqual(span_seconds, extractor.MAX_CHUNK_SPAN_SECONDS)
            self.assertLessEqual(len(chunk), extractor.CHUNK)
        self.assertEqual(sorted(f for c in chunks for f in c), frames)

    def test_dense_frames_still_chunk_by_count(self):
        fps = 30.0
        frames = list(range(250))  # consecutive frames, ~8.3s total -> count-bound, not time

        chunks = list(extractor.iter_chunks(frames, fps))

        self.assertEqual([len(c) for c in chunks], [100, 100, 50])

    def test_empty_input_yields_no_chunks(self):
        self.assertEqual(list(extractor.iter_chunks([], 30.0)), [])

    def test_single_frame_is_its_own_chunk(self):
        self.assertEqual(list(extractor.iter_chunks([42], 30.0)), [[42]])


class CollectJobsTests(unittest.TestCase):
    def test_pairs_shots_with_existing_videos_only(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shots_dir = root / "shots"
            videos_dir = root / "videos"
            keyframes_dir = root / "keyframes"
            (shots_dir / "M01").mkdir(parents=True)
            (videos_dir / "M01").mkdir(parents=True)
            (videos_dir / "M01" / "M01_V001.mp4").touch()

            (shots_dir / "M01" / "M01_V001.json").write_text(json.dumps({
                "fps": 25.0, "shots": [[0, 20]],
            }))
            (shots_dir / "M01" / "M01_V002.json").write_text(json.dumps({
                "fps": 25.0, "shots": [[0, 20]],
            }))  # no matching video -> must be skipped

            jobs = extractor.collect_jobs(shots_dir, videos_dir, keyframes_dir, groups=set())

            self.assertEqual(len(jobs), 1)
            video_path, frames, out_dir, fps = jobs[0]
            self.assertEqual(video_path.name, "M01_V001.mp4")
            self.assertEqual(frames, extractor.compute_keyframes([(0, 20)]))
            self.assertEqual(out_dir, keyframes_dir / "M01" / "M01_V001")
            self.assertEqual(fps, 25.0)

    def test_group_filter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            shots_dir, videos_dir, keyframes_dir = root / "shots", root / "videos", root / "kf"
            for group, video_id in (("M01", "M01_V001"), ("S01", "S01_V001")):
                (shots_dir / group).mkdir(parents=True)
                (videos_dir / group).mkdir(parents=True)
                (videos_dir / group / f"{video_id}.mp4").touch()
                (shots_dir / group / f"{video_id}.json").write_text(
                    json.dumps({"fps": 25.0, "shots": [[0, 10]]}))

            jobs = extractor.collect_jobs(shots_dir, videos_dir, keyframes_dir, groups={"S01"})

            self.assertEqual(len(jobs), 1)
            self.assertIn("S01_V001", str(jobs[0][2]))


if __name__ == "__main__":
    unittest.main()
