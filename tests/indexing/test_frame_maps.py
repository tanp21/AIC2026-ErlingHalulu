import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from backend import frame_maps
from scripts.indexing import build_frame_maps


class FrameMapTests(unittest.TestCase):
    def test_loads_pts_and_resolves_exact_frame_without_fps(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "V.npz"
            np.savez_compressed(
                path,
                pts=np.array([9000, 12003, 15000], dtype=np.int64),
                time_base_num=np.int64(1),
                time_base_den=np.int64(90000),
            )

            mapping = frame_maps.load_path(path)

            self.assertEqual(mapping.frame_time(0), 0)
            self.assertEqual(mapping.frame_time(1), 3003 / 90000)
            self.assertEqual(mapping.frame_at_media_time(6000 / 90000), 2)
            self.assertIsNone(mapping.frame_at_media_time(6001 / 90000))

    def test_rejects_duplicate_or_non_monotonic_pts(self):
        for pts in ([0, 1, 1], [0, 2, 1]):
            with self.subTest(pts=pts):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    frame_maps.FrameMap(np.array(pts), 1, 1000)

    def test_payload_uses_integer_pts_deltas(self):
        mapping = frame_maps.FrameMap(np.array([100, 104, 111]), 1, 1000)

        self.assertEqual(mapping.payload("V"), {
            "videoId": "V",
            "timeBaseNumerator": 1,
            "timeBaseDenominator": 1000,
            "firstPts": 100,
            "ptsDeltas": [4, 7],
            "frameCount": 3,
        })

    def test_parses_ffprobe_best_effort_pts_and_time_base(self):
        mapping = build_frame_maps.parse_probe({
            "streams": [{"time_base": "1/12800"}],
            "frames": [
                {"best_effort_timestamp": "512"},
                {"best_effort_timestamp": "1024"},
            ],
        })

        self.assertEqual(mapping.time_base_num, 1)
        self.assertEqual(mapping.time_base_den, 12800)
        np.testing.assert_array_equal(mapping.pts, [512, 1024])

    def test_repairs_a_decoded_timeline_that_repeats_or_steps_backwards(self):
        mapping = build_frame_maps.parse_probe({
            "streams": [{"time_base": "1/12800"}],
            "frames": [
                {"best_effort_timestamp": "512"},
                {"best_effort_timestamp": "1024"},
                {"best_effort_timestamp": "900"},
                {"best_effort_timestamp": "900"},
            ],
        })

        # Sorted, then a repeat of the frame before it bumped one tick: one entry per frame,
        # still strictly increasing, and every frame that did not need it keeps its own time.
        np.testing.assert_array_equal(mapping.pts, [512, 900, 901, 1024])

    def test_builds_times_aligned_with_index_rows(self):
        maps = {
            "A": frame_maps.FrameMap(np.array([10, 20]), 1, 100),
            "B": frame_maps.FrameMap(np.array([100, 130]), 1, 1000),
        }

        times = build_frame_maps.aligned_times(
            np.array([1, 0, 1]),
            np.array([0, 1, 1]),
            {0: "A", 1: "B"},
            maps.__getitem__,
        )

        np.testing.assert_allclose(times, [0, 0.1, 0.03])

    def test_packet_pts_must_match_entire_decoded_map_before_reuse(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "V.npz"
            np.savez_compressed(path, pts=np.array([0, 512, 1024], dtype=np.int64),
                                time_base_num=np.int64(1), time_base_den=np.int64(12800))
            packets = build_frame_maps.parse_packet_probe({
                "streams": [{"time_base": "1/12800"}],
                "packets": [{"pts": 1024}, {"pts": 0}, {"pts": 512}],
            })
            with patch.object(build_frame_maps, "probe_packets", return_value=packets), patch.object(build_frame_maps, "probe") as decode:
                mapping, verified = build_frame_maps.verify_or_probe("video.mp4", path)
                self.assertTrue(verified)
                np.testing.assert_array_equal(mapping.pts, [0, 512, 1024])
                decode.assert_not_called()

            changed = frame_maps.FrameMap(np.array([0, 513, 1024]), 1, 12800)
            with patch.object(build_frame_maps, "probe_packets", return_value=changed), patch.object(build_frame_maps, "probe", return_value=changed) as decode:
                mapping, verified = build_frame_maps.verify_or_probe("video.mp4", path)
                self.assertFalse(verified)
                self.assertIs(mapping, changed)
                decode.assert_called_once()

    def test_packet_pts_without_frame_equivalent_falls_back_to_decode(self):
        with self.assertRaisesRegex(ValueError, "packet PTS unavailable"):
            build_frame_maps.parse_packet_probe({"streams": [{"time_base": "1/1000"}], "packets": [{"pts": 0}, {}]})


if __name__ == "__main__":
    unittest.main()
