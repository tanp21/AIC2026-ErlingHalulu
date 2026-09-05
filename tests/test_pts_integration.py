import unittest
from unittest.mock import patch

import numpy as np

from backend import core
from backend import frame_maps
from backend import thumb
from backend.app import get_frame_map


class PtsIntegrationTests(unittest.TestCase):
    def setUp(self):
        core._frames_by_video.clear()

    @patch("backend.core.frame_maps.load")
    @patch("backend.core.load_meta")
    def test_audio_timestamp_chooses_indexed_frame_by_pts(self, load_meta, load_map):
        load_meta.return_value = (
            np.array([0, 0]),
            np.array([1, 2]),
            {0: {"video_id": "V", "video_idx": 0}},
            {"V": {"video_id": "V", "video_idx": 0}},
        )
        load_map.return_value = frame_maps.FrameMap(np.array([0, 10, 25, 60]), 1, 100)

        self.assertEqual(core.nearest_frame("V", 0.24), 2)

    @patch("backend.app.frame_maps.load")
    def test_frame_map_endpoint_returns_delta_payload(self, load_map):
        load_map.return_value = frame_maps.FrameMap(np.array([100, 104, 111]), 1, 1000)

        self.assertEqual(get_frame_map("V")["ptsDeltas"], [4, 7])

    def test_thumbnail_command_selects_exact_pts(self):
        mapping = frame_maps.FrameMap(np.array([9000, 12003]), 1, 90000)

        command = thumb.ffmpeg_command("video.mp4", 1, mapping, "out.jpg", 320)
        joined = " ".join(command)

        self.assertNotIn("fps", joined)
        self.assertIn("-copyts", command)
        self.assertIn("eq(pts\\,12003)", joined)
        self.assertIn("0.133366666666667", joined)


if __name__ == "__main__":
    unittest.main()
