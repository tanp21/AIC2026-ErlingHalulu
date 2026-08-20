import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import core
import frame_maps
import thumb


class RuntimeMountTests(unittest.TestCase):
    def test_video_metadata_path_is_rebased_to_data_mount(self):
        row = {"year": "2026", "group": "L21", "video_id": "L21_V015", "path": "/host/secret"}
        self.assertEqual(
            core.mounted_video_path(row, Path("/data")),
            Path("/data/2026/videos/L21/L21_V015.mp4"),
        )

    def test_frame_map_directory_follows_index_mount(self):
        self.assertEqual(
            frame_maps.map_directory(Path("/indexes")),
            Path("/indexes/frame_maps"),
        )


class ThumbnailConcurrencyTests(unittest.TestCase):
    def test_concurrent_same_thumbnail_is_generated_once(self):
        mapping = frame_maps.FrameMap(np.array([0, 10]), 1, 10)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            thumb, "CACHE_DIR", Path(directory)
        ), patch("thumb.subprocess.run") as run:
            def generate(command, **_):
                time.sleep(0.02)
                Path(command[-1]).write_bytes(b"jpeg")

            run.side_effect = generate
            results = []
            workers = [
                threading.Thread(target=lambda: results.append(thumb.extract("video.mp4", 1, mapping)))
                for _ in range(5)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        self.assertEqual(run.call_count, 1)
        self.assertEqual(results, [b"jpeg"] * 5)

    def test_ffmpeg_has_a_timeout(self):
        mapping = frame_maps.FrameMap(np.array([0, 10]), 1, 10)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            thumb, "CACHE_DIR", Path(directory)
        ), patch("thumb.subprocess.run") as run:
            run.side_effect = lambda command, **_: Path(command[-1]).write_bytes(b"jpeg")
            thumb.extract("video.mp4", 1, mapping)
        self.assertEqual(run.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
