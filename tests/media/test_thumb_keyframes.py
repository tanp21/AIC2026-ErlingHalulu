"""`/thumb` serves the organisers' own keyframe image when the offline store has one.

The N traffic-camera videos carry no usable frame timestamps, so ffmpeg extracts nothing from
them (a seek decodes no frame at all) -- the stored keyframes are why an N hit has a thumbnail.
"""
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from backend import thumb


class StoredKeyframeTests(unittest.TestCase):
    def test_reads_the_image_for_the_requested_width(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "320" / "N029" / "N029_V002" / "248.jpg"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"jpeg-bytes")
            with patch.object(thumb, "KEYFRAME_DIR", Path(tmp)):
                self.assertEqual(thumb.stored_keyframe("N029_V002", 248, 320), b"jpeg-bytes")
                self.assertIsNone(thumb.stored_keyframe("N029_V002", 249, 320),
                                  "a frame with no stored keyframe falls through to the video")
                self.assertIsNone(thumb.stored_keyframe("N029_V002", 248, 160),
                                  "widths are built separately, not resized at request time")

    def test_an_absent_store_is_not_an_error(self):
        with patch.object(thumb, "KEYFRAME_DIR", Path("/nonexistent-keyframe-store")):
            self.assertIsNone(thumb.stored_keyframe("N029_V002", 0, 320))


class ExtractionCommandTests(unittest.TestCase):
    """N is extracted by decoded position; every other group by exact PTS."""

    def test_a_traffic_camera_video_is_extracted_without_a_seek(self):
        command = thumb.ffmpeg_command("/data/2026/videos/N029/N029_V002.mov", 248,
                                       self._mapping(), "/tmp/out.jpg", 320)

        self.assertNotIn("-ss", command)
        self.assertIn("select=eq(n\\,248),scale=320:-2", command)

    def test_a_pts_video_keeps_the_seek_and_the_pts_match(self):
        command = thumb.ffmpeg_command("/data/2026/videos/L21/L21_V001.mp4", 1,
                                       self._mapping(), "/tmp/out.jpg", 320)

        self.assertIn("-ss", command)
        self.assertTrue(any(part.startswith("select=eq(pts\\,") for part in command))
        self.assertTrue(thumb.carries_no_frame_timestamps("/data/x/N029_V002.mov"))
        self.assertFalse(thumb.carries_no_frame_timestamps("/data/x/L21_V001.mp4"))

    @staticmethod
    def _mapping():
        class Mapping:
            pts = [1000, 2000, 3000]

            @staticmethod
            def source_time(frame_id):
                return frame_id / 10

        return Mapping()


class ThumbRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from backend.access_control import AccessConfig
        from backend.app import app

        os.environ.setdefault(
            "AIC_ALLOWED_USERS",
            "owner@example.com,u2@example.com,u3@example.com,u4@example.com,u5@example.com",
        )
        os.environ.setdefault("AIC_ADMIN_USER", "owner@example.com")
        app.state.access_config = AccessConfig.from_env()
        cls.client = TestClient(app)

    def get(self, url):
        return self.client.get(url, headers={"X-AIC-User": "owner@example.com"})

    def test_serves_a_stored_keyframe_without_touching_the_video(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "320" / "N029" / "N029_V002" / "248.jpg"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"jpeg-bytes")
            with patch.object(thumb, "KEYFRAME_DIR", Path(tmp)), \
                 patch("backend.app.resolve", side_effect=HTTPException(404, "no such video")):
                response = self.get("/thumb/N029_V002/248?w=320")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"jpeg-bytes")
        self.assertEqual(response.headers["content-type"], "image/jpeg")

    def test_frame_serves_the_largest_stored_keyframe(self):
        with TemporaryDirectory() as tmp:
            # 640 only: a host can hold a subset of the widths, and /frame takes the best there.
            path = Path(tmp) / "640" / "N029" / "N029_V002" / "248.jpg"
            path.parent.mkdir(parents=True)
            path.write_bytes(b"big-jpeg")
            with patch.object(thumb, "KEYFRAME_DIR", Path(tmp)), \
                 patch("backend.app.resolve", side_effect=HTTPException(404, "no such video")):
                response = self.get("/frame/N029_V002/248")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"big-jpeg")

    def test_frame_prefers_the_widest_stored_keyframe(self):
        with TemporaryDirectory() as tmp:
            for width, body in ((320, b"small"), (1280, b"large")):
                path = Path(tmp) / str(width) / "N029" / "N029_V002" / "248.jpg"
                path.parent.mkdir(parents=True)
                path.write_bytes(body)
            with patch.object(thumb, "KEYFRAME_DIR", Path(tmp)):
                self.assertEqual(thumb.stored_keyframe_largest("N029_V002", 248), b"large")

    def test_falls_back_to_the_video_when_nothing_is_stored(self):
        with patch.object(thumb, "KEYFRAME_DIR", Path("/nonexistent-keyframe-store")), \
             patch("backend.app.resolve", return_value={"path": "/data/L21_V001.mp4"}), \
             patch("backend.app.frame_maps.load", return_value=object()), \
             patch("backend.app.thumb.extract", return_value=b"from-video") as extract:
            response = self.get("/thumb/L21_V001/100?w=320")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"from-video")
        extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()
