"""Shot captions: the per-video lookup the agent reads as context.

The corpus is read one video at a time because `shot_captions.parquet` holds all
103,086 rows in a single row group — reading it costs ~5 s and leaves ~900 MB
resident, against ~20 ms and nothing for one video's part file. These tests pin
that lookup, including the two cases where there is simply no caption to give.
"""
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from backend import shots


def write_part(root: Path, video_id: str, rows: list[dict]) -> None:
    """One `parts/<group>/<video_id>.parquet` with the columns the lookup reads."""
    path = root / "2026" / "captions" / "parts" / video_id[:3] / f"{video_id}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [] for name in shots.CAPTION_COLUMNS}
    for row in rows:
        for name in columns:
            columns[name].append(row.get(name))
    pq.write_table(pa.Table.from_pydict(columns), path)


def window(shot_index, start, end, caption, window_index=0, window_count=1):
    return {"shot_index": shot_index, "window_index": window_index,
            "window_count": window_count, "window_start_sec": start,
            "window_end_sec": end, "start_sec": start, "end_sec": end,
            "caption": caption}


class CaptionPathTests(unittest.TestCase):
    def test_2026_videos_resolve_to_their_part_file(self):
        path = shots.captions_path("L22_V001", data_dir=Path("/data"))
        self.assertEqual(path, Path("/data/2026/captions/parts/L22/L22_V001.parquet"))

    def test_a_2025_video_resolves_to_nothing(self):
        # 2025 was never captioned. Returning a path that cannot exist would make
        # every miss look like a missing file instead of a known gap.
        self.assertIsNone(shots.captions_path("K01_V001", data_dir=Path("/data")))

    def test_an_unrecognised_prefix_resolves_to_nothing(self):
        self.assertIsNone(shots.captions_path("Z99_V001", data_dir=Path("/data")))


class CaptionLookupTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_the_window_containing_the_time_is_chosen(self):
        write_part(self.root, "L21_V001", [
            window(0, 0.0, 10.0, "First scene."),
            window(1, 10.0, 20.0, "Second scene."),
        ])
        found = shots.caption_for_frame("L21_V001", 12.5, data_dir=self.root)
        self.assertEqual(found["caption"], "Second scene.")
        self.assertEqual(found["shot_index"], 1)

    def test_a_split_shot_gives_the_window_the_frame_sits_deepest_in(self):
        # Windows of one shot overlap by 2 s; the frame at 9.0 is inside both, and
        # the first window is the one it belongs to.
        write_part(self.root, "L21_V001", [
            window(7, 0.0, 10.0, "Opening half.", window_index=0, window_count=2),
            window(7, 8.0, 18.0, "Closing half.", window_index=1, window_count=2),
        ])
        self.assertEqual(
            shots.caption_for_frame("L21_V001", 9.0, data_dir=self.root)["caption"],
            "Opening half.")
        self.assertEqual(
            shots.caption_for_frame("L21_V001", 17.0, data_dir=self.root)["caption"],
            "Closing half.")

    def test_a_time_outside_every_window_has_no_caption(self):
        write_part(self.root, "L21_V001", [window(0, 0.0, 10.0, "Only scene.")])
        self.assertIsNone(shots.caption_for_frame("L21_V001", 99.0, data_dir=self.root))

    def test_a_video_without_a_part_file_has_no_caption(self):
        self.assertIsNone(shots.caption_for_frame("L21_V777", 1.0, data_dir=self.root))

    def test_an_empty_caption_is_not_offered(self):
        write_part(self.root, "L21_V001", [window(0, 0.0, 10.0, "")])
        self.assertIsNone(shots.caption_for_frame("L21_V001", 5.0, data_dir=self.root))

    def test_the_captioners_preamble_is_stripped_on_the_way_out(self):
        write_part(self.root, "L21_V001", [window(0, 0.0, 10.0,
                                                  "Based on the provided video frames, here is "
                                                  "the detailed description in chronological "
                                                  "order: A red boat drifts downstream.")])
        found = shots.caption_for_frame("L21_V001", 5.0, data_dir=self.root)
        self.assertEqual(found["caption"], "A red boat drifts downstream.")

    def test_a_corrupt_part_file_is_treated_as_no_caption(self):
        path = self.root / "2026" / "captions" / "parts" / "L21" / "L21_V001.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a parquet file")
        self.assertIsNone(shots.caption_for_frame("L21_V001", 5.0, data_dir=self.root))


class CaptionRouteTests(unittest.TestCase):
    """The route the agent reads captions through.

    Both cases here used to answer 500: an out-of-range frame raised IndexError out
    of `FrameMap.frame_time`, and a video with no frame map raised FileNotFoundError
    from `np.load`, which the route's `except KeyError` did not catch. The agent
    tolerates a 500 only by reporting no captions, so the same request would be
    re-issued for every later batch touching that video.
    """

    @classmethod
    def setUpClass(cls):
        import os

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

    def test_an_indexed_video_answers_with_its_captions(self):
        response = self.get("/captions/L22_V001/frames?frames=1000")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["available"])
        self.assertIn("caption", body["captions"]["1000"])

    def test_a_frame_past_the_end_of_the_map_is_not_a_server_error(self):
        response = self.get("/captions/L22_V001/frames?frames=999999")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"video_id": "L22_V001", "available": False,
                                           "captions": {}})

    def test_a_video_without_a_frame_map_is_not_a_server_error(self):
        response = self.get("/captions/L99_V999/frames?frames=0")
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["available"])

    def test_a_2025_video_answers_unavailable(self):
        response = self.get("/captions/K01_V001/frames?frames=1000")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["captions"], {})

    def test_a_malformed_frame_list_is_rejected(self):
        self.assertEqual(self.get("/captions/L22_V001/frames?frames=abc").status_code, 422)


if __name__ == "__main__":
    unittest.main()
