import unittest

from scripts.captioning import frames, windows


def unit(duration=4.0, fps=25.0, frame_count=12, width=0):
    return windows.CaptionUnit(
        "2026", "L21", "L21_V001", 0, 0, int(duration * fps) - 1, fps, "/video.mp4",
        0, 1, 10.0, 10.0 + duration, frame_count,
    )


class FramePositionTests(unittest.TestCase):
    def test_positions_are_uniform_and_inset_off_both_edges(self):
        # A shot boundary is not exact -- TransNetV2 can leave a sliver of the
        # previous/next scene right at the cut, so sampling stays off both edges
        # (observed live: L26_V448 shot 39 opening on the previous shot's kitchen
        # wide-shot before its own peppercorn close-up). 4 s at 25 fps is 100 frames,
        # minus the 1-frame tail margin = 99 available; 12% inset is 11 frames each side.
        positions = frames.frame_positions(unit(duration=4.0, fps=25.0, frame_count=12))

        self.assertEqual(len(positions), 12)
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(positions[0], 0)
        self.assertLess(positions[-1], 98)
        self.assertEqual(positions[0], 11)
        self.assertEqual(positions[-1], 87)

    def test_a_very_short_window_never_asks_for_more_frames_than_exist(self):
        positions = frames.frame_positions(unit(duration=0.4, fps=25.0, frame_count=8))

        self.assertLessEqual(len(positions), 9)
        self.assertGreaterEqual(len(positions), 1)

    def test_a_handful_of_frames_skips_the_inset_rather_than_starve(self):
        # Under EDGE_INSET_FLOOR there is nothing to spare -- inset would either starve
        # the sample or become a no-op after rounding, so it is skipped outright and the
        # first frame is used, same as before this change.
        positions = frames.frame_positions(unit(duration=0.1, fps=25.0, frame_count=4))

        self.assertEqual(positions[0], 0)


class FfmpegCommandTests(unittest.TestCase):
    def test_seeks_before_input_and_selects_exact_indices(self):
        command = frames.ffmpeg_command(unit(), [0, 5, 10], width=0)

        self.assertEqual(command[command.index("-ss") + 1], "10.000000")
        # -ss must precede -i or ffmpeg decodes the file from the start.
        self.assertLess(command.index("-ss"), command.index("-i"))
        self.assertIn("select=eq(n\\,0)+eq(n\\,5)+eq(n\\,10)", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "3")

    def test_scale_filter_only_appears_when_a_width_is_requested(self):
        native = frames.ffmpeg_command(unit(), [0], width=0)
        scaled = frames.ffmpeg_command(unit(), [0], width=960)

        self.assertNotIn("scale", native[native.index("-vf") + 1])
        self.assertIn("scale=w=min(iw\\,960):h=-2", scaled[scaled.index("-vf") + 1])


class JpegStreamTests(unittest.TestCase):
    def test_splits_a_concatenated_mjpeg_stream_on_its_markers(self):
        payload = b"\xff\xd8AAA\xff\xd9\xff\xd8BB\xff\xd9"

        self.assertEqual(frames.split_jpeg_stream(payload), [
            b"\xff\xd8AAA\xff\xd9", b"\xff\xd8BB\xff\xd9",
        ])

    def test_a_truncated_trailing_frame_is_dropped(self):
        payload = b"\xff\xd8AAA\xff\xd9\xff\xd8BB"

        self.assertEqual(len(frames.split_jpeg_stream(payload)), 1)


class CacheKeyTests(unittest.TestCase):
    def test_key_changes_with_the_pixels_and_not_with_anything_else(self):
        base = unit()
        self.assertEqual(frames.cache_key(base, 960, 2), frames.cache_key(base, 960, 2))
        self.assertNotEqual(frames.cache_key(base, 960, 2),
                            frames.cache_key(base, 640, 2))
        self.assertNotEqual(frames.cache_key(base, 960, 2),
                            frames.cache_key(base, 960, 5))


if __name__ == "__main__":
    unittest.main()
