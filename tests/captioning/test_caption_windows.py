import math
import unittest

from scripts.captioning import caption_shots, windows


def record(shot_index=0, start_frame=0, frames=100, fps=25.0):
    return caption_shots.ShotRecord(
        "2026", "L21", "L21_V001", shot_index,
        start_frame, start_frame + frames - 1, fps, "/video.mp4",
    )


class WindowSpanTests(unittest.TestCase):
    def test_short_shot_stays_one_window(self):
        self.assertEqual(windows.window_spans(4.0), [(0.0, 4.0)])
        self.assertEqual(windows.window_count(20.0), 1)

    def test_windows_cover_the_shot_exactly_and_overlap_by_two_seconds(self):
        for duration in (20.5, 31.4, 60.0, 415.0, 662.6):
            spans = windows.window_spans(duration)
            self.assertAlmostEqual(spans[0][0], 0.0)
            self.assertAlmostEqual(spans[-1][1], duration)
            lengths = {round(end - start, 6) for start, end in spans}
            self.assertEqual(len(lengths), 1, f"uneven windows at {duration}")
            self.assertLessEqual(max(lengths), windows.WINDOW_SECONDS + 1e-9)
            for before, after in zip(spans, spans[1:]):
                self.assertAlmostEqual(before[1] - after[0], windows.OVERLAP_SECONDS)

    def test_a_barely_long_shot_does_not_produce_two_near_duplicate_windows(self):
        # A fixed 18 s stride would give (0, 20) and (0.5, 20.5): two captions of the
        # same thing. Shrinking the window instead splits 20.5 s down the middle.
        spans = windows.window_spans(20.5)
        self.assertEqual(len(spans), 2)
        self.assertAlmostEqual(spans[0][1] - spans[0][0], 11.25)
        self.assertAlmostEqual(spans[1][0], 9.25)

    def test_window_count_matches_the_stride_formula(self):
        stride = windows.WINDOW_SECONDS - windows.OVERLAP_SECONDS
        for duration in (21.0, 40.0, 137.0, 662.6, 1178.0):
            self.assertEqual(
                windows.window_count(duration),
                math.ceil((duration - windows.WINDOW_SECONDS) / stride) + 1,
            )


class FrameSamplingTests(unittest.TestCase):
    def test_frame_count_is_flat_at_eight(self):
        # 8 frames regardless of window length: the exact recipe of the
        # "1024-default-8frames" run picked as the quality baseline.
        for duration in (1.5, 2.0, 4.9, 9.9, 20.0):
            self.assertEqual(windows.frames_for(duration), 8)


class BuildUnitTests(unittest.TestCase):
    def test_unsplit_shot_keeps_the_one_row_per_shot_contract(self):
        units = windows.build_units([record(frames=100, fps=25.0)])

        self.assertEqual(len(units), 1)
        unit = units[0]
        self.assertEqual((unit.window_index, unit.window_count), (0, 1))
        # Same half-frame padding ShotRecord applies, so the ffmpeg window is unchanged.
        self.assertAlmostEqual(unit.window_start_sec, 0.0)
        self.assertAlmostEqual(unit.window_end_sec, (100 - 0.5 + 0.75) / 25.0, places=6)

    def test_long_shot_windows_stay_inside_the_shot(self):
        # 600 frames at 10 fps = 60 s, starting 1000 frames in.
        units = windows.build_units([record(start_frame=1000, frames=600, fps=10.0)])

        self.assertEqual(len(units), 4)
        self.assertTrue(all(u.window_count == 4 for u in units))
        self.assertEqual([u.window_index for u in units], [0, 1, 2, 3])
        self.assertGreaterEqual(units[0].window_start_sec, 100.0 - 0.05)
        self.assertLessEqual(units[-1].window_end_sec, 160.0 + 0.05)
        self.assertTrue(all(u.frame_count == 8 for u in units))

    def test_summary_counts_shots_units_and_splits(self):
        units = windows.build_units([
            record(shot_index=0, frames=50, fps=25.0),
            record(shot_index=1, start_frame=50, frames=2000, fps=25.0),
        ])

        summary = windows.summarize(units)

        self.assertEqual(summary["shots"], 2)
        self.assertEqual(summary["split_shots"], 1)
        self.assertEqual(summary["units"], len(units))
        self.assertGreater(summary["max_windows"], 1)


if __name__ == "__main__":
    unittest.main()
