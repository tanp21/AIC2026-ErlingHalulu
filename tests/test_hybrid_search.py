import unittest

from backend.app import _audio_time_window, filter_search_results, fuse_search_results


class HybridFusionTests(unittest.TestCase):
    def test_ocr_filters_by_video_and_keeps_hits_within_the_same_shot(self):
        primary = [
            {
                "rank": 4, "score": 0.9, "video_id": "L21_V015", "time": 11.0,
                "source": "both", "scores": {"audio_rank": 2},
                "matched_keywords": ["spoken"],
            },
            {"rank": 5, "score": 0.8, "video_id": "L21_V015", "time": 19.5},
            {"rank": 6, "score": 0.7, "video_id": "L21_V016", "time": 15.0},
        ]
        ocr = [{
            "video_id": "L21_V015", "score": 3.5,
            "ocr_snippet": "THANH PHO", "ocr_shot_index": 12,
            "ocr_start": 10.0, "ocr_end": 20.0,
            "scores": {"ocr_score": 3.5, "ocr_rank": 1},
            "ocr_matched_keywords": ["THANH PHO"],
        }]

        filtered = filter_search_results(primary, ocr)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(filtered[0]["rank"], 0)
        self.assertEqual(filtered[1]["rank"], 1)
        self.assertEqual(filtered[0]["score"], 0.9)
        self.assertEqual(filtered[0]["matched_keywords"], ["spoken"])
        self.assertEqual(filtered[0]["ocr_matched_keywords"], ["THANH PHO"])
        self.assertEqual(filtered[0]["ocr_shot_index"], 12)
        self.assertEqual(filtered[0]["scores"]["audio_rank"], 2)
        self.assertEqual(filtered[0]["scores"]["ocr_rank"], 1)

    def test_ocr_drops_hits_outside_the_matching_shot(self):
        # Reproduces the reported bug: a visual hit at 22:30 in a video and
        # an OCR match confined to the 3s shot 3:10-3:13 are different
        # shots -- only one keyframe per shot is OCR'd, so a hit outside
        # that exact shot must not inherit the match, even in the same
        # video.
        primary = [{"rank": 0, "score": 0.9, "video_id": "L22_V016", "time": 1350.0}]
        ocr = [{
            "video_id": "L22_V016", "ocr_snippet": "thit", "ocr_shot_index": 40,
            "ocr_start": 190.0, "ocr_end": 193.0, "scores": {"ocr_rank": 1},
        }]

        filtered = filter_search_results(primary, ocr)

        self.assertEqual(filtered, [])

    def test_ocr_includes_hits_right_at_the_shot_boundary(self):
        primary = [
            {"rank": 0, "score": 0.9, "video_id": "L21_V015", "time": 10.0},
            {"rank": 1, "score": 0.8, "video_id": "L21_V015", "time": 20.0},
        ]
        ocr = [{
            "video_id": "L21_V015", "ocr_snippet": "edge", "ocr_shot_index": 1,
            "ocr_start": 10.0, "ocr_end": 20.0, "scores": {"ocr_rank": 1},
        }]

        filtered = filter_search_results(primary, ocr)

        self.assertEqual(len(filtered), 2)

    def test_ocr_prefers_the_highest_ranked_shot_when_several_contain_the_hit(self):
        primary = [{"rank": 0, "score": 0.9, "video_id": "L21_V015", "time": 10.0}]
        ocr = [
            {"video_id": "L21_V015", "ocr_snippet": "worse", "ocr_shot_index": 1,
             "ocr_start": 0.0, "ocr_end": 20.0, "scores": {"ocr_rank": 2}},
            {"video_id": "L21_V015", "ocr_snippet": "better", "ocr_shot_index": 2,
             "ocr_start": 5.0, "ocr_end": 15.0, "scores": {"ocr_rank": 1}},
        ]

        filtered = filter_search_results(primary, ocr)

        self.assertEqual(filtered[0]["ocr_snippet"], "better")

    def test_a_higher_scored_fuzzy_only_hit_never_outranks_an_exact_hit(self):
        # The primary (visual/audio) score for the fuzzy-only false-positive
        # deliberately beats the exact hit's -- OCR exactness must still win.
        primary = [
            {"rank": 0, "score": 0.99, "video_id": "L23_V010", "time": 10.0},
            {"rank": 1, "score": 0.50, "video_id": "L23_V013", "time": 10.0},
        ]
        ocr = [
            {"video_id": "L23_V010", "ocr_snippet": "Ngo Minh Quan", "ocr_shot_index": 1,
             "ocr_start": 0.0, "ocr_end": 20.0, "scores": {"ocr_rank": 1}, "ocr_exact": False},
            {"video_id": "L23_V013", "ocr_snippet": "Da Nang - Tam Ky (Quang Nam)", "ocr_shot_index": 2,
             "ocr_start": 0.0, "ocr_end": 20.0, "scores": {"ocr_rank": 2}, "ocr_exact": True},
        ]

        filtered = filter_search_results(primary, ocr)

        self.assertEqual([item["video_id"] for item in filtered], ["L23_V013", "L23_V010"])
        self.assertEqual([item["rank"] for item in filtered], [0, 1])

    def test_matching_audio_and_visual_hits_merge_and_keep_visual_timestamp(self):
        visual = [
            {
                "video_id": "L21_V015",
                "time": 857.833,
                "frame_idx": 25735,
                "score": 0.8,
                "source": "visual",
            }
        ]
        audio = [
            {
                "video_id": "L21_V015",
                "time": 870.0,
                "frame_idx": 26100,
                "score": 0.1,
                "source": "transcript",
                "transcript_snippet": "chuyến bay tư nhân và cực quang",
                "transcript_start": 840.0,
                "transcript_end": 900.0,
            }
        ]

        fused = fuse_search_results(visual, audio, k=10)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["source"], "both")
        self.assertEqual(fused[0]["frame_idx"], 25735)
        self.assertEqual(fused[0]["time"], 857.833)
        self.assertEqual(fused[0]["transcript_start"], 840.0)
        self.assertEqual(fused[0]["scores"]["visual_rank"], 1)
        self.assertEqual(fused[0]["scores"]["audio_rank"], 1)

    def test_distant_hits_from_same_video_are_not_merged(self):
        visual = [{"video_id": "L21_V015", "time": 10.0, "source": "visual"}]
        audio = [{"video_id": "L21_V015", "time": 100.0, "source": "transcript"}]

        fused = fuse_search_results(visual, audio, k=10)

        self.assertEqual(len(fused), 2)
        self.assertEqual({hit["source"] for hit in fused}, {"visual", "transcript"})

    def test_long_passage_fuses_beyond_the_old_fixed_30s_window(self):
        # A 100s-long chunk's midpoint anchor can land far from the actual
        # matching moment inside it; the adaptive window (scaled to the
        # chunk's own span, capped at 120s) should still credit this as a
        # match even though the gap exceeds the old fixed 30s default.
        visual = [{"video_id": "L21_V015", "time": 150.0, "source": "visual"}]
        audio = [{
            "video_id": "L21_V015", "time": 100.0, "source": "transcript",
            "transcript_start": 50.0, "transcript_end": 150.0,  # 100s span
        }]

        fused = fuse_search_results(visual, audio, k=10)

        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0]["source"], "both")

    def test_adaptive_window_is_still_capped_at_the_ceiling(self):
        # Even a very long passage's window is capped (120s default), so an
        # implausibly distant visual hit still does not merge.
        visual = [{"video_id": "L21_V015", "time": 300.0, "source": "visual"}]
        audio = [{
            "video_id": "L21_V015", "time": 100.0, "source": "transcript",
            "transcript_start": 0.0, "transcript_end": 260.0,  # 260s span, gap=200s
        }]

        fused = fuse_search_results(visual, audio, k=10)

        self.assertEqual(len(fused), 2)
        self.assertEqual({hit["source"] for hit in fused}, {"visual", "transcript"})

    def test_audio_time_window_scales_with_passage_span_between_floor_and_ceiling(self):
        self.assertEqual(_audio_time_window({"transcript_start": 0.0, "transcript_end": 10.0}), 30.0)
        self.assertEqual(_audio_time_window({"transcript_start": 0.0, "transcript_end": 60.0}), 60.0)
        self.assertEqual(_audio_time_window({"transcript_start": 0.0, "transcript_end": 500.0}), 120.0)

    def test_audio_time_window_falls_back_to_floor_without_span_info(self):
        self.assertEqual(_audio_time_window({}), 30.0)
        self.assertEqual(_audio_time_window({"transcript_start": 5.0}), 30.0)


if __name__ == "__main__":
    unittest.main()
