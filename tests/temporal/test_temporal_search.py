import unittest

from backend.temporal_search import (
    build_sequences,
    deduplicate_candidates,
    fuse_audio,
    score_values,
    temporal_nms,
)


def hit(event, video, frame, time, score, rank=1):
    return {
        "event_index": event,
        "video_id": video,
        "frame_idx": frame,
        "time": time,
        "score": score,
        "event_rank": rank,
    }


class TemporalScoringTests(unittest.TestCase):
    def test_all_benchmark_scoring_methods(self):
        scores = [0.8, 0.4]
        normalized = [1.0, 0.5]
        percentiles = [1.0, 0.0]

        self.assertAlmostEqual(score_values(scores, normalized, percentiles, "raw_mean"), 0.6)
        self.assertAlmostEqual(score_values(scores, normalized, percentiles, "raw_min"), 0.4)
        self.assertAlmostEqual(score_values(scores, normalized, percentiles, "minmax_mean"), 0.75)
        self.assertAlmostEqual(score_values(scores, normalized, percentiles, "percentile_mean"), 0.5)
        shifted = [0.9, 0.7]
        self.assertAlmostEqual(
            score_values(scores, normalized, percentiles, "shifted_harmonic"),
            2 / sum(1 / value for value in shifted),
        )

    def test_unknown_scoring_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown temporal scoring method"):
            score_values([0.5], [0.5], [1.0], "mystery")


class TemporalDynamicProgrammingTests(unittest.TestCase):
    def test_beam_keeps_alternative_prefix_for_same_endpoint(self):
        candidates = [
            [hit(0, "A", 10, 10, 0.9), hit(0, "A", 100, 100, 0.8)],
            [hit(1, "A", 110, 110, 0.7)],
        ]

        greedy = build_sequences(candidates, 120, "raw_mean", 10, beam_size=1)
        beam = build_sequences(candidates, 120, "raw_mean", 10, beam_size=2)

        self.assertEqual([[event["frame_idx"] for event in path["events"]] for path in greedy], [[10, 110]])
        self.assertEqual(
            [[event["frame_idx"] for event in path["events"]] for path in beam],
            [[10, 110], [100, 110]],
        )

    def test_builds_best_path_for_each_final_endpoint_in_same_video(self):
        candidates = [
            [hit(0, "A", 10, 10, 0.8), hit(0, "A", 100, 100, 0.9), hit(0, "B", 1, 1, 1.0)],
            [hit(1, "A", 20, 20, 0.7), hit(1, "A", 110, 110, 0.95), hit(1, "B", 2, 2, 1.0)],
            [hit(2, "A", 30, 30, 0.6), hit(2, "A", 120, 120, 0.85)],
        ]

        paths = build_sequences(candidates, max_gap_seconds=30, scoring_method="raw_mean", candidate_k=10)

        self.assertEqual(
            [[event["frame_idx"] for event in path["events"]] for path in paths],
            [[100, 110, 120], [10, 20, 30]],
        )
        self.assertTrue(all(path["video_id"] == "A" for path in paths))

    def test_requires_strictly_increasing_frames_and_adjacent_gap(self):
        candidates = [
            [hit(0, "A", 10, 10, 0.8)],
            [hit(1, "A", 10, 10, 1.0), hit(1, "A", 30, 31, 0.9)],
        ]

        self.assertEqual(
            build_sequences(candidates, max_gap_seconds=20, scoring_method="raw_mean", candidate_k=10),
            [],
        )

    def test_min_gap_rejects_a_too_close_higher_scoring_candidate(self):
        candidates = [
            [hit(0, "A", 10, 10, 0.5)],
            [hit(1, "A", 20, 10.5, 0.9), hit(1, "A", 30, 20, 0.5)],
        ]

        unrestricted = build_sequences(candidates, 30, "raw_mean", 10, min_gap_seconds=0)
        restricted = build_sequences(candidates, 30, "raw_mean", 10, min_gap_seconds=5)

        self.assertEqual(
            [event["frame_idx"] for event in unrestricted[0]["events"]], [10, 20]
        )
        self.assertEqual(
            [event["frame_idx"] for event in restricted[0]["events"]], [10, 30]
        )

    def test_min_gap_candidate_remains_usable_for_a_later_more_distant_row(self):
        candidates = [
            [hit(0, "A", 10, 10, 0.5)],
            [hit(1, "A", 20, 10.5, 0.9), hit(1, "A", 40, 40, 0.1)],
        ]

        paths = build_sequences(candidates, 60, "raw_mean", 10, beam_size=2, min_gap_seconds=5)

        self.assertEqual(
            sorted(
                [event["frame_idx"] for event in path["events"]] for path in paths
            ),
            [[10, 40]],
        )

    def test_percentile_and_minmax_inputs_are_derived_per_event(self):
        candidates = [
            [hit(0, "A", 1, 1, 0.9, 1), hit(0, "A", 2, 2, 0.9, 2)],
            [hit(1, "A", 3, 3, 0.8, 1), hit(1, "B", 3, 3, 0.1, 2)],
        ]

        minmax = build_sequences(candidates, 10, "minmax_mean", candidate_k=2)
        percentile = build_sequences(candidates, 10, "percentile_mean", candidate_k=2)

        self.assertEqual(minmax[0]["visual_score"], 0.75)
        self.assertEqual(percentile[0]["visual_score"], 1.0)


class TemporalFusionAndNmsTests(unittest.TestCase):
    def test_audio_best_rank_in_expanded_interval_is_reusable(self):
        paths = [
            {"video_id": "A", "start_time": 40, "end_time": 60, "visual_rank": 1, "visual_score": 0.8},
            {"video_id": "A", "start_time": 80, "end_time": 100, "visual_rank": 2, "visual_score": 0.7},
        ]
        audio = [
            {"video_id": "A", "time": 70, "score": 0.4, "transcript_snippet": "shared"},
            {"video_id": "A", "time": 50, "score": 0.9, "transcript_snippet": "best"},
        ]

        fused = fuse_audio(paths, audio, window_seconds=30, rrf_k=60)

        self.assertEqual([path["audio_evidence"]["transcript_snippet"] for path in fused], ["shared", "shared"])
        self.assertEqual([path["source"] for path in fused], ["both", "both"])
        self.assertAlmostEqual(fused[0]["score"], 1 / 61 + 1 / 61)

    def test_nms_keeps_multiple_disjoint_intervals_and_suppresses_overlap(self):
        paths = [
            {"video_id": "A", "start_time": 0, "end_time": 20, "score": 0.8},
            {"video_id": "A", "start_time": 10, "end_time": 30, "score": 0.9},
            {"video_id": "A", "start_time": 40, "end_time": 50, "score": 0.7},
            {"video_id": "B", "start_time": 10, "end_time": 30, "score": 0.6},
        ]

        kept = temporal_nms(paths)

        self.assertEqual(
            [(path["video_id"], path["start_time"], path["end_time"]) for path in kept],
            [("A", 10, 30), ("A", 40, 50), ("B", 10, 30)],
        )

    def test_equal_score_prefers_earlier_interval(self):
        kept = temporal_nms([
            {"video_id": "A", "start_time": 10, "end_time": 30, "score": 0.8},
            {"video_id": "A", "start_time": 0, "end_time": 20, "score": 0.8},
        ])

        self.assertEqual((kept[0]["start_time"], kept[0]["end_time"]), (0, 20))


class TemporalCandidateDeduplicationTests(unittest.TestCase):
    def test_keeps_best_nearby_hit_and_backfills_to_limit(self):
        candidates = [
            hit(0, "A", 1, 10.0, 0.9, 1),
            hit(0, "A", 2, 11.0, 0.8, 2),
            hit(0, "B", 1, 10.0, 0.7, 3),
            hit(0, "A", 3, 20.0, 0.6, 4),
        ]

        kept = deduplicate_candidates(candidates, window_seconds=2, limit=3)

        self.assertEqual([(row["video_id"], row["time"]) for row in kept], [
            ("A", 10.0), ("B", 10.0), ("A", 20.0),
        ])
        self.assertEqual([row["event_rank"] for row in kept], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
