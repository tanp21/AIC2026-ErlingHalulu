import unittest
from unittest.mock import patch

from scripts.benchmarks.bench_temporal_search import _ranked_paths, evaluate_results, select_config


class TemporalBenchmarkMetricTests(unittest.TestCase):
    @patch("backend.app.temporal_visual_paths")
    def test_ranked_paths_reads_runtime_metadata_object(self, visual):
        path = {"video_id": "V1", "start_time": 1, "end_time": 2, "score": 0.5,
                "events": [{"frame_idx": 1}, {"frame_idx": 2}]}
        visual.return_value = {"paths": [path], "latency_ms": 3.0}

        ranked = _ranked_paths({"events": ["one", "two"]}, 500, "raw_mean")

        self.assertEqual(ranked, [path])

    def test_accepts_any_ground_truth_sequence_with_all_events_in_tolerance(self):
        results = [
            {"video_id": "wrong", "events": [{"time": 1}, {"time": 2}]},
            {"video_id": "V1", "events": [{"time": 102}, {"time": 198}]},
        ]
        ground_truth = [
            {"video_id": "V1", "event_times": [10, 20]},
            {"video_id": "V1", "event_times": [100, 200]},
        ]

        metrics = evaluate_results(results, ground_truth, tolerance_seconds=10)

        self.assertEqual(metrics, {"reciprocal_rank": 0.5, "recall_at_10": 1.0})

    def test_requires_same_video_event_count_and_every_event_within_tolerance(self):
        ground_truth = [{"video_id": "V1", "event_times": [100, 200]}]
        cases = [
            [{"video_id": "V2", "events": [{"time": 100}, {"time": 200}]}],
            [{"video_id": "V1", "events": [{"time": 100}]}],
            [{"video_id": "V1", "events": [{"time": 100}, {"time": 211}]}],
        ]

        for results in cases:
            with self.subTest(results=results):
                self.assertEqual(
                    evaluate_results(results, ground_truth, tolerance_seconds=10),
                    {"reciprocal_rank": 0.0, "recall_at_10": 0.0},
                )


class TemporalBenchmarkSelectionTests(unittest.TestCase):
    def test_selects_mrr_then_recall_then_latency_then_smaller_k(self):
        rows = [
            {"candidate_k": 2000, "scoring_method": "raw_mean", "min_gap_seconds": 0.0, "mrr": 0.7, "recall_at_10": 0.8, "p95_ms": 900},
            {"candidate_k": 1000, "scoring_method": "raw_min", "min_gap_seconds": 0.0, "mrr": 0.7, "recall_at_10": 0.9, "p95_ms": 1200},
            {"candidate_k": 2000, "scoring_method": "raw_min", "min_gap_seconds": 0.0, "mrr": 0.7, "recall_at_10": 0.9, "p95_ms": 800},
            {"candidate_k": 5000, "scoring_method": "percentile_mean", "min_gap_seconds": 0.0, "mrr": 0.6, "recall_at_10": 1.0, "p95_ms": 100},
        ]

        selected = select_config(rows)

        self.assertEqual(selected["candidate_k"], 2000)
        self.assertEqual(selected["scoring_method"], "raw_min")

    def test_smaller_k_is_final_tie_breaker(self):
        common = {"mrr": 0.7, "recall_at_10": 0.9, "p95_ms": 800, "min_gap_seconds": 0.0}
        selected = select_config([
            {**common, "candidate_k": 2000, "scoring_method": "raw_mean"},
            {**common, "candidate_k": 500, "scoring_method": "raw_mean"},
        ])

        self.assertEqual(selected["candidate_k"], 500)

    def test_smaller_min_gap_is_last_tie_breaker(self):
        common = {"mrr": 0.7, "recall_at_10": 0.9, "p95_ms": 800, "candidate_k": 2000, "scoring_method": "raw_mean"}
        selected = select_config([
            {**common, "min_gap_seconds": 1.5},
            {**common, "min_gap_seconds": 0.0},
        ])

        self.assertEqual(selected["min_gap_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
