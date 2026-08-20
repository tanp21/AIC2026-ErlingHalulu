import unittest

from app import fuse_search_results


class HybridFusionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
