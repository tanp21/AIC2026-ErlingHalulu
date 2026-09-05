import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np
from pydantic import ValidationError

from backend import core
from backend import app as server
from backend.access_control import AccessConfig, LatestWinsCoordinator
from backend.audio_search import AudioHit
from backend.app import (
    TemporalSearchReq,
    TemporalSnapshotStore,
    finalize_temporal_response,
    temporal_search_response,
    temporal_visual_search,
)


class TemporalRequestTests(unittest.TestCase):
    def test_accepts_two_to_six_events_and_optional_audio(self):
        request = TemporalSearchReq(
            events=[" first ", "second"], audio_text=" lời nói ", max_gap_seconds=600
        )

        self.assertEqual(request.events, ["first", "second"])
        self.assertEqual(request.audio_text, "lời nói")
        self.assertNotIn("index", TemporalSearchReq.model_fields)

    def test_rejects_event_count_lengths_whitespace_and_gap(self):
        invalid = (
            {"events": ["one"]},
            {"events": [str(i) for i in range(7)]},
            {"events": ["one", " "]},
            {"events": ["x" * 2001, "two"]},
            {"events": ["one", "two"], "max_gap_seconds": 0},
            {"events": ["one", "two"], "max_gap_seconds": 601},
            {"events": ["one", "two"], "audio_text": "x" * 2001},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                TemporalSearchReq(**payload)


class TemporalBatchEncodingTests(unittest.TestCase):
    def test_encode_texts_uses_one_model_batch(self):
        import torch

        tokenizer = Mock(return_value=torch.tensor([[1], [2]]))
        model = Mock()
        model.encode_text.return_value = torch.tensor([[3.0, 4.0], [0.0, 2.0]])

        with patch.object(core, "load_model", return_value=(model, tokenizer)):
            encoded = core.encode_texts(["one", "two"])

        tokenizer.assert_called_once_with(["one", "two"])
        model.encode_text.assert_called_once()
        self.assertEqual(encoded.shape, (2, 2))
        np.testing.assert_allclose(np.linalg.norm(encoded, axis=1), [1.0, 1.0])


class TemporalRuntimeTests(unittest.TestCase):
    def test_flat_search_maps_event_candidates_and_groups_multiple_sequences(self):
        request = TemporalSearchReq(events=["one", "two"], max_gap_seconds=20)
        index = Mock()
        index.search.return_value = (
            np.array([[0.9, 0.8], [0.95, 0.7]], dtype="float32"),
            np.array([[0, 2], [1, 3]], dtype="int64"),
        )
        id_video = np.array([0, 0, 0, 0])
        id_frame = np.array([10, 20, 100, 110])
        video_meta = {0: {"video_id": "A"}}
        mapping = Mock()
        mapping.pts = np.arange(200)
        mapping.frame_time.side_effect = lambda frame: float(frame)

        with (
            patch.object(core, "encode_texts", return_value=np.zeros((2, 4), dtype="float32")),
            patch.object(core, "get_index", return_value=index),
            patch.object(core, "load_meta", return_value=(id_video, id_frame, video_meta, {})),
            patch.object(core, "load_times", return_value=np.array([10.0, 20.0, 100.0, 110.0])),
            patch("backend.app.frame_maps.load", return_value=mapping),
        ):
            response = temporal_visual_search(
                request, candidate_k=2, scoring_method="raw_mean"
            )

        self.assertEqual(response["total_sequences"], 2)
        self.assertEqual(response["videos"][0]["video_id"], "A")
        self.assertEqual(
            [[event["frame_idx"] for event in sequence["events"]]
             for sequence in response["videos"][0]["sequences"]],
            [[10, 20], [100, 110]],
        )
        self.assertTrue(all(
            sequence["score"] == round(sequence["score"], 6)
            for sequence in response["videos"][0]["sequences"]
        ))

    def test_audio_fusion_happens_before_nms_and_can_change_overlap_winner(self):
        paths = [
            {"video_id": "A", "start_time": 0, "end_time": 100, "score": 0.9,
             "visual_score": 0.9, "visual_rank": 1, "source": "visual", "events": []},
            {"video_id": "A", "start_time": 90, "end_time": 200, "score": 0.8,
             "visual_score": 0.8, "visual_rank": 2, "source": "visual", "events": []},
        ]
        audio = [{
            "video_id": "A", "time": 230, "score": 0.7,
            "transcript_snippet": "matching later sequence",
        }]

        response = finalize_temporal_response(
            paths, {"faiss_latency_ms": 1.0}, "raw_mean", 1000, audio
        )

        self.assertEqual(response["total_sequences"], 1)
        sequence = response["videos"][0]["sequences"][0]
        self.assertEqual((sequence["start_time"], sequence["end_time"]), (90, 200))
        self.assertEqual(sequence["source"], "both")

    @patch("backend.app.temporal_visual_paths")
    @patch("backend.app.audio_search.search_audio")
    def test_audio_failure_falls_back_to_visual_with_warning(self, audio, visual):
        request = TemporalSearchReq(events=["one", "two"], audio_text="lời nói")
        visual.return_value = ({
            "paths": [{"video_id": "A", "start_time": 0, "end_time": 10,
                       "score": 0.8, "visual_score": 0.8, "visual_rank": 1,
                       "source": "visual", "events": []}],
            "faiss_latency_ms": 1.0,
        })
        audio.side_effect = RuntimeError("offline")

        response = temporal_search_response(request, 1000, "raw_mean")

        self.assertEqual(response["total_sequences"], 1)
        self.assertIn("audio_unavailable", response["warning"])


class TemporalSnapshotTests(unittest.TestCase):
    def test_pages_whole_video_groups_and_enforces_owner_and_ttl(self):
        now = 100.0
        store = TemporalSnapshotStore(page_size=2, ttl_seconds=10, clock=lambda: now)
        response = {
            "videos": [{"video_id": str(i), "sequences": [{"rank": i}]} for i in range(3)],
            "total_sequences": 3,
            "latency_ms": 1.0,
        }

        first = store.create("one@example.com", response)

        self.assertEqual([video["video_id"] for video in first["videos"]], ["0", "1"])
        self.assertEqual(first["total_videos"], 3)
        self.assertTrue(first["has_more"])
        with self.assertRaises(LookupError):
            store.next("two@example.com", first["next_cursor"])
        second = store.next("one@example.com", first["next_cursor"])
        self.assertEqual([video["video_id"] for video in second["videos"]], ["2"])

    def test_preserves_sequence_depth_truncation_metadata(self):
        store = TemporalSnapshotStore(page_size=20)

        page = store.create("one@example.com", {
            "videos": [{"video_id": "A", "sequences": []}],
            "total_sequences": 1000,
            "truncated": True,
        })

        self.assertTrue(page["truncated"])


class TemporalEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(
            frozenset({"one@example.com", "two@example.com"}), "one@example.com"
        )
        server.app.state.search_coordinator = LatestWinsCoordinator()
        server.app.state.temporal_snapshots = TemporalSnapshotStore(page_size=1)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_search_and_next_page_use_benchmark_config_and_owner_cursor(self):
        result = {
            "videos": [
                {"video_id": "A", "sequences": [{"rank": 0}]},
                {"video_id": "B", "sequences": [{"rank": 1}]},
            ],
            "total_sequences": 2,
            "latency_ms": 1.0,
        }

        async def inline(function, *args):
            return function(*args)

        with (
            patch("backend.app.load_temporal_config", return_value=(2000, "raw_min")),
            patch("backend.app.temporal_search_response", return_value=result) as search,
            patch("backend.app.run_in_threadpool", side_effect=inline),
        ):
            response = await self.client.post(
                "/search/temporal",
                headers={"X-AIC-User": "one@example.com"},
                json={"events": ["one", "two"]},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["videos"][0]["video_id"], "A")
        search.assert_called_once()
        self.assertEqual(search.call_args.args[1:], (2000, "raw_min"))
        cursor = response.json()["next_cursor"]

        denied = await self.client.post(
            "/search/temporal/next",
            headers={"X-AIC-User": "two@example.com"},
            json={"cursor": cursor},
        )
        self.assertEqual(denied.status_code, 410)

        next_page = await self.client.post(
            "/search/temporal/next",
            headers={"X-AIC-User": "one@example.com"},
            json={"cursor": cursor},
        )
        self.assertEqual(next_page.json()["videos"][0]["video_id"], "B")


if __name__ == "__main__":
    unittest.main()
