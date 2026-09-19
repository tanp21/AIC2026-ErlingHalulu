import unittest
from unittest.mock import Mock, patch

import httpx
import numpy as np
from pydantic import ValidationError

from backend import app as server
from backend.access_control import AccessConfig, LatestWinsCoordinator
from backend.app import AgentBatchReq, _agent_visual_hits, agent_search
from backend.audio_search import AudioHit
from backend.ocr_search import OcrHit


META = (
    np.array([0, 0, 0, 1, 1], dtype=np.int32),
    np.array([10, 20, 30, 5, 15], dtype=np.int32),
    {0: {"video_id": "L01_V001", "video_idx": 0}, 1: {"video_id": "L01_V002", "video_idx": 1}},
    {"L01_V001": {"video_id": "L01_V001", "video_idx": 0},
     "L01_V002": {"video_id": "L01_V002", "video_idx": 1}},
)
TIMES = np.array([0.4, 0.8, 1.2, 0.2, 0.6])
VECTORS = np.array([
    [1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6], [0.0, -1.0],
], dtype="float32")


def flat_index():
    index = Mock()
    index.reconstruct_batch.side_effect = lambda ids: VECTORS[np.asarray(ids)]

    def search(query, k):
        scores = VECTORS @ query[0]
        order = np.argsort(-scores)[:k]
        return scores[order][None, :], order[None, :]

    index.search.side_effect = search
    return index


class AgentBatchRequestTests(unittest.TestCase):
    def test_rejects_duplicate_ids_unknown_kinds_and_oversized_batches(self):
        invalid = (
            {"queries": [{"kind": "visual", "id": "a", "text": "x"},
                         {"kind": "ocr", "id": "a", "text": "y"}]},
            {"queries": [{"kind": "caption", "id": "a", "text": "x"}]},
            {"queries": [{"kind": "visual", "id": str(i), "text": "x"} for i in range(13)]},
            {"queries": [{"kind": "temporal", "id": "t", "events": ["only one"]}]},
            {"queries": [{"kind": "visual", "id": "a", "text": "x", "limit": 501}]},
            {"queries": [{"kind": "visual", "id": "a", "text": "x", "video_ids": ["bad"]}]},
        )
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                AgentBatchReq(**payload)


class AgentVisualHitsTests(unittest.TestCase):
    def setUp(self):
        self.patches = [
            patch("backend.app.core.get_index", return_value=flat_index()),
            patch("backend.app.core.load_meta", return_value=META),
            patch("backend.app.core.load_times", return_value=TIMES),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()

    def test_corpus_search_maps_rows_to_frames_and_times(self):
        hits = _agent_visual_hits(np.array([1.0, 0.0]), "l2026", 2)

        self.assertEqual(
            [(hit["video_id"], hit["frame_idx"], hit["time"], hit["rank"]) for hit in hits],
            [("L01_V001", 10, 0.4, 1), ("L01_V002", 5, 0.2, 2)],
        )

    def test_scoped_search_scores_every_keyframe_of_the_named_videos(self):
        hits = _agent_visual_hits(np.array([0.0, -1.0]), "l2026", 5, video_ids=["L01_V002"])

        self.assertEqual([hit["frame_idx"] for hit in hits], [15, 5])
        self.assertTrue(all(hit["video_id"] == "L01_V002" for hit in hits))

    def test_excludes_the_reference_frame(self):
        hits = _agent_visual_hits(np.array([1.0, 0.0]), "l2026", 2, exclude_global_id=0)

        self.assertNotIn(10, [hit["frame_idx"] for hit in hits if hit["video_id"] == "L01_V001"])
        self.assertEqual(len(hits), 2)


class AgentSearchTests(unittest.TestCase):
    def test_runs_each_modality_and_isolates_a_failing_sub_query(self):
        request = AgentBatchReq(queries=[
            {"kind": "visual", "id": "v", "text": "a red boat", "limit": 1},
            {"kind": "audio", "id": "a", "text": "cá nhám"},
            {"kind": "ocr", "id": "o", "text": "Nà Ní"},
        ])
        audio = [
            AudioHit("L01_V002", 10.0, 20.0, 15.0, "con cá nhám rất lớn", 3.0, bm25_rank=1),
            AudioHit("K01_V001", 0.0, 5.0, 2.0, "outside the profile", 2.0, bm25_rank=2),
        ]
        with (
            patch("backend.app.core.encode_texts", return_value=np.array([[1.0, 0.0]], dtype="float32")),
            patch("backend.app.core.get_index", return_value=flat_index()),
            patch("backend.app.core.load_meta", return_value=META),
            patch("backend.app.core.load_times", return_value=TIMES),
            patch("backend.app.core.nearest_frame", return_value=15),
            patch("backend.app.audio_search.search_audio_keyword", return_value=audio) as keyword,
            patch("backend.app.ocr_search.search_ocr_keyword", side_effect=RuntimeError("es down")),
        ):
            response = agent_search(request)

        results = response["results"]
        self.assertEqual(results["v"]["hits"][0]["frame_idx"], 10)
        keyword.assert_called_once_with("cá nhám", 100)
        self.assertEqual(len(results["a"]["hits"]), 1)
        self.assertEqual(results["a"]["hits"][0]["frame_idx"], 15)
        self.assertEqual(results["a"]["hits"][0]["text"], "con cá nhám rất lớn")
        self.assertEqual(results["o"], {"kind": "ocr", "error": "RuntimeError: es down"})

    def test_paraphrases_fuse_into_one_vector_with_the_caption_at_half_weight(self):
        request = AgentBatchReq(queries=[{
            "kind": "visual", "id": "v", "text": "caption", "paraphrases": ["p1", "p2"], "limit": 1,
        }])
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype="float32")
        index = flat_index()
        with (
            patch("backend.app.core.encode_texts", return_value=vectors) as encode,
            patch("backend.app.core.get_index", return_value=index),
            patch("backend.app.core.load_meta", return_value=META),
            patch("backend.app.core.load_times", return_value=TIMES),
        ):
            agent_search(request)

        encode.assert_called_once_with(["caption", "p1", "p2"])
        query = index.search.call_args.args[0]
        np.testing.assert_allclose(query, [[2 ** -0.5, 2 ** -0.5]], rtol=1e-6)
        with self.assertRaises(ValidationError):
            AgentBatchReq(queries=[{"kind": "visual", "id": "v", "text": "c",
                                    "paraphrases": ["p"] * 5}])

    def test_ocr_hits_flag_exact_phrase_matches(self):
        request = AgentBatchReq(queries=[{"kind": "ocr", "id": "o", "text": "Nà Ní"}])
        hit = OcrHit("L01_V001", 3, 20, 0.5, 1.5, 0.8, "Mèo NÀ NÍ", 4.0)
        with (
            patch("backend.app.core.load_meta", return_value=META),
            patch("backend.app.ocr_search.search_ocr_keyword", return_value=[hit]),
        ):
            response = agent_search(request)

        only = response["results"]["o"]["hits"][0]
        self.assertTrue(only["exact"])
        self.assertEqual((only["frame_idx"], only["start"], only["end"]), (20, 0.5, 1.5))

    def test_temporal_keeps_top_sequences_per_video_and_uses_benchmark_config(self):
        request = AgentBatchReq(queries=[
            {"kind": "temporal", "id": "t", "events": ["first", "second"], "limit": 1}
        ])
        event = {"frame_idx": 10, "time": 0.4, "score": 0.9}
        response = {"videos": [
            {"video_id": "L01_V001", "best_score": 0.9,
             "sequences": [{"rank": r, "score": 0.9 - r / 10, "events": [event, event]}
                           for r in range(5)]},
            {"video_id": "L01_V002", "best_score": 0.1, "sequences": []},
        ]}
        with (
            patch("backend.app.load_temporal_config", return_value=(2000, "raw_min", 1.5)),
            patch("backend.app.temporal_search_response", return_value=response) as search,
        ):
            result = agent_search(request)["results"]["t"]

        self.assertEqual(search.call_args.args[1:], (2000, "raw_min", 1.5))
        self.assertEqual(len(result["hits"]), 1)
        self.assertEqual(len(result["hits"][0]["sequences"]), 3)


class AgentSearchEndpointTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.app.state.access_config = AccessConfig(
            frozenset({"one@example.com", "two@example.com"}), "one@example.com"
        )
        server.app.state.search_coordinator = LatestWinsCoordinator()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_runs_under_a_separate_coordinator_identity(self):
        async def run(user, operation, **_):
            self.assertEqual(user, "two@example.com#agent")
            return {"results": {}}

        coordinator = Mock()
        coordinator.run.side_effect = run
        server.app.state.search_coordinator = coordinator

        response = await self.client.post(
            "/agent/search",
            headers={"X-AIC-User": "two@example.com"},
            json={"queries": [{"kind": "visual", "id": "v", "text": "boat"}]},
        )

        self.assertEqual(response.status_code, 200)
        coordinator.run.assert_called_once()

    async def test_requires_trusted_identity(self):
        response = await self.client.post(
            "/agent/search", json={"queries": [{"kind": "visual", "id": "v", "text": "boat"}]}
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
