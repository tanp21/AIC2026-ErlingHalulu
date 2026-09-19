import os
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from fastapi import HTTPException

from backend import search_core


class LocalModeTests(unittest.TestCase):
    def test_encode_texts_uses_the_in_process_batcher(self):
        with (
            patch.dict(os.environ, {"AIC_CORE_URL": ""}),
            patch.object(search_core, "_request") as request,
            patch.object(search_core.core, "encode_texts", return_value=np.array([[1.0]])) as encode,
        ):
            result = search_core.encode_texts(["a"])

        encode.assert_called_once_with(["a"])
        request.assert_not_called()
        np.testing.assert_allclose(result, [[1.0]])

    def test_known_videos_reads_local_metadata(self):
        meta = (None, None, None, {"K01_V001": {}})
        with (
            patch.dict(os.environ, {"AIC_CORE_URL": ""}),
            patch.object(search_core.core, "load_meta", return_value=meta),
        ):
            self.assertEqual(search_core.known_videos("l2026"), {"K01_V001"})

    def test_visual_search_delegates_to_the_local_index(self):
        result = {"hits": [], "index_info": {}, "faiss_latency_ms": 0.0, "profile": "l2026"}
        with (
            patch.dict(os.environ, {"AIC_CORE_URL": ""}),
            patch.object(search_core, "_request") as request,
            patch.object(search_core, "local_visual_search", return_value=result) as local,
        ):
            self.assertIs(search_core.visual_search("flat", "l2026", [[1.0]], 3), result)

        request.assert_not_called()
        local.assert_called_once()


class RemoteModeTests(unittest.TestCase):
    def setUp(self):
        search_core._fetch_catalog.cache_clear()
        self.env = patch.dict(os.environ, {"AIC_CORE_URL": "http://core.test"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        search_core._fetch_catalog.cache_clear()

    def test_encode_text_posts_a_single_text(self):
        with patch.object(search_core, "_request", return_value={"vectors": [[1.0, 2.0]]}) as request:
            vector = search_core.encode_text("hello")

        request.assert_called_once_with("POST", "/core/encode/text", json={"texts": ["hello"]})
        np.testing.assert_allclose(vector, [[1.0, 2.0]])

    def test_encode_image_posts_the_raw_bytes(self):
        with patch.object(search_core, "_request", return_value={"vector": [0.0, 1.0]}) as request:
            vector = search_core.encode_image(b"jpeg")

        self.assertEqual(request.call_args.args, ("POST", "/core/encode/image"))
        self.assertEqual(request.call_args.kwargs["content"], b"jpeg")
        np.testing.assert_allclose(vector, [[0.0, 1.0]])

    def test_visual_search_sends_vectors_and_flags(self):
        payload = {"hits": [], "index_info": {}, "faiss_latency_ms": 1.0, "profile": "l2026"}
        with patch.object(search_core, "_request", return_value=payload) as request:
            result = search_core.visual_search(
                "flat", None, [1.0, 0.0], 10, with_pts=True, exclude_global_id=3,
            )

        self.assertIs(result, payload)
        self.assertEqual(request.call_args.args, ("POST", "/core/visual/search"))
        body = request.call_args.kwargs["json"]
        self.assertEqual(body["vectors"], [[1.0, 0.0]])
        self.assertIsNone(body["profile"])
        self.assertTrue(body["with_pts"])
        self.assertEqual(body["exclude_global_id"], 3)

    def test_annotate_posts_points_and_returns_marks(self):
        marks = [{"frame_idx": 5, "pts": 10, "time": 1.0}]
        with patch.object(search_core, "_request", return_value={"marks": marks}) as request:
            result = search_core.annotate([{"video_id": "A", "time": 1.0}], "l2026")

        self.assertEqual(result, marks)
        body = request.call_args.kwargs["json"]
        self.assertEqual(body["points"], [{"video_id": "A", "time": 1.0}])
        self.assertTrue(body["with_pts"])

    def test_catalog_is_cached_for_video_ids_and_default_profile(self):
        catalog = {"default_profile": "l2026", "video_ids": ["A", "B"], "indexes": {}}
        with patch.object(search_core, "_request", return_value=catalog) as request:
            self.assertEqual(search_core.default_profile(), "l2026")
            self.assertEqual(search_core.known_videos("l2026"), {"A", "B"})

        self.assertEqual(request.call_count, 2)
        self.assertEqual(request.call_args_list[0].kwargs["params"], None)
        self.assertEqual(request.call_args_list[1].kwargs["params"], {"profile": "l2026"})


class AudioRemoteTests(unittest.TestCase):
    def test_keyword_search_goes_through_the_remote_es_shim(self):
        from backend import audio_search

        response = {"hits": {"hits": [{
            "_score": 3.0,
            "_source": {"video_id": "A", "start_sec": 0.0, "end_sec": 5.0,
                        "anchor_sec": 2.5, "text": "con cá"},
        }]}}
        with (
            patch.dict(os.environ, {"AIC_CORE_URL": "http://core.test"}),
            patch.object(search_core, "_request", return_value=response) as request,
        ):
            hits = audio_search.search_audio_keyword("cá", 5)

        self.assertEqual([(hit.video_id, hit.bm25_rank) for hit in hits], [("A", 1)])
        self.assertEqual(request.call_args.args, ("POST", "/core/es/search"))
        self.assertEqual(request.call_args.kwargs["json"]["index"], audio_search.AUDIO_INDEX)

    def test_semantic_search_uses_remote_qwen_and_flat_index(self):
        from backend import audio_search

        hit = {
            "video_id": "A", "start_sec": 0.0, "end_sec": 10.0, "anchor_sec": 5.0,
            "text": "con cá nhám", "score": 0.9, "bm25_rank": None, "vector_rank": 1,
            "passage_id": "A:0", "matched_keywords": [],
        }

        def fake_request(method, path, **kwargs):
            if path == "/core/encode/qwen":
                return {"vectors": [[0.0, 1.0, 0.0]]}
            if path == "/core/audio/flat-search":
                return {"hits": [hit]}
            raise AssertionError(f"unexpected request {path}")

        with (
            patch.dict(os.environ, {"AIC_CORE_URL": "http://core.test"}),
            patch.object(search_core, "_request", side_effect=fake_request),
        ):
            hits = audio_search.search_audio_semantic("cá nhám", 5)

        self.assertEqual([(h.video_id, h.vector_rank) for h in hits], [("A", 1)])


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"AIC_CORE_URL": "http://core.test"})
        self.env.start()

    def tearDown(self):
        search_core.close_client()
        self.env.stop()

    def _client(self, handler):
        return httpx.Client(base_url="http://core.test", transport=httpx.MockTransport(handler))

    def test_error_status_becomes_an_http_exception(self):
        client = self._client(lambda request: httpx.Response(400, json={"detail": "unknown index"}))
        with patch.object(search_core, "_client_for", return_value=client):
            with self.assertRaises(HTTPException) as caught:
                search_core._request("POST", "/core/visual/search", json={})

        self.assertEqual(caught.exception.status_code, 400)
        self.assertEqual(caught.exception.detail, "unknown index")

    def test_transport_failure_retries_then_reports_503(self):
        attempts = []

        def handler(request):
            attempts.append(request)
            raise httpx.ConnectError("boom", request=request)

        client = self._client(handler)
        with (
            patch.object(search_core, "_client_for", return_value=client),
            patch.object(search_core.time, "sleep"),
        ):
            with self.assertRaises(HTTPException) as caught:
                search_core._request("GET", "/core/healthz", retries=2)

        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(len(attempts), 3)


if __name__ == "__main__":
    unittest.main()
