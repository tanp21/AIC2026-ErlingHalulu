import unittest
from pathlib import Path
import base64
from unittest.mock import patch

import httpx
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from fastapi.testclient import TestClient

from backend import app as server
from backend import core_service


def core_app():
    application = FastAPI()
    core_service.install_core_service(application)
    return application


class CoreEndpointTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(core_app())

    def test_healthz_reports_the_api_version(self):
        response = self.client.get("/core/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["api_version"], core_service.CORE_API_VERSION)

    def test_encode_text_returns_vectors(self):
        vectors = np.array([[1.0, 2.0]], dtype="float32")
        with patch.object(core_service.core, "encode_texts", return_value=vectors):
            response = self.client.post("/core/encode/text", json={"texts": ["a"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"vectors": [[1.0, 2.0]]})

    def test_encode_image_decodes_and_encodes_the_uploaded_bytes(self):
        image = object()
        vector = np.array([[0.0, 1.0]], dtype="float32")
        with (
            patch.object(core_service.core, "decode_query_image", return_value=image),
            patch.object(core_service.core, "encode_image", return_value=vector) as encode,
        ):
            response = self.client.post(
                "/core/encode/image", content=b"jpeg", headers={"content-type": "image/jpeg"},
            )

        self.assertEqual(response.json(), {"vector": [0.0, 1.0]})
        encode.assert_called_once_with(image)

    def test_encode_image_rejects_undecodable_bytes(self):
        with patch.object(core_service.core, "decode_query_image", side_effect=ValueError("bad image")):
            response = self.client.post(
                "/core/encode/image", content=b"junk", headers={"content-type": "image/jpeg"},
            )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "bad image")

    def test_visual_search_runs_the_local_primitive(self):
        payload = {"hits": [], "index_info": {}, "faiss_latency_ms": 1.0, "profile": "l2026"}
        with patch.object(
            core_service.search_core, "local_visual_search", return_value=payload
        ) as local:
            response = self.client.post(
                "/core/visual/search",
                json={"index": "flat", "vectors": [[1.0, 0.0]], "k": 5},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(local.call_args.args[:2], ("flat", None))

    def test_visual_search_reports_an_unknown_index(self):
        with patch.object(core_service.search_core, "local_visual_search", side_effect=KeyError("nope")):
            response = self.client.post(
                "/core/visual/search", json={"index": "nope", "vectors": [[1.0]], "k": 5},
            )

        self.assertEqual(response.status_code, 400)

    def test_frame_annotate_passes_points_through(self):
        marks = [{"frame_idx": 5, "pts": 10, "time": 1.0}]
        with patch.object(core_service.search_core, "local_annotate", return_value=marks) as local:
            response = self.client.post(
                "/core/frame/annotate",
                json={"points": [{"video_id": "A", "time": 1.0}]},
            )

        self.assertEqual(response.json(), {"marks": marks})
        self.assertEqual(local.call_args.args[0], [{"video_id": "A", "time": 1.0}])

    def test_audio_flat_search_serializes_hits(self):
        from backend.audio_search import AudioHit

        hits = [AudioHit("A", 0.0, 10.0, 5.0, "text", 2.0, vector_rank=1)]
        with patch.object(core_service.audio_search, "search_audio_flat_local", return_value=hits):
            response = self.client.post("/core/audio/flat-search", json={"vector": [0.1], "k": 3})

        body = response.json()
        self.assertEqual(body["hits"][0]["video_id"], "A")
        self.assertEqual(body["hits"][0]["vector_rank"], 1)

    def test_audio_flat_search_reports_a_missing_flat_index(self):
        with patch.object(core_service.audio_search, "search_audio_flat_local", return_value=None):
            response = self.client.post("/core/audio/flat-search", json={"vector": [0.1], "k": 3})

        self.assertEqual(response.json(), {"hits": None})

    def test_es_search_rejects_unknown_indexes(self):
        response = self.client.post("/core/es/search", json={"index": "other", "body": {}})

        self.assertEqual(response.status_code, 400)

    def test_es_search_forwards_allowed_indexes(self):
        fake = type("Response", (), {"__iter__": lambda self: iter([("hits", {})])})()
        with patch.object(core_service.ocr_search, "get_client") as get_client:
            get_client.return_value.search.return_value = fake
            response = self.client.post(
                "/core/es/search",
                json={"index": core_service.ocr_search.OCR_INDEX, "body": {"query": {}}},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"hits": {}})
        get_client.return_value.search.assert_called_once_with(
            index=core_service.ocr_search.OCR_INDEX, body={"query": {}}
        )

    def test_catalog_reports_profile_metadata(self):
        meta = (None, None, None, {"A": {}, "B": {}})
        with (
            patch.object(core_service.core, "load_meta", return_value=meta),
            patch.object(core_service.core, "available_indexes", return_value=["flat"]),
            patch.object(core_service.core, "index_info", return_value={"type": "FlatIP"}),
            patch.object(core_service.core, "default_profile", return_value="l2026"),
            patch.object(core_service.core, "list_profiles", return_value=[]),
        ):
            response = self.client.get("/core/catalog?profile=l2026")

        self.assertEqual(response.json()["video_ids"], ["A", "B"])
        self.assertEqual(response.json()["indexes"], {"flat": {"type": "FlatIP"}})


class CoreRouteRegistrationTests(unittest.TestCase):
    def test_application_registers_the_core_routes(self):
        paths = {route.path for route in server.app.routes}
        for path in (
            "/core/healthz", "/core/catalog", "/core/encode/text", "/core/encode/qwen",
            "/core/encode/image", "/core/visual/search", "/core/visual/frame-embedding",
            "/core/frame/annotate", "/core/audio/flat-search", "/core/es/search",
        ):
            self.assertIn(path, paths)

    def test_container_includes_the_core_modules(self):
        dockerfile = Path("Dockerfile").read_text()
        self.assertIn("backend/core_service.py", dockerfile)
        self.assertIn("backend/search_core.py", dockerfile)


class CoreProxyTests(unittest.TestCase):
    def _upstream(self, seen):
        upstream = FastAPI()

        @upstream.api_route("/{path:path}", methods=["GET", "POST", "PATCH"])
        async def capture(path: str, request: Request):
            seen["url"] = str(request.url)
            seen["method"] = request.method
            seen["user"] = request.headers.get("x-aic-user")
            seen["body"] = await request.body()
            if path.startswith("thumb/"):
                return Response(b"jpeg-bytes", media_type="image/jpeg")
            if path.startswith("shots/"):
                return JSONResponse({"labelRevision": 1})
            return JSONResponse({"source": "upstream"})

        return upstream

    def _middleware(self, seen):
        application = FastAPI()

        @application.get("/thumb/{video_id}/{frame_idx}")
        def local_thumb(video_id: str, frame_idx: int):
            return {"source": "local"}

        @application.get("/search")
        def local_search():
            return {"source": "local"}

        middleware = core_service.CoreProxyMiddleware(application, "http://core.test", "me@x.com")
        middleware._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self._upstream(seen)),
        )
        return middleware

    def test_media_routes_are_forwarded_with_the_service_identity(self):
        seen = {}
        client = TestClient(self._middleware(seen))

        response = client.get("/thumb/L21_V001/10?w=320")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"jpeg-bytes")
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(seen["url"], "http://core.test/thumb/L21_V001/10?w=320")
        self.assertEqual(seen["user"], "me@x.com")

    def test_request_bodies_and_methods_are_forwarded(self):
        seen = {}
        client = TestClient(self._middleware(seen))

        response = client.patch("/shots/L21_V001/labels", json={"expected_revision": 0})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"labelRevision": 1})
        self.assertEqual(seen["method"], "PATCH")
        self.assertEqual(seen["body"], b'{"expected_revision":0}')

    def test_search_routes_stay_local(self):
        seen = {}
        client = TestClient(self._middleware(seen))

        response = client.get("/search")

        self.assertEqual(response.json(), {"source": "local"})
        self.assertEqual(seen, {})

    def test_a_gzipped_upstream_reply_survives_the_apps_own_gzip(self):
        # The compute host answers `Content-Encoding: gzip` with its compressed length. Passed
        # on with that capitalisation, GZipMiddleware missed it, compressed the body again under
        # the old Content-Length, and uvicorn dropped the connection ("Failed to fetch").
        import gzip

        from starlette.middleware.gzip import GZipMiddleware

        payload = b'{"cues": [' + b",".join(b'{"text": "xin chao"}' for _ in range(200)) + b"]}"
        packed = gzip.compress(payload)

        def handler(request: httpx.Request):
            return httpx.Response(200, headers=[("Content-Type", "application/json"),
                                                ("Content-Encoding", "gzip"),
                                                ("Content-Length", str(len(packed)))],
                                  stream=httpx.ByteStream(packed))

        middleware = core_service.CoreProxyMiddleware(FastAPI(), "http://core.test")
        middleware._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = TestClient(GZipMiddleware(middleware, minimum_size=10))

        response = client.get("/transcript/L21_V001", headers={"Accept-Encoding": "gzip"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, payload)
        self.assertEqual(response.headers.get_list("content-encoding"), ["gzip"])

    def test_upstream_failure_is_reported_as_502(self):
        def handler(request: httpx.Request):
            raise httpx.ConnectError("down", request=request)

        application = FastAPI()
        middleware = core_service.CoreProxyMiddleware(application, "http://core.test")
        middleware._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = TestClient(middleware)

        response = client.get("/videos")

        self.assertEqual(response.status_code, 502)


class _AsLogin:
    """Stands in for TrustedIngressAccessMiddleware, which runs outside the proxy."""

    def __init__(self, app, login):
        self.app = app
        self.login = login

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.login:
            scope.setdefault("state", {})["login"] = self.login
        return await self.app(scope, receive, send)


class MediaHostProxyTests(unittest.TestCase):
    """`mode="api"`: the media host answers the SPA and the heavy bytes, forwards the rest.

    The venue link carries only the small JSON the compute host returns, never a frame.
    """

    # The DRES routes are the media host's: it holds the credentials and the videos the
    # submission payload is built from, and the rented instance should hold neither. The
    # frame map is the media host's too -- it is a file in the `indexes/` it mounts, and
    # sending it across the tunnel only made the player wait for it.
    LOCAL = ("/", "/healthz", "/favicon.ico", "/thumb/L21_V001/10?w=320",
             "/frame/L21_V001/10", "/video/L21_V001", "/clip/L21_V001/3",
             "/static/assets/index.js", "/api/dres/status", "/api/dres/submit",
             "/api/dres/keyframes/L21_V001", "/frame-map/L21_V001")

    def _upstream(self, seen):
        upstream = FastAPI()

        @upstream.api_route("/{path:path}", methods=["GET", "POST", "PATCH"])
        async def capture(path: str, request: Request):
            seen["url"] = str(request.url)
            seen["delegated"] = request.headers.get("x-aic-delegated-user")
            return JSONResponse({"source": "upstream"})

        return upstream

    def _client(self, seen, login="mate@example.com"):
        application = FastAPI()

        @application.api_route("/{path:path}", methods=["GET", "POST", "PATCH"])
        def local(path: str):
            return {"source": "local"}

        proxy = core_service.CoreProxyMiddleware(application, "http://core.test", mode="api")
        proxy._client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self._upstream(seen)),
        )
        return TestClient(_AsLogin(proxy, login))

    def test_spa_and_media_stay_local(self):
        seen = {}
        client = self._client(seen)

        for path in self.LOCAL:
            self.assertEqual(client.get(path).json(), {"source": "local"}, path)

        self.assertEqual(seen, {})

    def test_api_calls_reach_the_compute_host_with_the_forwarded_login(self):
        seen = {}
        client = self._client(seen, login="mate@example.com")

        response = client.get("/search")

        self.assertEqual(response.json(), {"source": "upstream"})
        self.assertEqual(seen["url"], "http://core.test/search")
        self.assertEqual(seen["delegated"], "mate@example.com")

    def test_everything_the_media_host_cannot_answer_is_forwarded(self):
        seen = {}
        client = self._client(seen)

        for path in ("/search", "/api/agent/runs/active", "/api/collab/rounds", "/shots/L21_V001",
                     "/transcript/L21_V001", "/videos", "/profiles", "/core/encode/text"):
            client.get(path)
            self.assertEqual(seen["url"], f"http://core.test{path}", path)

    def test_a_client_supplied_delegation_is_dropped(self):
        seen = {}
        client = self._client(seen, login=None)

        client.get("/search", headers={"X-AIC-Delegated-User": "owner@example.com"})

        self.assertIsNone(seen["delegated"])


class VisionJudgeTests(unittest.TestCase):
    """The endpoint `AIC_VLM_URL` points a teammate's app at.

    Its contract is `agent.vlm.HttpVLM`'s: labelled base64 images plus a prompt and
    a response schema, answered with the parsed JSON the caller expects.
    """

    def setUp(self):
        self.client = TestClient(core_app())

    def post(self, **overrides):
        payload = {
            "prompt": "Đánh giá khung hình này",
            "schema": {"type": "object", "properties": {"id": {}, "score": {}}},
            "images": [{"label": "[C1] L21_V001 t=1.0s", "b64": base64.b64encode(b"jpeg").decode()}],
            "job": "verify",
        }
        return self.client.post("/core/vision/judge", json={**payload, **overrides})

    def test_a_judgement_round_trips_through_the_backend(self):
        seen = {}

        class Fake:
            def judge(self, images, prompt, schema, **kwargs):
                seen.update(images=images, prompt=prompt, schema=schema, **kwargs)
                return {"id": "C1", "score": 7}

        with patch.object(core_service, "_vision_backend", return_value=Fake()):
            response = self.post()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"reply": {"id": "C1", "score": 7}})
        self.assertEqual(seen["images"], [("[C1] L21_V001 t=1.0s", b"jpeg")])
        self.assertEqual(seen["job"], "verify")
        # The wire name is `schema`; the model attribute has to avoid the BaseModel
        # method of that name.
        self.assertEqual(seen["schema"]["type"], "object")

    def test_an_unreadable_image_is_rejected_before_the_model_is_touched(self):
        with patch.object(core_service, "_vision_backend") as backend:
            response = self.post(images=[{"label": "[C1]", "b64": "not base64!!"}])

        self.assertEqual(response.status_code, 422)
        backend.assert_not_called()

    def test_an_empty_image_list_is_rejected(self):
        self.assertEqual(self.post(images=[]).status_code, 422)

    def test_a_model_failure_is_reported_as_503_not_500(self):
        # The caller treats 503 as a degraded job and keeps its other candidates.
        from agent.vlm import VLMError

        class Broken:
            def judge(self, *_, **__):
                raise VLMError("local vision backend disabled after a failure")

        with patch.object(core_service, "_vision_backend", return_value=Broken()):
            response = self.post()

        self.assertEqual(response.status_code, 503)
        self.assertIn("disabled", response.json()["detail"])

    def test_the_backend_is_built_once_and_reused(self):
        class Fake:
            def judge(self, *_, **__):
                return {"ok": True}

        with (
            patch("agent.vlm.LocalVLM") as constructor,
            patch.object(core_service, "_vision", None),
        ):
            constructor.return_value = Fake()
            self.post()
            self.post()

        self.assertEqual(constructor.call_count, 1, "a reload per call would dominate the query budget")


if __name__ == "__main__":
    unittest.main()
