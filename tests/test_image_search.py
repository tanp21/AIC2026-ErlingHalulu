import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image
from pydantic import ValidationError

from backend import app
from backend import core
from backend import frame_maps
from backend import thumb


class ImageQueryVectorTests(unittest.TestCase):
    def test_fuses_text_and_image_then_normalizes_once(self):
        text = np.array([[1.0, 0.0]], dtype="float32")
        image = np.array([[0.0, 1.0]], dtype="float32")

        fused = core.fuse_query_vectors(text, image, 0.5)

        np.testing.assert_allclose(
            fused,
            np.array([[2**-0.5, 2**-0.5]], dtype="float32"),
            rtol=1e-6,
        )
        self.assertEqual(fused.dtype, np.float32)

    def test_image_only_query_is_normalized(self):
        image = np.array([[3.0, 4.0]], dtype="float32")

        fused = core.fuse_query_vectors(None, image, 1.0)

        np.testing.assert_allclose(fused, np.array([[0.6, 0.8]], dtype="float32"))


class UploadedImageTests(unittest.TestCase):
    @staticmethod
    def image_bytes(format="PNG", size=(8, 6)):
        output = io.BytesIO()
        Image.new("RGB", size, "red").save(output, format=format)
        return output.getvalue()

    def test_decodes_supported_static_image_as_oriented_rgb(self):
        image = core.decode_query_image(self.image_bytes())

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (8, 6))

    def test_rejects_invalid_and_oversized_images(self):
        with self.assertRaisesRegex(ValueError, "valid JPEG, PNG, or WebP"):
            core.decode_query_image(b"not an image")
        with self.assertRaisesRegex(ValueError, "20 megapixels"):
            core.decode_query_image(self.image_bytes(size=(5000, 5000)))


class IndexedFrameEmbeddingTests(unittest.TestCase):
    def test_reconstructs_the_exact_index_row_for_a_keyframe(self):
        index = Mock()
        index.reconstruct.return_value = np.array([3.0, 4.0], dtype="float32")
        meta = (
            np.array([7, 8, 7], dtype="int32"),
            np.array([10, 20, 30], dtype="int32"),
            {},
            {"K01_V001": {"video_idx": 7}},
        )

        with patch.object(core, "load_meta", return_value=meta), patch.object(
            core, "get_index", return_value=index
        ):
            vector, global_id = core.indexed_frame_embedding("flat", "K01_V001", 30)

        self.assertEqual(global_id, 2)
        np.testing.assert_allclose(vector, np.array([[0.6, 0.8]], dtype="float32"))
        index.reconstruct.assert_called_once_with(2)

    def test_rejects_a_frame_that_is_not_indexed(self):
        meta = (
            np.array([7], dtype="int32"),
            np.array([10], dtype="int32"),
            {},
            {"K01_V001": {"video_idx": 7}},
        )

        with patch.object(core, "load_meta", return_value=meta):
            with self.assertRaisesRegex(KeyError, "indexed frame"):
                core.indexed_frame_embedding("flat", "K01_V001", 99)


class ImageSearchRequestTests(unittest.TestCase):
    def test_accepts_visual_or_hybrid_with_bounded_weight_and_reference(self):
        request = app.ImageSearchReq(
            mode="hybrid",
            image_weight=0.75,
            reference={"video_id": "K01_V001", "frame_idx": 30},
        )

        self.assertEqual(request.reference.video_id, "K01_V001")
        self.assertEqual(request.image_weight, 0.75)

    def test_rejects_non_visual_modes_bad_weight_and_bad_reference(self):
        for payload in (
            {"mode": "audio"},
            {"mode": "visual", "image_weight": 1.1},
            {"mode": "visual", "reference": {"video_id": "../../bad", "frame_idx": 1}},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                app.ImageSearchReq(**payload)


class VisualVectorSearchTests(unittest.TestCase):
    def test_excludes_reference_and_backfills_result_limit(self):
        index = Mock()
        index.search.return_value = (
            np.array([[0.99, 0.8, 0.7]], dtype="float32"),
            np.array([[0, 1, 2]], dtype="int64"),
        )
        meta = (
            np.array([0, 0, 0], dtype="int32"),
            np.array([10, 20, 30], dtype="int32"),
            {0: {"video_id": "K01_V001"}},
            {},
        )
        mapping = Mock()
        mapping.pts = np.arange(31, dtype="int64")
        mapping.frame_time.side_effect = lambda frame: frame / 10

        with (
            patch.object(core, "get_index", return_value=index),
            patch.object(core, "index_info", return_value={"metric": "ip", "type": "FlatIP"}),
            patch.object(core, "load_meta", return_value=meta),
            patch.object(app.frame_maps, "load", return_value=mapping),
        ):
            response = app.visual_vector_search(
                app.SearchReq(mode="visual"),
                np.array([[1.0, 0.0]], dtype="float32"),
                exclude_global_id=0,
            )

        self.assertEqual([item["frame_idx"] for item in response["results"]], [20, 30])
        self.assertEqual([item["rank"] for item in response["results"]], [0, 1])
        index.search.assert_called_once()
        self.assertEqual(index.search.call_args.args[1], app.SEARCH_RESULT_LIMIT + 1)


class ImageConditionedSearchTests(unittest.TestCase):
    @staticmethod
    def visual_response(query, excluded):
        return {
            "results": [{"query": query.tolist(), "excluded": excluded}],
            "latency_ms": 2.0,
            "mode": "visual",
            "index": "flat",
            "index_info": {"type": "FlatIP"},
        }

    def test_uploaded_image_and_text_search_the_fused_vector(self):
        request = app.ImageSearchReq(mode="visual", visual_text="a red car", image_weight=0.5)
        with (
            patch.object(core, "decode_query_image", return_value=object()),
            patch.object(core, "encode_image", return_value=np.array([[0.0, 1.0]], dtype="float32")),
            patch.object(core, "encode_text", return_value=np.array([[1.0, 0.0]], dtype="float32")),
            patch.object(
                app,
                "visual_vector_search",
                side_effect=lambda _request, query, exclude_global_id=None: self.visual_response(
                    query, exclude_global_id
                ),
            ),
        ):
            response = app.image_conditioned_search(request, b"image bytes")

        self.assertEqual(response["results"][0]["excluded"], None)
        np.testing.assert_allclose(
            response["results"][0]["query"],
            [[2**-0.5, 2**-0.5]],
            rtol=1e-6,
        )

    def test_image_only_weight_skips_text_encoding(self):
        request = app.ImageSearchReq(mode="visual", visual_text="ignored", image_weight=1)
        image_vector = np.array([[0.0, 2.0]], dtype="float32")
        with (
            patch.object(core, "decode_query_image", return_value=object()),
            patch.object(core, "encode_image", return_value=image_vector),
            patch.object(
                core, "encode_text", return_value=np.array([[1.0, 0.0]], dtype="float32")
            ) as encode_text,
            patch.object(
                app,
                "visual_vector_search",
                side_effect=lambda _request, query, exclude_global_id=None: self.visual_response(
                    query, exclude_global_id
                ),
            ),
        ):
            response = app.image_conditioned_search(request, b"image bytes")

        encode_text.assert_not_called()
        np.testing.assert_array_equal(response["results"][0]["query"], [[0.0, 1.0]])

    def test_internal_reference_is_excluded_from_results(self):
        request = app.ImageSearchReq(
            mode="visual",
            reference={"video_id": "K01_V001", "frame_idx": 30},
        )
        with (
            patch.object(
                core,
                "indexed_frame_embedding",
                return_value=(np.array([[1.0, 0.0]], dtype="float32"), 42),
            ),
            patch.object(
                app,
                "visual_vector_search",
                side_effect=lambda _request, query, exclude_global_id=None: self.visual_response(
                    query, exclude_global_id
                ),
            ),
        ):
            response = app.image_conditioned_search(request)

        self.assertEqual(response["results"][0]["excluded"], 42)

    def test_unindexed_reference_is_encoded_from_its_actual_frame(self):
        request = app.ImageSearchReq(
            mode="visual",
            reference={"video_id": "K01_V001", "frame_idx": 99},
        )
        with (
            patch.object(core, "indexed_frame_embedding", side_effect=KeyError("no indexed frame")),
            patch.object(app, "resolve", return_value={"path": "video.mp4"}),
            patch.object(frame_maps, "load", return_value=Mock()),
            patch.object(thumb, "extract", return_value=b"image"),
            patch.object(core, "decode_query_image", return_value=object()),
            patch.object(core, "encode_image", return_value=np.array([[1.0, 0.0]], dtype="float32")),
            patch.object(app, "visual_vector_search", side_effect=lambda _request, query, exclude_global_id=None:
                self.visual_response(query, exclude_global_id)),
        ):
            response = app.image_conditioned_search(request)

        self.assertIsNone(response["results"][0]["excluded"])

    def test_hybrid_image_query_uses_ocr_as_a_filter_without_audio(self):
        request = app.ImageSearchReq(mode="hybrid", ocr_text="THÀNH PHỐ")
        with (
            patch.object(core, "decode_query_image", return_value=object()),
            patch.object(core, "encode_image", return_value=np.array([[1.0, 0.0]], dtype="float32")),
            patch.object(
                app,
                "visual_vector_search",
                return_value={
                    "results": [], "latency_ms": 2.0, "mode": "visual",
                    "index": "flat", "index_info": {"type": "FlatIP"},
                },
            ),
            patch.object(
                app,
                "ocr_search_response",
                return_value={"results": [], "latency_ms": 3.0, "mode": "ocr"},
            ) as ocr,
        ):
            response = app.image_conditioned_search(request, b"image bytes")

        ocr.assert_called_once_with(request, "THÀNH PHỐ", filter_mode=True)
        self.assertEqual(response["latencies"], {"visual_ms": 2.0, "ocr_ms": 3.0})

    def test_application_exposes_the_multipart_search_route(self):
        self.assertIn("/search/image", {route.path for route in app.app.routes})


class ImageSearchEndpointTests(unittest.IsolatedAsyncioTestCase):
    class Coordinator:
        async def run(self, _owner, operation, **_kwargs):
            return await operation()

    def request(self):
        state = SimpleNamespace(
            search_coordinator=self.Coordinator(),
            search_snapshots=app.SearchSnapshotStore(page_size=2),
        )
        return SimpleNamespace(
            state=SimpleNamespace(login="mate@example.com", is_admin=False),
            app=SimpleNamespace(state=state),
        )

    async def test_reference_request_returns_the_normal_search_snapshot_shape(self):
        response = {
            "results": [{"rank": 0}],
            "latency_ms": 1.0,
            "mode": "visual",
            "index": "flat",
            "index_info": {"type": "FlatIP"},
        }
        payload = app.ImageSearchReq(
            reference={"video_id": "K01_V001", "frame_idx": 10}
        ).model_dump_json()
        with patch.object(app, "image_conditioned_search", return_value=response):
            result = await app.image_search_endpoint(self.request(), payload, None)

        self.assertEqual(result["results"], [{"rank": 0}])
        self.assertEqual(result["total_results"], 1)
        self.assertFalse(result["has_more"])

    async def test_rejects_two_sources_and_files_over_six_mib(self):
        payload = app.ImageSearchReq(
            reference={"video_id": "K01_V001", "frame_idx": 10}
        ).model_dump_json()
        small = UploadFile(file=io.BytesIO(b"image"), filename="query.png")
        with self.assertRaises(HTTPException) as two_sources:
            await app.image_search_endpoint(self.request(), payload, small)
        self.assertEqual(two_sources.exception.status_code, 422)

        upload_payload = app.ImageSearchReq().model_dump_json()
        large = UploadFile(
            file=io.BytesIO(b"x" * (app.MAX_QUERY_IMAGE_BYTES + 1)),
            filename="large.png",
        )
        with self.assertRaises(HTTPException) as oversized:
            await app.image_search_endpoint(self.request(), upload_payload, large)
        self.assertEqual(oversized.exception.status_code, 413)


if __name__ == "__main__":
    unittest.main()
