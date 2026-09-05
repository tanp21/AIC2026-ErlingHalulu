import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend import app


def response(mode):
    return {
        "results": [],
        "latency_ms": 1.0,
        "mode": mode,
        "index": "flat",
        "index_info": {"type": "test"},
        "profile": "default",
    }


class SeparateHybridQueryTests(unittest.TestCase):
    @patch("backend.app.fuse_search_results", return_value=[])
    @patch("backend.app.audio_search_response", return_value=response("audio"))
    @patch("backend.app.visual_search", return_value=response("visual"))
    def test_hybrid_routes_each_language_to_its_own_search(self, visual, audio, _fuse):
        request = app.SearchReq(
            mode="hybrid",
            visual_text="four astronauts wearing black",
            audio_text="nghiên cứu ánh sáng cực quang",
        )

        result = app.search(request)

        visual.assert_called_once_with(request, "four astronauts wearing black")
        audio.assert_called_once_with(request, "nghiên cứu ánh sáng cực quang")
        self.assertEqual(result["mode"], "hybrid")

    @patch("backend.app.ocr_search_response", return_value=response("ocr"))
    @patch("backend.app.fuse_search_results", return_value=[])
    @patch("backend.app.audio_search_response", return_value=response("audio"))
    @patch("backend.app.visual_search", return_value=response("visual"))
    def test_visual_audio_and_ocr_run_together(self, visual, audio, fuse, ocr):
        request = app.SearchReq(
            mode="hybrid",
            visual_text="four astronauts wearing black",
            audio_text="nghiên cứu ánh sáng cực quang",
            ocr_text="THÀNH PHỐ",
        )

        result = app.search(request)

        visual.assert_called_once_with(request, "four astronauts wearing black")
        audio.assert_called_once_with(request, "nghiên cứu ánh sáng cực quang")
        ocr.assert_called_once_with(request, "THÀNH PHỐ", filter_mode=True)
        fuse.assert_called_once()
        self.assertEqual(result["mode"], "hybrid")
        self.assertEqual(
            result["latencies"],
            {"visual_ms": 1.0, "audio_ms": 1.0, "ocr_ms": 1.0},
        )

    @patch("backend.app.audio_search_response")
    @patch("backend.app.visual_search", return_value=response("visual"))
    def test_hybrid_with_only_visual_query_degrades_to_visual(self, visual, audio):
        request = app.SearchReq(mode="hybrid", visual_text="private spacecraft")

        result = app.search(request)

        visual.assert_called_once_with(request, "private spacecraft")
        audio.assert_not_called()
        self.assertEqual(result["mode"], "visual")

    @patch("backend.app.audio_search_response", return_value=response("audio"))
    @patch("backend.app.visual_search")
    def test_hybrid_with_only_audio_query_degrades_to_audio(self, visual, audio):
        request = app.SearchReq(mode="hybrid", audio_text="tàu vũ trụ tư nhân")

        result = app.search(request)

        audio.assert_called_once_with(request, "tàu vũ trụ tư nhân")
        visual.assert_not_called()
        self.assertEqual(result["mode"], "audio")

    @patch("backend.app.ocr_search_response", return_value=response("ocr"))
    @patch("backend.app.visual_search", return_value=response("visual"))
    def test_visual_query_with_ocr_filters_the_visual_response(self, visual, ocr):
        request = app.SearchReq(
            mode="hybrid", visual_text="city street", ocr_text="THÀNH PHỐ"
        )

        result = app.search(request)

        visual.assert_called_once_with(request, "city street")
        ocr.assert_called_once_with(request, "THÀNH PHỐ", filter_mode=True)
        self.assertEqual(result["mode"], "visual")
        self.assertEqual(result["latencies"], {"visual_ms": 1.0, "ocr_ms": 1.0})

    @patch("backend.app.ocr_search_response", return_value=response("ocr"))
    @patch("backend.app.audio_search_response", return_value=response("audio"))
    def test_audio_query_with_ocr_filters_the_audio_response(self, audio, ocr):
        request = app.SearchReq(
            mode="hybrid", audio_text="lời thoại", ocr_text="THÀNH PHỐ"
        )

        result = app.search(request)

        audio.assert_called_once_with(request, "lời thoại")
        ocr.assert_called_once_with(request, "THÀNH PHỐ", filter_mode=True)
        self.assertEqual(result["mode"], "audio")

    @patch("backend.app.ocr_search_response", return_value=response("ocr"))
    def test_hybrid_with_only_ocr_query_degrades_to_ocr(self, ocr):
        request = app.SearchReq(mode="hybrid", ocr_text="THÀNH PHỐ")

        result = app.search(request)

        ocr.assert_called_once_with(request, "THÀNH PHỐ")
        self.assertEqual(result["mode"], "ocr")

    @patch("backend.app.ocr_search_response", side_effect=RuntimeError("index offline"))
    @patch("backend.app.visual_search", return_value=response("visual"))
    def test_ocr_filter_failure_does_not_return_unfiltered_visual_results(self, _visual, _ocr):
        request = app.SearchReq(
            mode="hybrid", visual_text="city street", ocr_text="THÀNH PHỐ"
        )

        with self.assertRaises(HTTPException) as raised:
            app.search(request)

        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("ocr search unavailable", raised.exception.detail)

    @patch("backend.app.visual_search", return_value=response("visual"))
    def test_legacy_text_remains_supported(self, visual):
        request = app.SearchReq(mode="visual", text="legacy query")

        app.search(request)

        visual.assert_called_once_with(request, "legacy query")

    def test_empty_request_is_rejected(self):
        with self.assertRaises(HTTPException) as raised:
            app.search(app.SearchReq(mode="hybrid"))

        self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
