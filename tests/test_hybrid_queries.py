import unittest
from unittest.mock import patch

from fastapi import HTTPException

import app


def response(mode):
    return {
        "results": [],
        "latency_ms": 1.0,
        "mode": mode,
        "index": "flat",
        "index_info": {"type": "test"},
    }


class SeparateHybridQueryTests(unittest.TestCase):
    @patch("app.fuse_search_results", return_value=[])
    @patch("app.audio_search_response", return_value=response("audio"))
    @patch("app.visual_search", return_value=response("visual"))
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

    @patch("app.audio_search_response")
    @patch("app.visual_search", return_value=response("visual"))
    def test_hybrid_with_only_visual_query_degrades_to_visual(self, visual, audio):
        request = app.SearchReq(mode="hybrid", visual_text="private spacecraft")

        result = app.search(request)

        visual.assert_called_once_with(request, "private spacecraft")
        audio.assert_not_called()
        self.assertEqual(result["mode"], "visual")

    @patch("app.audio_search_response", return_value=response("audio"))
    @patch("app.visual_search")
    def test_hybrid_with_only_audio_query_degrades_to_audio(self, visual, audio):
        request = app.SearchReq(mode="hybrid", audio_text="tàu vũ trụ tư nhân")

        result = app.search(request)

        audio.assert_called_once_with(request, "tàu vũ trụ tư nhân")
        visual.assert_not_called()
        self.assertEqual(result["mode"], "audio")

    @patch("app.visual_search", return_value=response("visual"))
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
