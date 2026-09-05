import unittest
from unittest.mock import patch

import httpx

from backend import audio_search
from backend.access_control import AccessConfig
from backend.app import app


class TranscriptCueTests(unittest.TestCase):
    def test_groups_timed_tokens_into_sentence_like_cues(self):
        cues = audio_search.build_transcript_cues([
            {"offset": 0.0, "token": " Xin"},
            {"offset": 1.0, "token": " chào."},
            {"offset": 2.2, "token": " Các"},
            {"offset": 3.0, "token": " bạn!"},
            {"offset": 20.0, "token": " Tin"},
            {"offset": 29.0, "token": " mới"},
        ])

        self.assertEqual(cues, [
            {"start": 0.0, "end": 2.2, "text": "Xin chào."},
            {"start": 2.2, "end": 5.0, "text": "Các bạn!"},
            {"start": 20.0, "end": 22.0, "text": "Tin"},
            {"start": 29.0, "end": 31.0, "text": "mới"},
        ])

    def test_empty_rows_produce_no_cues(self):
        self.assertEqual(audio_search.build_transcript_cues([]), [])

    def test_ignores_invalid_timed_tokens(self):
        self.assertEqual(audio_search.build_transcript_cues([
            {"offset": None, "token": " lỗi"},
            {"offset": "bad", "token": " lỗi"},
            {"offset": 1, "token": None},
            {"offset": 2, "token": " hợp lệ."},
        ]), [{"start": 2.0, "end": 4.0, "text": "hợp lệ."}])


class TranscriptApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app.state.access_config = AccessConfig(
            frozenset({"owner@example.com", "mate@example.com"}), "owner@example.com"
        )
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_transcript_route_requires_identity_and_returns_cues(self):
        self.assertEqual((await self.client.get("/transcript/L21_V001")).status_code, 401)
        with patch("backend.app.audio_search.load_transcript_cues", return_value=[
            {"start": 1.0, "end": 3.0, "text": "xin chào"}
        ]):
            response = await self.client.get(
                "/transcript/L21_V001",
                headers={"X-AIC-User": "mate@example.com"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "videoId": "L21_V001",
            "cues": [{"start": 1.0, "end": 3.0, "text": "xin chào"}],
        })

    async def test_transcript_route_reports_unavailable_source(self):
        with patch("backend.app.audio_search.load_transcript_cues", side_effect=FileNotFoundError):
            response = await self.client.get(
                "/transcript/L21_V001",
                headers={"X-AIC-User": "mate@example.com"},
            )
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "transcript data unavailable")


if __name__ == "__main__":
    unittest.main()
