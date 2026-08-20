import asyncio
import os
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

import app


class StartupWarmupTests(unittest.TestCase):
    def test_warmup_initializes_search_components_in_safe_order(self):
        events = Mock()
        query = np.zeros((1, 1024), dtype="float32")
        index = Mock()
        index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        with (
            patch("app.core.load_meta", side_effect=lambda: events.meta()),
            patch("app.core.encode_text", side_effect=lambda text: (events.visual(text), query)[1]),
            patch("app.core.get_index", side_effect=lambda name: (events.index(name), index)[1]),
            patch("app.audio_search.encode_query", side_effect=lambda text: events.audio(text)),
        ):
            app.warmup_search()

        self.assertEqual(
            events.mock_calls,
            [
                call.meta(),
                call.visual("warmup visual search"),
                call.index("flat"),
                call.audio("khởi động tìm kiếm lời nói"),
            ],
        )
        index.search.assert_called_once_with(query, 1)

    def test_warmup_propagates_component_failure(self):
        with (
            patch("app.core.load_meta"),
            patch("app.core.encode_text", side_effect=RuntimeError("PE-Core unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "PE-Core unavailable"):
                app.warmup_search()

    def test_warmup_reports_timings_through_uvicorn_logger(self):
        query = np.zeros((1, 1024), dtype="float32")
        index = Mock()
        with (
            patch("app.core.load_meta"),
            patch("app.core.encode_text", return_value=query),
            patch("app.core.get_index", return_value=index),
            patch("app.audio_search.encode_query"),
            self.assertLogs("uvicorn.error", level="INFO") as logs,
        ):
            app.warmup_search()

        self.assertTrue(any("search warm-up complete" in line for line in logs.output))

    def test_lifespan_blocks_on_warmup_before_serving(self):
        async def enter_lifespan():
            async with app.lifespan(app.app):
                pass

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with patch.dict(os.environ, auth_env), patch("app.warmup_search") as warmup:
            asyncio.run(enter_lifespan())

        warmup.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
