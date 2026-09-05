import asyncio
import os
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from backend import app


class StartupWarmupTests(unittest.TestCase):
    def tearDown(self):
        app.core.start_text_batcher()

    def test_warmup_initializes_search_components_in_safe_order(self):
        events = Mock()
        query = np.zeros((1, 1024), dtype="float32")
        index = Mock()
        index.search.return_value = (np.zeros((1, 1)), np.zeros((1, 1)))

        with (
            patch.dict(os.environ, {"AIC_WARMUP": "full"}),
            patch("backend.app.core.load_meta", side_effect=lambda: events.meta()),
            patch("backend.app.core.encode_text", side_effect=lambda text: (events.visual(text), query)[1]),
            patch("backend.app.core.get_index", side_effect=lambda name: (events.index(name), index)[1]),
            patch("backend.app.audio_search.encode_query", side_effect=lambda text: events.audio(text)),
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
            patch.dict(os.environ, {"AIC_WARMUP": "full"}),
            patch("backend.app.core.load_meta"),
            patch("backend.app.core.encode_text", side_effect=RuntimeError("PE-Core unavailable")),
        ):
            with self.assertRaisesRegex(RuntimeError, "PE-Core unavailable"):
                app.warmup_search()

    def test_warmup_reports_timings_through_uvicorn_logger(self):
        query = np.zeros((1, 1024), dtype="float32")
        index = Mock()
        with (
            patch.dict(os.environ, {"AIC_WARMUP": "full"}),
            patch("backend.app.core.load_meta"),
            patch("backend.app.core.encode_text", return_value=query),
            patch("backend.app.core.get_index", return_value=index),
            patch("backend.app.audio_search.encode_query"),
            self.assertLogs("uvicorn.error", level="INFO") as logs,
        ):
            app.warmup_search()

        self.assertTrue(any("search warm-up complete" in line for line in logs.output))

    def test_default_warmup_only_loads_metadata(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("backend.app.core.load_meta") as metadata,
            patch("backend.app.core.encode_text") as visual,
            patch("backend.app.core.get_index") as index,
            patch("backend.app.audio_search.encode_query") as audio,
        ):
            app.warmup_search()

        metadata.assert_called_once_with()
        visual.assert_not_called()
        index.assert_not_called()
        audio.assert_not_called()

    def test_lifespan_blocks_on_warmup_before_serving(self):
        async def enter_lifespan():
            async with app.lifespan(app.app):
                pass

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with patch.dict(os.environ, auth_env), patch("backend.app.warmup_search") as warmup:
            asyncio.run(enter_lifespan())

        warmup.assert_called_once_with()

    def test_lifespan_closes_text_batcher_after_shutdown(self):
        async def enter_lifespan():
            async with app.lifespan(app.app):
                close.assert_not_called()

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with (
            patch.dict(os.environ, auth_env),
            patch("backend.app.warmup_search"),
            patch("backend.app.core.close_text_batcher") as close,
        ):
            asyncio.run(enter_lifespan())

        close.assert_called_once_with()

    def test_lifespan_closes_text_batcher_when_application_work_fails(self):
        async def enter_lifespan():
            async with app.lifespan(app.app):
                raise RuntimeError("request failed")

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with (
            patch.dict(os.environ, auth_env),
            patch("backend.app.warmup_search"),
            patch("backend.app.core.close_text_batcher") as close,
            self.assertRaisesRegex(RuntimeError, "request failed"),
        ):
            asyncio.run(enter_lifespan())

        close.assert_called_once_with()

    def test_lifespan_configures_admin_search_priority(self):
        async def exercise_admin():
            async with app.lifespan(app.app):
                for _ in range(11):
                    await app.app.state.search_coordinator.run(
                        "owner@example.com", lambda: asyncio.sleep(0)
                    )

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        try:
            with patch.dict(os.environ, auth_env), patch("backend.app.warmup_search"):
                asyncio.run(exercise_admin())
        except Exception as exc:
            self.fail(f"lifespan did not configure admin priority: {exc}")

    def test_lifespan_allows_admin_and_four_teammates_to_search_together(self):
        async def exercise_team():
            started = 0
            all_started = asyncio.Event()
            release = asyncio.Event()

            async def operation():
                nonlocal started
                started += 1
                if started == 5:
                    all_started.set()
                await release.wait()

            async with app.lifespan(app.app):
                coordinator = app.app.state.search_coordinator
                tasks = [
                    asyncio.create_task(coordinator.run(user, operation))
                    for user in (
                        "one@example.com",
                        "two@example.com",
                        "three@example.com",
                        "four@example.com",
                        "owner@example.com",
                    )
                ]
                try:
                    await asyncio.wait_for(all_started.wait(), 0.5)
                    concurrent = True
                except TimeoutError:
                    concurrent = False
                release.set()
                await asyncio.gather(*tasks)
                self.assertTrue(concurrent)

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with patch.dict(os.environ, auth_env), patch("backend.app.warmup_search"):
            asyncio.run(exercise_team())

    def test_lifespan_does_not_rate_limit_teammate_searches(self):
        async def exercise_teammate():
            async with app.lifespan(app.app):
                coordinator = app.app.state.search_coordinator
                for _ in range(11):
                    await coordinator.run(
                        "one@example.com", lambda: asyncio.sleep(0)
                    )

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        try:
            with patch.dict(os.environ, auth_env), patch("backend.app.warmup_search"):
                asyncio.run(exercise_teammate())
        except Exception as exc:
            self.fail(f"lifespan rate limited a trusted teammate: {exc}")

    def test_lifespan_does_not_starve_teammate_after_admin_activity(self):
        async def exercise_teammate():
            async with app.lifespan(app.app):
                coordinator = app.app.state.search_coordinator
                coordinator.touch_admin_activity()
                await asyncio.wait_for(
                    coordinator.run("one@example.com", lambda: asyncio.sleep(0)),
                    0.1,
                )

        auth_env = {
            "AIC_ALLOWED_USERS": "owner@example.com,one@example.com,two@example.com,three@example.com,four@example.com",
            "AIC_ADMIN_USER": "owner@example.com",
        }
        with patch.dict(os.environ, auth_env), patch("backend.app.warmup_search"):
            asyncio.run(exercise_teammate())


if __name__ == "__main__":
    unittest.main()
