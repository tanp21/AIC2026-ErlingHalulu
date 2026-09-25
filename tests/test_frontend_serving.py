import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi import Response

from backend import app as app_module, frame_maps
from backend.app import FRONTEND_DIST, app, index


class FrontendServingTests(unittest.TestCase):
    def test_root_and_static_mount_use_the_vite_distribution(self):
        response = index()
        static_mount = next(route for route in app.routes if route.path == "/static")

        self.assertEqual(Path(response.path), FRONTEND_DIST / "index.html")
        self.assertEqual(Path(static_mount.app.directory), FRONTEND_DIST)

    def test_app_registers_collaboration_routes(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/collab/rounds", paths)
        self.assertIn("/api/collab/queries/{query_id}", paths)

    def test_container_includes_temporal_runtime_artifacts(self):
        dockerfile = Path("Dockerfile").read_text()

        self.assertIn("backend/temporal_search.py", dockerfile)
        self.assertIn("benchmarks/temporal/temporal_config.json", dockerfile)

    def test_container_includes_the_paraphrase_fusion_module(self):
        # /agent/search imports it for embedding-mean paraphrase fusion.
        dockerfile = Path("Dockerfile").read_text()
        self.assertIn("backend/subquery_fusion.py", dockerfile)
        self.assertIn("backend/caption_search.py", dockerfile)

    def test_container_includes_the_clip_export_module(self):
        self.assertIn("backend/shots.py", Path("Dockerfile").read_text())

    def test_container_includes_the_server_hosted_agent(self):
        dockerfile = Path("Dockerfile").read_text()

        self.assertIn("backend/agent_runner.py", dockerfile)
        self.assertIn("COPY agent ./agent", dockerfile)

    def test_container_includes_the_on_demand_checker(self):
        # The Search tab's "Check by AI". `agent/` is copied wholesale, so only the backend
        # half needs naming here.
        dockerfile = Path("Dockerfile").read_text()

        self.assertIn("backend/checker.py", dockerfile)
        self.assertIn("COPY agent ./agent", dockerfile)

    def test_app_registers_check_routes(self):
        paths = {route.path for route in app.routes}

        self.assertIn("/api/check", paths)
        self.assertIn("/api/check/lookup", paths)

    def test_app_registers_agent_run_routes(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/api/agent/runs", paths)
        self.assertIn("/api/agent/runs/{run_id}", paths)
        self.assertIn("/api/agent/runs/{run_id}/submission.zip", paths)

    def test_app_registers_clip_export_routes(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/videos", paths)
        self.assertIn("/shots/{video_id}", paths)
        self.assertIn("/shots/{video_id}/labels", paths)
        self.assertIn("/clip/{video_id}/{shot_index}", paths)

    def test_app_registers_the_shot_caption_route(self):
        """The agent reads shot captions as context over HTTP, like every other channel."""
        from backend.core_service import PROXY_PREFIXES

        paths = {route.path for route in app.routes}
        # Two routes, one path segment apart: the whole video in time order, and the
        # frames a verify chunk is about to judge. Registering both under the same
        # path silently dropped whichever FastAPI did not match first.
        self.assertIn("/captions/{video_id}", paths)
        self.assertIn("/captions/{video_id}/frames", paths)
        self.assertIn("/captions/{video_id}/windows", paths)
        # Split development forwards media/data routes to the compute host; a route
        # missing here works locally and 404s for a teammate.
        self.assertIn("/captions/", PROXY_PREFIXES)

    def test_app_registers_the_ocr_route_and_forwards_it(self):
        """The agent's review view reads a shot's on-screen text whether or not a query matched it."""
        from backend.core_service import PROXY_PREFIXES

        self.assertIn("/ocr/{video_id}", {route.path for route in app.routes})
        self.assertIn("/ocr/", PROXY_PREFIXES)

    def test_context_can_return_every_indexed_keyframe_in_original_video_time(self):
        mapping = frame_maps.FrameMap(np.array([100, 140, 200, 260, 350]), 1, 1000)
        with patch("backend.app.core.frame_ids", return_value=np.arange(5)), \
             patch("backend.app.frame_maps.load", return_value=mapping):
            nearby = app_module.keyframe_context(Response(), "L01_V001", 2, "l2026", 1, False)
            complete = app_module.keyframe_context(Response(), "L01_V001", 2, "l2026", 1, True)
        self.assertEqual([frame["frame_idx"] for frame in nearby["frames"]], [1, 2, 3])
        self.assertEqual([frame["frame_idx"] for frame in complete["frames"]], [0, 1, 2, 3, 4])
        self.assertEqual([frame["time"] for frame in complete["frames"]], [0, .04, .1, .16, .25])


if __name__ == "__main__":
    unittest.main()
