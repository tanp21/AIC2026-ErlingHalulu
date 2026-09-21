import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
