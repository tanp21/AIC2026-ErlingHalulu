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
        self.assertIn("backend/subquery_fusion.py", Path("Dockerfile").read_text())

    def test_container_includes_the_clip_export_module(self):
        self.assertIn("backend/shots.py", Path("Dockerfile").read_text())

    def test_app_registers_clip_export_routes(self):
        paths = {route.path for route in app.routes}
        self.assertIn("/videos", paths)
        self.assertIn("/shots/{video_id}", paths)
        self.assertIn("/shots/{video_id}/labels", paths)
        self.assertIn("/clip/{video_id}/{shot_index}", paths)


if __name__ == "__main__":
    unittest.main()
