import unittest
from pathlib import Path

from app import FRONTEND_DIST, app, index


class FrontendServingTests(unittest.TestCase):
    def test_root_and_static_mount_use_the_vite_distribution(self):
        response = index()
        static_mount = next(route for route in app.routes if route.path == "/static")

        self.assertEqual(Path(response.path), FRONTEND_DIST / "index.html")
        self.assertEqual(Path(static_mount.app.directory), FRONTEND_DIST)


if __name__ == "__main__":
    unittest.main()
