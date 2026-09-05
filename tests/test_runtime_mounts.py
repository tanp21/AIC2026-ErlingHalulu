import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend import core
from backend import frame_maps
from backend import thumb

ROOT = Path(__file__).resolve().parents[1]


class RuntimeMountTests(unittest.TestCase):
    def test_video_metadata_path_is_rebased_to_data_mount(self):
        row = {"year": "2026", "group": "L21", "video_id": "L21_V015", "path": "/host/secret"}
        self.assertEqual(
            core.mounted_video_path(row, Path("/data")),
            Path("/data/2026/videos/L21/L21_V015.mp4"),
        )

    def test_frame_map_directory_follows_index_mount(self):
        self.assertEqual(
            frame_maps.map_directory(Path("/indexes")),
            Path("/indexes/frame_maps"),
        )

    def test_collaboration_database_has_a_persistent_writable_mount(self):
        compose = (ROOT / "compose.yaml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("AIC_COLLAB_DB: /workspace/aic-collab.sqlite3", compose)
        self.assertIn("collab-data:/workspace", compose)
        self.assertIn("collab-data:", compose)
        self.assertIn("backend/collaboration.py", dockerfile)
        self.assertIn("mkdir /cache /workspace", dockerfile)
        self.assertIn("chown app:app /cache /workspace", dockerfile)

    def test_shot_labels_have_a_persistent_writable_mount(self):
        compose = (ROOT / "compose.yaml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("AIC_SHOT_LABEL_DIR: /labels", compose)
        self.assertIn("source: ./artifacts/shot-labels", compose)
        self.assertIn("target: /labels", compose)
        self.assertIn("create_host_path: false", compose)
        self.assertIn("backend/shot_labels.py", dockerfile)
        self.assertIn("mkdir /cache /workspace /labels", dockerfile)

    def test_public_ingress_uses_basic_auth_and_overwrites_identity(self):
        compose = (ROOT / "compose.yaml").read_text()
        haproxy = (ROOT / "haproxy.cfg").read_text()
        gitignore = (ROOT / ".gitignore").read_text().splitlines()

        self.assertIn('"0.0.0.0:8000:8000"', compose)
        self.assertIn("./haproxy-users.cfg:/usr/local/etc/haproxy/users.cfg:ro", compose)
        self.assertGreaterEqual(
            compose.count('user: "${AIC_UID:-1000}:${AIC_GID:-1000}"'), 2
        )
        self.assertIn('"/usr/local/etc/haproxy/users.cfg"', compose)
        self.assertIn("acl authenticated http_auth(aic_users)", haproxy)
        self.assertIn("http-request del-header X-AIC-User", haproxy)
        self.assertIn("http-request auth realm AIC if !health !authenticated", haproxy)
        self.assertIn(
            "http-request set-header X-AIC-User %[http_auth_user] if authenticated",
            haproxy,
        )
        self.assertIn("/haproxy-users.cfg", gitignore)

    def test_teammate_video_has_a_shared_wan_budget_but_admin_does_not(self):
        compose = (ROOT / "compose.yaml").read_text()
        haproxy = (ROOT / "haproxy.cfg").read_text()

        ingress = compose.split("  ingress:\n", 1)[1].split("\nvolumes:\n", 1)[0]
        self.assertIn("env_file:\n      - .env.secure", ingress)
        self.assertIn('acl admin http_auth_user -i "${AIC_ADMIN_USER}"', haproxy)
        self.assertIn("acl video path_beg /video/", haproxy)
        self.assertIn("use_backend teammate_video if video !admin", haproxy)
        self.assertIn("backend teammate_video", haproxy)
        self.assertIn(
            "filter bwlim-out teammate_video_limit limit 24m key int(1)",
            haproxy,
        )
        self.assertIn(
            "http-response set-bandwidth-limit teammate_video_limit", haproxy
        )
        self.assertNotIn("set-bandwidth-limit", haproxy.split("backend search_app", 1)[1].split("backend teammate_video", 1)[0])

    def test_ingress_capacity_does_not_block_admin_behind_video_streams(self):
        haproxy = (ROOT / "haproxy.cfg").read_text()

        self.assertIn("maxconn 512", haproxy)
        self.assertNotIn("maxconn 64", haproxy)

class ThumbnailConcurrencyTests(unittest.TestCase):
    def test_concurrent_same_thumbnail_is_generated_once(self):
        mapping = frame_maps.FrameMap(np.array([0, 10]), 1, 10)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            thumb, "CACHE_DIR", Path(directory)
        ), patch("backend.thumb.subprocess.run") as run:
            def generate(command, **_):
                time.sleep(0.02)
                Path(command[-1]).write_bytes(b"jpeg")

            run.side_effect = generate
            results = []
            workers = [
                threading.Thread(target=lambda: results.append(thumb.extract("video.mp4", 1, mapping)))
                for _ in range(5)
            ]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()

        self.assertEqual(run.call_count, 1)
        self.assertEqual(results, [b"jpeg"] * 5)

    def test_ffmpeg_has_a_timeout(self):
        mapping = frame_maps.FrameMap(np.array([0, 10]), 1, 10)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            thumb, "CACHE_DIR", Path(directory)
        ), patch("backend.thumb.subprocess.run") as run:
            run.side_effect = lambda command, **_: Path(command[-1]).write_bytes(b"jpeg")
            thumb.extract("video.mp4", 1, mapping)
        self.assertEqual(run.call_args.kwargs["timeout"], 30)


if __name__ == "__main__":
    unittest.main()
