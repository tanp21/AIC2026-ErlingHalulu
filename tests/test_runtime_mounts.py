import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend import core
from backend import frame_maps
from backend import shots
from backend import thumb

ROOT = Path(__file__).resolve().parents[1]


class RuntimeMountTests(unittest.TestCase):
    def test_video_metadata_path_is_rebased_to_data_mount(self):
        row = {"year": "2026", "group": "L21", "video_id": "L21_V015", "path": "/host/secret"}
        self.assertEqual(
            core.mounted_video_path(row, Path("/data")),
            Path("/data/2026/videos/L21/L21_V015.mp4"),
        )

    def test_streaming_path_prefers_faststart_copy_when_present(self):
        row = {"year": "2026", "group": "L21", "video_id": "L21_V015", "path": "/data/2026/videos/L21/L21_V015.mp4"}
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            self.assertEqual(shots.streaming_path(row, data_dir), Path(row["path"]))

            faststart = data_dir / "2026" / "videos_faststart" / "L21" / "L21_V015.mp4"
            faststart.parent.mkdir(parents=True)
            faststart.write_bytes(b"remuxed")
            self.assertEqual(shots.streaming_path(row, data_dir), faststart)

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

    def test_agent_output_has_a_persistent_writable_mount(self):
        compose = (ROOT / "compose.yaml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("AIC_AGENT_OUT_DIR: /workspace/agent", compose)
        self.assertIn("backend/agent_runner.py", dockerfile)

    def test_local_vlm_snapshots_have_a_read_only_mount(self):
        """A local vision backend loads weights from a materialised directory, not
        the blob cache, and the container is offline — so the mount is the only way
        it sees them."""
        compose = (ROOT / "compose.yaml").read_text()
        self.assertIn("AIC_VLM_DIR: /models/vlm", compose)
        self.assertIn(":/models/vlm:ro", compose)
        self.assertIn("AIC_VLM_DIR:?Set AIC_VLM_DIR", compose)

    def test_agent_runs_have_an_egress_network(self):
        compose = (ROOT / "compose.yaml").read_text()
        app = compose.split("  app:\n", 1)[1].split("  ingress:\n", 1)[0]
        self.assertIn("- aic-egress", app)
        self.assertIn("aic-egress:", compose)

    def test_shot_labels_have_a_persistent_writable_mount(self):
        compose = (ROOT / "compose.yaml").read_text()
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn("AIC_SHOT_LABEL_DIR: /labels", compose)
        self.assertIn("source: ./artifacts/shot-labels", compose)
        self.assertIn("target: /labels", compose)
        self.assertIn("create_host_path: false", compose)
        self.assertIn("backend/shot_labels.py", dockerfile)
        self.assertIn("mkdir /cache /workspace /labels", dockerfile)

    def test_public_ingress_grants_identity_by_tailscale_serve_header(self):
        compose = (ROOT / "compose.yaml").read_text()
        haproxy = (ROOT / "haproxy.cfg").read_text()
        gitignore = (ROOT / ".gitignore").read_text().splitlines()

        self.assertIn('"127.0.0.1:8000:8000"', compose)
        self.assertNotIn("haproxy-users.cfg", compose)
        self.assertGreaterEqual(
            compose.count('user: "${AIC_UID:-1000}:${AIC_GID:-1000}"'), 2
        )
        self.assertIn("acl local_proxy src 127.0.0.1 172.16.0.0/12", haproxy)
        self.assertIn(
            "acl has_ts_login req.hdr(Tailscale-User-Login) -m found", haproxy
        )
        self.assertIn("http-request del-header X-AIC-User", haproxy)
        self.assertIn("http-request deny deny_status 403 if !health !local_proxy", haproxy)
        self.assertIn(
            'http-request set-header X-AIC-User "%[req.hdr(Tailscale-User-Login)]" '
            "if local_proxy has_ts_login",
            haproxy,
        )
        self.assertIn(
            'http-request set-header X-AIC-User "${AIC_ADMIN_USER}" if local_proxy !has_ts_login',
            haproxy,
        )
        self.assertNotIn("http_auth", haproxy)
        # The Basic Auth password file is no longer wired in, but stays gitignored in case
        # it is ever reintroduced.
        self.assertIn("/haproxy-users.cfg", gitignore)

    def test_teammate_video_has_a_shared_wan_budget_but_admin_does_not(self):
        compose = (ROOT / "compose.yaml").read_text()
        haproxy = (ROOT / "haproxy.cfg").read_text()

        ingress = compose.split("  ingress:\n", 1)[1].split("\nvolumes:\n", 1)[0]
        self.assertIn("env_file:\n      - .env.secure", ingress)
        self.assertIn(
            'acl is_admin_login req.hdr(Tailscale-User-Login) -i "${AIC_ADMIN_USER}"', haproxy
        )
        self.assertIn("acl video path_beg /video/", haproxy)
        self.assertIn("use_backend teammate_video if video !is_admin_login", haproxy)
        self.assertIn("backend teammate_video", haproxy)
        self.assertIn("stick-table type ip size 10k expire 1m store bytes_out_rate(1s)", haproxy)
        self.assertIn(
            "filter bwlim-out teammate_video_limit limit 20m key src",
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

    def test_thumbnail_cache_key_ignores_the_mount_prefix(self):
        mapping = frame_maps.FrameMap(np.array([0, 10]), 1, 10)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            thumb, "CACHE_DIR", Path(directory)
        ), patch("backend.thumb.subprocess.run") as run:
            run.side_effect = lambda command, **_: Path(command[-1]).write_bytes(b"jpeg")
            host = thumb.extract("/home/user/AIC_HCMC/data/2026/videos/L22/L22_V024.mp4", 1, mapping)
            container = thumb.extract("/data/2026/videos/L22/L22_V024.mp4", 1, mapping)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(host, container)

    def test_ffmpeg_pins_filter_threads(self):
        mapping = frame_maps.FrameMap(np.array([0, 10]), 1, 10)

        command = thumb.ffmpeg_command("video.mp4", 1, mapping, "out.jpg", 320)

        self.assertIn("-filter_threads", command)
        self.assertIn("1", command[command.index("-filter_threads") + 1])


if __name__ == "__main__":
    unittest.main()
