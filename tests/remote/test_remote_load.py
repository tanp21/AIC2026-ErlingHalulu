import json
import subprocess
import sys
import threading
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


class RemoteLoadTests(unittest.TestCase):
    def test_load_credentials_reads_five_tab_separated_accounts(self):
        try:
            from scripts.benchmarks.remote_load import load_credentials
        except ImportError as exc:
            self.fail(f"remote load benchmark is missing: {exc}")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.txt"
            path.write_text(
                "AIC initial credentials - keep private\n\n"
                "owner@example.com\towner-secret\n"
                "one@example.com\tone-secret\n"
                "two@example.com\ttwo-secret\n"
                "three@example.com\tthree-secret\n"
                "four@example.com\tfour-secret\n"
            )

            credentials = load_credentials(path)

        self.assertEqual(credentials[0], ("owner@example.com", "owner-secret"))
        self.assertEqual(len(credentials), 5)

    def test_load_credentials_rejects_wrong_account_count(self):
        from scripts.benchmarks.remote_load import load_credentials

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.txt"
            path.write_text("owner@example.com\tsecret\n")
            with self.assertRaisesRegex(ValueError, "exactly five"):
                load_credentials(path)

    def test_load_credentials_rejects_duplicate_accounts(self):
        from scripts.benchmarks.remote_load import load_credentials

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "credentials.txt"
            path.write_text("same@example.com\tsecret\n" * 5)
            with self.assertRaisesRegex(ValueError, "unique"):
                load_credentials(path)

    def test_multipart_payload_contains_keyframe_reference(self):
        try:
            from scripts.benchmarks.remote_load import multipart_payload
        except ImportError as exc:
            self.fail(f"multipart helper is missing: {exc}")

        body, content_type = multipart_payload({
            "mode": "visual",
            "index": "flat",
            "nprobe": 0,
            "ef": 0,
            "image_weight": 1,
            "reference": {"video_id": "L21_V001", "frame_idx": 42},
        })

        self.assertIn("multipart/form-data; boundary=", content_type)
        self.assertIn(b'name="payload"', body)
        self.assertIn(b'"video_id": "L21_V001"', body)

    def test_summary_labels_mean_request_throughput_and_contains_no_credentials(self):
        try:
            from scripts.benchmarks.remote_load import Metric, summarize
        except ImportError as exc:
            self.fail(f"summary helper is missing: {exc}")

        report = summarize([
            Metric("mixed", "admin", "search", 200, 100, 50, 0, 500),
            Metric("mixed", "admin", "search", 200, 300, 60, 0, 700),
        ])
        encoded = json.dumps(report)

        self.assertEqual(report["mixed/admin/search"]["count"], 2)
        self.assertEqual(report["mixed/admin/search"]["p95_ms"], 300)
        self.assertEqual(report["mixed/admin/search"]["mean_request_mbps"], 0.02)
        self.assertNotIn("mbps", report["mixed/admin/search"])
        self.assertNotIn("secret", encoded)

    def test_run_cycle_exercises_search_video_and_rerank(self):
        try:
            from scripts.benchmarks.remote_load import HttpClient, run_cycle
        except ImportError as exc:
            self.fail(f"workload runner is missing: {exc}")

        requests = []
        search_response = json.dumps({
            "results": [{
                "video_id": "L21_V001",
                "frame_idx": 42,
                "thumb": "/thumb/L21_V001/42",
                "frame": "/frame/L21_V001/42",
                "video": "/video/L21_V001",
            }],
        }).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def _send(self, status, body, **headers):
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name.replace("_", "-"), value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                requests.append(("POST", self.path, self.headers.get("Authorization")))
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                self._send(200, search_response, Content_Type="application/json")

            def do_GET(self):
                requests.append(("GET", self.path, self.headers.get("Authorization")))
                if self.path.startswith("/video/"):
                    requested = self.headers["Range"].removeprefix("bytes=")
                    start, requested_end = (int(value) for value in requested.split("-"))
                    total = 8192
                    end = min(requested_end, total - 1)
                    self._send(
                        206,
                        b"v" * (end - start + 1),
                        Content_Range=f"bytes {start}-{end}/{total}",
                    )
                elif self.path.startswith("/frame-map/"):
                    self._send(200, b"{}", Content_Type="application/json")
                elif self.path.startswith("/transcript/"):
                    self._send(200, b'{"cues": []}', Content_Type="application/json")
                else:
                    self._send(200, b"image")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        metrics = []
        try:
            client = HttpClient(
                f"http://127.0.0.1:{server.server_port}",
                ("owner@example.com", "secret"),
                "admin",
                "mixed",
                metrics.append,
            )
            run_cycle(client, "người đi xe máy qua giao lộ")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        paths = [(method, path) for method, path, _ in requests]
        self.assertIn(("POST", "/search"), paths)
        self.assertIn(("GET", "/video/L21_V001"), paths)
        self.assertIn(("POST", "/search/image"), paths)
        self.assertTrue(all(auth and auth.startswith("Basic ") for _, _, auth in requests))
        self.assertEqual(sum(metric.operation == "video_range" for metric in metrics), 2)

    def test_client_rejects_cross_origin_asset_url_before_sending_credentials(self):
        from scripts.benchmarks.remote_load import HttpClient

        client = HttpClient(
            "https://aic.example",
            ("owner@example.com", "secret"),
            "admin",
            "mixed",
            lambda _: None,
        )

        with self.assertRaisesRegex(ValueError, "same origin"):
            client.request("thumb", "https://attacker.example/collect")

    def test_client_rejects_cross_origin_redirect_without_forwarding_credentials(self):
        from scripts.benchmarks.remote_load import HttpClient

        received = []

        class TargetHandler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                received.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

        target = ThreadingHTTPServer(("127.0.0.1", 0), TargetHandler)

        class RedirectHandler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(302)
                self.send_header(
                    "Location", f"http://127.0.0.1:{target.server_port}/collect"
                )
                self.end_headers()

        source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        threads = [
            threading.Thread(target=server.serve_forever, daemon=True)
            for server in (source, target)
        ]
        for thread in threads:
            thread.start()
        try:
            client = HttpClient(
                f"http://127.0.0.1:{source.server_port}",
                ("owner@example.com", "secret"),
                "admin",
                "mixed",
                lambda _: None,
            )
            with self.assertRaisesRegex(ValueError, "cross-origin redirect"):
                client.request("thumb", "/redirect")
        finally:
            source.shutdown()
            target.shutdown()
            source.server_close()
            target.server_close()
            for thread in threads:
                thread.join()

        self.assertEqual(received, [])

    def test_video_range_requires_partial_response_and_matching_content_range(self):
        from scripts.benchmarks.remote_load import HttpClient

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", "5")
                self.end_headers()
                self.wfile.write(b"video")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        metrics = []
        try:
            client = HttpClient(
                f"http://127.0.0.1:{server.server_port}",
                ("owner@example.com", "secret"),
                "admin",
                "mixed",
                metrics.append,
            )
            client.request(
                "video_range",
                "/video/test",
                headers={"Range": "bytes=0-4"},
                expected_range=(0, 4),
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(metrics[-1].status, 599)

    def test_video_range_requires_body_length_to_match_content_range(self):
        from scripts.benchmarks.remote_load import HttpClient

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                self.send_response(206)
                self.send_header("Content-Range", "bytes 0-4/10")
                self.send_header("Content-Length", "1")
                self.end_headers()
                self.wfile.write(b"x")

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        metrics = []
        try:
            client = HttpClient(
                f"http://127.0.0.1:{server.server_port}",
                ("owner@example.com", "secret"),
                "admin",
                "mixed",
                metrics.append,
            )
            client.request(
                "video_range",
                "/video/test",
                headers={"Range": "bytes=0-4"},
                expected_range=(0, 4),
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertEqual(metrics[-1].status, 599)

    def test_evaluate_requires_admin_latency_and_video_slo(self):
        try:
            from scripts.benchmarks.remote_load import evaluate
        except ImportError as exc:
            self.fail(f"SLO evaluator is missing: {exc}")

        passing = {
            "baseline/admin/search": {"p50_ms": 800, "p95_ms": 1000, "errors": 0},
            "baseline/admin/rerank": {"p50_ms": 900, "p95_ms": 1200, "errors": 0},
            "mixed/admin/search": {"p50_ms": 900, "p95_ms": 1200, "errors": 0},
            "mixed/admin/rerank": {"p50_ms": 1000, "p95_ms": 1400, "errors": 0},
            "mixed/admin/video_range": {
                "count": 2,
                "p95_ms": 1800,
                "ttfb_p95_ms": 500,
                "max_gap_ms": 1000,
                "errors": 0,
            },
            "baseline/admin/video_range": {
                "count": 2,
                "p95_ms": 1500,
                "errors": 0,
            },
        }
        for operation in ("thumb", "frame", "frame_map", "transcript"):
            passing[f"baseline/admin/{operation}"] = {
                "count": 1, "errors": 0, "p95_ms": 50,
            }
            passing[f"mixed/admin/{operation}"] = {
                "count": 1, "errors": 0, "p95_ms": 60,
            }
        for operation in (
            "search", "thumb", "frame", "frame_map", "transcript", "video_range", "rerank"
        ):
            passing[f"mixed/teammate/{operation}"] = {"count": 4, "errors": 0}
        passing["mixed/teammate/video_range"]["count"] = 8
        failing = {
            **passing,
            "mixed/admin/video_range": {
                "count": 2,
                "ttfb_p95_ms": 2500,
                "max_gap_ms": 6000,
                "errors": 0,
            },
        }

        self.assertTrue(evaluate(passing)["passed"])
        self.assertFalse(evaluate(failing)["passed"])

        missing_teammate_work = {
            key: value for key, value in passing.items()
            if not key.startswith("mixed/teammate/")
        }
        self.assertFalse(evaluate(missing_teammate_work)["passed"])

        failing_admin_asset = {
            **passing,
            "mixed/admin/frame": {"count": 1, "errors": 1, "p95_ms": 60},
        }
        self.assertFalse(evaluate(failing_admin_asset)["passed"])

        slow_admin_asset = {
            **passing,
            "mixed/admin/frame": {"count": 1, "errors": 0, "p95_ms": 2500},
        }
        self.assertFalse(evaluate(slow_admin_asset)["passed"])

        slow_admin_video = {
            **passing,
            "mixed/admin/video_range": {
                "count": 2,
                "p95_ms": 6000,
                "ttfb_p95_ms": 500,
                "max_gap_ms": 1000,
                "errors": 0,
            },
        }
        self.assertFalse(evaluate(slow_admin_video)["passed"])

    def test_evaluate_allows_rare_subsecond_admin_queue_wait(self):
        from scripts.benchmarks.remote_load import evaluate

        summary = {
            "baseline/admin/search": {"p50_ms": 450, "p95_ms": 540, "errors": 0},
            "baseline/admin/rerank": {"p50_ms": 380, "p95_ms": 620, "errors": 0},
            "mixed/admin/search": {"p50_ms": 490, "p95_ms": 920, "errors": 0},
            "mixed/admin/rerank": {"p50_ms": 410, "p95_ms": 680, "errors": 0},
            "mixed/admin/video_range": {
                "count": 2, "p95_ms": 80, "ttfb_p95_ms": 50,
                "max_gap_ms": 50, "errors": 0,
            },
            "baseline/admin/video_range": {
                "count": 2, "p95_ms": 60, "errors": 0,
            },
        }
        for operation in ("thumb", "frame", "frame_map", "transcript"):
            summary[f"baseline/admin/{operation}"] = {
                "count": 1, "errors": 0, "p95_ms": 50,
            }
            summary[f"mixed/admin/{operation}"] = {
                "count": 1, "errors": 0, "p95_ms": 60,
            }
        for operation in (
            "search", "thumb", "frame", "frame_map", "transcript", "rerank"
        ):
            summary[f"mixed/teammate/{operation}"] = {"count": 4, "errors": 0}
        summary["mixed/teammate/video_range"] = {"count": 8, "errors": 0}

        self.assertTrue(evaluate(summary)["passed"])

    def test_main_refuses_plain_http_without_explicit_opt_in(self):
        from scripts.benchmarks.remote_load import main

        with self.assertRaisesRegex(SystemExit, "allow-insecure-http"):
            main(["--base-url", "http://171.227.204.189"])

    def test_main_returns_failure_when_slo_fails(self):
        from scripts.benchmarks import remote_load

        with tempfile.TemporaryDirectory() as directory:
            credentials = Path(directory) / "credentials.txt"
            credentials.write_text(
                "owner@example.com\towner-secret\n"
                "one@example.com\tone-secret\n"
                "two@example.com\ttwo-secret\n"
                "three@example.com\tthree-secret\n"
                "four@example.com\tfour-secret\n"
            )
            output = Path(directory) / "report.json"
            with (
                mock.patch.object(remote_load, "run_phase"),
                mock.patch.object(remote_load, "summarize", return_value={}),
            ):
                status = remote_load.main([
                    "--base-url", "http://127.0.0.1",
                    "--allow-insecure-http",
                    "--credentials", str(credentials),
                    "--baseline-seconds", "1",
                    "--duration", "1",
                    "--output", str(output),
                ])

        self.assertEqual(status, 1)

    def test_cli_documents_full_and_teammates_only_modes(self):
        result = subprocess.run(
            [sys.executable, "scripts/benchmarks/remote_load.py", "--help"],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertIn("--duration", result.stdout)
        self.assertIn("teammates-only", result.stdout)
        self.assertIn("--allow-insecure-http", result.stdout)


if __name__ == "__main__":
    unittest.main()
