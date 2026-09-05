#!/usr/bin/env python3
"""Exercise the public AIC endpoint from a machine outside the home LAN."""

import argparse
import json
import math
import re
import secrets
import time
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


def load_credentials(path: Path) -> list[tuple[str, str]]:
    credentials = []
    for line in path.read_text().splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and "@" in parts[0] and parts[1]:
            credentials.append((parts[0].strip().lower(), parts[1].strip()))
    if len(credentials) != 5:
        raise ValueError("credentials file must contain exactly five accounts")
    if len({user for user, _ in credentials}) != 5:
        raise ValueError("credentials file must contain five unique accounts")
    return credentials


def multipart_payload(payload: dict) -> tuple[bytes, str]:
    boundary = f"aic-{secrets.token_hex(12)}"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="payload"\r\n\r\n'
        f"{json.dumps(payload)}\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    return body, f"multipart/form-data; boundary={boundary}"


@dataclass(frozen=True)
class Metric:
    phase: str
    role: str
    operation: str
    status: int
    elapsed_ms: float
    ttfb_ms: float
    max_gap_ms: float
    size: int


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def summarize(metrics: list[Metric]) -> dict[str, dict]:
    groups: dict[str, list[Metric]] = {}
    for metric in metrics:
        groups.setdefault(
            f"{metric.phase}/{metric.role}/{metric.operation}", []
        ).append(metric)

    report = {}
    for key, items in sorted(groups.items()):
        elapsed = [item.elapsed_ms for item in items]
        duration_seconds = sum(elapsed) / 1000
        report[key] = {
            "count": len(items),
            "errors": sum(not 200 <= item.status < 300 for item in items),
            "p50_ms": round(_percentile(elapsed, 0.50), 1),
            "p95_ms": round(_percentile(elapsed, 0.95), 1),
            "max_ms": round(max(elapsed), 1),
            "ttfb_p95_ms": round(
                _percentile([item.ttfb_ms for item in items], 0.95), 1
            ),
            "max_gap_ms": round(max(item.max_gap_ms for item in items), 1),
            "mib": round(sum(item.size for item in items) / 1024 / 1024, 2),
            "mean_request_mbps": round(
                sum(item.size for item in items) * 8 / duration_seconds / 1_000_000,
                2,
            ) if duration_seconds else 0,
        }
    return report


def evaluate(summary: dict[str, dict]) -> dict:
    checks = {}
    for operation in ("search", "rerank"):
        baseline = summary.get(f"baseline/admin/{operation}", {})
        mixed = summary.get(f"mixed/admin/{operation}", {})
        baseline_p50 = baseline.get("p50_ms", 0)
        mixed_p50 = mixed.get("p50_ms", math.inf)
        mixed_p95 = mixed.get("p95_ms", math.inf)
        checks[f"admin_{operation}"] = bool(
            baseline_p50
            and baseline.get("errors") == 0
            and mixed.get("errors") == 0
            and mixed_p50 <= baseline_p50 * 1.25
            and mixed_p95 <= 1500
        )

    for operation in ("thumb", "frame", "frame_map", "transcript"):
        baseline = summary.get(f"baseline/admin/{operation}", {})
        metric = summary.get(f"mixed/admin/{operation}", {})
        baseline_p95 = baseline.get("p95_ms", 0)
        mixed_p95 = metric.get("p95_ms", math.inf)
        checks[f"admin_{operation}"] = bool(
            baseline.get("count", 0) >= 1
            and baseline.get("errors") == 0
            and metric.get("count", 0) >= 1
            and metric.get("errors") == 0
            and mixed_p95 <= 2000
            and mixed_p95 <= max(250, baseline_p95 * 2)
        )

    baseline_video = summary.get("baseline/admin/video_range", {})
    baseline_video_p95 = baseline_video.get("p95_ms", 0)
    video = summary.get("mixed/admin/video_range", {})
    video_p95 = video.get("p95_ms", math.inf)
    checks["admin_video"] = bool(
        baseline_video.get("count", 0) >= 2
        and baseline_video.get("errors") == 0
        and baseline_video_p95
        and video.get("count", 0) >= 2
        and video.get("errors") == 0
        and video_p95 <= 5000
        and video_p95 <= max(500, baseline_video_p95 * 1.5)
        and video.get("ttfb_p95_ms", math.inf) <= 2000
        and video.get("max_gap_ms", math.inf) <= 5000
    )
    for operation in (
        "search", "thumb", "frame", "frame_map", "transcript", "video_range", "rerank"
    ):
        metric = summary.get(f"mixed/teammate/{operation}", {})
        minimum = 8 if operation == "video_range" else 4
        checks[f"teammate_{operation}"] = bool(
            metric.get("count", 0) >= minimum and metric.get("errors") == 0
        )
    return {"passed": all(checks.values()), "checks": checks}


def _origin(url: str) -> tuple[str, str, int | None]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    port = parsed.port or {"http": 80, "https": 443}.get(scheme)
    return scheme, (parsed.hostname or "").lower(), port


def _content_range(value: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", value.strip())
    if not match:
        return None
    start, end, total = (int(part) for part in match.groups())
    if start > end or end >= total:
        return None
    return start, end, total


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    def __init__(self, allowed_origin):
        super().__init__()
        self.allowed_origin = allowed_origin

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _origin(newurl) != self.allowed_origin:
            raise ValueError("cross-origin redirect refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class HttpClient:
    def __init__(
        self,
        base_url: str,
        credential: tuple[str, str],
        role: str,
        phase: str,
        record,
        timeout: float = 30,
    ):
        self.base_url = base_url.rstrip("/") + "/"
        self.origin = _origin(self.base_url)
        self.opener = build_opener(_SameOriginRedirectHandler(self.origin))
        token = b64encode(f"{credential[0]}:{credential[1]}".encode()).decode()
        self.authorization = f"Basic {token}"
        self.role = role
        self.phase = phase
        self.record = record
        self.timeout = timeout

    def request(
        self,
        operation: str,
        path: str,
        *,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
        expected_range: tuple[int, int] | None = None,
    ) -> tuple[int, bytes, object]:
        target = urljoin(self.base_url, path)
        if _origin(target) != self.origin:
            raise ValueError("request URL must use the configured server's same origin")
        request_headers = {"Authorization": self.authorization, **(headers or {})}
        request = Request(
            target,
            data=data,
            headers=request_headers,
            method="POST" if data is not None else "GET",
        )
        started = time.perf_counter()
        status = 0
        body = bytearray()
        response_headers = {}
        first_byte_at = started
        max_gap = 0.0
        try:
            try:
                response = self.opener.open(request, timeout=self.timeout)
            except HTTPError as exc:
                response = exc
            with response:
                status = response.status
                response_headers = response.headers
                previous = time.perf_counter()
                while chunk := response.read(64 * 1024):
                    now = time.perf_counter()
                    if not body:
                        first_byte_at = now
                    max_gap = max(max_gap, now - previous)
                    previous = now
                    body.extend(chunk)
        finally:
            finished = time.perf_counter()
            metric_status = status
            if expected_range is not None:
                parsed_range = _content_range(response_headers.get("Content-Range", ""))
                expected_start, requested_end = expected_range
                if (
                    status != 206
                    or parsed_range is None
                    or parsed_range[0] != expected_start
                    or parsed_range[1] != min(requested_end, parsed_range[2] - 1)
                    or len(body) != parsed_range[1] - parsed_range[0] + 1
                ):
                    metric_status = 599
            self.record(Metric(
                self.phase,
                self.role,
                operation,
                metric_status,
                (finished - started) * 1000,
                (first_byte_at - started) * 1000,
                max_gap * 1000,
                len(body),
            ))
        return status, bytes(body), response_headers

    def post_json(self, operation: str, path: str, payload: dict):
        return self.request(
            operation,
            path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )


def run_cycle(client: HttpClient, query: str) -> None:
    search_request = {
        "mode": "visual",
        "index": "flat",
        "nprobe": 0,
        "ef": 0,
        "visual_text": query,
    }
    status, body, _ = client.post_json("search", "/search", search_request)
    if status != 200:
        return
    results = json.loads(body).get("results", [])
    if not results:
        return
    result = results[0]

    client.request("thumb", result["thumb"])
    client.request("frame", result["frame"])
    client.request("frame_map", f'/frame-map/{result["video_id"]}')
    client.request("transcript", f'/transcript/{result["video_id"]}')

    status, _, headers = client.request(
        "video_range",
        result["video"],
        headers={"Range": "bytes=0-2097151"},
        expected_range=(0, 2097151),
    )
    parsed_range = _content_range(headers.get("Content-Range", ""))
    if status != 206 or parsed_range is None:
        return
    total = parsed_range[2]
    offset = total // 2
    end = min(total - 1, offset + 2 * 1024 * 1024 - 1)
    client.request(
        "video_range",
        result["video"],
        headers={"Range": f"bytes={offset}-{end}"},
        expected_range=(offset, end),
    )

    rerank = {
        **search_request,
        "image_weight": 1,
        "reference": {
            "video_id": result["video_id"],
            "frame_idx": result["frame_idx"],
        },
    }
    rerank_body, content_type = multipart_payload(rerank)
    client.request(
        "rerank",
        "/search/image",
        data=rerank_body,
        headers={"Content-Type": content_type},
    )


QUERIES = (
    "người đi xe máy qua giao lộ",
    "đám đông đứng trước tòa nhà",
    "xe buýt chạy trên đường phố",
    "người mặc áo sáng màu đang nói chuyện",
    "cảnh quay ngoài trời vào ban ngày",
)


def run_phase(
    base_url: str,
    credentials: list[tuple[str, str]],
    admin_user: str,
    phase: str,
    duration: float,
    metrics: list[Metric],
) -> None:
    deadline = time.monotonic() + duration

    def worker(index: int, credential: tuple[str, str]):
        role = "admin" if credential[0] == admin_user else "teammate"
        client = HttpClient(base_url, credential, role, phase, metrics.append)
        cycle = 0
        while time.monotonic() < deadline:
            try:
                run_cycle(client, QUERIES[(index + cycle) % len(QUERIES)])
            except Exception as exc:
                print(f"{phase}/{role}: {type(exc).__name__}: {exc}")
                time.sleep(0.2)
            cycle += 1

    with ThreadPoolExecutor(max_workers=len(credentials)) as executor:
        futures = [
            executor.submit(worker, index, credential)
            for index, credential in enumerate(credentials)
        ]
        for future in futures:
            future.result()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Stress the public AIC server with search, video Range and rerank traffic."
    )
    parser.add_argument("--base-url", default="http://171.227.204.189")
    parser.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="Explicitly allow Basic Auth over unencrypted HTTP (credentials can be intercepted)",
    )
    parser.add_argument(
        "--credentials", type=Path, default=Path("aic-initial-passwords.txt")
    )
    parser.add_argument("--admin-user", help="Defaults to the first credential")
    parser.add_argument("--duration", type=float, default=600)
    parser.add_argument("--baseline-seconds", type=float, default=60)
    parser.add_argument(
        "--mode", choices=("full", "teammates-only"), default="full"
    )
    parser.add_argument("--output", type=Path, default=Path("wan-load-report.json"))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.duration <= 0 or args.baseline_seconds <= 0:
        raise SystemExit("durations must be positive")
    scheme = urlsplit(args.base_url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise SystemExit("base URL must use http or https")
    if scheme == "http" and not args.allow_insecure_http:
        raise SystemExit(
            "plain HTTP exposes Basic Auth credentials; use HTTPS or pass "
            "--allow-insecure-http to accept that risk"
        )
    credentials = load_credentials(args.credentials)
    admin_user = (args.admin_user or credentials[0][0]).strip().lower()
    if admin_user not in {user for user, _ in credentials}:
        raise SystemExit("admin user is not present in the credentials file")

    metrics: list[Metric] = []
    started = datetime.now(timezone.utc).isoformat()
    if args.mode == "full":
        admin_credential = next(item for item in credentials if item[0] == admin_user)
        print(f"baseline: admin only for {args.baseline_seconds:g}s")
        run_phase(
            args.base_url,
            [admin_credential],
            admin_user,
            "baseline",
            args.baseline_seconds,
            metrics,
        )
        active_credentials = credentials
        phase = "mixed"
    else:
        active_credentials = [item for item in credentials if item[0] != admin_user]
        phase = "teammates_only"

    print(f"{phase}: {len(active_credentials)} clients for {args.duration:g}s")
    run_phase(
        args.base_url,
        active_credentials,
        admin_user,
        phase,
        args.duration,
        metrics,
    )
    summary = summarize(metrics)
    report = {
        "started_at": started,
        "base_url": args.base_url,
        "mode": args.mode,
        "duration_seconds": args.duration,
        "baseline_seconds": args.baseline_seconds if args.mode == "full" else 0,
        "summary": summary,
        "slo": evaluate(summary) if args.mode == "full" else None,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["slo"] or {"report": str(args.output)}, indent=2))
    print(f"report: {args.output}")
    return 0 if report["slo"] is None or report["slo"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
