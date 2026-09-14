"""Simulate the team's 5-way concurrent load: hybrid search + video open.

    uv run python -m scripts.loadtest.simulate_five_users --video-id L21_V015 --rounds 5 \\
        --credentials user1@example.com:pw1 user2@example.com:pw2 user3@example.com:pw3 \\
                       user4@example.com:pw4 user5@example.com:pw5

Each credential runs `--rounds` iterations of: one hybrid /search call (latency),
then one Range GET on /video/{video-id} for the first MB (time to first byte).
All credentials run concurrently, matching a real competition moment where
everyone searches and opens videos around the same time.

Compare the printed p50/p95 against docs/operations.md's documented targets:
hybrid search comfortably under 4s even at full 5-way concurrency, and video
TTFB in the low single digits of seconds (not the multi-minute stalls this
was written to diagnose). Passwords are only ever supplied on the command
line / read from the environment -- never hardcode them here.
"""
import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


def one_user(base_url, auth, video_id, rounds, verify):
    search_ms, video_ttfb_ms = [], []
    with httpx.Client(base_url=base_url, auth=auth, timeout=30, verify=verify) as client:
        for _ in range(rounds):
            started = time.perf_counter()
            response = client.post(
                "/search",
                json={"mode": "hybrid", "visual_text": "person walking", "audio_text": "xin chào"},
            )
            response.raise_for_status()
            search_ms.append((time.perf_counter() - started) * 1000)

            started = time.perf_counter()
            with client.stream(
                "GET", f"/video/{video_id}", headers={"Range": "bytes=0-1048575"}
            ) as stream:
                next(stream.iter_bytes())
            video_ttfb_ms.append((time.perf_counter() - started) * 1000)
    return search_ms, video_ttfb_ms


def percentile(values, pct):
    ordered = sorted(values)
    index = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[index]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--credentials", nargs="+", required=True,
        help="user:pass pairs, one per simulated teammate",
    )
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification")
    args = parser.parse_args(argv)

    creds = [tuple(pair.split(":", 1)) for pair in args.credentials]

    with ThreadPoolExecutor(max_workers=len(creds)) as pool:
        futures = [
            pool.submit(one_user, args.base_url, cred, args.video_id, args.rounds, not args.insecure)
            for cred in creds
        ]
        results = [future.result() for future in futures]

    all_search = [ms for search, _ in results for ms in search]
    all_video = [ms for _, video in results for ms in video]

    print(f"{len(creds)} users x {args.rounds} rounds = {len(all_search)} requests per metric")
    print(
        f"search    p50={statistics.median(all_search):.0f}ms  "
        f"p95={percentile(all_search, 0.95):.0f}ms  max={max(all_search):.0f}ms"
    )
    print(
        f"video-ttfb p50={statistics.median(all_video):.0f}ms  "
        f"p95={percentile(all_video, 0.95):.0f}ms  max={max(all_video):.0f}ms"
    )


if __name__ == "__main__":
    main()
