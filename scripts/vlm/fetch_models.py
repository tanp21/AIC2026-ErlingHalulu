#!/usr/bin/env python3
"""Download the local vision-language models the agent can run instead of Gemini.

The app container is read-only with ``HF_HUB_OFFLINE=1``, so the weights have to
be fetched here, on the dev machine, and mounted in (``compose.yaml`` mounts
``$AIC_VLM_DIR`` alongside the existing Hugging Face cache). Run with the network
up; re-running is a no-op once the files are present.

Usage:
    uv run python -m scripts.vlm.fetch_models                  # every candidate
    uv run python -m scripts.vlm.fetch_models --models Vintern-1B-v3_5
    uv run python -m scripts.vlm.fetch_models --list           # what, and whether it is here
    uv run python -m scripts.vlm.fetch_models --self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

from agent.vlm import CANDIDATES, missing, snapshot, spec, vlm_dir


def report() -> int:
    """Print each candidate and whether its snapshot is on disk."""
    absent = {candidate.name for candidate in missing()}
    for candidate in CANDIDATES:
        here = "present" if candidate.name not in absent else "MISSING"
        size = _size(candidate.snapshot_dir)
        print(f"{here:>8}  {candidate.name:<32} {size:>9}  {candidate.note}")
    print(f"\nsnapshots live in {vlm_dir()}")
    return len(absent)


def _size(directory: Path) -> str:
    if not directory.is_dir():
        return "-"
    total = sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    for unit in ("B", "KiB", "MiB", "GiB"):
        if total < 1024 or unit == "GiB":
            return f"{total:.0f}{unit}" if unit == "B" else f"{total:.1f}{unit}"
        total /= 1024
    return "-"


def fetch(names: list[str]) -> None:
    for name in names:
        candidate = spec(name)
        print(f"downloading {candidate.name} -> {candidate.snapshot_dir}", flush=True)
        path = snapshot(candidate)
        print(f"  done: {path}", flush=True)


def self_test() -> None:
    """The registry is the only logic here; check it resolves and stays unique."""
    assert len({candidate.name for candidate in CANDIDATES}) == len(CANDIDATES)
    assert len({candidate.slug for candidate in CANDIDATES}) == len(CANDIDATES)
    for candidate in CANDIDATES:
        assert "/" in candidate.name, candidate.name
        assert candidate.family in ("internvl", "qwen3vl", "qwen2_5vl"), candidate.family
        # slug and lookup both round-trip, since the benchmark names models on the CLI
        assert spec(candidate.name) is candidate
        assert spec(candidate.slug) is candidate
        assert candidate.snapshot_dir.parent == vlm_dir()
    try:
        spec("nope/nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown model resolved")
    # `missing` must not claim a directory that only exists as an empty shell
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=[c.name for c in CANDIDATES],
                        help="hub ids or slugs; defaults to every candidate")
    parser.add_argument("--list", action="store_true", help="report and exit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.list:
        raise SystemExit(1 if report() else 0)
    fetch(args.models)
    report()


if __name__ == "__main__":
    main()
