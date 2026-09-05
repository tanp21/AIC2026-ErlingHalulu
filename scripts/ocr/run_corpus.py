#!/usr/bin/env python3
"""Run the OCR pipeline over every keyframe in the corpus, keeping per-line boxes.

The prior extraction that produced ``data/2026/ocr/ocr_2026.json`` (see
``docs/reports/ocr-shot-audit.md``) OCR'd only one representative keyframe per
shot and flattened each frame to a single string, discarding the per-line
boxes ``scripts.ocr.postprocess.suppress_static_overlays`` needs to tell a
scrolling news ticker apart from real on-screen text. This run instead OCRs
**every** pre-extracted visual-search keyframe (not just one per shot,
per the user's explicit ask), and writes the raw per-line
``{text, box, conf}`` records. Extraction is decoupled from shot assignment on
purpose: a keyframe's shot is resolved later, at indexing time
(``scripts.indexing.build_ocr_index``), by comparing its frame index against
shot boundaries -- that keeps this script a plain "OCR this directory of
images" tool, and lets the indexer pick whichever keyframe in a shot survives
suppression best.

Meant to run per shard on rented GPU instances (e.g. vast.ai): ``--groups``
restricts to a comma-separated list of top-level groups (``L21,L22``), and
``--shard i/n`` splits the remaining video list round-robin so several
instances can split one group between them. Output is resumable -- keyframes
already present in ``--out`` are skipped, and the file is flushed
periodically so an interrupted rented instance doesn't lose all its progress.

Usage (module form, matches the repo's other offline tooling):
    uv run python -m scripts.ocr.run_corpus --groups L21,L22 --out data/2026/ocr/raw_v2/out_L21_L22.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ocr.engines import OcrConfig, OcrPipeline  # noqa: E402


def iter_keyframes(keyframes_dir: Path, groups: list[str] | None, shard: tuple[int, int] | None):
    """Yields ``(group, video_id, frame_idx, keyframe_path)`` for every keyframe, in a stable order."""
    group_dirs = sorted(p for p in keyframes_dir.iterdir() if p.is_dir())
    if groups:
        wanted = set(groups)
        group_dirs = [p for p in group_dirs if p.name in wanted]

    videos = []
    for group_dir in group_dirs:
        for video_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            videos.append((group_dir.name, video_dir))

    if shard:
        index, total = shard
        videos = videos[index::total]

    for group, video_dir in videos:
        video_id = video_dir.name
        keyframe_files = sorted(
            (p for p in video_dir.glob("*.png") if not p.name.endswith(".tmp.png")),
            key=lambda p: int(p.stem),
        )
        for frame_path in keyframe_files:
            yield group, video_id, int(frame_path.stem), frame_path


def load_existing(out_path: Path) -> dict:
    if out_path.exists():
        return json.loads(out_path.read_text())
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyframes-dir", type=Path, default=ROOT / "data" / "2026" / "keyframes")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--groups", default="", help="comma-separated group filter, e.g. L21,L22")
    parser.add_argument("--shard", default="", help="i/n -- process every n-th video starting at i")
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--limit", type=int, default=0, help="cap number of keyframes (0 = no cap)")
    args = parser.parse_args()

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    shard = None
    if args.shard:
        index_str, total_str = args.shard.split("/")
        shard = (int(index_str), int(total_str))

    results = load_existing(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    pipeline = OcrPipeline(OcrConfig())

    import cv2

    processed = 0
    skipped_existing = 0
    start_time = time.monotonic()
    for group, video_id, frame_idx, frame_path in iter_keyframes(
        args.keyframes_dir, groups, shard
    ):
        key = f"{group}/{video_id}/{frame_idx}"
        if key in results:
            skipped_existing += 1
            continue

        image = cv2.imread(str(frame_path))
        if image is None:
            print(f"[run-corpus] unreadable frame, skipping {key}: {frame_path}", file=sys.stderr)
            continue

        try:
            lines = pipeline.run(image)
        except Exception as exc:  # pragma: no cover -- keep the corpus run alive
            print(f"[run-corpus] error on {key}: {exc}", file=sys.stderr)
            lines = []

        results[key] = {"lines": lines}
        processed += 1

        if processed % args.flush_every == 0:
            args.out.write_text(json.dumps(results))
            elapsed = time.monotonic() - start_time
            print(
                f"[run-corpus] {processed} keyframes this run "
                f"({skipped_existing} already done), {elapsed / max(processed, 1):.2f}s/keyframe",
                flush=True,
            )

        if args.limit and processed >= args.limit:
            break

    args.out.write_text(json.dumps(results))
    print(f"[run-corpus] DONE: {len(results)} keyframes total in {args.out}")


if __name__ == "__main__":
    main()
