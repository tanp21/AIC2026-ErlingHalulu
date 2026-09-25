"""Merge per-rank JSONL OCR shards downloaded from Kaggle into the flat dict schema
``scripts/indexing/build_ocr_index.py`` expects.

    uv run python -m scripts.ocr.merge_ocr_kaggle_shards \\
        --shards "data/2026/ocr/kaggle_parts/**/*.jsonl" \\
        --out data/2026/ocr/raw_v2/ocr_batch2_m01_m05.json \\
        --expected-count 135114

Each shard is one line per keyframe, written by the Kaggle notebook's ``OcrShardWriter``:
``{"key": "<group>/<video_id>/<frame_idx>", "lines": [{"text", "box", "conf"}, ...]}``. A run
split across multiple Kaggle sessions (``PART_INDEX``/``PART_COUNT``) or GPUs (one shard per
rank) produces several such files; this merges them all into the single
``{"<key>": {"lines": [...]}}`` dict ``build_ocr_index.py`` reads, unchanged -- no downstream
code needs to know shards ever existed.

Fail-closed on purpose: whole-video partitioning across ranks/parts should make every key
unique, so a duplicate key is a bug (double-processed video, or two parts overlapping), not
something to silently overwrite.
"""
import argparse
import json
from pathlib import Path


def iter_shard_records(path):
    """Yields ``(key, lines)`` from one JSONL shard file, in file order."""
    with open(path) as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: malformed JSON line") from exc
            yield record["key"], record["lines"]


def merge_shards(paths):
    """Merges every shard's records into one dict. Raises on any cross-shard key collision."""
    merged = {}
    for path in paths:
        for key, lines in iter_shard_records(path):
            if key in merged:
                raise ValueError(f"duplicate key across shards: {key} (seen again in {path})")
            merged[key] = {"lines": lines}
    return merged


def validate_merged(merged, expected_count=None):
    """Fail-closed sanity checks on the merged dict. Raises on any violation."""
    if expected_count is not None and len(merged) != expected_count:
        raise ValueError(f"expected {expected_count} keyframes, merged {len(merged)}")
    for key, value in merged.items():
        lines = value.get("lines")
        if not isinstance(lines, list):
            raise ValueError(f"{key}: 'lines' is not a list")
        for line in lines:
            box = line.get("box")
            conf = line.get("conf")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{key}: malformed box {box!r}")
            if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
                raise ValueError(f"{key}: malformed conf {conf!r}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, nargs="+",
                        help="one or more JSONL shard paths, or glob patterns")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None,
                        help="e.g. the manifest's keyframe_count, minus any skipped videos")
    args = parser.parse_args(argv)

    paths = sorted({
        p for pattern in args.shards
        for p in (Path(pattern).parent.glob(Path(pattern).name)
                  if any(c in pattern for c in "*?[")
                  else [Path(pattern)])
    })
    if not paths:
        raise SystemExit(f"no shard files matched {args.shards}")

    merged = merge_shards(paths)
    validate_merged(merged, expected_count=args.expected_count)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(merged))
    print(f"[merge-ocr-kaggle-shards] {len(paths)} shards -> {len(merged)} keyframes "
         f"-> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
