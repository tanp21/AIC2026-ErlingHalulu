#!/usr/bin/env python3
"""Export the pilot captions as one JSON payload for the review page.

`pilot_report.py` renders Markdown for reading in an editor; this produces the same
material as data, so a page can pivot it by shot, window and config.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts" / "captions" / "1024-default-8frames" / "captions.parquet"

# Why each shot is in the pilot. Ranks are PE-Core's, from
# docs/reports/visual-search-recall-failure-analysis-2026.md.
NOTES = {
    ("L27_V010", 83): ("target", "Manual review target"),
    ("L30_V078", 20): ("target", "Manual review target - a 2.4 s shot containing a cut"),
    ("L30_V078", 21): ("target", "Manual review target - recipe card, dense Vietnamese"),
    ("L30_V072", 19): ("target", "Manual review target"),
    ("K17_V003", 102): ("target", "Manual review target (2025, AV1 1080p)"),
    ("K19_V016", 158): ("target", "Manual review target (2025, AV1 1080p)"),
    ("L23_V013", 20): ("target", "Manual review target"),
    ("L23_V013", 21): ("target", "Manual review target - earlier runs used this index"),
    ("L25_V060", 66): ("miss", "R2 p2-20-kis, rank >1000. Geography lecture slide inside a 662 s shot"),
    ("L25_V012", 81): ("miss", "R2 p2-23-qa, rank >1000. Biology exam question in a 460 s shot"),
    ("L25_V041", 25): ("miss", "R1 p1-22-kis, rank >1000. English grammar lecture, 447 s shot"),
    ("L25_V062", 43): ("miss", "R2 p2-26-kis, rank >1000. Tug-of-war slide, 415 s shot"),
    ("L24_V006", 38): ("miss", "R3 p2-34-trake, ranks 978/>1000/>1000/140. Four events in one 31 s shot"),
    ("L24_V024", 28): ("miss", "R1 p1-16-trake:1, rank >1000. Dragon dance"),
    ("L24_V024", 29): ("miss", "R1 p1-16-trake:2, rank >1000. Lion dancer landing on a pillar"),
    ("L26_V156", 82): ("miss", "R3 p2-21-trake:4, rank >1000. 'A fourth orange segment' - ordinal counting"),
    ("L26_V156", 61): ("deep", "R3 p2-21-trake:3, rank 311"),
    ("L26_V481", 87): ("miss", "R3 p2-16-kis, rank >1000. Fine-grained cooking action"),
    ("L28_V018", 14): ("deep", "R1 p1-2-kis, rank 868. Ground truth is suspect - a graphics map, not a dam"),
    ("L21_V003", 224): ("deep", "R1 p1-9-qa, rank 370"),
    ("L26_V389", 86): ("deep", "R1 p1-18-kis, rank 332"),
    ("L26_V482", 28): ("deep", "R3 p2-17-kis, rank 229"),
    ("L30_V003", 46): ("deep", "R1 p1-25-kis, rank 184"),
    ("L29_V021", 43): ("deep", "R1 p1-13-kis, rank 155"),
    ("L22_V024", 230): ("deep", "R2 p2-6-kis, rank 153"),
    ("L26_V254", 8): ("target", "Added on request. Neighbour of R2 p2-30-qa, whose gold "
                                "frame 529 sits in shot 7 (19.24-21.96 s)"),
    ("L26_V448", 38): ("target", "Added on request. 1.04 s shot, the shortest in the pilot. "
                                 "R3 p2-24-qa's gold frame 2819 sits in shot 44"),
    ("L26_V448", 39): ("target", "Added on request. Follows shot 38 in the same video."),
}


def config_meta(rows):
    head = rows[0]
    return {
        "width": head["frame_width"],
        "kv": head["kv_cache_dtype"],
        "max_tokens": head["max_tokens"],
        "units": len(rows),
        "mean_prompt": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
        "mean_output": round(sum(r["output_tokens"] for r in rows) / len(rows)),
        "truncated": sum(1 for r in rows if r["finish_reason"] != "stop"),
        "mean_elapsed": round(sum(r["elapsed_sec"] for r in rows) / len(rows), 1),
    }


def build(paths, rates):
    configs, by_key = {}, defaultdict(dict)
    shots = {}
    for path in paths:
        label = Path(path).parent.name
        rows = pq.read_table(path).to_pylist()
        configs[label] = config_meta(rows)
        configs[label]["units_per_second"] = rates.get(label)
        for row in rows:
            shot_key = f"{row['video_id']}#{row['shot_index']}"
            kind, note = NOTES.get((row["video_id"], row["shot_index"]), ("other", ""))
            shots.setdefault(shot_key, {
                "video_id": row["video_id"],
                "shot_index": row["shot_index"],
                "group": row["group"],
                "year": row["year"],
                "start_sec": round(row["start_sec"], 2),
                "end_sec": round(row["end_sec"], 2),
                "duration_sec": round(row["duration_sec"], 2),
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "window_count": row["window_count"],
                "kind": kind,
                "note": note,
                "windows": {},
            })
            window = shots[shot_key]["windows"].setdefault(str(row["window_index"]), {
                "index": row["window_index"],
                "start": round(row["window_start_sec"], 2),
                "end": round(row["window_end_sec"], 2),
                "captions": {},
            })
            window["captions"][label] = {
                "summary": row["summary"],
                "description": row["caption"],
                "onscreen": row["onscreen_text"],
                "raw": row["raw_output"],
                "frames": row["frames_used"],
                "prompt_tokens": row["prompt_tokens"],
                "output_tokens": row["output_tokens"],
                "finish": row["finish_reason"],
                "elapsed": round(row["elapsed_sec"], 1),
            }

    baseline = {}
    if BASELINE.exists():
        for row in pq.read_table(BASELINE).to_pylist():
            baseline[f"{row['video_id']}#{row['shot_index']}"] = {
                "caption": row["caption"],
                "output_tokens": row["output_tokens"],
                "elapsed": round(row["elapsed_sec"], 1),
            }
    return {"configs": configs, "shots": shots, "baseline": baseline}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--rates", default="",
                        help="label=units_per_second pairs, comma separated")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rates = {}
    for pair in args.rates.split(","):
        if "=" in pair:
            label, value = pair.split("=", 1)
            rates[label.strip()] = float(value)
    payload = build(args.parquet, rates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    print(f"{len(payload['shots'])} shots, {len(payload['configs'])} configs "
          f"-> {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
