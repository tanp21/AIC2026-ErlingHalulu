#!/usr/bin/env python3
"""Package the 2026 shot captions for the Kaggle Qwen3-Embedding runs.

Reads ``data/2026/captions/shot_captions.parquet``, keeps the columns an
offline embedding run needs, sorts by ``(video_id, shot_index, window_index)``
(the row order every embedding artifact and eval result is keyed on), and writes
a single parquet plus the 137-case eval fixtures and per-video fps. Upload the
output directory as a private Kaggle dataset (one copy per account that runs a
notebook) and point the notebook at it.

Usage:
    uv run python -m scripts.embedding.prepare_caption_embed_dataset
    uv run python -m scripts.embedding.prepare_caption_embed_dataset --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAPTIONS = ROOT / "data/2026/captions/shot_captions.parquet"
DEFAULT_OUTPUT = ROOT / "data/2026/captions/embed_dataset"
DEFAULT_SHOTS = ROOT / "data/2026/shots"
DEFAULT_QUERIES = ROOT / "artifacts/official_benchmark_queries_en_v2.json"
DEFAULT_ANSWERS = ROOT / "artifacts/official_benchmark_answers_v2.json"
EXPECTED_ROWS = 103_086
EXPECTED_VIDEOS = 873
COLUMNS = [
    "group", "video_id", "shot_index", "window_index", "window_count",
    "start_frame", "end_frame", "start_sec", "end_sec", "duration_sec",
    "window_start_sec", "window_end_sec", "caption", "model_id", "prompt_sha256",
]
KEY_COLUMNS = ("video_id", "shot_index", "window_index")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fps(shots_root: Path, group: str, video_id: str) -> float:
    path = Path(shots_root) / group / f"{video_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"shot intervals missing for {video_id}: {path}")
    return float(json.loads(path.read_text())["fps"])


def build(captions: Path, output: Path, shots_root: Path, queries: Path, answers: Path,
          strict: bool = True):
    table = pq.read_table(captions, columns=COLUMNS)
    nulls = pc.sum(pc.is_null(table["caption"])).as_py()
    if nulls:
        raise RuntimeError(f"{nulls} captions are null")
    table = table.sort_by([(column, "ascending") for column in KEY_COLUMNS])
    keys = list(zip(*[table[column].to_pylist() for column in KEY_COLUMNS]))
    if len(set(keys)) != table.num_rows:
        raise RuntimeError("duplicate (video_id, shot_index, window_index) rows")
    videos = set(table["video_id"].to_pylist())
    if strict:
        if table.num_rows != EXPECTED_ROWS:
            raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {table.num_rows}")
        if len(videos) != EXPECTED_VIDEOS:
            raise RuntimeError(f"expected {EXPECTED_VIDEOS} videos, got {len(videos)}")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    captions_out = output / "shot_captions_2026.parquet"
    pq.write_table(table, captions_out, compression="zstd")

    fps = {}
    for group, video_id in sorted({(row["group"], row["video_id"]) for row in
                                   table.select(["group", "video_id"]).to_pylist()}):
        fps[video_id] = load_fps(shots_root, group, video_id)
    (output / "video_fps.json").write_text(json.dumps(fps, indent=2, sort_keys=True) + "\n")

    shutil.copyfile(queries, output / Path(queries).name)
    shutil.copyfile(answers, output / Path(answers).name)

    manifest = {
        "source": str(captions),
        "rows": table.num_rows,
        "videos": len(videos),
        "fps_entries": len(fps),
        "columns": COLUMNS,
        "model_id": sorted(set(table["model_id"].to_pylist())),
        "prompt_sha256": sorted(set(table["prompt_sha256"].to_pylist())),
        "sha256": sha256_file(captions_out),
        "size_bytes": captions_out.stat().st_size,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def self_test():
    import tempfile

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        shots = root / "shots" / "L99"
        shots.mkdir(parents=True)
        for video_id, fps in (("L99_V001", 25.0), ("L99_V002", 30.0)):
            (shots / f"{video_id}.json").write_text(json.dumps({"fps": fps, "shots": []}))
        rows = []
        for index, video_id in enumerate(["L99_V002", "L99_V001", "L99_V002"]):
            rows.append({
                "group": "L99", "video_id": video_id, "shot_index": index,
                "window_index": 0, "window_count": 1, "start_frame": 0, "end_frame": 10,
                "start_sec": 0.0, "end_sec": 1.0, "duration_sec": 1.0,
                "window_start_sec": 0.0, "window_end_sec": 1.0,
                "caption": f"caption {index}", "model_id": "test", "prompt_sha256": "abc",
            })
        source = root / "captions.parquet"
        pq.write_table(pa.Table.from_pylist(rows), source)
        queries, answers = root / "q.json", root / "a.json"
        queries.write_text("{}")
        answers.write_text("{}")
        manifest = build(source, root / "out", root / "shots", queries, answers, strict=False)
        assert manifest["rows"] == 3 and manifest["videos"] == 2
        ordered = pq.read_table(root / "out/shot_captions_2026.parquet")["video_id"].to_pylist()
        assert ordered == ["L99_V001", "L99_V002", "L99_V002"], ordered
        assert json.loads((root / "out/video_fps.json").read_text()) == {
            "L99_V001": 25.0, "L99_V002": 30.0}
        try:
            build(source, root / "out2", root / "shots", queries, answers, strict=True)
        except RuntimeError as error:
            assert "expected 103086 rows" in str(error)
        else:
            raise AssertionError("strict build accepted the fixture")
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=DEFAULT_CAPTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shots", type=Path, default=DEFAULT_SHOTS)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--answers", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    manifest = build(args.captions, args.output, args.shots, args.queries, args.answers)
    print(f"prepared {args.output}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
