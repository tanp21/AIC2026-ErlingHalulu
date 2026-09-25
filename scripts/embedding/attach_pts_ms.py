"""Attach the organisers' real keyframe times to a sharded embedding ``meta.parquet``.

    uv run python -m scripts.embedding.attach_pts_ms \
        --embedding-dir data/2026/embeddings_n2026/pe-core \
        --points data/2026/keyframes_btc/points.parquet

``scripts/extraction/ingest_btc_keyframes.py`` renames every N-video keyframe to the
``frame_idx`` its ``map-keyframes`` row carries, so the two embedders (which take
``int(path.stem)`` as the frame id) need no change at all. That ``frame_idx`` is only a
relative label on this corpus, though -- N001-N100 is VFR, and BTC's own guidance is to submit
``pts_time * 1000`` for KIS and QA rather than a frame index. This joins that real time back
in, by ``(video_id, frame_idx)``, as a ``pts_ms`` column beside the four
``ShardWriter`` writes -- the same field ``notebooks/kaggle_pe_core_n2026_2s.py`` carries for
the same videos.

Run it **once, after embedding is finished**: ``ShardWriter._write_meta()`` rewrites
``meta.parquet`` with its own four columns at every checkpoint, so a ``pts_ms`` column added
mid-run would be dropped by the next shard that rolls over.

Every row must match. A row with no entry in ``points.parquet`` is a keyframe the ingest step
never accounted for, and guessing a time for it would put a wrong second into a submission --
so that fails, loudly, listing the first few.
"""
import argparse
import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

BASE_COLUMNS = ("shard_idx", "row_idx", "video_id", "frame_idx")
EXTRA_COLUMNS = {"pts_ms": pa.int64(), "keyframe_n": pa.int32()}


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_points(path):
    """``{(video_id, frame_idx): {"pts_ms": …, "keyframe_n": …}}`` from the ingest side table."""
    table = pq.read_table(path, columns=["video_id", "frame_idx", "pts_ms", "keyframe_n"])
    return {(row["video_id"], int(row["frame_idx"])): row for row in table.to_pylist()}


def attach(meta_path, points):
    """Rewrite one ``meta.parquet`` with ``pts_ms``/``keyframe_n`` joined in; returns rows."""
    meta_path = Path(meta_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"no meta.parquet at {meta_path}")
    table = pq.read_table(meta_path)
    missing = [name for name in BASE_COLUMNS if name not in table.column_names]
    if missing:
        raise ValueError(f"{meta_path}: missing column(s) {missing}; is this a ShardWriter "
                         f"output?")
    rows = table.to_pylist()

    unmatched = []
    out = []
    for row in rows:
        key = (row["video_id"], int(row["frame_idx"]))
        point = points.get(key)
        if point is None:
            unmatched.append(key)
            continue
        out.append({**{name: row[name] for name in BASE_COLUMNS},
                    "pts_ms": int(point["pts_ms"]),
                    "keyframe_n": int(point["keyframe_n"])})
    if unmatched:
        raise ValueError(
            f"{meta_path}: {len(unmatched)} of {len(rows)} row(s) have no entry in the points "
            f"table, first {unmatched[:5]} -- re-run the ingest for those videos rather than "
            f"letting them through without a time")

    # Row order is left exactly as the writer produced it: shard_idx/row_idx are what locate
    # each vector, and reordering them would only make the two files harder to compare.
    types = {"shard_idx": pa.int32(), "row_idx": pa.int32(), "video_id": pa.string(),
             "frame_idx": pa.int32(), **EXTRA_COLUMNS}
    pq.write_table(pa.table({name: pa.array([row[name] for row in out], type=types[name])
                             for name in (*BASE_COLUMNS, *EXTRA_COLUMNS)}), meta_path)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True, type=Path,
                        help="a sharded embedding directory (emb_*.npy + meta.parquet)")
    parser.add_argument("--points", required=True, type=Path,
                        help="the ingest side table, e.g. data/2026/keyframes_btc/"
                             "points.parquet")
    args = parser.parse_args(argv)

    points = load_points(args.points)
    log(f"{len(points)} keyframe row(s) in {args.points}")

    rows = attach(args.embedding_dir / "meta.parquet", points)
    info_path = args.embedding_dir / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
        info["pts_ms"] = ("from the organisers' map-keyframes (pts_time * 1000); "
                          "N001-N100 is VFR, so frame_idx is only a relative label")
        info_path.write_text(json.dumps(info, indent=2) + "\n")

    per_video = {}
    for row in rows:
        per_video[row["video_id"]] = per_video.get(row["video_id"], 0) + 1
    log(f"DONE: {len(rows)} row(s) across {len(per_video)} video(s) now carry pts_ms -> "
        f"{args.embedding_dir / 'meta.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
