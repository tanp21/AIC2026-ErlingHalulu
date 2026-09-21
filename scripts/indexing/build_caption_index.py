#!/usr/bin/env python3
"""Build the Elasticsearch shot-caption index from the CapRL caption parquet.

Reads ``artifacts/captions/shot_captions.parquet`` (either schema in
``scripts.captioning.caption_shots``: the window schema's
``(video_id, shot_index, window_index)`` rows, or the older one-row-per-shot
schema, which is read as window 0 of 1), resolves each caption's time window to
a real frame through the same frame-map machinery the rest of the app uses
(never fps arithmetic), and bulk-indexes one document per caption.

BM25 by default. ``--dense`` additionally embeds every caption with the same
Qwen3 model the transcript channel queries with, which needs the GPU and about
an hour for the 2026 corpus; only then does
``backend.caption_search.search_captions_semantic`` work.

    uv run python -m scripts.indexing.build_caption_index --recreate
    uv run python -m scripts.indexing.build_caption_index --recreate --dense
    uv run python -m scripts.indexing.build_caption_index --dry-run --limit 5
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import frame_maps
from backend.caption_search import CAPTION_INDEX, build_index_definition, get_client

DEFAULT_SOURCE = ROOT / "artifacts" / "captions" / "shot_captions.parquet"
#: Read from the parquet; `caption` is the contract of both schemas, the rest of
#: the window columns are optional so the older per-shot files still load.
COLUMNS = (
    "year", "group", "video_id", "shot_index", "window_index",
    "start_sec", "end_sec", "window_start_sec", "window_end_sec", "caption",
)


def read_rows(source: Path, limit: int | None = None):
    """Caption rows as dicts, tolerating either parquet schema."""
    import pyarrow.parquet as pq

    table = pq.read_table(source)
    available = [name for name in COLUMNS if name in table.column_names]
    table = table.select(available)
    rows = table.to_pylist()
    return rows[:limit] if limit else rows


def _window_bounds(row) -> tuple[float, float]:
    """The caption's own time window: the split window when there is one."""
    start = row.get("window_start_sec")
    end = row.get("window_end_sec")
    if start is None or end is None:
        start, end = row.get("start_sec"), row.get("end_sec")
    return float(start), float(end)


def iter_documents(rows, *, log=print):
    """Yields ``(doc_id, source_dict)``; rows whose video has no frame map are skipped.

    The anchor is the middle of the caption's window -- what the model spent its
    8 frames on -- mapped to that video's nearest real frame, so a hit can be
    thumbnailed and sits inside the span the agent attaches evidence to.
    """
    maps: dict[str, object] = {}
    skipped_empty = skipped_errors = 0
    for row in rows:
        video_id = row["video_id"]
        text = " ".join((row.get("caption") or "").split())
        if not text:
            skipped_empty += 1
            continue
        if video_id not in maps:
            try:
                maps[video_id] = frame_maps.load(video_id)
            except (OSError, FileNotFoundError, KeyError, ValueError) as exc:
                log(f"[caption-index] skipping video {video_id}: {exc}", file=sys.stderr)
                maps[video_id] = None
        mapping = maps[video_id]
        if mapping is None:
            skipped_errors += 1
            continue
        start_sec, end_sec = _window_bounds(row)
        anchor_sec = (start_sec + end_sec) / 2
        try:
            frame_idx = int(mapping.frame_at_media_time(anchor_sec))
        except (IndexError, KeyError, ValueError) as exc:
            log(f"[caption-index] skipping {video_id}:{row.get('shot_index')}: {exc}",
                file=sys.stderr)
            skipped_errors += 1
            continue
        shot_idx = int(row.get("shot_index") or 0)
        window_idx = int(row.get("window_index") or 0)
        yield f"{video_id}:{shot_idx}:{window_idx}", {
            "unit_id": f"{video_id}:{shot_idx}:{window_idx}",
            "video_id": video_id,
            "group": row.get("group") or video_id.split("_")[0],
            "year": str(row.get("year") or "2026"),
            "shot_idx": shot_idx,
            "window_idx": window_idx,
            "frame_idx": frame_idx,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "anchor_sec": round(anchor_sec, 3),
            "text": text,
        }
    log(f"[caption-index] skipped {skipped_empty} empty captions, "
        f"{skipped_errors} unresolvable rows", file=sys.stderr)


def _embedder(batch_size: int):
    """Document embeddings from the transcript channel's model, so queries match."""
    from backend import audio_search

    model = audio_search.load_embedding_model()

    def embed(texts):
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=False)

    return embed


def build(source: Path, index: str, bulk_size: int, recreate: bool, dense: bool,
          limit: int | None, batch_size: int):
    from elasticsearch.helpers import bulk

    client = get_client()
    if client.indices.exists(index=index):
        if not recreate:
            raise SystemExit(f"index {index!r} already exists; pass --recreate")
        client.indices.delete(index=index)
    client.indices.create(index=index, body=build_index_definition(dense=dense))
    embed = _embedder(batch_size) if dense else None

    total = 0
    batch: list[dict] = []

    def flush():
        nonlocal total
        if not batch:
            return
        if embed is not None:
            vectors = embed([action["_source"]["text"] for action in batch])
            for action, vector in zip(batch, vectors):
                action["_source"]["embedding"] = [float(value) for value in vector]
        bulk(client.options(request_timeout=300), batch)
        total += len(batch)
        print(f"[caption-index] indexed {total} captions", flush=True)
        batch.clear()

    for doc_id, source_doc in iter_documents(read_rows(source, limit)):
        batch.append({"_index": index, "_id": doc_id, "_source": source_doc})
        if len(batch) >= bulk_size:
            flush()
    flush()
    client.indices.refresh(index=index)

    count = client.count(index=index)["count"]
    if count != total:
        raise RuntimeError(f"Elasticsearch count {count} != generated {total}")
    print(f"[caption-index] DONE: {count} captions -> {index}")


def dry_run(source: Path, limit: int | None):
    rows = read_rows(source, limit)
    documents = list(iter_documents(rows))
    print(f"[caption-index] {len(rows)} rows -> {len(documents)} documents")
    for _, document in documents[:3]:
        print(f"  {document['unit_id']} frame={document['frame_idx']} "
              f"{document['start_sec']}-{document['end_sec']}s: {document['text'][:200]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", default=CAPTION_INDEX)
    parser.add_argument("--bulk-size", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32, help="--dense encoder batch")
    parser.add_argument("--limit", type=int, help="only the first N parquet rows")
    parser.add_argument("--dense", action="store_true", help="also embed captions (GPU)")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="resolve rows, index nothing")
    args = parser.parse_args()
    if args.dry_run:
        dry_run(args.source, args.limit)
        return
    build(args.source, args.index, args.bulk_size, args.recreate, args.dense,
          args.limit, args.batch_size)


if __name__ == "__main__":
    main()
