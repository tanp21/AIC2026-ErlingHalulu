#!/usr/bin/env python3
"""Build the Elasticsearch transcript index for 2026 L21-L30 videos."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from audio_search import (
    AUDIO_INDEX,
    MODEL_NAME,
    build_index_definition,
    get_client,
    make_passages,
)

ROOT = Path(__file__).resolve().parents[2]
STEP_DATA = ROOT / "data" / "2026" / "audio" / "step_data.parquet"
TRANSCRIPTIONS = ROOT / "data" / "2026" / "audio" / "transcription.parquet"


def iter_passages(rows, window_seconds: int = 60, stride_seconds: int = 15):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["audio_key"]].append(
            {"offset": float(row["offset"]), "token": row["token"]}
        )
    for video_id in sorted(grouped):
        timed_tokens = sorted(grouped[video_id], key=lambda row: row["offset"])
        for passage in make_passages(
            video_id, timed_tokens, window_seconds, stride_seconds
        ):
            yield {
                "passage_id": passage.passage_id,
                "video_id": passage.video_id,
                "group": passage.video_id[:3],
                "year": "2026",
                "start_sec": passage.start_sec,
                "end_sec": passage.end_sec,
                "anchor_sec": passage.anchor_sec,
                "text": passage.text,
            }


def load_rows(path: Path = STEP_DATA):
    columns = pq.read_table(path, columns=["audio_key", "offset", "token"]).to_pydict()
    return (
        {"audio_key": key, "offset": offset, "token": token}
        for key, offset, token in zip(
            columns["audio_key"], columns["offset"], columns["token"]
        )
    )


def load_transcript_ids(path: Path = TRANSCRIPTIONS):
    return set(pq.read_table(path, columns=["audio_key"]).column("audio_key").to_pylist())


def validate_coverage(source_ids, indexed_ids, expected_total=873):
    if len(source_ids) != expected_total:
        raise RuntimeError(
            f"expected {expected_total} source transcripts, got {len(source_ids)}"
        )
    unknown = set(indexed_ids) - set(source_ids)
    if unknown:
        raise RuntimeError(f"indexed videos missing from source transcripts: {sorted(unknown)}")
    return set(source_ids) - set(indexed_ids)


def build(index: str, batch_size: int, bulk_size: int, recreate: bool, device: str):
    import numpy as np
    from elasticsearch.helpers import bulk
    from sentence_transformers import SentenceTransformer

    client = get_client()
    if client.indices.exists(index=index):
        if not recreate:
            raise SystemExit(f"index {index!r} already exists; pass --recreate")
        client.indices.delete(index=index)
    client.indices.create(index=index, body=build_index_definition())

    model = SentenceTransformer(MODEL_NAME, device=device)
    passages = iter_passages(load_rows())
    total = 0
    videos = set()
    batch = []

    def flush(items):
        nonlocal total
        if not items:
            return
        vectors = model.encode(
            [item["text"] for item in items],
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        actions = []
        for item, vector in zip(items, vectors):
            source = dict(item)
            source["embedding"] = vector.tolist()
            actions.append(
                {"_index": index, "_id": item["passage_id"], "_source": source}
            )
            videos.add(item["video_id"])
        bulk(client.options(request_timeout=120), actions)
        total += len(items)
        print(f"[audio-index] indexed {total} passages", flush=True)

    for passage in passages:
        batch.append(passage)
        if len(batch) >= bulk_size:
            flush(batch)
            batch = []
    flush(batch)
    client.indices.refresh(index=index)

    count = client.count(index=index)["count"]
    if count != total:
        raise RuntimeError(f"Elasticsearch count {count} != generated {total}")
    skipped = validate_coverage(load_transcript_ids(), videos)
    print(
        f"[audio-index] DONE: {count} passages, {len(videos)} searchable videos, "
        f"{len(skipped)} without usable speech ({', '.join(sorted(skipped))}) -> {index}"
    )


def validate_existing(index: str):
    client = get_client()
    response = client.search(
        index=index,
        size=0,
        aggs={"videos": {"terms": {"field": "video_id", "size": 1000}}},
    )
    videos = {bucket["key"] for bucket in response["aggregations"]["videos"]["buckets"]}
    skipped = validate_coverage(load_transcript_ids(), videos)
    count = client.count(index=index)["count"]
    print(
        f"[audio-index] VALID: {count} passages, {len(videos)} searchable videos, "
        f"{len(skipped)} without usable speech ({', '.join(sorted(skipped))})"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=AUDIO_INDEX)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bulk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    if args.validate_existing:
        validate_existing(args.index)
    else:
        build(args.index, args.batch_size, args.bulk_size, args.recreate, args.device)


if __name__ == "__main__":
    main()
