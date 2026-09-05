#!/usr/bin/env python3
"""Build the Elasticsearch transcript index for 2026 L21-L30 videos."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import audio_search
from backend.audio_search import (
    AUDIO_INDEX,
    MODEL_NAME,
    SemanticChunkConfig,
    build_index_definition,
    get_client,
    make_passages,
)

# Mirrors audio_search's own AIC_DATA_DIR resolution so this still finds the
# corpus when run from a container, where data is mounted at /data rather
# than beside this script.
AUDIO_DIR = audio_search._AUDIO_DIR
STEP_DATA = AUDIO_DIR / "step_data.parquet"
TRANSCRIPTIONS = AUDIO_DIR / "transcription.parquet"


def iter_passages(rows, *, strategy: str = "sliding", window_seconds: int = 60,
                  stride_seconds: int = 15, embed_cues=None, similarity_threshold: float = 0.5,
                  chunk_config: SemanticChunkConfig | None = None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["audio_key"]].append(
            {"offset": float(row["offset"]), "token": row["token"]}
        )
    for video_id in sorted(grouped):
        timed_tokens = sorted(grouped[video_id], key=lambda row: row["offset"])
        if strategy == "semantic_duration":
            passages = audio_search.make_bounded_semantic_passages(
                video_id, timed_tokens, embed_cues, config=chunk_config
            )
        elif strategy == "semantic":
            passages = audio_search.make_semantic_passages(
                video_id, timed_tokens, embed_cues, similarity_threshold=similarity_threshold
            )
        else:
            passages = make_passages(video_id, timed_tokens, window_seconds, stride_seconds)
        for passage in passages:
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


def build(index: str, batch_size: int, bulk_size: int, recreate: bool, device: str, *,
         strategy: str = "sliding", similarity_threshold: float = 0.5,
         window_seconds: int = 60, stride_seconds: int = 15,
         chunk_config: SemanticChunkConfig | None = None):
    import numpy as np
    from elasticsearch.helpers import bulk
    from sentence_transformers import SentenceTransformer

    client = get_client()
    if client.indices.exists(index=index):
        if not recreate:
            raise SystemExit(f"index {index!r} already exists; pass --recreate")
        client.indices.delete(index=index)
    client.indices.create(index=index, body=build_index_definition())

    # Qwen3-Embedding-4B is intentionally kept in FP16: this is the same
    # serving dtype and avoids silently building a prohibitively large FP32 job.
    import torch
    model = SentenceTransformer(
        MODEL_NAME, device=device, model_kwargs={"torch_dtype": torch.float16},
    )

    def embed_cues(texts):
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False)

    # audio_search owns the choice of transcript: the raw ASR parquet unless
    # AIC_TRANSCRIPT_PATH names another file.
    source = audio_search.transcript_path()
    print(f"[audio-index] indexing from {source.name} (strategy={strategy})")
    passages = iter_passages(load_rows(source), strategy=strategy,
                             window_seconds=window_seconds, stride_seconds=stride_seconds,
                             embed_cues=embed_cues, similarity_threshold=similarity_threshold,
                             chunk_config=chunk_config)
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


def export_flat(index: str, output: Path):
    import faiss
    import numpy as np
    from elasticsearch.helpers import scan

    flat = faiss.IndexFlatIP(audio_search.EMBEDDING_DIM)
    metadata = []
    vectors = []

    def flush():
        if vectors:
            batch = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(batch)
            flat.add(batch)
            vectors.clear()

    fields = ["passage_id", "video_id", "start_sec", "end_sec", "anchor_sec", "text", "embedding"]
    for hit in scan(get_client(), index=index, query={"query": {"match_all": {}}}, _source=fields):
        source = hit["_source"]
        vector = source.pop("embedding")
        if len(vector) != audio_search.EMBEDDING_DIM:
            raise RuntimeError(f"invalid vector dimension for {hit['_id']}")
        source.setdefault("passage_id", hit["_id"])
        metadata.append(source)
        vectors.append(vector)
        if len(vectors) == 256:
            flush()
    flush()

    output.parent.mkdir(parents=True, exist_ok=True)
    index_path = output.with_suffix(".flat.ip")
    metadata_path = output.with_suffix(".json")
    index_tmp = index_path.with_suffix(index_path.suffix + ".tmp")
    metadata_tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    faiss.write_index(flat, str(index_tmp))
    metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")))
    os.replace(index_tmp, index_path)
    os.replace(metadata_tmp, metadata_path)
    print(f"[audio-index] FLAT: {flat.ntotal} passages, dim={flat.d} -> {index_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=AUDIO_INDEX)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bulk-size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--export-flat", type=Path, metavar="PATH",
                        help="export stored Elasticsearch vectors to PATH.flat.ip and PATH.json")
    # Default strategy and duration bounds per benchmarks/audio/semantic_chunk_config.json:
    # a hybrid benchmark (44 KIS/QA queries, real PE-Core+FAISS visual search, real
    # app.fuse_search_results after its adaptive max_time_delta fix) found this
    # duration scale ties or beats sliding on every metric while indexing the
    # fewest passages of any config tried, and a follow-up TRAKE benchmark against
    # temporal_search.py's real per-event DP found no meaningful regression. See
    # docs/audio-search.md for the full comparison.
    parser.add_argument("--strategy", choices=["sliding", "semantic", "semantic_duration"],
                        default="semantic_duration")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                        help="cue-merge threshold for --strategy semantic")
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int, default=15)
    # --strategy semantic_duration only.
    parser.add_argument("--context-window-sentences", type=int, default=3)
    parser.add_argument("--breakpoint-percentile", type=float, default=15.0)
    parser.add_argument("--min-scores-for-percentile", type=int, default=4)
    parser.add_argument("--min-duration-seconds", type=float, default=10.0)
    parser.add_argument("--target-duration-seconds", type=float, default=40.0)
    parser.add_argument("--max-duration-seconds", type=float, default=75.0)
    parser.add_argument("--overlap-sentences", type=int, default=1)
    # Phase 2, off by default -- not exercised by the benchmark that chose
    # the duration bounds above. An earlier ungated design (pause alone made
    # a boundary eligible) let pause dominate ~86%/70% of all boundaries at
    # the 75th/85th percentile and read as a false topic transition on
    # manual review roughly half the time, concentrated in interview- and
    # commentary-style content -- see docs/audio-search.md. Pause can now
    # only promote a boundary that already clears the wider
    # --pause-borderline-semantic-percentile, never create one on its own.
    pause = parser.add_mutually_exclusive_group()
    pause.add_argument("--pause-boost", dest="pause_boost", action="store_true", default=False)
    pause.add_argument("--no-pause-boost", dest="pause_boost", action="store_false")
    parser.add_argument("--pause-relative-percentile", type=float, default=75.0)
    parser.add_argument("--pause-borderline-semantic-percentile", type=float, default=30.0)
    args = parser.parse_args()
    if args.export_flat:
        export_flat(args.index, args.export_flat)
    elif args.validate_existing:
        validate_existing(args.index)
    else:
        chunk_config = SemanticChunkConfig(
            context_window_sentences=args.context_window_sentences,
            breakpoint_percentile=args.breakpoint_percentile,
            min_scores_for_percentile=args.min_scores_for_percentile,
            min_duration_seconds=args.min_duration_seconds,
            target_duration_seconds=args.target_duration_seconds,
            max_duration_seconds=args.max_duration_seconds,
            overlap_sentences=args.overlap_sentences,
            pause_boost_enabled=args.pause_boost,
            pause_relative_percentile=args.pause_relative_percentile,
            pause_borderline_semantic_percentile=args.pause_borderline_semantic_percentile,
        )
        build(args.index, args.batch_size, args.bulk_size, args.recreate, args.device,
              strategy=args.strategy, similarity_threshold=args.similarity_threshold,
              window_seconds=args.window_seconds, stride_seconds=args.stride_seconds,
              chunk_config=chunk_config)


if __name__ == "__main__":
    main()
