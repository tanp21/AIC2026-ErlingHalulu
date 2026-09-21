#!/usr/bin/env python3
"""Full-corpus contrastive-long cleaning (150 words) of the 2026 CapRL captions.

The originals in data/2026/captions/shot_captions.parquet are never modified. Outputs:

    artifacts/caption_vs_pe_core/full/neighbors.npy   5 nearest other-video neighbours per row
    artifacts/caption_vs_pe_core/full/cleaned_long.jsonl   resumable raw DeepSeek output
    data/2026/captions/cleaned/contrastive_long.parquet    final cleaned captions + originals

Stages:
    neighbors  FAISS kNN over the existing Qwen3-4B caption embeddings
    clean      DeepSeek `deepseek-flash` rewrite, batched and resumable
    build      merge the JSONL into the cleaned parquet (originals kept as a column)
"""
import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.bench_caption_retrieval import CAPTION_DIR, CAPTIONS_PARQUET, caption_documents
from scripts.benchmarks.clean_captions_probe import (CONTRAST_LONG_SYSTEM, parse_items, truncate)

FULL_DIR = ROOT / "artifacts/caption_vs_pe_core/full"
NEIGHBORS_NPY = FULL_DIR / "neighbors.npy"
CLEANED_JSONL = FULL_DIR / "cleaned_long.jsonl"
CLEANED_PARQUET = ROOT / "data/2026/captions/cleaned/contrastive_long.parquet"
META_COLUMNS = ("group", "video_id", "shot_index", "window_index", "window_count",
                "window_start_sec", "window_end_sec")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def neighbors_stage(args):
    import faiss

    docs, meta = caption_documents(CAPTION_DIR)
    video_of = np.asarray([row["video_id"] for row in meta])
    log(f"building FAISS index over {docs.shape[0]} x {docs.shape[1]}")
    index = faiss.IndexFlatIP(docs.shape[1])
    index.add(docs)
    log(f"searching {args.search_k} neighbours per row")
    _, found = index.search(docs, args.search_k)
    neighbors = np.empty((len(meta), args.neighbors), dtype=np.int32)
    for row in range(len(meta)):
        mask = video_of[found[row]] != video_of[row]
        picked = found[row][mask][:args.neighbors]
        if len(picked) < args.neighbors:
            log(f"row {row}: only {len(picked)} other-video neighbours")
            picked = np.pad(picked, (0, args.neighbors - len(picked)), constant_values=-1)
        neighbors[row] = picked
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(NEIGHBORS_NPY, neighbors)
    log(f"wrote {NEIGHBORS_NPY}")


def clean_stage(args):
    from agent.config import read_dotenv
    from agent.llm import DeepSeekClient, LLMError

    dotenv = read_dotenv(ROOT / ".env")
    key = (os.environ.get("DEEPSEEK_API_KEY") or dotenv.get("DEEPSEEK_API_KEY", "")).strip()
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    model = args.model or os.environ.get("DEEPSEEK_MODEL") or dotenv.get(
        "DEEPSEEK_MODEL") or "deepseek-flash"
    client = DeepSeekClient(key, model=model, thinking=False)

    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    neighbors = np.load(NEIGHBORS_NPY)
    done = set()
    if CLEANED_JSONL.exists():
        with open(CLEANED_JSONL) as handle:
            for line in handle:
                try:
                    done.add(int(json.loads(line)["index"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    pending = [index for index in range(len(captions)) if index not in done]
    if args.limit:
        pending = pending[:args.limit]
    log(f"{len(captions)} rows, {len(done)} done, {len(pending)} pending, "
        f"{args.workers} workers ({model})")

    lock = threading.Lock()
    handle = open(CLEANED_JSONL, "a", buffering=1)
    batches = [pending[start:start + args.batch_size]
               for start in range(0, len(pending), args.batch_size)]
    state = {"done": 0, "failed": 0}
    system = CONTRAST_LONG_SYSTEM.replace("{words}", str(args.words))

    def build(batch):
        return json.dumps([{
            "id": index,
            "caption": captions[index],
            "similar_other_videos": [truncate(captions[n]) for n in neighbors[index] if n >= 0],
        } for index in batch], ensure_ascii=False)

    def work(batch):
        requested = set(batch)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": build(batch)}]
        parsed = {}
        for attempt in (1, 2):
            try:
                parsed = parse_items(client.chat_json(messages, max_tokens=8192),
                                     requested, ("contrastive_long",))
            except LLMError as error:
                log(f"batch {batch[0]}..{batch[-1]} attempt {attempt}: {error}")
                parsed = {}
            missing = requested - set(parsed)
            if not missing:
                break
            messages[1]["content"] = build(sorted(missing))
        with lock:
            for index in batch:
                row = parsed.get(index)
                record = ({"index": index, "contrastive_long": row["contrastive_long"]}
                          if row else {"index": index, "failed": True})
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            state["done"] += len(batch)
            state["failed"] += len(batch) - len(parsed)
            if (state["done"] // args.batch_size) % 100 == 0:
                log(f"{state['done']}/{len(pending)} batches written, "
                    f"failed {state['failed']}, usage {client.usage.as_dict()}")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(work, batches))
    finally:
        handle.close()
    log(f"done: {state['done']} windows, failed {state['failed']}, "
        f"usage {client.usage.as_dict()}")


def read_cleaned(path):
    cleaned, failed = {}, 0
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("failed"):
                failed += 1
                continue
            cleaned[int(record["index"])] = record["contrastive_long"]
    return cleaned, failed


def build_stage(args):
    captions = pq.read_table(CAPTIONS_PARQUET, columns=list(META_COLUMNS) + ["caption"]).to_pylist()
    cleaned, failed = read_cleaned(CLEANED_JSONL)
    if len(cleaned) < len(captions) * 0.97:
        log(f"warning: only {len(cleaned)}/{len(captions)} cleaned, building anyway")
    columns = {name: pa.array([row[name] for row in captions]) for name in META_COLUMNS}
    columns["caption"] = pa.array([row["caption"] for row in captions])
    columns["contrastive_long"] = pa.array(
        [cleaned.get(index, row["caption"]) for index, row in enumerate(captions)])
    columns["fallback"] = pa.array([index not in cleaned for index in range(len(captions))])
    CLEANED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), CLEANED_PARQUET)
    log(f"wrote {CLEANED_PARQUET}: {len(captions)} rows, {len(cleaned)} cleaned, "
        f"{len(captions) - len(cleaned)} fallback, {failed} failed records")


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "cleaned.jsonl"
        path.write_text('{"index": 1, "contrastive_long": "a"}\n'
                        '{"index": 2, "failed": true}\n'
                        '{"index": 0, "contrastive_long": "b"}\n')
        cleaned, failed = read_cleaned(path)
        assert cleaned == {1: "a", 0: "b"} and failed == 1
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("neighbors", "clean", "build", "all"), default="all")
    parser.add_argument("--neighbors", type=int, default=5)
    parser.add_argument("--search-k", type=int, default=64)
    parser.add_argument("--model", default=None)
    parser.add_argument("--words", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.stage in ("neighbors", "all"):
        neighbors_stage(args)
    if args.stage in ("clean", "all"):
        clean_stage(args)
    if args.stage in ("build", "all"):
        build_stage(args)


if __name__ == "__main__":
    main()
