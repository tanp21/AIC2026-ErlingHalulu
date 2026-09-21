#!/usr/bin/env python3
"""Regenerate captions with a local instruct model (no n-grams, no diffs).

Each window caption is rewritten by the model into a new caption that keeps the specific
visual details and drops generic broadcast boilerplate. Outputs are appended to a fresh
timestamped JSONL so a previous run is never overwritten.

Run with the captioning venv:
    .venv-captioning/bin/python -m scripts.benchmarks.local_rewrite_captions --limit 24
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.bench_caption_retrieval import CAPTIONS_PARQUET
from scripts.benchmarks.clean_captions_probe import PROBE_DIR

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SYSTEM = """You rewrite a video-shot caption into a NEW caption that is easy to tell apart from the 5 similar captions of OTHER videos provided.

Rules:
- KEEP almost all of the original caption: every action in order, every person, object, clothing, color, count, position, quoted on-screen text, timing and camera movement.
- Only REWORD or REMOVE the parts that also appear in the similar captions: generic setting, program names, channel logos, timecodes, camera/editing commentary, repeated commentary.
- Do NOT summarise and do NOT shorten the distinctive parts. The new caption should be nearly as long as the original (up to 600 words).
- Never invent facts that are not in the original caption.
Return only the new caption text, no preamble."""


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--indices-file", type=Path,
                        default=PROBE_DIR / "local_screen_subset.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    neighbors_path = PROBE_DIR / "caption_neighbors.json"
    neighbors = json.loads(neighbors_path.read_text()) if neighbors_path.exists() else {}
    if args.indices_file.exists():
        indices = json.loads(args.indices_file.read_text())[:args.limit]
    else:
        indices = list(range(args.limit))
    stamp = time.strftime("%Y%m%d_%H%M")
    model_tag = args.model.split("/")[-1]
    out_path = PROBE_DIR / f"contrastive_rewrite_{model_tag}_{stamp}.jsonl"
    log(f"model {args.model} | {len(indices)} windows | writing {out_path}")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                 device_map=args.device).eval()

    def user_content(index):
        payload = {"caption": captions[index]}
        if str(index) in neighbors:
            payload["similar_other_videos"] = [
                captions[neighbor][:400] for neighbor in neighbors[str(index)]]
        return json.dumps(payload, ensure_ascii=False)

    messages = [[{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": user_content(index)}] for index in indices]
    prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
               for m in messages]

    handle = open(out_path, "a", buffering=1)
    done = 0
    started = time.monotonic()
    tokens_out = 0
    try:
        with torch.inference_mode():
            for start in range(0, len(prompts), args.batch):
                batch = prompts[start:start + args.batch]
                encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                    max_length=4096).to(args.device)
                generated = model.generate(**encoded, max_new_tokens=args.max_new_tokens,
                                           do_sample=False)
                new_tokens = generated[:, encoded["input_ids"].shape[1]:]
                tokens_out += int(new_tokens.numel())
                texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
                for offset, text in enumerate(texts):
                    index = indices[start + offset]
                    handle.write(json.dumps({"id": index, "original": captions[index],
                                             "rewritten": text.strip()},
                                            ensure_ascii=False) + "\n")
                done += len(texts)
                elapsed = max(time.monotonic() - started, 1e-6)
                log(f"{done}/{len(prompts)} done, {tokens_out / elapsed:.0f} tok/s")
    finally:
        handle.close()
    log(f"wrote {out_path}")

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    for row in rows[:3]:
        print("=" * 90)
        print("ORIGINAL:", row["original"][:300].replace("\n", " "))
        print("REWRITTEN:", row["rewritten"][:300].replace("\n", " "))


if __name__ == "__main__":
    main()
