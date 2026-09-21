#!/usr/bin/env python3
"""Contrastive caption cleaning with a local vLLM model (diff format).

Instead of asking the model to rewrite the whole caption (slow, loses details), it returns a
diff: exact substrings to drop plus rewrite pairs. The cleaned text is reconstructed from the
original by `apply_diff`, so distinctive details cannot be dropped or hallucinated — only the
listed shared phrases are touched.

Run with the captioning venv (has vLLM):
    .venv-captioning/bin/python -m scripts.benchmarks.clean_captions_local --self-test
    .venv-captioning/bin/python -m scripts.benchmarks.clean_captions_local --stage shared
    .venv-captioning/bin/python -m scripts.benchmarks.clean_captions_local --stage screen --limit 200
    .venv-captioning/bin/python -m scripts.benchmarks.clean_captions_local --stage clean
"""
import argparse
import json
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.llm import parse_json
from scripts.benchmarks.bench_caption_retrieval import CAPTIONS_PARQUET
from scripts.benchmarks.clean_captions_probe import (NEIGHBORS_JSON, PROBE_DIR, pool_windows,
                                                    truncate)

SHARED_JSON = PROBE_DIR / "shared_phrases_probe.json"
LOCAL_JSONL = PROBE_DIR / "cleaned_local.jsonl"
LOCAL_JSON = PROBE_DIR / "cleaned_local_probe.json"
SCREEN_SUBSET = PROBE_DIR / "local_screen_subset.json"
SCREEN_REPORT = PROBE_DIR / "local_screen_report.json"
DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
STOP = frozenset(
    "the a an of in on at to and or for with is are was were be been being this that these those "
    "it its as by from into during over under up down out off then when where which who whom whose "
    "what how not no yes there here also has have had do does did but if while".split())

SYSTEM = """You clean ONE video-shot caption for text-to-video retrieval.

Input: the caption, excerpts of the 5 most similar captions from OTHER videos, and "suspected_shared_phrases": lowercase phrases that also appear in those similar captions.

Decide which suspected phrases are generic/shared and should NOT stay in this caption:
- DROP a phrase only when it is scene-setting or boilerplate: background/setting descriptions, program names, channel logos, timecodes, camera/editing commentary, or repeated commentary that any shot of the same show would share.
- NEVER drop or reword a phrase that contains a person, animal, vehicle, food item, or main object — the subject of the shot is what identifies this video. If such a phrase also appears in the similar captions, KEEP it.
- REWORD (replace) a phrase instead of dropping it only when removing it would leave the sentence ungrammatical; "to" must keep the same meaning and be no longer than "from".
- KEEP every phrase that carries distinctive content: clothing, colors, counts, positions, actions, camera movement, on-screen text, timings.
- Use ONLY phrases from suspected_shared_phrases, copied exactly as given. Never invent phrases or facts.

Return compact JSON only (no whitespace, no explanations): {"items": [{"id": <id>, "drop": ["phrase from the list", ...], "replace": [{"from": "phrase from the list", "to": "<short phrase>"}]}]}
- At most 12 drop phrases and 5 replacements per item. Empty lists are fine. One item per input id."""


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def grams(tokens, sizes=(3, 4, 5)):
    for size in sizes:
        for start in range(len(tokens) - size + 1):
            yield " ".join(tokens[start:start + size])


def phrase_pattern(phrase):
    """Case-insensitive matcher tolerating punctuation/whitespace between tokens."""
    tokens = [re.escape(token) for token in phrase.split()]
    separator = r"[\s,;:/\-–—]+"
    return re.compile(r"(?<!\w)" + separator.join(tokens) + r"(?!\w)", re.IGNORECASE)


def apply_diff(caption, drop, replace):
    """Apply phrases picked from the shared list; returns (cleaned, invalid_phrases).

    Phrases are validated against the ORIGINAL caption, so an overlapping shorter phrase
    that was already removed by a longer one is valid (just a no-op on the current text)."""
    invalid = []
    text = caption
    for phrase in drop or []:
        if not isinstance(phrase, str) or not phrase.strip():
            invalid.append(phrase)
            continue
        pattern = phrase_pattern(phrase)
        if not pattern.search(caption):
            invalid.append(phrase)
            continue
        text = pattern.sub(" ", text, count=1)
    for pair in replace or []:
        source = pair.get("from") if isinstance(pair, dict) else None
        target = pair.get("to") if isinstance(pair, dict) else None
        if not isinstance(source, str) or not isinstance(target, str):
            invalid.append(source)
            continue
        pattern = phrase_pattern(source)
        if not pattern.search(caption):
            invalid.append(source)
            continue
        text = pattern.sub(lambda match: target, text, count=1)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    text = text.strip().strip(",;:").strip()
    return text, invalid


def content_words(text):
    return {word for word in tokenize(text) if word not in STOP and len(word) > 2}


def detail_metrics(original, cleaned, replacements):
    original_words = content_words(original)
    cleaned_words = content_words(cleaned)
    new_words = content_words(" ".join(pair.get("to", "") for pair in replacements
                                       if isinstance(pair, dict)))
    return {
        "retention": round(len(original_words & cleaned_words) / max(len(original_words), 1), 3),
        "hallucination": round(len(new_words - original_words) / max(len(new_words), 1), 3),
    }


def shared_phrases_stage(args):
    windows, _ = pool_windows()
    neighbors = json.loads(NEIGHBORS_JSON.read_text())
    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    needed = set(windows) | {n for values in neighbors.values() for n in values}
    tokens = {index: tokenize(captions[index]) for index in needed}
    gram_sets = {index: set(grams(tokens[index])) for index in needed}
    output = {}
    for index in sorted(windows):
        own = gram_sets[index]
        own_tokens = tokens[index]
        counts = Counter()
        for neighbor in neighbors[str(index)]:
            counts.update(gram_sets[neighbor] & own)
        candidates = []
        seen = set()
        for size in (5, 4, 3):
            for start in range(len(own_tokens) - size + 1):
                gram = " ".join(own_tokens[start:start + size])
                if gram in seen:
                    continue
                seen.add(gram)
                if counts.get(gram, 0) >= 2:
                    candidates.append((gram, start, start + size))
        candidates.sort(key=lambda item: -(item[2] - item[1]))
        kept, used = [], []
        for gram, start, end in candidates:
            if any(not (end <= low or start >= high) for low, high in used):
                continue
            kept.append(gram)
            used.append((start, end))
        output[str(index)] = kept[:args.max_phrases]
    SHARED_JSON.write_text(json.dumps(output, indent=1) + "\n")
    sizes = [len(value) for value in output.values()]
    log(f"wrote {SHARED_JSON}: median {int(np.median(sizes))} shared phrases/window, "
        f"max {max(sizes)}")


def build_messages(index, captions, neighbors, shared, neighbor_chars=300):
    payload = {
        "id": index,
        "caption": captions[index],
        "similar_other_videos": [truncate(captions[neighbor], neighbor_chars)
                                 for neighbor in neighbors[str(index)]],
        "suspected_shared_phrases": shared[str(index)],
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def format_prompts(tokenizer, messages_list):
    prompts = []
    for messages in messages_list:
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
        prompts.append(text)
    return prompts


def load_backend(model, args):
    if args.backend == "deepseek":
        import os

        from agent.config import read_dotenv
        from agent.llm import DeepSeekClient

        dotenv = read_dotenv(ROOT / ".env")
        key = (os.environ.get("DEEPSEEK_API_KEY") or dotenv.get("DEEPSEEK_API_KEY", "")).strip()
        if not key:
            raise SystemExit("DEEPSEEK_API_KEY is not set")
        name = model if model != DEFAULT_MODEL else "deepseek-flash"
        log(f"using DeepSeek {name}")
        return DeepSeekClient(key, model=name, thinking=False)
    if args.backend == "vllm":
        from vllm import LLM

        log(f"loading {model} with vLLM")
        return LLM(model=model, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory,
                   max_model_len=args.max_model_len, disable_log_stats=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log(f"loading {model} with transformers")
    tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left")
    model_obj = AutoModelForCausalLM.from_pretrained(
        model, dtype=torch.float16, device_map=args.device).eval()
    return (tokenizer, model_obj)


def generate_batch(backend, prompts, args):
    if args.backend == "vllm":
        from vllm import SamplingParams

        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
        started = time.monotonic()
        outputs = backend.generate(prompts, sampling)
        elapsed = max(time.monotonic() - started, 1e-6)
        tokens_out = sum(len(output.outputs[0].token_ids) for output in outputs)
        log(f"generated {len(outputs)} items in {elapsed:.0f}s = {tokens_out / elapsed:.0f} tok/s")
        return [output.outputs[0].text for output in outputs]
    import torch

    tokenizer, model_obj = backend
    texts = []
    started = time.monotonic()
    tokens_out = 0
    with torch.inference_mode():
        for start in range(0, len(prompts), args.batch):
            batch = prompts[start:start + args.batch]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=args.max_model_len).to(args.device)
            generated = model_obj.generate(**encoded, max_new_tokens=args.max_tokens,
                                           do_sample=False)
            new_tokens = generated[:, encoded["input_ids"].shape[1]:]
            tokens_out += int(new_tokens.numel())
            texts.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    elapsed = max(time.monotonic() - started, 1e-6)
    log(f"generated {len(texts)} items in {elapsed:.0f}s = {tokens_out / elapsed:.0f} tok/s")
    return texts


def generate_diffs(model, indices, args, out_path):
    from transformers import AutoTokenizer

    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    neighbors = json.loads(NEIGHBORS_JSON.read_text())
    shared = json.loads(SHARED_JSON.read_text())
    tokenizer = None if args.backend == "deepseek" else AutoTokenizer.from_pretrained(model)
    done = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                row = json.loads(line)
                done[int(row["id"])] = row
            except (json.JSONDecodeError, KeyError):
                continue
    pending = [index for index in indices if index not in done]
    log(f"{len(indices)} requested, {len(done)} done, {len(pending)} pending with {model}")
    handle = open(out_path, "a", buffering=1)
    backend = load_backend(model, args) if pending else None
    if args.backend == "deepseek":
        lock = threading.Lock()
        state = {"count": 0}

        def work(index):
            messages = build_messages(index, captions, neighbors, shared, args.neighbor_chars)
            item = {}
            try:
                payload = backend.chat_json(messages, max_tokens=args.max_tokens)
                for candidate in (payload.get("items") if isinstance(payload, dict) else []) or []:
                    try:
                        if int(candidate.get("id")) == index:
                            item = candidate
                            break
                    except (TypeError, ValueError):
                        continue
            except Exception as error:
                log(f"item {index}: {type(error).__name__}: {error}")
            if isinstance(item.get("drop"), list) and item["drop"] and isinstance(item["drop"][0], str):
                drops = item["drop"]
                replaces = item.get("replace") if isinstance(item.get("replace"), list) else []
                errors = []
            else:
                drops, replaces, errors = resolve_item(item, shared[str(index)])
            cleaned, invalid = apply_diff(captions[index], drops, replaces)
            record = {"id": index, "caption": captions[index], "drop": drops,
                      "replace": replaces, "cleaned": cleaned,
                      "invalid_spans": invalid + errors,
                      "raw": "" if item else json.dumps(item)[:200],
                      "parsed": bool(item)}
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[index] = record
                state["count"] += 1
                if state["count"] % 250 == 0:
                    log(f"{state['count']}/{len(pending)} items done")

        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(work, pending))
        finally:
            handle.close()
        return done
    try:
        for start in range(0, len(pending), args.batch):
            batch = pending[start:start + args.batch]
            messages = [build_messages(index, captions, neighbors, shared, args.neighbor_chars) for index in batch]
            prompts = format_prompts(tokenizer, messages)
            texts = generate_batch(backend, prompts, args)
            parsed = parse_diffs(texts, set(batch))
            failed = [index for index in batch if index not in parsed]
            if failed:
                log(f"retrying {len(failed)} items that did not parse")
                retry_messages = []
                for index in failed:
                    messages = build_messages(index, captions, neighbors, shared, args.neighbor_chars)
                    messages[-1]["content"] += "\n\nReturn ONLY the compact JSON object. Do not explain."
                    retry_messages.append(messages)
                retry_texts = generate_batch(backend, format_prompts(tokenizer, retry_messages), args)
                parsed.update(parse_diffs(retry_texts, set(failed)))
            for position, index in enumerate(batch):
                item = parsed.get(index, {})
                if isinstance(item.get("drop"), list) and item["drop"] and isinstance(item["drop"][0], str):
                    drops = item["drop"]
                    replaces = item.get("replace") if isinstance(item.get("replace"), list) else []
                    errors = []
                else:
                    drops, replaces, errors = resolve_item(item, shared[str(index)])
                cleaned, invalid = apply_diff(captions[index], drops, replaces)
                record = {"id": index, "caption": captions[index], "drop": drops,
                          "replace": replaces, "cleaned": cleaned,
                          "invalid_spans": invalid + errors,
                          "raw": texts[position][:400] if index not in parsed else "",
                          "parsed": index in parsed}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[index] = record
            log(f"{min(start + args.batch, len(pending))}/{len(pending)} batches done")
    finally:
        handle.close()
    return done


def screen_subset(args):
    if SCREEN_SUBSET.exists():
        return json.loads(SCREEN_SUBSET.read_text())[:args.limit]
    pool = json.loads((PROBE_DIR / "pool.json").read_text())
    targets = set()
    for entry in pool["cases"].values():
        targets.update(int(index) for index in entry["targets"])
    bad_control = set(pool["bad"]) | set(pool["control"])
    chosen = set()
    for case_id, entry in pool["cases"].items():
        if case_id in bad_control:
            chosen.update(int(index) for index in entry["targets"])
    windows = sorted(int(index) for index in pool["windows"])
    rng = np.random.default_rng(0)
    extra = [index for index in rng.permutation(windows) if index not in chosen]
    chosen.update(int(index) for index in extra[: max(0, args.limit - len(chosen))])
    subset = sorted(chosen)[:args.limit]
    SCREEN_SUBSET.write_text(json.dumps(subset) + "\n")
    return subset


def resolve_item(item, shared_list):
    """Convert index-based diffs into (drop phrases, replace pairs, index errors)."""
    drops, replaces, errors = [], [], []
    for value in item.get("drop") or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            errors.append(value)
            continue
        if 0 <= index < len(shared_list):
            drops.append(shared_list[index])
        else:
            errors.append(value)
    for pair in item.get("replace") or []:
        index = pair.get("index") if isinstance(pair, dict) else None
        target = pair.get("to") if isinstance(pair, dict) else None
        try:
            index = int(index)
        except (TypeError, ValueError):
            errors.append(index)
            continue
        if 0 <= index < len(shared_list) and isinstance(target, str) and target.strip():
            replaces.append({"from": shared_list[index], "to": target.strip()})
        else:
            errors.append(index)
    return drops, replaces, errors


def parse_diffs(texts, requested):
    parsed = {}
    for text in texts:
        try:
            payload = parse_json(text)
        except Exception:
            continue
        items = payload.get("items") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            continue
        for item in items:
            try:
                index = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if index in requested:
                parsed[index] = item
    return parsed


def write_screen_report(model, records):
    parsed = [row for row in records.values() if row["parsed"]]
    total_phrases = sum(len(row["drop"]) + len(row["replace"]) for row in records.values())
    invalid_phrases = sum(len(row["invalid_spans"]) for row in records.values())
    retention, hallucination = [], []
    for row in records.values():
        metrics = detail_metrics(row["caption"], row["cleaned"], row["replace"])
        retention.append(metrics["retention"])
        hallucination.append(metrics["hallucination"])
    report = {
        "model": model, "items": len(records), "parsed": len(parsed),
        "parse_rate": round(len(parsed) / max(len(records), 1), 3),
        "phrases": total_phrases,
        "phrase_valid_rate": round(1 - invalid_phrases / max(total_phrases, 1), 3),
        "invalid_phrases": invalid_phrases,
        "records_with_invalid": sum(1 for row in records.values() if row["invalid_spans"]),
        "retention_mean": round(float(np.mean(retention)), 3),
        "hallucination_mean": round(float(np.mean(hallucination)), 3),
        "empty_diff": sum(1 for row in records.values() if not row["drop"] and not row["replace"]),
    }
    SCREEN_REPORT.write_text(json.dumps(report, indent=1) + "\n")
    log(f"screen report: {report}")
    log(f"wrote {SCREEN_REPORT}")


def screen_stage(args):
    subset = screen_subset(args)
    model = args.model
    out_path = PROBE_DIR / f"cleaned_local_screen_{model.split('/')[-1]}.jsonl"
    records = generate_diffs(model, subset, args, out_path)
    write_screen_report(model, records)


def rule_stage(args):
    """No-LLM baseline: drop every shared phrase found in the caption."""
    windows, _ = pool_windows()
    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    shared = json.loads(SHARED_JSON.read_text())
    output = {}
    for index in sorted(windows):
        cleaned, invalid = apply_diff(captions[index], shared[str(index)], [])
        output[str(index)] = {"caption": captions[index], "drop": shared[str(index)],
                              "cleaned": cleaned, "invalid_spans": invalid}
    path = PROBE_DIR / "cleaned_rule_probe.json"
    path.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    log(f"wrote {path}")


def reapply_stage(args):
    """Recompute cleaned text from stored diffs with the current matcher (no model needed)."""
    path = PROBE_DIR / f"cleaned_local_screen_{args.model.split('/')[-1]}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    records = {}
    for row in rows:
        cleaned, invalid = apply_diff(row["caption"], row["drop"], row["replace"])
        row["cleaned"] = cleaned
        row["invalid_spans"] = invalid
        records[row["id"]] = row
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records.values()) + "\n")
    write_screen_report(args.model, records)


def clean_stage(args):
    windows, _ = pool_windows()
    indices = sorted(windows)
    model = args.model
    out_path = PROBE_DIR / f"cleaned_local_{model.split('/')[-1]}.jsonl"
    records = generate_diffs(model, indices, args, out_path)
    output = {str(index): {"caption": row["caption"], "drop": row["drop"],
                           "replace": row["replace"], "cleaned": row["cleaned"]}
              for index, row in records.items()}
    LOCAL_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    log(f"wrote {LOCAL_JSON}: {len(output)} windows")


def self_test():
    text, invalid = apply_diff("A chef in a white coat stirs the pan in a studio kitchen.",
                               ["in a studio kitchen"], [{"from": "stirs the pan", "to": "stirs"}])
    assert invalid == [] and text == "A chef in a white coat stirs."
    text, invalid = apply_diff("abc def", ["zzz"], [{"from": "nope", "to": "x"}])
    assert text == "abc def" and invalid == ["zzz", "nope"]
    assert content_words("The chef cooks rice") == {"chef", "cooks", "rice"}
    grams_list = list(grams(["a", "b", "c", "d"], sizes=(3,)))
    assert grams_list == ["a b c", "b c d"]
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("shared", "screen", "reapply", "clean", "rule", "all"),
                        default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("vllm", "transformers", "deepseek"), default="vllm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--neighbor-chars", type=int, default=300)
    parser.add_argument("--max-phrases", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.stage in ("shared", "all"):
        shared_phrases_stage(args)
    if args.stage in ("screen", "all"):
        screen_stage(args)
    if args.stage == "reapply":
        reapply_stage(args)
    if args.stage == "rule":
        rule_stage(args)
    if args.stage in ("clean", "all"):
        clean_stage(args)


if __name__ == "__main__":
    main()
