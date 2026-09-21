#!/usr/bin/env python3
"""Split CapRL shot captions into 2-5 short self-contained "core captions" with DeepSeek.

Reads the chunking-probe pool (unique windows), calls DeepSeek in batches with JSON mode,
validates the reply, and falls back to a sentence split for any window whose cores stay
invalid after a retry. Resumable: windows already present in the output are skipped.

Usage:
    export DEEPSEEK_API_KEY=...
    uv run python -m scripts.benchmarks.generate_core_captions --limit 8   # smoke test
    uv run python -m scripts.benchmarks.generate_core_captions
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))

from scripts.benchmarks.probe_caption_chunking import POOL_JSON, PROBE_DIR, split_sentences

SYSTEM_PROMPT = """You split one long, dense video-shot caption into 2-5 short self-contained "core captions" for text-to-video retrieval.

Rules for every core caption:
- One moment or one coherent visual aspect per core; do not mix unrelated moments.
- Self-contained: name the subject explicitly; avoid pronouns (he/she/it) and words like "then" that need the previous sentence.
- At most 40 words, English, present tense.
- Keep discriminative visual specifics: people, clothing, colors, counts, positions, objects, actions, and any on-screen text (quote it verbatim).
- Drop broadcast boilerplate: channel logos, timecodes, program titles, "the video opens with", camera/editing commentary, and scene-setting that every shot of the same show would share.

Return JSON only: {"items": [{"id": <id>, "cores": ["...", "..."]}]} with exactly one item per input id."""


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def valid_cores(cores):
    return (isinstance(cores, list) and 1 <= len(cores) <= 6
            and all(isinstance(core, str) and core.strip() for core in cores))


def parse_items(payload, requested):
    items = (payload or {}).get("items") or []
    by_id = {}
    for item in items:
        try:
            index = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if index in requested and valid_cores(item.get("cores")):
            by_id[index] = [core.strip() for core in item["cores"]]
    return by_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=POOL_JSON)
    parser.add_argument("--output", type=Path, default=PROBE_DIR / "core_captions_probe.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert valid_cores(["a", "b"]) and not valid_cores([]) and not valid_cores([" "])
        assert parse_items({"items": [{"id": "3", "cores": ["x"]}, {"id": 4, "cores": []}]},
                           {3, 4}) == {3: ["x"]}
        assert split_sentences("One short. Another much longer sentence goes here now.") == \
            ["One short.", "Another much longer sentence goes here now."]
        print("self-test OK")
        return

    from agent.config import read_dotenv
    from agent.llm import DeepSeekClient, LLMError

    dotenv = read_dotenv(ROOT / ".env")
    api_key = (__import__("os").environ.get("DEEPSEEK_API_KEY") or dotenv.get("DEEPSEEK_API_KEY", "")).strip()
    if not api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set (environment or .env)")
    model = args.model or __import__("os").environ.get("DEEPSEEK_MODEL") or dotenv.get("DEEPSEEK_MODEL") or "deepseek-flash"
    client = DeepSeekClient(api_key, model=model, thinking=False)

    pool = json.loads(Path(args.pool).read_text())
    windows = {int(index): row["caption"] for index, row in pool["windows"].items()}
    output = json.loads(args.output.read_text()) if args.output.exists() else {}
    pending = [index for index in sorted(windows) if str(index) not in output]
    if args.limit:
        pending = pending[:args.limit]
    log(f"{len(windows)} pool windows, {len(output)} done, {len(pending)} pending with {model}")

    lock = threading.Lock()
    batches = [pending[start:start + args.batch] for start in range(0, len(pending), args.batch)]

    def work(batch):
        requested = set(batch)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(
                [{"id": index, "caption": windows[index]} for index in batch], ensure_ascii=False)},
        ]
        for attempt in (1, 2):
            try:
                parsed = parse_items(client.chat_json(messages, max_tokens=4096), requested)
            except LLMError as error:
                log(f"batch {batch[0]}..{batch[-1]} attempt {attempt} failed: {error}")
                parsed = {}
            missing = requested - set(parsed)
            if not missing:
                break
            log(f"batch {batch[0]}..{batch[-1]} attempt {attempt}: {len(missing)} invalid/missing")
            messages[1]["content"] = json.dumps(
                [{"id": index, "caption": windows[index]} for index in sorted(missing)],
                ensure_ascii=False)
        with lock:
            for index in batch:
                cores = parsed.get(index)
                output[str(index)] = {"cores": cores, "fallback": False} if cores else \
                    {"cores": split_sentences(windows[index]), "fallback": True}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
        log(f"batch {batch[0]}..{batch[-1]} done ({len(batch)} windows)")

    with ThreadPoolExecutor(max_workers=args.workers) as pool_executor:
        list(pool_executor.map(work, batches))

    fallbacks = sum(1 for row in output.values() if row.get("fallback"))
    core_counts = [len(row["cores"]) for row in output.values() if not row.get("fallback")]
    import statistics
    log(f"done: {len(output)} windows, {fallbacks} fallbacks, "
        f"cores/window median {statistics.median(core_counts) if core_counts else 0}")
    log(f"usage: {client.usage.as_dict()}")


if __name__ == "__main__":
    main()
