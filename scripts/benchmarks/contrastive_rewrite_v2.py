#!/usr/bin/env python3
"""Contrastive rewrite v2 — local LLM, full-caption regeneration.

The previous local rewrite (`local_rewrite_captions.py`) preserved detail well (retention 0.923)
but stayed 60.8% shared with its nearest neighbours, so retrieval did not improve. Per-phrase
survival showed the failure is structured: boilerplate died (0/61, 0/224) while position, colour
and relational phrases survived almost untouched ("on the left" 321/325, "the woman in the"
40/40) — the old prompt's "KEEP every colour, count, position" rule was directly protecting the
very phrases that needed to change.

v2 therefore hands the model the explicit `suspected_shared_phrases` list and makes rewording each
listed phrase the primary objective, while forbidding changes to anything else. A second pass
repairs whatever survived.

Nothing existing is overwritten: the original captions are only ever read, every artefact this
script writes is new and pinned by a run stamp, and pass 1 lines are never modified by pass 2.

Run with the captioning venv (needs the flake env, hence `direnv exec .`; bare
`.venv-captioning/bin/python` fails with `ImportError: libz.so.1`):
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 direnv exec . .venv-captioning/bin/python \\
        -m scripts.benchmarks.contrastive_rewrite_v2 --self-test
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 direnv exec . .venv-captioning/bin/python \\
        -m scripts.benchmarks.contrastive_rewrite_v2 --stage rewrite --limit 8
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
from scripts.benchmarks.clean_captions_local import (SHARED_JSON, content_words, format_prompts,
                                                     generate_batch, load_backend, phrase_pattern)
from scripts.benchmarks.clean_captions_probe import (NEIGHBORS_JSON, PROBE_DIR, pool_windows,
                                                     truncate)

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PROBE_JSON = PROBE_DIR / "cleaned_local_contrastive_v2_probe.json"
AUDIT_JSON = PROBE_DIR / "contrastive_v2_audit.json"
AUDIT_MD = PROBE_DIR / "contrastive_v2_audit.md"

SYSTEM = """You rewrite ONE video-shot caption into a NEW caption for text-to-video retrieval. \
Many other videos look almost identical, so your rewrite must use different WORDING from the \
similar captions of OTHER videos while describing exactly the same shot.

You receive JSON with: "caption", excerpts of the 5 most similar captions from other videos, and \
"suspected_shared_phrases" (lowercase phrases that also occur in those similar captions).

Step 1 — every phrase in suspected_shared_phrases must be REWORDED into different wording with the \
same meaning. This is the main job: the phrases are usually positions, colours, counts, relative \
locations and scene framing ("on the left", "with a black", "in front of the"). Keep the fact \
(which side, which colour, how many, which object) and change the words ("on the left" -> "along \
the left-hand side of the frame"). Do NOT keep a listed phrase verbatim.
Step 2 — DELETE a listed phrase only when rewording it would break the sentence grammar. If the \
phrase can be reworded, reword it; deletion is the last resort.
Step 3 — NEVER mangle the subject. If a listed phrase contains the person, animal, object, \
garment or colour the shot is about (e.g. "the fish is", "the man in the"), keep that concrete \
noun UNCHANGED and reword only the words around it. Every concrete noun in a listed phrase must \
still appear in your caption.
Step 4 — every detail NOT in the list must be kept: all actions in the order they happen, people, \
objects, clothing, colours, counts, positions, quoted on-screen text (quote it), timing and \
camera movement. Do not summarise, do not compress, do not drop sentences. Aim for a caption at \
least 90% as long as the original.
Step 5 — never invent facts, never guess, never add anything not in the original caption.

Return ONLY the new caption as plain text. No JSON, no preamble, no explanation."""

REPAIR_SYSTEM = SYSTEM.replace(
    """Step 1 — every phrase in suspected_shared_phrases must be REWORDED into different wording with the \
same meaning. This is the main job: the phrases are usually positions, colours, counts, relative \
locations and scene framing ("on the left", "with a black", "in front of the"). Keep the fact \
(which side, which colour, how many, which object) and change the words ("on the left" -> "along \
the left-hand side of the frame"). Do NOT keep a listed phrase verbatim.""",
    """Step 1 — your previous rewrite is given as "previous_rewrite", and "phrases_still_shared" lists \
phrases that are still present in it verbatim. Reword EVERY phrase in phrases_still_shared into \
different wording with the same meaning. Change NOTHING else: keep every other word of \
"previous_rewrite" as it is.""")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def surviving(rewritten, phrases):
    """Listed phrases that still appear in `rewritten` under the repo's standard matcher."""
    return [phrase for phrase in phrases if phrase_pattern(phrase).search(rewritten)]


def rewrite_metrics(original, rewritten, shared):
    """Retention / hallucination / length / contrastive audit for a free-form rewrite.

    `detail_metrics` is not reused here: its hallucination term only sees diff targets, and a
    rewrite has none. Synonyms count as non-retained, so retention is a lower bound.
    """
    original_words = content_words(original)
    rewritten_words = content_words(rewritten)
    return {
        "retention": round(len(original_words & rewritten_words) / max(len(original_words), 1), 3),
        "hallucination": round(len(rewritten_words - original_words) / max(len(rewritten_words), 1), 3),
        "length_ratio": round(len(rewritten.split()) / max(len(original.split()), 1), 3),
        "shared_remaining": round(len(surviving(rewritten, shared)) / max(len(shared), 1), 3),
    }


def payload_pass1(index, captions, neighbors, shared, args):
    body = {
        "id": index,
        "caption": captions[index],
        "similar_other_videos": [truncate(captions[neighbor], args.neighbor_chars)
                                 for neighbor in neighbors[str(index)]],
        "suspected_shared_phrases": shared[str(index)],
    }
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def payload_pass2(index, captions, neighbors, shared, args, pass1_text, still_shared):
    body = {
        "id": index,
        "caption": captions[index],
        "similar_other_videos": [truncate(captions[neighbor], args.neighbor_chars)
                                 for neighbor in neighbors[str(index)]],
        "suspected_shared_phrases": shared[str(index)],
        "previous_rewrite": pass1_text,
        "phrases_still_shared": still_shared,
    }
    return [{"role": "system", "content": REPAIR_SYSTEM},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def read_records(path):
    """{id: {tag: record}} from an append-only JSONL; tolerates a torn last line."""
    records = {}
    if not path.exists():
        return records
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(row.get("rewritten", "")).strip():
            records.setdefault(int(row["id"]), {})[row["tag"]] = row["rewritten"].strip()
    return records


def run_pass(backend, tokenizer, path, indices, build, args, tag):
    """Generate for `indices`, appending one JSON line per window (resumable)."""
    done = set(read_records(path))
    pending = [index for index in indices if index not in done]
    log(f"{tag}: {len(indices)} requested, {len(done)} already done, {len(pending)} pending")
    if not pending:
        return
    handle = open(path, "a", buffering=1)
    started = time.monotonic()
    try:
        for start in range(0, len(pending), args.chunk):
            batch = pending[start:start + args.chunk]
            prompts = format_prompts(tokenizer, [build(index) for index in batch])
            texts = generate_batch(backend, prompts, args)
            for index, text in zip(batch, texts):
                handle.write(json.dumps({"id": index, "tag": tag, "rewritten": text.strip()},
                                        ensure_ascii=False) + "\n")
            finished = min(start + args.chunk, len(pending))
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = finished / elapsed
            log(f"{tag}: {finished}/{len(pending)} windows, {rate * 60:.1f}/min, "
                f"eta {(len(pending) - finished) / rate / 60:.0f}m")
    finally:
        handle.close()
    log(f"wrote {path}")


def load_context(args):
    windows, _ = pool_windows()
    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    neighbors = json.loads(NEIGHBORS_JSON.read_text())
    shared = json.loads(SHARED_JSON.read_text())
    indices = sorted(windows)
    if args.indices_file:
        subset = set(json.loads(Path(args.indices_file).read_text()))
        indices = [index for index in indices if index in subset]
    if args.limit:
        indices = indices[:args.limit]
    return captions, neighbors, shared, indices


def load_model(args):
    from transformers import AutoTokenizer

    log(f"loading {args.model} with {args.backend}")
    return load_backend(args.model, args), AutoTokenizer.from_pretrained(args.model)


def stage_rewrite(args):
    captions, neighbors, shared, indices = load_context(args)
    backend, tokenizer = load_model(args)
    path = PROBE_DIR / f"contrastive_v2_pass1_{args.run}.jsonl"
    run_pass(backend, tokenizer, path, indices,
             lambda index: payload_pass1(index, captions, neighbors, shared, args), args, "pass1")


def stage_repair(args):
    captions, neighbors, shared, indices = load_context(args)
    pass1_path = PROBE_DIR / f"contrastive_v2_pass1_{args.run}.jsonl"
    pass1 = read_records(pass1_path)
    missing = [index for index in indices if index not in pass1]
    if missing:
        raise SystemExit(f"{pass1_path.name} is missing {len(missing)} windows (first {missing[0]}); "
                         f"finish --stage rewrite first")
    todo, survivors_of = [], {}
    for index in indices:
        still = surviving(pass1[index], shared[str(index)])
        survivors_of[index] = still
        triggers = len(still) >= args.repair_min
        triggers = triggers or (len(still) >= 2
                                and len(still) / max(len(shared[str(index)]), 1)
                                >= args.repair_ratio)
        if triggers:
            todo.append(index)
    if args.repair_max:
        todo = todo[:args.repair_max]
    log(f"repair: {len(todo)}/{len(indices)} windows still share "
        f">= {args.repair_min} phrases (or >= {args.repair_ratio:.0%} of a short list)")
    backend, tokenizer = load_model(args)
    path = PROBE_DIR / f"contrastive_v2_pass2_{args.run}.jsonl"
    run_pass(backend, tokenizer, path, todo,
             lambda index: payload_pass2(index, captions, neighbors, shared, args,
                                         pass1[index], survivors_of[index]), args, "pass2")


def stage_merge(args):
    captions, neighbors, shared, indices = load_context(args)
    pass1 = read_records(PROBE_DIR / f"contrastive_v2_pass1_{args.run}.jsonl")
    pass2 = read_records(PROBE_DIR / f"contrastive_v2_pass2_{args.run}.jsonl")
    final_path = PROBE_DIR / f"contrastive_v2_final_{args.run}.jsonl"
    output, repaired = {}, 0
    handle = open(final_path, "w", buffering=1)
    try:
        for index in indices:
            if index not in pass1:
                continue
            text, source = pass1[index], "pass1"
            if index in pass2:
                text, source = pass2[index], "pass2"
                repaired += 1
            handle.write(json.dumps({"id": index, "source": source, "rewritten": text},
                                    ensure_ascii=False) + "\n")
            output[str(index)] = {"caption": captions[index], "rewritten": text}
    finally:
        handle.close()
    PROBE_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
    log(f"merged {len(output)} windows ({repaired} repaired by pass 2) -> {PROBE_JSON}")


def stage_audit(args):
    captions, _, shared, indices = load_context(args)
    final = read_records(PROBE_DIR / f"contrastive_v2_final_{args.run}.jsonl")
    pass1 = read_records(PROBE_DIR / f"contrastive_v2_pass1_{args.run}.jsonl")
    rows = {index: rewrite_metrics(captions[index], final[index], shared[str(index)])
            for index in indices if index in final}
    keys = ("retention", "hallucination", "length_ratio", "shared_remaining")

    def mean_of(records, key):
        return round(float(np.mean([row[key] for row in records.values()])), 3) if records else None

    pass1_rows = {index: rewrite_metrics(captions[index], pass1[index], shared[str(index)])
                  for index in indices if index in pass1}
    baseline = {}
    previous = PROBE_DIR / "cleaned_local_rewrite_probe.json"
    if previous.exists():  # the v1 rewrite this run is meant to beat, for comparison only
        v1 = json.loads(previous.read_text())
        baseline["v1_rewrite"] = {
            key: round(float(np.mean([rewrite_metrics(row["original"], row["rewritten"],
                                                      shared[str(index)])[key]
                                      for index, row in v1.items()
                                      if str(index) in shared and row.get("rewritten")])), 3)
            for key in keys}
    report = {
        "run": args.run, "model": args.model, "windows": len(rows),
        "pass1_windows": len(pass1_rows), "repaired_windows": len(pass1_rows) - len(
            [index for index in pass1_rows if pass1_rows[index] == rows.get(index)]),
        "pass1": {key: mean_of(pass1_rows, key) for key in keys},
        "final": {key: mean_of(rows, key) for key in keys},
        "baseline": baseline,
    }
    AUDIT_JSON.write_text(json.dumps(report, indent=1) + "\n")
    lines = [f"# Contrastive rewrite v2 audit — run {args.run}", "",
             f"model `{args.model}`, {len(rows)} windows", "",
             "| stage | retention | hallucination | length ratio | shared remaining |",
             "|---|---:|---:|---:|---:|"]
    for label, block in (("pass 1", report["pass1"]), ("final (after repair)", report["final"]),
                         *(("v1 rewrite (baseline)", value) for _, value in baseline.items())):
        lines.append(f"| {label} | {block['retention']} | {block['hallucination']} | "
                     f"{block['length_ratio']} | {block['shared_remaining']} |")
    worst = sorted(rows, key=lambda index: -rows[index]["shared_remaining"])[:20]
    lines += ["", "## Windows still most shared after repair", "",
              "| id | shared remaining | retention |", "|---|---:|---:|"]
    lines += [f"| {index} | {rows[index]['shared_remaining']} | {rows[index]['retention']} |"
              for index in worst]
    AUDIT_MD.write_text("\n".join(lines) + "\n")
    log(f"final: {report['final']}")
    for name, block in baseline.items():
        log(f"{name}: {block}")
    log(f"wrote {AUDIT_JSON} and {AUDIT_MD}")


def stage_show(args):
    captions, _, shared, indices = load_context(args)
    final = read_records(PROBE_DIR / f"contrastive_v2_final_{args.run}.jsonl")
    for index in indices[:args.show]:
        if index not in final:
            continue
        metrics = rewrite_metrics(captions[index], final[index], shared[str(index)])
        print("=" * 100)
        print(f"window {index} | {metrics}")
        print("--- still shared:", surviving(final[index], shared[str(index)]))
        print("--- ORIGINAL:", captions[index].replace("\n", " "))
        print("--- REWRITTEN:", final[index].replace("\n", " "))


def self_test():
    assert surviving("a blue shirt is worn", ["blue shirt"]) == ["blue shirt"]
    assert surviving("A blue garment is worn", ["blue shirt"]) == []
    metrics = rewrite_metrics("the chef cooks rice in a studio", "the chef cooks rice in a studio",
                              ["in a studio"])
    assert metrics["retention"] == 1.0 and metrics["shared_remaining"] == 1.0
    metrics = rewrite_metrics("the chef cooks rice in a studio", "the cook prepares rice indoors",
                              ["in a studio"])
    assert metrics["shared_remaining"] == 0.0 and 0 < metrics["retention"] < 1.0
    assert "phrases_still_shared" in REPAIR_SYSTEM and "Step 1 — every phrase in suspected" \
        not in REPAIR_SYSTEM
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("rewrite", "repair", "merge", "audit", "show", "all"),
                        default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--run", default=None, help="run stamp pinning the output files")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--indices-file", default=None)
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64, help="transformers backend only")
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--max-model-len", type=int, default=5120)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--neighbor-chars", type=int, default=400)
    parser.add_argument("--repair-min", type=int, default=4,
                        help="pass 2 triggers at this many surviving phrases")
    parser.add_argument("--repair-ratio", type=float, default=0.20)
    parser.add_argument("--repair-max", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    args.run = args.run or time.strftime("%Y%m%d_%H%M")
    log(f"run {args.run} | model {args.model} | backend {args.backend}")
    if args.stage in ("rewrite", "all"):
        stage_rewrite(args)
    if args.stage in ("repair", "all"):
        stage_repair(args)
    if args.stage in ("merge", "all"):
        stage_merge(args)
    if args.stage in ("audit", "all", "show"):
        stage_audit(args)
    if args.stage in ("show", "all"):
        stage_show(args)


if __name__ == "__main__":
    main()
