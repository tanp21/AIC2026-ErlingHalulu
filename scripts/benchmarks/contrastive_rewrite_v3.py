#!/usr/bin/env python3
"""Contrastive caption v3 — aggressive deletion of shared content, maximum retained detail.

Continues `contrastive_rewrite_v2.py` (written by the previous session, never run). v2 *rewords*
listed shared phrases while keeping their meaning. Two problems, both load-bearing here:

  1. A paraphrase embeds almost identically under Qwen3, so rewording cannot reduce cross-video
     similarity — the thing that actually costs retrieval.
  2. v2's `shared_remaining` audit is *lexical* (`phrase_pattern`), so it would report success
     while the embedding never moved. The metric is gameable by exactly v2's strategy.

v3 therefore (a) DELETES shared content rather than rewording it, and (b) is judged by an
embedding-level audit, never the lexical one alone.

Stage `paraphrase-probe` is the cheap decisive pre-test that justifies (a): it paraphrases the
mined shared phrases with the local model, substitutes them back into every pool caption, and
measures whether cross-video near-duplication actually falls. That is v2's strategy simulated at
caption level for near-zero generation cost — if near-dup does not drop, delete-first is proven
necessary without spending hours generating rewrites.

Nothing existing is overwritten: original captions are only ever read, and every artefact written
here is new and pinned by a run stamp.

Run with the captioning venv (needs the flake env, hence `direnv exec .`):
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 direnv exec . .venv-captioning/bin/python \\
        -m scripts.benchmarks.contrastive_rewrite_v3 --self-test
    VLLM_WSL2_ENABLE_PIN_MEMORY=1 direnv exec . .venv-captioning/bin/python \\
        -m scripts.benchmarks.contrastive_rewrite_v3 --stage paraphrase-probe --limit-phrases 200
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.benchmarks.bench_caption_retrieval import CAPTIONS_PARQUET, CACHE_DIR, CAPTION_DIR
from scripts.benchmarks.clean_captions_local import (SHARED_JSON, format_prompts, generate_batch,
                                                     load_backend, tokenize)
from scripts.benchmarks.clean_captions_probe import (NEIGHBORS_JSON, PROBE_DIR, metrics_report,
                                                     pairwise_similarity, pool_windows, rank_pool,
                                                     truncate)
from scripts.benchmarks.probe_caption_chunking import QWEN_SMALL, SmallEncoder
from scripts.benchmarks.bench_visual_embeddings import load_cases
from scripts.benchmarks.bench_caption_retrieval import QUERIES

QWEN_4B = "Qwen/Qwen3-Embedding-4B"

DEFAULT_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
RUN = time.strftime("%Y%m%d_%H%M")

# The repo matcher tolerates punctuation *between* phrase tokens, but its separator class omits
# apostrophes and sentence-final punctuation. Measured on 9,929 mined phrases: 88.0% of phrases
# matched back into their own source caption with the shipped class, 94.2% with this one (and it
# does not over-match unrelated captions). The residual misses are n-grams whose tokens were split
# by non-ASCII text, which v3's span-based mining (mine_shared_axes.py) avoids entirely.
SEPARATOR = r"[\s,;:/\-–—'’\.!\?]+"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def phrase_pattern(phrase):
    """Case-insensitive phrase matcher; see SEPARATOR for why this differs from the repo's."""
    tokens = [re.escape(token) for token in phrase.split()]
    return re.compile(r"(?<!\w)" + SEPARATOR.join(tokens) + r"(?!\w)", re.IGNORECASE)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def captions_and_shared():
    captions = pq.read_table(CAPTIONS_PARQUET, columns=["caption"]).column("caption").to_pylist()
    shared = json.loads(SHARED_JSON.read_text())
    return captions, shared


PARAPHRASE_SYSTEM = """You reword short phrases from video captions so that they mean exactly the same thing but use different words.

You receive a JSON list of {"id": <int>, "phrase": "<lowercase phrase>"}.

For each phrase, produce a natural English rewording that:
- keeps the SAME meaning and the SAME kind of information (if it says which side, which colour, how many, or which object, the rewording must still say that);
- uses clearly DIFFERENT wording — do not merely add or drop a word, or swap in a synonym of one word;
- keeps every concrete noun that names a person, animal, object, garment, food, or on-screen text;
- is about the same length or slightly longer.

Return ONLY compact JSON: {"items": [{"id": <int>, "paraphrase": "<rewording>"}]} with one item per input id."""


def parse_paraphrases(text, requested):
    from agent.llm import parse_json

    try:
        payload = parse_json(text)
    except Exception:
        return {}
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {}
    found = {}
    for item in items:
        try:
            index = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        value = item.get("paraphrase")
        if index in requested and isinstance(value, str) and value.strip():
            found[index] = value.strip()
    return found


def generate_paraphrases(phrases, args):
    """Unique phrase -> paraphrase, via the local instruct model. Resumable."""
    path = PROBE_DIR / f"phrase_paraphrases_{args.run}.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    pending = [index for index in range(len(phrases)) if str(index) not in done]
    log(f"paraphrase: {len(phrases)} unique phrases, {len(done)} done, {len(pending)} pending")
    if not pending:
        return done
    backend, tokenizer = None, None
    if args.backend == "vllm":
        from transformers import AutoTokenizer

        backend = load_backend(args.model, args)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        from transformers import AutoTokenizer

        backend = load_backend(args.model, args)
        tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    started = time.monotonic()
    for start in range(0, len(pending), args.phrase_chunk):
        batch = pending[start:start + args.phrase_chunk]
        body = json.dumps([{"id": index, "phrase": phrases[index]} for index in batch],
                          ensure_ascii=False)
        messages = [{"role": "system", "content": PARAPHRASE_SYSTEM},
                    {"role": "user", "content": body}]
        texts = generate_batch(backend, format_prompts(tokenizer, [messages]), args)
        found = parse_paraphrases(texts[0], set(batch))
        for index in batch:
            done[str(index)] = found.get(index, "")
        path.write_text(json.dumps(done, ensure_ascii=False, indent=1) + "\n")
        finished = min(start + args.phrase_chunk, len(pending))
        rate = finished / max(time.monotonic() - started, 1e-6)
        log(f"paraphrase {finished}/{len(pending)} ({rate * 60:.0f}/min)")
    valid = sum(1 for value in done.values() if value)
    log(f"wrote {path.name}: {valid}/{len(done)} valid")
    # vLLM holds the GPU hard; the 0.6B/4B encoders run later in this same process on a 12 GB card.
    del backend, tokenizer
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    return done


def substitute(caption, phrases, paraphrases, mapping):
    """Replace each phrase occurrence with its paraphrase; returns (text, applied, missed)."""
    text, applied, missed = caption, 0, 0
    for phrase in phrases:
        replacement = mapping.get(phrase)
        if not replacement:
            continue
        pattern = phrase_pattern(phrase)
        if not pattern.search(text):
            missed += 1
            continue
        text = pattern.sub(lambda _: replacement, text)
        applied += 1
    return text, applied, missed


def stage_paraphrase_probe(args):
    captions, shared = captions_and_shared()
    windows, _ = pool_windows()
    indices = sorted(windows)

    unique = sorted({phrase for index in shared for phrase in shared[str(index)]})
    if args.limit_phrases:
        # Deterministic sample, biased to the most reused phrases: those carry the most shared
        # mass, so they are the ones whose paraphrasing would matter most.
        from collections import Counter

        counts = Counter(phrase for index in shared for phrase in shared[str(index)])
        unique = [phrase for phrase, _ in counts.most_common(args.limit_phrases)]
    log(f"{len(unique)} unique phrases to paraphrase")

    generated = generate_paraphrases(unique, args)
    mapping = {phrase: generated[str(position)] for position, phrase in enumerate(unique)
               if generated.get(str(position))}
    log(f"{len(mapping)}/{len(unique)} phrases have a paraphrase")

    originals, cleaned = [], []
    total_applied = total_missed = 0
    for index in indices:
        caption = captions[index]
        text, applied, missed = substitute(caption, shared[str(index)], None, mapping)
        originals.append(caption)
        cleaned.append(text if applied else caption)
        total_applied += applied
        total_missed += missed
    log(f"substituted {total_applied} phrases, {total_missed} did not match (kept original text)")

    # Phrase-level: does a paraphrase embed like its original?
    phrase_pairs = [(phrase, mapping[phrase]) for phrase in unique if phrase in mapping]
    encoder = SmallEncoder(QWEN_SMALL, args.dtype, args.device)
    phrase_vectors = encoder.encode([phrase for pair in phrase_pairs for phrase in pair])
    left, right = phrase_vectors[0::2], phrase_vectors[1::2]
    phrase_cosine = (left * right).sum(axis=1)

    from scripts.benchmarks.bench_caption_retrieval import CAPTION_DIR

    cap_meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    video_of = np.asarray([row["video_id"] for row in cap_meta])[np.asarray(indices)]

    vectors = {}
    for name, texts in (("original", originals), ("paraphrase_shared", cleaned)):
        cache = PROBE_DIR / f"v3_vectors_06b_{name}.npy"
        if cache.exists():
            vectors[name] = np.load(cache)
        else:
            log(f"encoding {len(texts)} {name} captions")
            vectors[name] = encoder.encode(list(texts))
            np.save(cache, vectors[name])

    similarity = {name: pairwise_similarity(matrix, video_of)
                  for name, matrix in vectors.items()}
    for name, block in similarity.items():
        log(f"{name}: near-dup {block}")

    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads((PROBE_DIR / "pool.json").read_text())
    position_of = {int(index): position for position, index in enumerate(indices)}
    query_vectors = np.load(PROBE_DIR / "query_vectors_06b.npy")
    ranks = rank_pool(cases, case_ids, pool, position_of, np.asarray(
        [row["video_id"] for row in cap_meta]), query_vectors, vectors)
    metrics = metrics_report(ranks, set(pool["bad"]), set(pool["control"]))

    report = {
        "run": args.run, "model": args.model,
        "phrases_unique": len(unique), "phrases_paraphrased": len(mapping),
        "phrases_substituted": total_applied, "phrases_unmatched": total_missed,
        "phrase_cosine": {
            "mean": round(float(phrase_cosine.mean()), 4),
            "median": round(float(np.median(phrase_cosine)), 4),
            "over_0.90": round(float((phrase_cosine > 0.90).mean()), 4),
            "over_0.95": round(float((phrase_cosine > 0.95).mean()), 4),
            "over_0.99": round(float((phrase_cosine > 0.99).mean()), 4),
        },
        "similarity": similarity,
        "metrics": {name: {key: value for key, value in block.items()
                            if key.startswith(("bad_", "control_"))}
                    for name, block in metrics.items()},
    }
    out = PROBE_DIR / f"paraphrase_probe_{args.run}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {out}")
    log(f"phrase cosine mean {report['phrase_cosine']['mean']}, "
        f">{0.95}: {report['phrase_cosine']['over_0.95']:.1%}")
    for name in vectors:
        log(f"{name}: bad_oracle {metrics[name]['bad_oracle']}")
        log(f"{name}: control_realistic {metrics[name]['control_realistic']}")


def build_arms(captions, indices, args):
    """Length-matched deterministic arms — no LLM, no mining.

    Every contrastive variant so far is much SHORTER than the original, and BAD recall rises
    monotonically as retention falls (original .017 -> diff .025 -> rule .035 -> clean150 .069 ->
    clean60 .103, all at ~100/87/87/31/12% retention). That pattern is equally consistent with a
    plain document-length / query-likeness effect as with contrastive surgery, and nothing in the
    repo has ruled it out. These arms are the control that separates the two.
    """
    arms = {}
    for words in args.control_words:
        arms[f"prefix_{words}w"] = [
            " ".join(captions[index].split()[:words]) for index in indices]
    for sentences in args.control_sentences:
        arms[f"lead_{sentences}s"] = [
            " ".join(re.split(r"(?<=[.!?])\s+", captions[index].strip())[:sentences])
            for index in indices]
    return arms


def stage_controls(args):
    """Embed length-matched truncation controls and score them on the pool.

    If a plain prefix reaches the retention level of a cleaned variant and matches its retrieval,
    then the contrastive rewriting is buying nothing that truncation does not — which is a far
    cheaper thing to ship. Run this before spending GPU-hours on generation.
    """
    captions, _ = captions_and_shared()
    windows, _ = pool_windows()
    indices = sorted(windows)
    from scripts.benchmarks.bench_caption_retrieval import CAPTION_DIR

    cap_meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    video_of = np.asarray([row["video_id"] for row in cap_meta])[np.asarray(indices)]

    arms = build_arms(captions, indices, args)
    model = QWEN_4B if args.encoder == "4b" else QWEN_SMALL
    query_path = (CACHE_DIR / "caption_query_vectors.npy" if args.encoder == "4b"
                  else PROBE_DIR / "query_vectors_06b.npy")
    query_vectors = np.load(query_path).astype(np.float32)
    encoder = SmallEncoder(model, args.dtype, args.device)

    vectors, lengths = {}, {}
    texts = {"original": [captions[index] for index in indices]}
    texts.update(arms)
    for name, items in texts.items():
        lengths[name] = round(float(np.mean([len(text.split()) for text in items])), 1)
        cache = PROBE_DIR / f"v3_vectors_{args.encoder}_{name}.npy"
        if cache.exists():
            vectors[name] = np.load(cache)
        else:
            log(f"encoding {len(items)} {name} (mean {lengths[name]} words)")
            vectors[name] = encoder.encode(list(items))
            np.save(cache, vectors[name])
        log(f"{name}: {pairwise_similarity(vectors[name], video_of)}")

    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads((PROBE_DIR / "pool.json").read_text())
    position_of = {int(index): position for position, index in enumerate(indices)}
    ranks = rank_pool(cases, case_ids, pool, position_of, np.asarray(
        [row["video_id"] for row in cap_meta]), query_vectors, vectors)
    metrics = metrics_report(ranks, set(pool["bad"]), set(pool["control"]))

    report = {"run": args.run, "encoder": args.encoder, "model": model,
              "mean_words": lengths, "metrics": metrics}
    out = PROBE_DIR / f"v3_controls_{args.encoder}_{args.run}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {out}")
    log(f"{'arm':22s} {'words':>6s} {'BAD mr':>7s} {'BAD@1':>6s} {'BAD@50':>7s} "
        f"{'CTRL mr':>8s} {'CTRL@1':>7s}")
    for name in vectors:
        bad, ctrl = metrics[name]["bad_oracle"], metrics[name]["control_realistic"]
        log(f"{name:22s} {lengths[name]:6.0f} {bad['mrr']:7.3f} {bad['top1']:6.3f} "
            f"{bad['top50']:7.3f} {ctrl['mrr']:8.3f} {ctrl['top1']:7.3f}")


def mine_uncapped(neighbors, captions, indices, min_neighbors, sizes=(3, 4, 5)):
    """Shared phrases per window with NO cap, longest-first, non-overlapping.

    The shipped miner keeps only 25 phrases (median = max = 25: 95.7% of windows are truncated
    by the cap), so it covers ~16% of the caption. This is the same signal, uncapped.
    """
    from collections import Counter

    needed = set(indices) | {n for values in neighbors.values() for n in values}
    tokens = {index: tokenize(captions[index]) for index in needed}
    grams = {}
    for index in needed:
        seen = set()
        for size in sizes:
            for start in range(len(tokens[index]) - size + 1):
                seen.add(" ".join(tokens[index][start:start + size]))
        grams[index] = seen
    out = {}
    for index in indices:
        own_tokens = tokens[index]
        counts = Counter()
        for neighbor in neighbors[str(index)]:
            counts.update(grams[neighbor] & grams[index])
        candidates, used = [], []
        for size in sizes:
            for start in range(len(own_tokens) - size + 1):
                gram = " ".join(own_tokens[start:start + size])
                if counts.get(gram, 0) >= min_neighbors:
                    candidates.append((gram, start, start + size))
        candidates.sort(key=lambda item: -(item[2] - item[1]))
        kept = []
        for gram, low, high in candidates:
            if any(not (high <= a or low >= b) for a, b in used):
                continue
            kept.append(gram)
            used.append((low, high))
        out[str(index)] = kept
    return out


def stage_prune(args):
    """Delete every mined shared phrase, no LLM. The user's hypothesis in its purest form.

    This is the `rule` arm without its 25-phrase cap: if "aggressively delete duplicated content"
    is the lever, coverage should matter and this should beat both `rule` and the truncation
    controls. If it does not, then what is removed is not what is driving retrieval.
    """
    from scripts.benchmarks.clean_captions_local import apply_diff

    captions, _ = captions_and_shared()
    windows, _ = pool_windows()
    indices = sorted(windows)
    neighbors = json.loads(NEIGHBORS_JSON.read_text())
    from scripts.benchmarks.bench_caption_retrieval import CAPTION_DIR

    cap_meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    video_of = np.asarray([row["video_id"] for row in cap_meta])[np.asarray(indices)]
    model = QWEN_4B if args.encoder == "4b" else QWEN_SMALL
    query_path = (CACHE_DIR / "caption_query_vectors.npy" if args.encoder == "4b"
                  else PROBE_DIR / "query_vectors_06b.npy")

    texts, stats = {}, {}
    for label, min_neighbors in (("prune_any1", 1), ("prune_any2", 2)):
        phrases = mine_uncapped(neighbors, captions, indices, min_neighbors)
        cleaned, covered, total, kept_words = [], 0, 0, 0
        for index in indices:
            listed = phrases[str(index)]
            text, invalid = apply_diff(captions[index], listed, [])
            cleaned.append(text or captions[index])
            covered += len(listed) - len(invalid)
            total += len(listed)
            kept_words += len(cleaned[-1].split())
        texts[label] = cleaned
        stats[label] = {"phrases_mean": round(total / len(indices), 1),
                        "applied_mean": round(covered / len(indices), 1),
                        "surviving_words": round(kept_words / len(indices), 1)}
        log(f"{label}: {stats[label]}")

    query_vectors = np.load(query_path).astype(np.float32)
    encoder = SmallEncoder(model, args.dtype, args.device)
    vectors = {"original": encoder.encode([captions[index] for index in indices])}
    for label, items in texts.items():
        vectors[label] = encoder.encode([item if item.strip() else captions[index]
                                         for index, item in zip(indices, items)])
        log(f"{label}: {pairwise_similarity(vectors[label], video_of)}")

    cases = load_cases(QUERIES)
    case_ids = [case["id"] for case in cases]
    pool = json.loads((PROBE_DIR / "pool.json").read_text())
    position_of = {int(index): position for position, index in enumerate(indices)}
    ranks = rank_pool(cases, case_ids, pool, position_of, np.asarray(
        [row["video_id"] for row in cap_meta]), query_vectors, vectors)
    metrics = metrics_report(ranks, set(pool["bad"]), set(pool["control"]))
    out = PROBE_DIR / f"v3_prune_{args.encoder}_{args.run}.json"
    out.write_text(json.dumps({"run": args.run, "encoder": args.encoder, "model": model,
                               "stats": stats, "metrics": metrics}, indent=1) + "\n")
    log(f"wrote {out}")
    for name in vectors:
        b, c = metrics[name]["bad_oracle"], metrics[name]["control_realistic"]
        log(f"{name:14s} BAD mr {b['mrr']:.3f} @1 {b['top1']:.3f} @50 {b['top50']:.3f} | "
            f"CTRL mr {c['mrr']:.3f} @1 {c['top1']:.3f}")


REWRITE_SYSTEM = """You rewrite ONE video-shot caption into a much SHORTER caption for text-to-video retrieval.

Many other videos look almost identical, so a caption that spells out the shared setting is useless: it matches every similar video equally well. Only what is specific to THIS exact moment can identify it.

You receive JSON with "caption", "similar_other_videos" (excerpts of the most similar captions from OTHER videos), "similar_same_video_shots" (excerpts from other shots of the SAME video) and "shared_phrases" (phrases that also occur in those similar captions).

Write a new caption of AT MOST {words} words. Rules:

1. DELETE the content behind every shared_phrases item. If a shared phrase sits inside a sentence about the setting, the programme, the studio, the kitchen, the lighting, the camera, or anything else that any shot of a similar video would also have, delete that WHOLE sentence. Do not merely reword it: a rewording carries the same meaning and so still matches the same videos — only removal changes what this caption matches.
2. KEEP every detail that is NOT shared: the specific people, clothing, colours, counts, positions, objects, the actions in the order they happen, and any on-screen text (quote it). These are the only things that identify this moment — never drop one just to save words.
3. Never invent anything that is not in the original caption.
4. Write a plain description, not a bullet list.

Spend the budget on rule 2. If you must cut something, cut shared content first, never specific detail. Return ONLY the new caption. No JSON, no preamble."""

REWRITE_COMBINED = """You rewrite ONE video-shot caption into several SHORTER versions of different lengths for text-to-video retrieval.

Many other videos look almost identical, so a caption that spells out the shared setting is useless: it matches every similar video equally well. Only what is specific to THIS exact moment can identify it.

You receive JSON with "caption", "similar_other_videos" (excerpts of the most similar captions from OTHER videos), "similar_same_video_shots" (excerpts from other shots of the SAME video) and "shared_phrases" (phrases that also occur in those similar captions).

Produce {count} versions, at most {budgets} words respectively. In every version:

1. DELETE the content behind every shared_phrases item. If a shared phrase sits inside a sentence about the setting, the programme, the studio, the kitchen, the lighting, the camera, or anything else that any shot of a similar video would also have, delete that WHOLE sentence. Do not merely reword it: a rewording carries the same meaning and so still matches the same videos — only removal changes what this caption matches.
2. KEEP every detail that is NOT shared: the specific people, clothing, colours, counts, positions, objects, the actions in the order they happen, and any on-screen text (quote it). These are the only things that identify this moment — never drop one just to save words.
3. Never invent anything that is not in the original caption.

The versions must be CLEARLY different lengths, and each one must land CLOSE TO its own budget — being much shorter than its budget is a failure, not a safe choice. Count as you write. If a version cannot reach its budget using only non-shared detail, keep it short and say nothing extra; never fill the space with shared content.
Return ONLY compact JSON: {"versions": [{"words": <budget>, "caption": "<text>"}]} shortest first."""


def run_generation(backend, tokenizer, path, indices, build, parse, args, tag):
    """Generate one JSON line per window; resumable by skipping ids already present."""
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("text", "")).strip():
                done.add(int(row["id"]))
    pending = [index for index in indices if index not in done]
    log(f"{tag}: {len(indices)} requested, {len(done)} done, {len(pending)} pending")
    if not pending:
        return
    handle = open(path, "a", buffering=1)
    started, generated = time.monotonic(), 0
    try:
        for start in range(0, len(pending), args.chunk):
            batch = pending[start:start + args.chunk]
            prompts = format_prompts(tokenizer, [build(index) for index in batch])
            texts = generate_batch(backend, prompts, args)
            for index, text in zip(batch, texts):
                parsed = parse(text)
                if parsed is None:
                    continue
                handle.write(json.dumps({"id": index, "text": parsed}, ensure_ascii=False) + "\n")
                generated += 1
            finished = min(start + args.chunk, len(pending))
            rate = finished / max(time.monotonic() - started, 1e-6)
            log(f"{tag}: {finished}/{len(pending)}, {rate * 60:.1f}/min, "
                f"eta {(len(pending) - finished) / rate / 60:.0f}m")
    finally:
        handle.close()
    log(f"wrote {path} ({generated} new)")


def rewrite_context(args):
    """Captions, per-window shared phrases (both axes when available) and neighbours.

    `shared_phrases_axes.json` is AXIS-major: {"other_videos": {window: [...]}, ...}. Reading it
    window-major silently yields empty lists for every window, which is invisible in the generated
    text but removes the entire signal the prompt is built on — so the shape is asserted here.
    """
    captions, flat = captions_and_shared()
    axes_path = PROBE_DIR / "shared_phrases_axes.json"
    if axes_path.exists():
        axes = json.loads(axes_path.read_text())
        other = axes.get("other_videos") or {}
        same = axes.get("same_video_other_shots") or {}
        if other and not isinstance(next(iter(other.values())), list):
            raise SystemExit(f"{axes_path.name} is not axis-major "
                             f"(expected {{axis: {{window: [phrases]}}}})")
    else:
        other = flat
        same = {}
    neighbors_path = PROBE_DIR / "caption_neighbors_axes.json"
    if neighbors_path.exists():
        neighbors = json.loads(neighbors_path.read_text())
    else:
        neighbors = {key: {"other_videos": value, "same_video_other_shots": []}
                     for key, value in json.loads(NEIGHBORS_JSON.read_text()).items()}
    return captions, other, same, neighbors


def build_messages(index, captions, other, same, neighbors, args, system, budget=None, count=None):
    entry = neighbors.get(str(index), {})
    # `--axes cross` withholds the intra-video list, so the two axes can be measured separately:
    # 62.7% of same-video phrases are absent from the cross-video list, and that residue is the
    # video's own recurring content (sets, costumes, presenter) — a video-signature cue whose
    # deletion may cost video top-1 rather than help it.
    same_list = [] if args.axes == "cross" else same.get(str(index), [])
    body = {
        "id": index,
        "caption": captions[index],
        "similar_other_videos": [truncate(captions[n], args.neighbor_chars)
                                 for n in entry.get("other_videos", [])],
        "similar_same_video_shots": [truncate(captions[n], args.neighbor_chars)
                                     for n in entry.get("same_video_other_shots", [])]
        if args.axes == "both" else [],
        "shared_phrases": other.get(str(index), []) + same_list,
    }
    text = system.replace("{words}", str(budget)) if budget else system
    if count:
        text = text.replace("{count}", str(count)).replace("{budgets}", " / ".join(
            str(value) for value in args.budgets))
    return [{"role": "system", "content": text},
            {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def stage_show_prompt(args):
    """Print the exact prompt for one window, so the input can be eyeballed before a long run."""
    captions, other, same, neighbors = rewrite_context(args)
    windows, _ = pool_windows()
    indices = sorted(int(index) for index in windows)
    index = indices[args.show]
    system = REWRITE_COMBINED if args.combined else REWRITE_SYSTEM
    messages = build_messages(index, captions, other, same, neighbors, args, system,
                              budget=None if args.combined else args.budgets[0],
                              count=len(args.budgets) if args.combined else None)
    body = json.loads(messages[1]["content"])
    print("=" * 100)
    print(f"window {index}")
    print(f"  caption words        : {len(body['caption'].split())}")
    print(f"  similar_other_videos : {len(body['similar_other_videos'])}")
    print(f"  similar_same_video   : {len(body['similar_same_video_shots'])}")
    print(f"  shared_phrases       : {len(body['shared_phrases'])}")
    print("=" * 100)
    print("SYSTEM:"); print(messages[0]["content"][:1200])
    print("-" * 100)
    print("USER (truncated):")
    print(json.dumps({**body, "caption": body["caption"][:700] + " …"},
                     ensure_ascii=False, indent=1)[:2500])


def stage_rewrite(args):
    captions, other, same, neighbors = rewrite_context(args)
    windows, _ = pool_windows()
    indices = sorted(windows)
    if args.limit:
        indices = indices[:args.limit]
    # Fail fast if the shared-phrase lists did not arrive: an empty list is invisible in the
    # generated text, so without this check a whole run can silently train on no signal at all.
    covered = sum(1 for index in indices
                  if other.get(str(index)) or same.get(str(index)))
    log(f"shared phrases present for {covered}/{len(indices)} windows "
        f"({covered / max(len(indices), 1):.1%})")
    if covered < 0.5 * len(indices):
        raise SystemExit("refusing to generate: shared_phrases are empty for most windows")
    backend, tokenizer = load_model(args)
    if args.combined:
        path = PROBE_DIR / f"v3_rewrite_combined_{args.run}.jsonl"
        build = lambda index: build_messages(index, captions, other, same, neighbors, args,
                                             REWRITE_COMBINED, count=len(args.budgets))
        run_generation(backend, tokenizer, path, indices, build,
                       lambda text: parse_versions(text, args.budgets), args, "combined")
    else:
        for budget in args.budgets:
            path = PROBE_DIR / f"v3_rewrite_{budget}_{args.run}.jsonl"
            build = lambda index, budget=budget: build_messages(
                index, captions, other, same, neighbors, args, REWRITE_SYSTEM, budget=budget)
            run_generation(backend, tokenizer, path, indices, build, parse_plain, args,
                           f"budget{budget}")


def parse_plain(text):
    value = (text or "").strip()
    return value or None


def parse_versions(text, budgets):
    """Map the returned versions onto the requested budgets.

    The model echoes a `words` field that is sometimes off by a few (118 for 120), so keying on it
    exactly would silently drop rows. Versions are requested shortest-first, so they are matched
    positionally after sorting by whatever number was returned.
    """
    from agent.llm import parse_json

    try:
        payload = parse_json(text)
    except Exception:
        return None
    versions = payload.get("versions") if isinstance(payload, dict) else None
    if not isinstance(versions, list) or not versions:
        return None
    entries = []
    for position, entry in enumerate(versions):
        if not isinstance(entry, dict):
            continue
        caption = entry.get("caption")
        if not (isinstance(caption, str) and caption.strip()):
            continue
        try:
            stated = int(entry.get("words"))
        except (TypeError, ValueError):
            stated = None
        entries.append((stated if stated is not None else position, position, caption.strip()))
    if not entries:
        return None
    entries.sort(key=lambda item: (item[0], item[1]))
    ordered = [caption for _, _, caption in entries]
    wanted = sorted(budgets)
    if len(ordered) == len(wanted):
        return dict(zip(wanted, ordered))
    # Fewer or more versions than asked for: take the shortest ones against the smallest budgets.
    return {budget: ordered[rank] for rank, budget in enumerate(wanted) if rank < len(ordered)}


def load_model(args):
    from transformers import AutoTokenizer

    log(f"loading {args.model} with {args.backend}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left") if args.backend == "transformers" \
        else AutoTokenizer.from_pretrained(args.model)
    return load_backend(args.model, args), tokenizer


def read_rewrites(path):
    """{id: text} from an append-only JSONL; the combined form stores {budget: text} per id."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            out[int(row["id"])] = row.get("text")
        except (TypeError, ValueError):
            continue
    return out


def per_budget_texts(args, indices):
    """{budget: {id: text}} from whichever rewrite files exist."""
    result = {}
    for budget in args.budgets:
        if args.combined:
            rows = read_rewrites(PROBE_DIR / f"v3_rewrite_combined_{args.run}.jsonl")
            result[budget] = {index: (rows.get(index) or {}).get(str(budget))
                              for index in indices}
        else:
            rows = read_rewrites(PROBE_DIR / f"v3_rewrite_{budget}_{args.run}.jsonl")
            result[budget] = {index: rows.get(index) for index in indices}
    return result


def rewrite_audit(original, rewritten, shared):
    """Retention / hallucination / length / lexical shared survival for one window."""
    original_words = {word for word in tokenize(original) if len(word) > 2}
    rewritten_words = {word for word in tokenize(rewritten) if len(word) > 2}
    return {
        "retention": len(original_words & rewritten_words) / max(len(original_words), 1),
        "hallucination": len(rewritten_words - original_words) / max(len(rewritten_words), 1),
        "length_ratio": len(rewritten.split()) / max(len(original.split()), 1),
        "shared_remaining": (sum(1 for phrase in shared if phrase_pattern(phrase).search(rewritten))
                             / max(len(shared), 1)),
    }


def stage_merge(args):
    captions, other, same, _ = rewrite_context(args)
    windows, _ = pool_windows()
    indices = sorted(int(index) for index in windows)
    texts = per_budget_texts(args, indices)
    for budget, rows in texts.items():
        output = {str(index): {"caption": captions[index], "rewritten": text}
                  for index, text in rows.items() if isinstance(text, str) and text.strip()}
        path = PROBE_DIR / f"cleaned_local_contrastive_v3_{budget}_probe.json"
        path.write_text(json.dumps(output, ensure_ascii=False, indent=1) + "\n")
        log(f"wrote {path.name}: {len(output)}/{len(indices)} windows "
            f"({len(output) / max(len(indices), 1):.1%} coverage)")


def stage_audit(args):
    captions, other, same, _ = rewrite_context(args)
    windows, _ = pool_windows()
    indices = sorted(int(index) for index in windows)
    texts = per_budget_texts(args, indices)
    report = {"run": args.run, "model": args.model, "budgets": {}}
    for budget, rows in texts.items():
        rows = {index: text for index, text in rows.items()
                if isinstance(text, str) and text.strip()}
        if not rows:
            log(f"budget {budget}: no rewrites found, skipped")
            continue
        keys = ("retention", "hallucination", "length_ratio", "shared_remaining")
        metrics = {key: [] for key in keys}
        axis = {"other_videos": [], "same_video_other_shots": []}
        for index, text in rows.items():
            shared = other.get(str(index), []) + same.get(str(index), [])
            audit = rewrite_audit(captions[index], text, shared)
            for key in keys:
                metrics[key].append(audit[key])
            for name, phrases in (("other_videos", other.get(str(index), [])),
                                  ("same_video_other_shots", same.get(str(index), []))):
                if phrases:
                    axis[name].append(
                        sum(1 for phrase in phrases if phrase_pattern(phrase).search(text))
                        / len(phrases))
        mean = lambda values: round(float(np.mean(values)), 3) if values else None
        block = {key: mean(values) for key, values in metrics.items()}
        block["mean_words"] = round(float(np.mean([len(text.split()) for text in rows.values()])), 1)
        block["windows"] = len(rows)
        block["shared_remaining_other_videos"] = mean(axis["other_videos"])
        block["shared_remaining_same_video_shots"] = mean(axis["same_video_other_shots"])
        report["budgets"][str(budget)] = block
        log(f"budget {budget}: {block}")
    path = PROBE_DIR / f"v3_audit_{args.run}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    log(f"wrote {path}")


def corpus_index():
    """Row-aligned corpus views: video ids, shot keys and the 4B vectors.

    Note `CAPTION_DIR` is the *embedding* directory (`data/2026/embeddings/qwen3-4b-fp16`), whose
    `meta.parquet` is row-aligned with `shot_captions.parquet` — that alignment is what lets these
    arrays be indexed by caption row.
    """
    meta = pq.read_table(CAPTION_DIR / "meta.parquet").to_pylist()
    video_of = [row["video_id"] for row in meta]
    shot_of = [(row["video_id"], row["shot_index"]) for row in meta]
    shards = sorted(CAPTION_DIR.glob("emb_*.npy"))
    vectors = np.vstack([np.load(path) for path in shards]).astype(np.float32)
    return meta, video_of, shot_of, vectors


def mine_intra_neighbors(indices, k, vectors, video_of, shot_of):
    """Top-k most similar windows from the SAME video but a DIFFERENT shot.

    Drawn from the whole corpus rather than from the pool: the pool holds a median of only 2
    windows per video against ~118 real ones, so pool-only mining would have almost no signal.
    """
    by_video = {}
    for position, video in enumerate(video_of):
        by_video.setdefault(video, []).append(position)
    out = {}
    for index in indices:
        video = video_of[index]
        peers = np.asarray([p for p in by_video[video]
                            if p != index and shot_of[p] != shot_of[index]], dtype=np.int64)
        if not len(peers):
            out[str(index)] = []
            continue
        similarity = vectors[peers] @ vectors[index]
        out[str(index)] = [int(p) for p in peers[np.argsort(-similarity)[:k]]]
    return out


def mine_axis_phrases(captions, indices, neighbor_map, min_neighbors, sizes=(3, 4, 5)):
    """Shared phrases between each window and the neighbours on one axis."""
    from collections import Counter

    needed = set(indices)
    for value in neighbor_map.values():
        needed.update(int(v) for v in value)
    grams = {}
    for index in needed:
        tokens = tokenize(captions[index])
        seen = set()
        for size in sizes:
            for start in range(len(tokens) - size + 1):
                seen.add(" ".join(tokens[start:start + size]))
        grams[index] = seen
    out = {}
    for index in indices:
        tokens = tokenize(captions[index])
        counts = Counter()
        for neighbor in neighbor_map[str(index)]:
            counts.update(grams[int(neighbor)] & grams[index])
        candidates, used = [], []
        for size in sizes:
            for start in range(len(tokens) - size + 1):
                gram = " ".join(tokens[start:start + size])
                if counts.get(gram, 0) >= min_neighbors:
                    candidates.append((gram, start, start + size))
        candidates.sort(key=lambda item: -(item[2] - item[1]))
        kept = []
        for gram, low, high in candidates:
            if any(not (high <= a or low >= b) for a, b in used):
                continue
            kept.append(gram)
            used.append((low, high))
        out[str(index)] = kept
    return out


def stage_mine_axes(args):
    """Build neighbours and shared phrases for BOTH axes, into run-stamped new files."""
    captions, _ = captions_and_shared()
    windows, _ = pool_windows()
    indices = sorted(int(index) for index in windows)
    log(f"loading corpus vectors for {len(indices)} pool windows")
    _, video_of, shot_of, vectors = corpus_index()

    cross = json.loads(NEIGHBORS_JSON.read_text())            # existing, cross-video, top-5
    neighbors = {str(index): {"other_videos": [int(n) for n in cross[str(index)]],
                              "same_video_other_shots": []} for index in indices}
    if args.intra_k:
        log(f"mining top-{args.intra_k} same-video different-shot neighbours")
        intra = mine_intra_neighbors(indices, args.intra_k, vectors, video_of, shot_of)
        for index in indices:
            neighbors[str(index)]["same_video_other_shots"] = intra[str(index)]
        have = sum(1 for index in indices if intra[str(index)])
        log(f"intra-video neighbours: {have}/{len(indices)} windows "
            f"({have / len(indices):.1%})")
    del vectors
    import gc

    gc.collect()

    phrases = {}
    for axis, threshold in (("other_videos", args.threshold_cross),
                            ("same_video_other_shots", args.threshold_same)):
        if axis == "same_video_other_shots" and not args.intra_k:
            phrases[axis] = {str(index): [] for index in indices}
            continue
        log(f"mining shared phrases on axis {axis} (>= {threshold} neighbours)")
        mapped = {str(index): neighbors[str(index)][axis] for index in indices}
        mined = mine_axis_phrases(captions, indices, mapped, threshold)
        phrases[axis] = mined
        sizes = [len(value) for value in mined.values()]
        log(f"  {axis}: median {int(np.median(sizes))} phrases/window, max {max(sizes)}")

    neighbor_path = PROBE_DIR / "caption_neighbors_axes.json"
    neighbor_path.write_text(json.dumps(neighbors) + "\n")
    phrase_path = PROBE_DIR / "shared_phrases_axes.json"
    phrase_path.write_text(json.dumps(phrases) + "\n")
    log(f"wrote {neighbor_path.name} and {phrase_path.name}")

    # How much of each caption the two axes cover, and how much they overlap: the numbers that say
    # whether the intra-video axis adds deletions the cross-video axis missed, or re-deletes the
    # same content.
    report = {}
    for axis, mined in phrases.items():
        covered, overlap = [], []
        for index in indices:
            own = tokenize(captions[index])
            spans, used = 0, []
            for phrase in mined[str(index)]:
                size = len(phrase.split())
                found = next((s for s in range(len(own) - size + 1)
                              if " ".join(own[s:s + size]) == phrase
                              and not any(not (s + size <= a or s >= b) for a, b in used)), None)
                if found is not None:
                    used.append((found, found + size))
                    spans += size
            covered.append(spans / max(len(own), 1))
        report[axis] = {"coverage_mean": round(float(np.mean(covered)), 4)}
    cross_sets = {index: set(phrases["other_videos"][str(index)]) for index in indices}
    same_sets = {index: set(phrases["same_video_other_shots"][str(index)]) for index in indices}
    both = [len(cross_sets[i] & same_sets[i]) for i in indices]
    total_same = sum(len(same_sets[i]) for i in indices)
    report["intra_shared_with_cross"] = round(sum(both) / max(total_same, 1), 4)
    report["thresholds"] = {"cross": args.threshold_cross, "same": args.threshold_same}
    report["intra_k"] = args.intra_k
    out = PROBE_DIR / f"v3_mining_report_{args.run}.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    log(f"mining report: {report}")


def stage_table(args):
    """Print every arm measured on the pool, side by side, from the reports on disk.

    Reads only files this work produced plus the pre-existing 0.6B probe reports, so the table can
    be regenerated without re-running anything.
    """
    rows = []
    probe = PROBE_DIR / "cleaning_probe.json"
    if probe.exists():
        metrics = json.loads(probe.read_text())["metrics"]
        for name in ("original", "stripped", "first_sentence", "distinctive", "contrastive",
                     "contrastive_long"):
            if name in metrics:
                rows.append((f"{name}  (0.6B)", metrics[name]))
    controls = PROBE_DIR / f"v3_controls_0b_{args.run}.json"
    if not controls.exists():  # accept whichever control run exists
        found = sorted(PROBE_DIR.glob("v3_controls_0b_*.json"))
        controls = found[-1] if found else None
    if controls:
        report = json.loads(controls.read_text())
        for name, block in report["metrics"].items():
            rows.append((f"{name}  (0.6B, no-LLM)", block))
    prune = sorted(PROBE_DIR.glob("v3_prune_0b_*.json"))
    if prune:
        report = json.loads(prune[-1].read_text())
        for name, block in report["metrics"].items():
            rows.append((f"{name}  (0.6B, no-LLM)", block))
    score = PROBE_DIR / "cleaning_probe_4b_v3.json"
    if score.exists():
        report = json.loads(score.read_text())
        for name, block in report["metrics"].items():
            rows.append((f"{name}  (4B)", block))

    header = (f"{'arm':38s} {'BAD mr':>7s} {'BAD@1':>6s} {'BAD@5':>6s} {'BAD@25':>7s} "
              f"{'BAD@50':>7s} {'CTRL mr':>8s} {'CTRL@1':>7s}")
    print(header)
    print("-" * len(header))
    for name, block in rows:
        bad, ctrl = block["bad_oracle"], block["control_realistic"]
        print(f"{name:38s} {bad['mrr']:7.3f} {bad['top1']:6.3f} {bad['top5']:6.3f} "
              f"{bad['top25']:7.3f} {bad['top50']:7.3f} {ctrl['mrr']:8.3f} {ctrl['top1']:7.3f}")


def stage_verify_originals(args):
    """Record/check the caption parquet hash so 'originals untouched' is verified, not promised."""
    path = PROBE_DIR / "v3_originals_hash.json"
    digest = sha256_file(CAPTIONS_PARQUET)
    previous = json.loads(path.read_text()) if path.exists() else None
    if previous is None:
        path.write_text(json.dumps({"path": str(CAPTIONS_PARQUET), "sha256": digest}, indent=1) + "\n")
        log(f"recorded baseline sha256 {digest[:16]}… -> {path.name}")
        return
    if previous["sha256"] == digest:
        log(f"OK: {CAPTIONS_PARQUET} unchanged ({digest[:16]}…)")
    else:
        raise SystemExit(f"ORIGINAL CAPTIONS CHANGED: {previous['sha256'][:16]}… -> {digest[:16]}…")


def self_test():
    assert phrase_pattern("chef s").search("a white chef's apron")
    assert phrase_pattern("in the background the").search("in the background. The scene")
    assert not phrase_pattern("the lion s head").search("a woman cooks rice")
    mapping = {"on the left": "along the left-hand side"}
    text, applied, missed = substitute("A pan on the left of the stove.", ["on the left"], None, mapping)
    assert applied == 1 and missed == 0 and "left-hand" in text
    text, applied, missed = substitute("A pan on the right.", ["on the left"], None, mapping)
    assert applied == 0 and missed == 1 and text == "A pan on the right."
    log("self-test OK")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",
                        choices=("paraphrase-probe", "controls", "prune", "rewrite", "merge",
                                 "audit", "mine-axes", "table", "show-prompt",
                                 "verify-originals"),
                        default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=[120, 200, 300],
                        help="rewrite stage: absolute word budgets to sweep")
    parser.add_argument("--combined", action="store_true",
                        help="rewrite stage: ask for all budgets in one call (shares the prompt)")
    parser.add_argument("--chunk", type=int, default=32,
                        help="rewrite stage: windows per generation call")
    parser.add_argument("--neighbor-chars", type=int, default=400)
    parser.add_argument("--axes", choices=("cross", "both"), default="both",
                        help="rewrite: which shared-phrase axes to hand the model")
    parser.add_argument("--intra-k", type=int, default=5,
                        help="mine-axes: same-video different-shot neighbours (0 disables)")
    parser.add_argument("--threshold-cross", type=int, default=2,
                        help="mine-axes: neighbours a phrase must appear in, cross-video")
    parser.add_argument("--threshold-same", type=int, default=2,
                        help="mine-axes: neighbours a phrase must appear in, same-video")
    parser.add_argument("--limit", type=int, default=0, help="rewrite stage: cap windows")
    parser.add_argument("--show", type=int, default=0,
                        help="show-prompt stage: index into the pool window list")
    parser.add_argument("--encoder", choices=("0b", "4b"), default="4b",
                        help="controls stage: which embedder to score the arms with")
    parser.add_argument("--control-words", type=int, nargs="+", default=[60, 150, 300],
                        help="controls stage: prefix-truncation arms, in words")
    parser.add_argument("--control-sentences", type=int, nargs="+", default=[3],
                        help="controls stage: leading-sentence arms")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--run", default=RUN)
    parser.add_argument("--limit-phrases", type=int, default=0)
    parser.add_argument("--phrase-chunk", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16, help="transformers backend only")
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--gpu-memory", type=float, default=0.85)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    log(f"run {args.run} | model {args.model} | backend {args.backend}")
    if args.stage == "paraphrase-probe":
        stage_paraphrase_probe(args)
    elif args.stage == "controls":
        stage_controls(args)
    elif args.stage == "prune":
        stage_prune(args)
    elif args.stage == "rewrite":
        stage_rewrite(args)
    elif args.stage == "merge":
        stage_merge(args)
    elif args.stage == "audit":
        stage_audit(args)
    elif args.stage == "mine-axes":
        stage_mine_axes(args)
    elif args.stage == "table":
        stage_table(args)
    elif args.stage == "show-prompt":
        stage_show_prompt(args)
    elif args.stage == "verify-originals":
        stage_verify_originals(args)
    else:
        parser.error("choose --stage or --self-test")


if __name__ == "__main__":
    main()
