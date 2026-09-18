#!/usr/bin/env python3
"""Generate and validate holistic-caption paraphrases for the paraphrase benchmark
(docs/reports/visual-holistic-paraphrase-benchmark-2026.md).

Input is only the user-approved English caption per case
(artifacts/official_benchmark_queries_en_v2.json) -- never the Vietnamese prompt, so no
detail can leak back in. Each caption gets 4 paraphrases in a fixed strategy order
(reorder, synonym, restructure, combined); benchmark subsets of size n use the first n.
Every paraphrase passes deterministic checks (scripts/benchmarks/paraphrase_validation.py)
and a separate Gemini equivalence judge, with up to 2 targeted regenerations.

Writes paraphrases.json, review.txt and judge_log.jsonl for the user's review gate.
No benchmark metrics are computed here.
"""
import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from google import genai
from google.genai import errors as genai_errors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import query_expansion as qe  # noqa: E402
from scripts.benchmarks import paraphrase_validation as pv  # noqa: E402
from scripts.benchmarks.bench_visual_embeddings import round_of  # noqa: E402

PROMPT_VERSION = "v1"

STRATEGY_TEXT = (
    "1. reorder - change the order of phrases or clauses, or switch between active and passive voice.\n"
    "2. synonym - keep the sentence structure but replace ordinary words with natural synonyms.\n"
    "3. restructure - change the grammatical construction (for example a clause becomes a "
    "participle or prepositional phrase, or a relative clause is used).\n"
    "4. combined - both change the order and use synonyms.\n"
)

HARD_RULES = (
    "Hard rules for every paraphrase:\n"
    "- The meaning must be exactly the same as the caption. Keep every person, animal, object, "
    "count, colour, clothing item, action, relation, setting, on-screen text and proper noun.\n"
    "- Do not add any detail, inference, adjective, adverb, intensifier or interpretation that is "
    "not in the caption. Do not remove any detail.\n"
    "- Keep colour words, proper nouns and quoted or on-screen text exactly as written. A count may "
    "be written as a digit or a word, but it must stay the same number.\n"
    "- Do not add framing words such as photo, image, picture, video, clip, scene, footage, shot or "
    "frame unless the caption already uses them.\n"
    "- Each paraphrase must read naturally and be worded clearly differently from the caption and "
    "from the other paraphrases.\n"
    "- Keep each paraphrase under 30 words.\n"
)

PARAPHRASE_PROMPT = (
    "You rewrite short English image-retrieval captions. For each caption write exactly 4 "
    "paraphrases, one per strategy:\n" + STRATEGY_TEXT + HARD_RULES +
    "- Keep each id unchanged.\n\nCaptions:\n"
)

REGENERATE_PROMPT = (
    "Each item is a paraphrase of an English image-retrieval caption that failed validation. "
    "Write one new paraphrase of the caption using the item's strategy:\n" + STRATEGY_TEXT +
    "Fix every listed failure, and make the new paraphrase clearly different from "
    "other_paraphrases.\n" + HARD_RULES + "- Keep each key unchanged.\n\nItems:\n"
)

JUDGE_PROMPT = (
    "You check whether each paraphrase has exactly the same visual meaning as its source caption, "
    "for an image search benchmark. Compare detail by detail: people and animals, counts, colours, "
    "clothing, objects, actions, relations between things, setting, on-screen text and proper "
    "nouns.\n"
    "- added: details present in the paraphrase but not in the source, including new adjectives, "
    "framing words such as 'photo' or 'video', and inferences.\n"
    "- removed: details in the source that are missing from the paraphrase.\n"
    "- changed: details present in both but altered (a different count, colour, action, relation "
    "or name).\n"
    "Rewording, word order, active/passive voice and true synonyms are NOT differences. Set "
    "equivalent=true only if added, removed and changed are all empty. Keep each key unchanged.\n\n"
    "Pairs:\n"
)

GENERATE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "paraphrases": {
                "type": "array", "minItems": 4, "maxItems": 4,
                "items": {
                    "type": "object",
                    "properties": {"strategy": {"type": "string", "enum": list(pv.STRATEGIES)},
                                   "text": {"type": "string"}},
                    "required": ["strategy", "text"], "additionalProperties": False,
                },
            },
        },
        "required": ["id", "paraphrases"], "additionalProperties": False,
    },
}
REGENERATE_SCHEMA = {
    "type": "array",
    "items": {"type": "object",
              "properties": {"key": {"type": "string"}, "text": {"type": "string"}},
              "required": ["key", "text"], "additionalProperties": False},
}
_STRINGS = {"type": "array", "items": {"type": "string"}}
JUDGE_SCHEMA = {
    "type": "array",
    "items": {"type": "object",
              "properties": {"key": {"type": "string"}, "equivalent": {"type": "boolean"},
                             "added": _STRINGS, "removed": _STRINGS, "changed": _STRINGS},
              "required": ["key", "equivalent", "added", "removed", "changed"],
              "additionalProperties": False},
}


def prompt_hash():
    joined = "\n---\n".join((PARAPHRASE_PROMPT, REGENERATE_PROMPT, JUDGE_PROMPT))
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]


class GeminiPool:
    """Round-robin over GEMINI_API_KEY, GEMINI_API_KEY_2..5; retries a call on API or parse errors."""

    def __init__(self, timeout_s=60.0, max_attempts=8):
        keys = [os.environ.get("GEMINI_API_KEY")] + [os.environ.get(f"GEMINI_API_KEY_{i}") for i in range(2, 6)]
        keys = [k for k in keys if k]
        if not keys:
            raise SystemExit("no GEMINI_API_KEY* set")
        self.clients = itertools.cycle([genai.Client(api_key=k) for k in keys])
        self.client = next(self.clients)
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts

    def call(self, prompt, payload, schema, validate):
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                rows = qe._call_gemini(self.client, prompt, payload, schema, self.timeout_s)
                validate(rows)
                return rows
            except (genai_errors.APIError, ValueError) as exc:
                last_error = exc
                print(f"  gemini attempt {attempt} failed ({type(exc).__name__}: {str(exc)[:160]}); rotating key",
                      flush=True)
                self.client = next(self.clients)
                time.sleep(min(2 * attempt, 10))
        raise RuntimeError(f"gemini call failed after {self.max_attempts} attempts: {last_error}")


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def generate_initial(pool, cases, batch_size):
    results = {}
    for batch in chunks(cases, batch_size):
        payload = [{"id": c["id"], "caption": c["caption"]} for c in batch]
        expected = {c["id"] for c in batch}

        def validate(rows):
            if {row["id"] for row in rows} != expected:
                raise ValueError("generated ids differ from the batch")
            for row in rows:
                if sorted(p["strategy"] for p in row["paraphrases"]) != sorted(pv.STRATEGIES):
                    raise ValueError(f"{row['id']}: strategies must be exactly {pv.STRATEGIES}")

        rows = pool.call(PARAPHRASE_PROMPT, payload, GENERATE_SCHEMA, validate)
        for row in rows:
            by_strategy = {p["strategy"]: p["text"].strip() for p in row["paraphrases"]}
            results[row["id"]] = [{"strategy": s, "text": by_strategy[s]} for s in pv.STRATEGIES]
        print(f"generated {len(results)}/{len(cases)}", flush=True)
    return results


def make_regenerate_batch(pool, batch_size):
    def regenerate_batch(requests):
        out = {}
        for batch in chunks(requests, batch_size):
            # Simple sequential keys: composite ids like "Round 1 - 2026/query-p1-1-kis:1#2"
            # are long, punctuation-heavy strings that the model doesn't reliably echo back
            # verbatim, which showed up as persistent "keys differ from the batch" failures.
            index_of = {f"req_{i}": (r["id"], r["slot"]) for i, r in enumerate(batch)}
            payload = [{"key": key, "strategy": r["strategy"], "caption": r["source"],
                        "failed_paraphrase": r["failed_text"], "failures": r["failures"],
                        "other_paraphrases": r["other_paraphrases"]}
                       for key, r in zip(index_of, batch)]
            expected = set(index_of)

            def validate(rows):
                if {row["key"] for row in rows} != expected:
                    raise ValueError("regenerated keys differ from the batch")

            for row in pool.call(REGENERATE_PROMPT, payload, REGENERATE_SCHEMA, validate):
                out[index_of[row["key"]]] = row["text"].strip()
        print(f"  regenerated {len(out)} slots", flush=True)
        return out
    return regenerate_batch


def make_judge_batch(pool, batch_size, log_handle):
    def judge_batch(pairs):
        out = {}
        for batch in chunks(pairs, batch_size):
            index_of = {f"req_{i}": (p["id"], p["slot"]) for i, p in enumerate(batch)}
            payload = [{"key": key, "source": p["source"], "paraphrase": p["paraphrase"]}
                       for key, p in zip(index_of, batch)]
            expected = set(index_of)

            def validate(rows):
                if {row["key"] for row in rows} != expected:
                    raise ValueError("judged keys differ from the batch")

            item_by_key = dict(zip(index_of, batch))
            for row in pool.call(JUDGE_PROMPT, payload, JUDGE_SCHEMA, validate):
                case_id, slot = index_of[row["key"]]
                verdict = {k: row[k] for k in ("equivalent", "added", "removed", "changed")}
                out[(case_id, slot)] = verdict
                log_handle.write(json.dumps({**item_by_key[row["key"]], **verdict}, ensure_ascii=False) + "\n")
        print(f"  judged {len(out)} paraphrases", flush=True)
        return out
    return judge_batch


def add_cosines(state, cases_by_id):
    """PE-Core text-text cosine between each accepted paraphrase and its caption (diagnostic only)."""
    from scripts.benchmarks.bench_query_expansion import encode_and_cache

    texts = set()
    for case_id, case in state.items():
        texts.add(cases_by_id[case_id]["caption"])
        texts.update(s["text"] for s in case["slots"] if s["status"] == "ok")
    vectors = encode_and_cache(sorted(texts))
    for case_id, case in state.items():
        caption_vec = vectors[cases_by_id[case_id]["caption"]]
        for slot in case["slots"]:
            slot["cosine_to_caption"] = (float(np.dot(caption_vec, vectors[slot["text"]]))
                                         if slot["status"] == "ok" else None)


def assert_output_valid(state):
    for case_id, case in state.items():
        accepted = []
        for slot in case["slots"]:
            if slot["status"] != "ok":
                continue
            checks = pv.deterministic_checks(case["source"], slot["text"], accepted, qe.count_tokens)
            assert not checks.hard, (case_id, slot["strategy"], checks.hard)
            assert checks.tokens <= qe.TOKEN_LIMIT, (case_id, slot["strategy"], checks.tokens)
            assert not pv.judge_failures(slot["judge"]), (case_id, slot["strategy"], slot["judge"])
            accepted.append(slot["text"])


def summarize(state):
    slots = [slot for case in state.values() for slot in case["slots"]]
    ok = [s for s in slots if s["status"] == "ok"]
    cosines = [s["cosine_to_caption"] for s in ok if s.get("cosine_to_caption") is not None]
    flags = Counter(flag for s in ok for flag in s.get("flags", []))
    return {
        "cases": len(state),
        "slots_total": len(slots),
        "slots_ok": len(ok),
        "slots_rejected": len(slots) - len(ok),
        "rejected_by_strategy": dict(Counter(s["strategy"] for s in slots if s["status"] != "ok")),
        "cases_n_effective_below_4": sorted(cid for cid, c in state.items() if c["n_effective"] < 4),
        "attempts_histogram": dict(sorted(Counter(s["attempts"] for s in slots).items())),
        "soft_flags": dict(flags),
        "max_tokens": max((s["tokens"] for s in ok), default=None),
        "cosine_to_caption": {
            "min": min(cosines), "median": statistics.median(cosines),
            "mean": statistics.fmean(cosines), "max": max(cosines),
        } if cosines else None,
    }


def write_review(state, cases_by_id, path):
    order = ["Round 1 - 2026", "Round 2 - 2026", "Round 3 - 2026", "Test"]
    lines = []
    for round_name in order:
        ids = [cid for cid in state if round_of(cid) == round_name]
        if not ids:
            continue
        lines.append(f"===== {round_name} ({len(ids)} cases) =====\n")
        for case_id in ids:
            case = state[case_id]
            lines.append(f"[{case_id}]  n_effective={case['n_effective']}")
            lines.append(f"  CAPTION ({qe.count_tokens(case['source'])} tok): {case['source']}")
            for slot in case["slots"]:
                if slot["status"] == "ok":
                    flags = ",".join(slot.get("flags", [])) or "-"
                    lines.append(
                        f"  {slot['strategy']:11s} ({slot['tokens']} tok, dist={slot['edit_dist_source']:.2f}, "
                        f"cos={slot['cosine_to_caption']:.3f}, tries={slot['attempts']}, flags={flags}): {slot['text']}")
                else:
                    last = slot["history"][-1]
                    lines.append(f"  {slot['strategy']:11s} REJECTED after {slot['attempts']} tries: "
                                 f"{last['text']}  <- {'; '.join(last['failures'])}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captions", type=Path, default=ROOT / "artifacts/official_benchmark_queries_en_v2.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / f"artifacts/holistic_paraphrase/{PROMPT_VERSION}")
    parser.add_argument("--limit", type=int, default=None, help="only the first N cases (smoke test)")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--judge-batch-size", type=int, default=24)
    args = parser.parse_args()

    source = json.loads(args.captions.read_text())
    cases = [{"id": c["id"], "caption": c["text"]} for c in source["cases"]]
    if args.limit:
        cases = cases[:args.limit]
    cases_by_id = {c["id"]: c for c in cases}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pool = GeminiPool()
    initial = generate_initial(pool, cases, args.batch_size)
    with open(args.out_dir / "judge_log.jsonl", "w", encoding="utf-8") as judge_log:
        state = pv.run_validation_rounds(
            {c["id"]: {"source": c["caption"], "slots": initial[c["id"]]} for c in cases},
            regenerate_batch=make_regenerate_batch(pool, args.batch_size),
            judge_batch=make_judge_batch(pool, args.judge_batch_size, judge_log),
            count_tokens=qe.count_tokens,
        )
    add_cosines(state, cases_by_id)
    assert_output_valid(state)

    stats = summarize(state)
    output_cases = []
    for c in cases:
        case = state[c["id"]]
        output_cases.append({
            "id": c["id"], "round": round_of(c["id"]), "caption": case["source"],
            "caption_tokens": qe.count_tokens(case["source"]), "n_effective": case["n_effective"],
            "paraphrases": [{
                "strategy": s["strategy"], "text": s["text"], "status": s["status"],
                "attempts": s["attempts"], "tokens": s.get("tokens"),
                "edit_dist_source": s.get("edit_dist_source"), "min_dist_prev": s.get("min_dist_prev"),
                "cosine_to_caption": s.get("cosine_to_caption"), "flags": s.get("flags", []),
                "judge": s.get("judge"), "history": s["history"],
            } for s in case["slots"]],
        })
    header = {
        "prompt_version": PROMPT_VERSION, "prompt_hash": prompt_hash(), "model": qe.GEMINI_MODEL,
        "source": str(args.captions.relative_to(ROOT)), "strategies": list(pv.STRATEGIES),
        "token_target": qe.TOKEN_TARGET, "token_limit": qe.TOKEN_LIMIT,
        "diversity_thresholds": {"vs_source": pv.DIVERSITY_VS_SOURCE, "vs_previous": pv.DIVERSITY_VS_PREVIOUS},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stats": stats,
    }
    (args.out_dir / "paraphrases.json").write_text(
        json.dumps({**header, "cases": output_cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_review(state, cases_by_id, args.out_dir / "review.txt")
    print(json.dumps(stats, indent=2))
    print(f"wrote {args.out_dir / 'paraphrases.json'}")


if __name__ == "__main__":
    main()
