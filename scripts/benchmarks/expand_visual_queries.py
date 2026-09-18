#!/usr/bin/env python3
"""Generate holistic-caption + aspect-query expansions for the query-expansion
benchmark (Phase A of docs/reports/visual-query-expansion-benchmark-2026.md's
plan). Reads the original Vietnamese prompts from
artifacts/official_benchmark_answers_v2.json and writes an expansions.json +
review.txt for the user to approve before any tuning/benchmark work.
"""
import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

from google import genai
from google.genai import errors as genai_errors

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend import query_expansion as qe  # noqa: E402


def api_keys():
    keys = [os.environ.get("GEMINI_API_KEY")]
    for i in range(2, 6):
        key = os.environ.get(f"GEMINI_API_KEY_{i}")
        if key:
            keys.append(key)
    return [k for k in keys if k]


def expand_with_rotation(cases, prompt_version, batch_size, timeout_s):
    keys = api_keys()
    if not keys:
        raise SystemExit("no GEMINI_API_KEY* set")
    clients = itertools.cycle(genai.Client(api_key=key) for key in keys)
    client = next(clients)
    results = {}
    remaining = list(cases)
    attempt = 0
    while remaining:
        attempt += 1
        try:
            batch_results = qe.expand_batch(remaining, client, prompt_version=prompt_version,
                                            batch_size=batch_size, timeout_s=timeout_s)
            results.update(batch_results)
            remaining = [case for case in remaining if case["id"] not in results]
        except genai_errors.ClientError as exc:
            if attempt > len(keys) * 2:
                raise
            print(f"  rotating key after error: {exc}", flush=True)
            client = next(clients)
            time.sleep(2)
    return results


def summarize(expansions):
    aspect_counts = Counter()
    shortened = over_target = dropped = 0
    max_tokens = 0
    by_round = Counter()
    for case_id, expansion in expansions.items():
        round_name = case_id.split("/", 1)[0]
        by_round[round_name] += 1
        aspect_counts[expansion.aspect_count_used] += 1
        items = [expansion.holistic, *expansion.aspects]
        for item in items:
            if item.dropped:
                dropped += 1
                continue
            shortened += int(item.was_shortened)
            over_target += int(item.over_target)
            max_tokens = max(max_tokens, item.tokens_after)
    return {
        "cases_by_round": dict(by_round),
        "aspect_count_histogram": dict(sorted(aspect_counts.items())),
        "shortened_items": shortened,
        "over_target_kept_items": over_target,
        "dropped_items": dropped,
        "max_tokens_after": max_tokens,
    }


def item_dict(item):
    d = {"text": item.text, "tokens_before": item.tokens_before, "tokens_after": item.tokens_after,
         "was_shortened": item.was_shortened, "over_target": item.over_target}
    if item.text_original:
        d["text_original"] = item.text_original
    if item.dropped:
        d["dropped"] = item.dropped
    facet = getattr(item, "facet", None)
    if facet:
        d["facet"] = facet
    return d


def write_review_txt(cases, expansions, path):
    lines = []
    for round_name in ["Round 1 - 2026", "Round 2 - 2026", "Round 3 - 2026", "Test"]:
        round_cases = [c for c in cases if c["id"].split("/", 1)[0] == round_name]
        if not round_cases:
            continue
        lines.append(f"===== {round_name} ({len(round_cases)} cases) =====\n")
        for case in round_cases:
            expansion = expansions[case["id"]]
            lines.append(f"[{case['id']}]")
            lines.append(f"  VI: {case['text'][:200]}")
            h = expansion.holistic
            tag = " (SHORTENED)" if h.was_shortened else (" (OVER TARGET)" if h.over_target else "")
            lines.append(f"  HOLISTIC ({h.tokens_after} tok{tag}): {h.text}")
            for a in expansion.aspects:
                if a.dropped:
                    lines.append(f"  ASPECT [dropped: {a.dropped}]: {a.text_original}")
                    continue
                tag = " (SHORTENED)" if a.was_shortened else (" (OVER TARGET)" if a.over_target else "")
                lines.append(f"  ASPECT [{a.facet}] ({a.tokens_after} tok{tag}): {a.text}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/official_benchmark_answers_v2.json")
    parser.add_argument("--approved-captions", type=Path,
                        default=ROOT / "artifacts/official_benchmark_queries_en_v2.json")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    out_dir = args.out_dir or (ROOT / f"artifacts/query_expansion/{args.prompt_version}")
    out_dir.mkdir(parents=True, exist_ok=True)

    source = json.loads(args.input.read_text())
    cases = [{"id": case["id"], "text": case["text"], "targets": case["targets"]} for case in source["cases"]]
    approved = json.loads(args.approved_captions.read_text())
    approved_by_id = {case["id"]: case["text"] for case in approved["cases"]}

    print(f"expanding {len(cases)} cases with prompt {args.prompt_version}...", flush=True)
    expansions = expand_with_rotation(cases, args.prompt_version, args.batch_size, args.timeout_s)

    output_cases = []
    for case in cases:
        expansion = expansions[case["id"]]
        output_cases.append({
            "id": case["id"], "round": case["id"].split("/", 1)[0], "text_vi": case["text"],
            "caption_en_approved": approved_by_id.get(case["id"]),
            "holistic": item_dict(expansion.holistic),
            "aspects": [item_dict(a) for a in expansion.aspects],
            "aspect_count_generated": expansion.aspect_count_generated,
            "aspect_count_used": expansion.aspect_count_used,
            "targets": case["targets"],
        })

    stats = summarize(expansions)
    header = {
        "model": qe.GEMINI_MODEL, "prompt_version": args.prompt_version,
        "token_target": qe.TOKEN_TARGET, "token_limit": qe.TOKEN_LIMIT, "stats": stats,
    }
    (out_dir / "expansions.json").write_text(
        json.dumps({**header, "cases": output_cases}, ensure_ascii=False, indent=2) + "\n")
    write_review_txt(cases, expansions, out_dir / "review.txt")

    print(json.dumps(stats, indent=2))
    print(f"wrote {out_dir / 'expansions.json'}")
    print(f"wrote {out_dir / 'review.txt'}")


if __name__ == "__main__":
    main()
