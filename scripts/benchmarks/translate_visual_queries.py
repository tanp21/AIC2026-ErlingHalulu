#!/usr/bin/env python3
"""Translate Vietnamese visual-retrieval query prompts into short, caption-style
English queries suitable for CLIP-style text towers (PE-Core/NaFlex/GOPT).

Why this exists (not scripts/benchmarks/translate_official_queries.py):
- These text towers have a 64-72 token context window, and truncate silently
  past it (no error). Checked against the *raw Vietnamese* prompts: 80/95
  (84%) of official-benchmark cases already exceed 72 tokens, so most queries
  in a naive Vietnamese-prompt benchmark are truncated mid-sentence.
- Even the existing "natural, precise" English translation
  (translate_official_queries.py, artifacts/official_benchmark_queries_en.json)
  still overflows 64 tokens for 30/95 (32%) of cases, because it preserves
  meta-commentary ("Find the exact clip where...", "The video describes...")
  that isn't how CLIP's training captions read and burns tokens without adding
  searchable visual content.
- This script instead asks for a short, single-sentence, caption-style
  description of the most distinctive visual moment (no instructional framing,
  no meta-commentary), verifies the result against each model's tokenizer, and
  re-shortens (up to --max-retries times) any case still over budget.
"""
import argparse
import json
import os
import sys
from pathlib import Path

from google import genai

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = "gemini-3.1-flash-lite"
TARGET_TOKENS = 50  # safety margin under the tightest text tower (NaFlex/GOPT: 64)

TRANSLATE_PROMPT = (
    "These are Vietnamese video-retrieval query prompts, written as instructions to a "
    "human competitor (e.g. \"Find the exact clip where...\", \"The video describes...\", "
    "\"This is the moment when...\"). Rewrite each one as a SHORT, single-sentence English "
    "image caption describing only the single most distinctive visual moment -- the way a "
    "photo caption or alt-text would read, not an instruction.\n"
    "Rules:\n"
    "- Drop ALL meta-commentary and instructional framing (\"find\", \"identify\", \"the video "
    "shows/describes\", \"this is the moment when\", question phrasing).\n"
    "- Keep concrete, searchable visual details: object/person counts, colors, actions, "
    "clothing, on-screen text, proper nouns, spatial relations, numbers.\n"
    "- If the original mentions multiple moments/details, keep only the most distinctive and "
    "rare ones (rare colors/objects/text beat generic scene-setting) so the result stays under "
    "~35 English words.\n"
    "- Do not answer questions, explain, or add information not present in the original.\n"
    "- Keep each id unchanged.\n\n"
)

SHORTEN_PROMPT = (
    "These English image captions for a CLIP-style retrieval model are still too long "
    "(over the text tower's token budget). Shorten each one further to at most 20 English "
    "words, keeping only the single most distinctive, rare visual detail (a specific color, "
    "count, object, on-screen text, or proper noun) plus its immediate action. Drop everything "
    "generic. Keep each id unchanged.\n\n"
)

SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
        "required": ["id", "text"],
        "additionalProperties": False,
    },
}


def call_gemini(client, prompt_prefix, payload):
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_prefix + json.dumps(payload, ensure_ascii=False),
        config={"temperature": 0.0, "thinking_config": {"thinking_level": "minimal"},
                "response_mime_type": "application/json", "response_json_schema": SCHEMA},
    )
    rows = json.loads(response.text)
    if [row["id"] for row in rows] != [item["id"] for item in payload]:
        raise RuntimeError("translated IDs changed/reordered in a batch")
    return {row["id"]: row["text"] for row in rows}


def token_lengths(texts):
    import open_clip
    tokenizer = open_clip.get_tokenizer("hf-hub:timm/PE-Core-bigG-14-448")
    tokens = tokenizer(texts)
    return [int((row != 0).sum()) for row in tokens]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/official_benchmark_answers_v2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/official_benchmark_queries_en_v2.json")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    source = json.loads(args.input.read_text())
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cases = source["cases"]
    texts = {case["id"]: case["text"] for case in cases}

    for start in range(0, len(cases), args.batch_size):
        batch = cases[start:start + args.batch_size]
        payload = [{"id": case["id"], "text": case["text"]} for case in batch]
        result = call_gemini(client, TRANSLATE_PROMPT, payload)
        texts.update(result)
        print(f"translated {min(start + args.batch_size, len(cases))}/{len(cases)}", flush=True)

    for attempt in range(args.max_retries):
        ids = list(texts.keys())
        lengths = token_lengths([texts[i] for i in ids])
        over = [i for i, length in zip(ids, lengths) if length >= TARGET_TOKENS]
        if not over:
            break
        print(f"pass {attempt + 1}: {len(over)} cases still >= {TARGET_TOKENS} tokens, shortening...", flush=True)
        for start in range(0, len(over), args.batch_size):
            batch_ids = over[start:start + args.batch_size]
            payload = [{"id": i, "text": texts[i]} for i in batch_ids]
            result = call_gemini(client, SHORTEN_PROMPT, payload)
            texts.update(result)

    final_ids = list(texts.keys())
    final_lengths = dict(zip(final_ids, token_lengths([texts[i] for i in final_ids])))
    still_over = [i for i in final_ids if final_lengths[i] >= TARGET_TOKENS]
    if still_over:
        print(f"WARNING: {len(still_over)} cases still >= {TARGET_TOKENS} tokens after "
              f"{args.max_retries} shortening passes: {still_over}")

    translated = [{**case, "text_vi": case["text"], "text": texts[case["id"]],
                  "tokens_pe_core": final_lengths[case["id"]]} for case in cases]
    output = {**source, "source": str(args.input.relative_to(ROOT)),
              "query_language": "English (caption-style, meta-commentary stripped)",
              "translation_model": MODEL, "target_tokens": TARGET_TOKENS, "cases": translated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(translated)} cases -> {args.output}")


if __name__ == "__main__":
    main()
