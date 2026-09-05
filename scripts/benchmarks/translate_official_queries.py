#!/usr/bin/env python3
"""Translate the frozen official benchmark prompts to English with Gemini."""
import argparse
import json
import os
from pathlib import Path

from google import genai

ROOT = Path(__file__).resolve().parents[2]
MODEL = "gemini-3.1-flash-lite"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/official_benchmark_answers.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/official_benchmark_queries_en.json")
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()
    source = json.loads(args.input.read_text())
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    translated = []
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "text": {"type": "string"}},
            "required": ["id", "text"],
            "additionalProperties": False,
        },
    }
    for start in range(0, len(source["cases"]), args.batch_size):
        batch = source["cases"][start:start + args.batch_size]
        payload = [{"id": case["id"], "text": case["text"]} for case in batch]
        response = client.models.generate_content(
            model=MODEL,
            contents=(
                "Translate each Vietnamese video-retrieval query into natural, precise English. "
                "Preserve every visible detail, action order, proper noun, number, question, and temporal cue. "
                "Do not answer questions, explain, summarize, or add information. Keep each id unchanged.\n\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
            config={"temperature": 0.0, "thinking_config": {"thinking_level": "minimal"},
                    "response_mime_type": "application/json", "response_json_schema": schema},
        )
        rows = json.loads(response.text)
        if [row["id"] for row in rows] != [case["id"] for case in batch]:
            raise RuntimeError(f"translated IDs changed in batch starting at {start}")
        translated.extend({**case, "text_vi": case["text"], "text": row["text"]}
                          for case, row in zip(batch, rows))
        print(f"translated {len(translated)}/{len(source['cases'])}", flush=True)
    output = {**source, "source": str(args.input.relative_to(ROOT)),
              "query_language": "English", "translation_model": MODEL, "cases": translated}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
