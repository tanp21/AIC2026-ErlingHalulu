#!/usr/bin/env python3
"""Write cleaned transcript parquets beside the raw ones.

Reads ``data/2026/audio/step_data.parquet``, restores capitalisation and
punctuation via the Gemini API (:mod:`scripts.audio.gemini_clean`), and writes
``step_data_clean.parquet`` and ``transcription_clean.parquet`` in the same
schema. Raw files are never touched, so a bad run is undone by deleting the
output.

Same schema is the point: ``backend.audio_search.make_passages`` and
``backend.audio_search.build_transcript_cues`` consume the cleaned rows unchanged, and
``backend.audio_search.TRANSCRIPT_PATH`` falls back to the raw file when the clean one
is absent. Nothing downstream has to know which it got.

    export GEMINI_API_KEY=...
    uv run python -m scripts.audio.build_clean_transcripts --gemini-model gemini-flash-lite-latest

Needs the network and a Gemini API key; every other step is pure Python. Run
this on the dev machine, never in the container.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audio.pipeline import CleanConfig, clean_corpus, load_words
from scripts.audio.postprocess import words_to_tokens

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT / "data" / "2026" / "audio"
RAW_STEP_DATA = AUDIO_DIR / "step_data.parquet"
#: Output lands in a staging directory, not beside the raw parquet the runtime
#: reads. Generating cleaned transcripts must not by itself change what a
#: running server serves — point ``AIC_TRANSCRIPT_PATH`` at the file to adopt it.
GENERATED_DIR = AUDIO_DIR / "generated"
CLEAN_STEP_DATA = GENERATED_DIR / "step_data_clean.parquet"
CLEAN_TRANSCRIPTS = GENERATED_DIR / "transcription_clean.parquet"
CONFIG = ROOT / "benchmarks" / "audio" / "audio_config.json"


def load_config(path: Path) -> CleanConfig:
    """Read the configured pipeline, or fall back to an inert one."""
    if not path.exists():
        return CleanConfig()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return CleanConfig.from_json(payload.get("clean", {}))


def write(words_by_video, step_data_path: Path, transcript_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    step_data_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    keys, offsets, tokens, nlls = [], [], [], []
    transcript_keys, texts = [], []
    for video_id, words in sorted(words_by_video.items()):
        rows = words_to_tokens(video_id, words)
        for row in rows:
            keys.append(row["audio_key"])
            offsets.append(row["offset"])
            tokens.append(row["token"])
            nlls.append(row["log_prob"])
        transcript_keys.append(video_id)
        texts.append(" ".join(word.text for word in words))

    pq.write_table(
        pa.table({
            "audio_key": pa.array(keys, pa.large_string()),
            "offset": pa.array(offsets, pa.float64()),
            "token": pa.array(tokens, pa.large_string()),
            "log_prob": pa.array(nlls, pa.float64()),
        }),
        step_data_path,
    )
    pq.write_table(
        pa.table({
            "audio_key": pa.array(transcript_keys, pa.large_string()),
            "text": pa.array(texts, pa.large_string()),
        }),
        transcript_path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=RAW_STEP_DATA)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--step-data-out", type=Path, default=CLEAN_STEP_DATA)
    parser.add_argument("--transcripts-out", type=Path, default=CLEAN_TRANSCRIPTS)
    parser.add_argument("--videos", default=None,
                        help="comma-separated audio_key allowlist, e.g. "
                             "L27_V010,L23_V013 — clean only these videos "
                             "instead of the whole corpus (for pilot runs)")
    parser.add_argument("--gemini-model", default=None,
                        help="overrides --config's gemini_model for this run")
    parser.add_argument("--gemini-concurrency", type=int, default=None)
    parser.add_argument("--gemini-rpm", type=float, default=None,
                        help="cap dispatch to this many requests/minute across "
                             "all threads combined, e.g. 14 to stay just under "
                             "the free tier's 15/min; omit for reactive-only "
                             "throttling (retry after a 429)")
    parser.add_argument("--gemini-cache-dir", type=Path, default=None)
    parser.add_argument("--gemini-api-key", default=None,
                        help="falls back to the GEMINI_API_KEY env var")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.gemini_model:
        config = CleanConfig(gemini_model=args.gemini_model)
    print(f"[clean] config: {config.label}")

    raw = load_words(args.source)
    if args.videos:
        requested = {video_id.strip() for video_id in args.videos.split(",") if video_id.strip()}
        missing = requested - raw.keys()
        if missing:
            raise SystemExit(f"--videos names videos not in {args.source.name}: {sorted(missing)}")
        raw = {key: words for key, words in raw.items() if key in requested}
    before = sum(len(words) for words in raw.values())
    print(f"[clean] {len(raw)} videos, {before:,} words from {args.source.name}")

    gemini = None
    if config.gemini_model:
        from scripts.audio.gemini_clean import GeminiCleaner, CONCURRENCY

        print(f"[clean] loading gemini model {config.gemini_model}")
        gemini = GeminiCleaner(
            config.gemini_model,
            api_key=args.gemini_api_key,
            cache_dir=args.gemini_cache_dir,
            concurrency=args.gemini_concurrency or CONCURRENCY,
            requests_per_minute=args.gemini_rpm,
        ).load()

    cleaned = clean_corpus(raw, config, gemini=gemini)
    after = sum(len(words) for words in cleaned.values())
    print(f"[clean] {after:,} words after cleaning "
          f"({after - before:+,}, {after / before:.1%} of source)")

    write(cleaned, args.step_data_out, args.transcripts_out)
    print(f"[clean] wrote {args.step_data_out}")
    print(f"[clean] wrote {args.transcripts_out}")


if __name__ == "__main__":
    main()
