"""Topic/story segmentation of Vietnamese transcripts via the DeepSeek API.

An alternative to ``backend.audio_search``'s embedding-change-score chunking
(``semantic_duration``): instead of deciding a chunk boundary from a
duration-bounded local maximum of embedding change, an LLM is shown a
video's sentences and asked where one topic/story ends and an unrelated one
begins -- e.g. a TV news bulletin cutting from one story to a completely
different one. This was written after ``semantic_duration``'s "growth zone"
rule (no boundary before ``target_duration_seconds``, however strong) was
found to merge two unrelated news stories on a real video (``M07_V008``),
diluting the embedding of the passage a competition query needed to match.

**The LLM never sees or produces timestamps.** It only picks sentence-index
boundaries over a numbered list of sentence texts (see :func:`build_prompt`);
``backend.audio_search._stitch_topic_boundaries``/``make_topic_passages``
always derive ``start_sec``/``end_sec`` from the sentences' own already-known
offsets. This module supplies the callback (``DeepSeekTopicSegmenter``,
matching ``backend.audio_search.LlmSegmentFn``'s contract) that
``make_topic_passages`` is injected with -- ``backend/audio_search.py``
itself never imports DeepSeek, mirroring how the two existing semantic
chunkers take an injected ``embed_cues`` rather than importing a model.

Like :mod:`scripts.audio.gemini_clean`, this is a network-bound, API-call-
per-batch script, not GPU batching:

* **Bounded concurrency and proactive rate limiting** are the caller's job
  (see ``build_segmenter``'s ``requests_per_minute``); this module's
  ``DeepSeekClient`` (``agent.llm.DeepSeekClient``) already retries 429/5xx
  with backoff on top of that.
* **A resumable on-disk cache**, keyed by a content hash of
  ``(model, prompt, sentences)``, so a benchmark sweep that calls the same
  video's segmentation repeatedly (once per ``--lexical``/``--dense`` run)
  never re-pays for it, and a killed corpus run resumes cheaply. Only a
  *validated* response is cached -- a rejected or failed one is retried on
  the next call instead of being frozen as "no boundaries found" forever.

Requires ``DEEPSEEK_API_KEY`` in the environment -- a dev-machine-only secret
for offline scripts, never part of ``.env.secure`` and never read inside the
Docker image. Pilot a few videos before a full-corpus run:

    export DEEPSEEK_API_KEY=...
    uv run python -m scripts.audio.deepseek_topic_segment --videos M07_V008
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data")) / "2026" / "audio"

#: Per-window boundary decisions, content-addressed. Gitignored data mount,
#: not the repo -- see module docstring.
DEEPSEEK_TOPIC_CACHE_DIR = Path(os.environ.get(
    "AIC_DEEPSEEK_TOPIC_CACHE_DIR", AUDIO_DIR / ".deepseek_topic_cache"
))

DEFAULT_MODEL = "deepseek-flash"
DEFAULT_CONCURRENCY = 4

#: Deliberately written as a short list of permitted decisions followed by an
#: explicit list of forbidden ones -- the same convention
#: ``scripts.audio.gemini_clean``'s prompt uses, and for the same reason: a
#: single caveat buried in a paragraph is followed far less reliably than an
#: itemised "must NOT" list. Changing this string invalidates every cached
#: response automatically, via :func:`prompt_fingerprint`.
_PROMPT_HEADER = (
    "You are splitting a Vietnamese speech transcript into segments for a "
    "search index that embeds one segment per passage. A passage that runs "
    "on too long -- even about one thing -- makes its own embedding vague "
    "and hurts search, so every long stretch needs internal boundaries, not "
    "just the obvious ones.\n\n"
    "Two kinds of boundary both count:\n"
    "1. TOPIC boundary: one story/topic ends and a new, unrelated one "
    "begins -- e.g. a TV news bulletin cutting from one story to a "
    "completely different one.\n"
    "2. BEAT boundary: a natural internal shift WITHIN one continuous "
    "topic or event -- live commentary on a single race, a lecture, a long "
    "interview, a monologue. A new phase of the event, a change of focus "
    "(a different competitor, a different sub-point), a change of speaker, "
    "or a pause all count. Mark a beat boundary roughly every 15-30 "
    "sentences of continuous commentary or narration, so a long excerpt is "
    "never returned as a single unbroken block just because it never "
    "changes topic.\n\n"
    "You must NOT do any of the following:\n"
    "- Do not invent a boundary in the middle of one sentence's idea or a "
    "single short exchange -- both kinds of boundary must fall at a "
    "genuine shift, however small.\n"
    "- Do not output timestamps, explanations, or markdown.\n"
    "- Do not invent sentence indices outside [0, {max_index}].\n"
    "- Indices must be strictly increasing, with no duplicates.\n\n"
    "Each boundary index is the index of the LAST sentence before a shift "
    "-- topic or beat. Only return an empty list if the entire excerpt "
    "truly has no internal shift of any kind at all; that should be rare "
    "for an excerpt this long.\n\n"
    "Numbered sentences:\n{numbered}"
)

_SCHEMA = {
    "type": "object",
    "properties": {"boundaries": {"type": "array", "items": {"type": "integer"}}},
    "required": ["boundaries"],
}


def prompt_fingerprint() -> str:
    """Short hash of the instruction, so a prompt change invalidates the cache.

    Same rationale as ``scripts.audio.gemini_clean.prompt_fingerprint``: a
    cache entry is keyed by content, and the prompt is as much a part of
    that content as the sentences are.
    """
    return hashlib.sha256(_PROMPT_HEADER.encode("utf-8")).hexdigest()[:8]


def build_prompt(sentences: list[str]) -> str:
    from agent.vlm import format_instruction

    numbered = "\n".join(f"{index}: {text}" for index, text in enumerate(sentences))
    header = _PROMPT_HEADER.format(max_index=max(0, len(sentences) - 2), numbered=numbered)
    return header + "\n\n" + format_instruction(_SCHEMA)


class _RateLimiter:
    """Paces dispatch to at most ``requests_per_minute``, shared across threads.

    Identical in spirit to ``scripts.audio.gemini_clean._RateLimiter``
    (provider-agnostic "next allowed time" pacer) -- copied rather than
    imported to keep this module's only coupling to ``gemini_clean`` at zero,
    since the two providers' retry/quota-error shapes are otherwise unrelated.
    """

    def __init__(self, requests_per_minute: float | None):
        self._min_interval = None if not requests_per_minute else 60.0 / requests_per_minute
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self._min_interval is None:
            return
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_allowed)
            self._next_allowed = start + self._min_interval
        delay = start - now
        if delay > 0:
            time.sleep(delay)


def segment_batch(client, sentences: list[str], *, max_retries: int = 1) -> list[int] | None:
    """One DeepSeek call, validated.

    Returns the validated boundary list (possibly empty -- "no topic change
    in this window" is a legitimate answer) on success, or ``None`` when
    every attempt failed or came back malformed. Callers must not cache a
    ``None`` result: a failed/rejected batch is meant to be retried on the
    next call, not permanently frozen as "no boundaries found".
    """
    from backend.audio_search import _validate_topic_boundaries

    if len(sentences) < 2:
        return []
    prompt = build_prompt(sentences)
    last_raw = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat_json([{"role": "user", "content": prompt}], salvage=True)
        except Exception as exc:
            if attempt == max_retries:
                print(f"[deepseek-topic] giving up after {max_retries + 1} "
                      f"attempts: {exc}", flush=True)
                return None
            continue
        boundaries = response.get("boundaries") if isinstance(response, dict) else None
        validated = _validate_topic_boundaries(boundaries, len(sentences))
        if validated is not None:
            return validated
        last_raw = boundaries
    print(f"[deepseek-topic] rejecting malformed response: {last_raw!r}", flush=True)
    return None


def _cache_key(model_id: str, sentences: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_fingerprint().encode("utf-8"))
    digest.update(b"\0")
    # \x1f (unit separator) is vanishingly unlikely to appear in transcript
    # text, unlike a space or newline, so it can't shift a boundary between
    # adjacent sentences into a collision with a different batch.
    digest.update("\x1f".join(sentences).encode("utf-8"))
    return digest.hexdigest()[:16]


def _read_cache(cache_dir: Path, key: str) -> list[int] | None:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(cache_dir: Path, key: str, boundaries: list[int]) -> None:
    (cache_dir / f"{key}.json").write_text(json.dumps(boundaries), encoding="utf-8")


class DeepSeekTopicSegmenter:
    """A loaded DeepSeek client + on-disk cache, callable as an
    ``backend.audio_search.LlmSegmentFn``.

    Caching is required, not optional, here: the benchmark harness calls this
    repeatedly across ``--lexical``/``--dense`` runs over the same
    ground-truth videos, and a full corpus is hundreds of videos. See
    :func:`segment_batch` for why only a validated response is cached.
    """

    def __init__(self, client, *, cache_dir: Path | None = None,
                 rate_limiter: "_RateLimiter | None" = None):
        self._client = client
        self.cache_dir = Path(cache_dir) if cache_dir else DEEPSEEK_TOPIC_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limiter = rate_limiter

    def __call__(self, sentences: list[str]) -> list[int]:
        if len(sentences) < 2:
            return []
        key = _cache_key(self._client.model, sentences)
        cached = _read_cache(self.cache_dir, key)
        if cached is not None:
            return cached
        if self._rate_limiter is not None:
            self._rate_limiter.wait()
        boundaries = segment_batch(self._client, sentences)
        if boundaries is None:
            return []
        _write_cache(self.cache_dir, key, boundaries)
        return boundaries


def build_segmenter(*, model: str | None = None, api_key: str | None = None,
                     base_url: str | None = None, cache_dir: Path | None = None,
                     requests_per_minute: float | None = None) -> DeepSeekTopicSegmenter:
    """Construct a ready-to-call segmenter from env-or-argument DeepSeek config.

    Same env vars ``agent/config.py`` already reads (``DEEPSEEK_API_KEY``,
    ``DEEPSEEK_MODEL``, ``DEEPSEEK_BASE_URL``), so a machine already set up
    to run the retrieval agent needs no new secret for this.
    """
    from agent.llm import DeepSeekClient

    key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not set; see scripts/audio/deepseek_topic_segment.py"
        )
    client = DeepSeekClient(
        key,
        model=model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL),
        base_url=base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        # A boundary list is a bounded classification, not a task that
        # benefits from a reasoning budget -- same call as `judge()`'s own
        # `thinking=False` for a bounded 0-10 score (agent/llm.py).
        thinking=False,
    )
    return DeepSeekTopicSegmenter(
        client, cache_dir=cache_dir, rate_limiter=_RateLimiter(requests_per_minute)
    )


def main():
    import argparse

    from backend import audio_search
    from scripts.audio.pipeline import load_words
    from scripts.audio.postprocess import words_to_tokens

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=None,
                        help="step_data parquet; defaults to audio_search.transcript_path()")
    parser.add_argument("--videos", default=None,
                        help="comma-separated audio_key allowlist, e.g. "
                             "M07_V008,L21_V015 -- segment only these videos "
                             "instead of the whole corpus (for pilot runs)")
    parser.add_argument("--deepseek-model", default=None)
    parser.add_argument("--deepseek-base-url", default=None)
    parser.add_argument("--deepseek-api-key", default=None,
                        help="falls back to the DEEPSEEK_API_KEY env var")
    parser.add_argument("--deepseek-rpm", type=float, default=None,
                        help="cap dispatch to this many requests/minute; "
                             "omit for reactive-only throttling")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--sentences-per-batch", type=int, default=60)
    parser.add_argument("--batch-overlap-sentences", type=int, default=10)
    args = parser.parse_args()

    source = args.source or audio_search.transcript_path()
    raw = load_words(source)
    if args.videos:
        requested = {video_id.strip() for video_id in args.videos.split(",") if video_id.strip()}
        missing = requested - raw.keys()
        if missing:
            raise SystemExit(f"--videos names videos not in {source.name}: {sorted(missing)}")
        raw = {key: words for key, words in raw.items() if key in requested}

    segmenter = build_segmenter(
        model=args.deepseek_model, api_key=args.deepseek_api_key,
        base_url=args.deepseek_base_url, cache_dir=args.cache_dir,
        requests_per_minute=args.deepseek_rpm,
    )

    batch = max(2, args.sentences_per_batch)
    overlap = max(0, min(args.batch_overlap_sentences, batch - 1))
    step = max(1, batch - overlap)

    for video_id, words in sorted(raw.items()):
        rows = words_to_tokens(video_id, words)
        sentences = audio_search._merge_cue_fragments_into_sentences(
            audio_search.build_transcript_cues(rows)
        )
        if len(sentences) <= batch:
            window_count = 1
        else:
            window_count = len(range(0, len(sentences) - batch, step)) + 1
        boundaries = audio_search._stitch_topic_boundaries(
            sentences, segmenter,
            sentences_per_batch=args.sentences_per_batch,
            batch_overlap_sentences=args.batch_overlap_sentences,
        )
        segments, start = [], 0
        for boundary in sorted(boundaries):
            segments.append(sentences[start:boundary + 1])
            start = boundary + 1
        segments.append(sentences[start:])
        print(f"[deepseek-topic] {video_id}: {len(sentences)} sentences, "
              f"{window_count} window(s), {len(boundaries)} boundaries "
              f"-> {len(segments)} segments")
        for segment in segments:
            duration = segment[-1]["end"] - segment[0]["start"]
            preview = segment[0]["text"][:60]
            print(f"    {segment[0]['start']:8.1f}-{segment[-1]['end']:8.1f} "
                  f"({duration:6.1f}s)  {preview}")


if __name__ == "__main__":
    main()
