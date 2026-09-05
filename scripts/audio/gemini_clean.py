"""Vietnamese transcript cleaning via the Gemini API.

Same job as :mod:`scripts.audio.capu` — restore capitalisation and punctuation
over a timed word stream — but conservatively more capable: Gemini can also
fix an obvious ASR mistranscription (a word that makes no sense in context),
not just add case and punctuation. It is a drop-in alternative to
``capu_model`` in :class:`scripts.audio.pipeline.CleanConfig`, not a new
pipeline stage.

**Free-form prompting, not word-constrained JSON.** An earlier version of
this module asked Gemini for a JSON array of exactly one string per input
word, so restoration could be applied index-by-index the same way
:func:`scripts.audio.postprocess.apply_capu_output` applies capu's output.
That constraint measurably hurt quality — a model asked to fill exactly N
independent JSON slots gets historical names and set phrases wrong far more
often than the same model asked to just rewrite the passage as prose, even
though nothing else about the prompt changed. So this module asks for free
prose instead, then reconstructs word timing with :func:`realign_words`: a
diff between the raw and cleaned word streams (after folding away diacritics
and punctuation, since Gemini adds both) that copies each unchanged word's
real offset across and interpolates a plausible one for anything Gemini
edited. This is deliberately approximate — it cannot be exact, because a
free-form rewrite no longer has a 1:1 word correspondence to the input — but
it is close enough for subtitle cues and passage boundaries, both of which
already tolerate second-level slop elsewhere in this codebase.

Unlike ``capu.py``, this is not a local model: every chunk is one Gemini API
call, so a whole corpus run is bounded by network latency and API rate limits
rather than GPU memory. Two things follow from that:

* **Bounded concurrency**, not GPU batching — chunks are dispatched across a
  small thread pool, with retry/backoff on transient errors.
* **A resumable on-disk cache**, keyed by a content hash of the request, so a
  killed run (or a deliberate ``--videos`` pilot subset) never re-pays for a
  chunk it already has an answer for. Same convention as
  ``scripts.benchmarks.bench_audio_search``'s ``.bench_embeddings`` cache: a
  gitignored directory under the data mount, not the repo.

Requires ``GEMINI_API_KEY`` in the environment. This is a dev-machine-only
secret for offline scripts — never part of ``.env.secure`` (the running app's
access-control config) and never read by anything under the Docker image.
Get a key at https://aistudio.google.com/apikey and run:

    export GEMINI_API_KEY=...
    uv run python -m scripts.audio.build_clean_transcripts --gemini-model gemini-flash-lite-latest

This is the only module in :mod:`scripts.audio` that talks to a model over
the network, so everything else stays testable offline (mirrors how
``capu.py`` is the only module that loads a local model).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import difflib
import hashlib
import os
from pathlib import Path
import random
import re
import threading
import time

from scripts.audio.postprocess import Word
from scripts.ocr.scoring import fold

ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data")) / "2026" / "audio"

#: Per-chunk restored text, content-addressed. Gitignored data mount, not the
#: Hugging Face cache and not the repo — see module docstring.
GEMINI_CACHE_DIR = Path(os.environ.get(
    "AIC_GEMINI_CACHE_DIR", AUDIO_DIR / ".gemini_cache"
))

DEFAULT_MODEL = "gemini-flash-lite-latest"

#: Words per request. gemini-flash-lite-latest's output limit is 65,536
#: tokens (checked against the live API), and Vietnamese prose runs at most
#: a few tokens per word, so this comfortably covers every video in the 2026
#: corpus in one call — the longest transcript is 10,653 words. One call
#: per video, not several, is what fixed the earlier version's habit of
#: losing track of a name it had already resolved a paragraph earlier (see
#: module docstring): a video split across chunks re-decides ambiguous names
#: independently in each piece, with no memory of what an earlier piece
#: picked. If a future corpus has a longer outlier, it still degrades
#: gracefully — that video splits and loses a little cross-chunk
#: consistency, which is not something this module tries to fully solve.
CHUNK_WORDS = 12_000

#: Bounded parallel HTTP calls, not GPU batching — see module docstring.
#: Kept modest by default because the Gemini free tier's per-minute quota
#: (15 requests/minute for gemini-flash-lite-latest, as of this writing) is
#: shared across every in-flight thread — a higher-tier key can raise this
#: via ``--gemini-concurrency``.
CONCURRENCY = 4

#: 429s are expected and routine on the free tier, not exceptional — each one
#: costs one wait (honouring the server's own ``retryDelay`` hint below, see
#: ``_retry_delay_seconds``), so the budget needs enough attempts to sit out
#: a full quota window, not just a couple of quick network blips.
MAX_RETRIES = 8
BASE_BACKOFF_SECONDS = 1.0

#: Below this fraction of matched (folded, punctuation-stripped) tokens
#: between the raw chunk and Gemini's response, the response is treated as
#: broken rather than trusted — see :func:`realign_words`. A real cleaning
#: pass, however heavily it corrects individual words, still shares most of
#: its vocabulary with the input; a response this different from the input
#: is far more likely a truncated, off-topic, or otherwise mangled answer.
MIN_SIMILARITY_RATIO = 0.5

_PROMPT = (
    "You are cleaning a Vietnamese speech-to-text transcript for a search "
    "index. Below is the raw transcript: no punctuation, no capitalisation, "
    "and occasional speech-recognition mistakes.\n\n"
    "Rewrite it as clean, natural Vietnamese prose:\n"
    "- Add correct punctuation and capitalisation.\n"
    "- Fix an obvious speech-recognition mistranscription (a word that makes "
    "no sense in context) with the word that was almost certainly spoken. "
    "If you are unsure, leave it as-is rather than guessing.\n"
    "- Preserve the original wording, meaning, and order otherwise. This is "
    "a correction pass, not a summary or a creative rewrite: do not add "
    "information that is not already there, and do not remove any content.\n"
    "- Output plain prose only: no markdown formatting (no asterisks, no "
    "bullet points, no headings), no speaker labels, no added structure "
    "beyond ordinary paragraph breaks.\n\n"
    "Raw transcript:\n{text}"
)


def _retryable_errors() -> tuple[type[BaseException], ...]:
    """Resolved lazily so importing this module never requires the SDK.

    ``google.genai.errors.APIError`` is the base for both ``ClientError``
    (4xx, including 429 rate limits) and ``ServerError`` (5xx) and carries a
    ``.code``, so catching just it and filtering in :func:`_is_retryable`
    covers every HTTP-shaped failure without needing the subclasses too.
    """
    errors: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        from google.genai import errors as genai_errors

        errors.append(genai_errors.APIError)
    except ImportError:
        pass
    return tuple(errors)


def _is_retryable(error: BaseException) -> bool:
    """A 429 or 5xx is worth retrying; a 400-class request error is not."""
    if isinstance(error, (TimeoutError, ConnectionError)):
        return True
    code = getattr(error, "code", None)
    if code is None:
        return True  # unknown shape: err on the side of retrying once more
    return code == 429 or code >= 500


class _RateLimiter:
    """Paces dispatch to at most ``requests_per_minute``, shared across threads.

    Reactive retry-on-429 (see :meth:`GeminiCleaner.restore_or_none`) is
    still the safety net, but relying on it alone wastes most of a free-tier
    run: every thread bursts as fast as it can, the burst blows through the
    per-minute quota almost immediately, and every request after that sits
    out a ~30-60s server-dictated wait it could have avoided by never
    bursting in the first place. Spacing dispatch to just under the quota
    keeps requests flowing continuously instead of in stop-and-wait bursts.

    A single shared "next allowed time", advanced under a lock each time a
    thread claims a slot, rather than a per-thread sleep — that is what
    keeps the *combined* rate across every thread at the target, regardless
    of how many threads are contending for slots.
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


class DailyQuotaExhausted(RuntimeError):
    """The API key's *daily* request quota is exhausted, not just per-minute.

    A per-minute 429 is worth waiting out — the window it names resets in
    seconds. A per-day 429 does not: the server's own ``retryDelay`` hint on
    one of these is typically ~60s, which is true of nothing about when the
    daily quota actually resets (Google's free tier resets it at midnight
    Pacific). Retrying it burns through every attempt in ``max_retries``
    doing nothing, and — worse — the caller that swallows the eventual
    ``None`` and moves on to the next video ends up silently leaving every
    remaining video in the corpus uncleaned instead of stopping. So this is
    raised instead of returned, and deliberately left to propagate all the
    way up through :func:`restore_words_gemini` and
    ``build_clean_transcripts.main`` uncaught: the run has to actually stop,
    visibly, so a human (or a script rotating in a fresh API key) notices.
    """


def _is_daily_quota_error(error: BaseException) -> bool:
    """Whether ``error`` names a ``...PerDay...`` quota, not a per-minute one."""
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return False
    body = details.get("error", details)
    entries = body.get("details") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for violation in entry.get("violations") or []:
            if not isinstance(violation, dict):
                continue
            if "PerDay" in str(violation.get("quotaId", "")):
                return True
    return False


def _retry_delay_seconds(error: BaseException) -> float | None:
    """Pull the server's own ``retryDelay`` hint out of a 429/5xx, if present.

    A quota error's ``details`` carries a ``RetryInfo`` entry like
    ``{"@type": "...RetryInfo", "retryDelay": "29s"}`` — the API is telling
    us exactly how long its quota window has left, which is far more
    reliable than guessing with exponential backoff. Falls back to ``None``
    (caller picks its own backoff) for any error that doesn't carry one.
    """
    details = getattr(error, "details", None)
    if not isinstance(details, dict):
        return None
    body = details.get("error", details)
    entries = body.get("details") if isinstance(body, dict) else None
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("@type", "")).endswith("RetryInfo"):
            delay = entry.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                try:
                    return float(delay[:-1])
                except ValueError:
                    return None
    return None


class GeminiCleaner:
    """A loaded Gemini client. :meth:`restore` takes plain text and returns it
    as cleaned prose, same shape as ``CapuModel.restore`` — but see the
    module docstring: unlike capu's output, this is not word-for-word
    aligned to the input, so callers go through :func:`realign_words` rather
    than applying the result by word index.

    Never raises — an unrecoverable failure (exhausted retries, empty
    response) returns the input unchanged, which :func:`realign_words` turns
    into a safe no-op (aligning text against itself is exact).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, *, api_key: str | None = None,
                 cache_dir: Path | None = None, concurrency: int = CONCURRENCY,
                 max_retries: int = MAX_RETRIES, timeout: float = 30.0,
                 requests_per_minute: float | None = None):
        self.model_id = model_id
        self.api_key = api_key
        self.cache_dir = Path(cache_dir) if cache_dir else GEMINI_CACHE_DIR
        self.concurrency = concurrency
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None
        self._rate_limiter = _RateLimiter(requests_per_minute)

    def load(self) -> "GeminiCleaner":
        from google import genai

        key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set; see scripts/audio/gemini_clean.py"
            )
        self._client = genai.Client(api_key=key)
        return self

    def restore_or_none(self, text: str) -> str | None:
        """One chunk, one (possibly retried) API call. ``None`` on failure.

        Free-form prose out, deliberately not constrained to a per-word JSON
        schema — see the module docstring for why. ``thinking_level:
        "minimal"`` and ``temperature: 0`` were what actually reproduced
        AI-Studio-quality output in manual testing during development; both
        matter (a higher thinking budget was not the difference, and the
        constrained-JSON version was worse at every thinking level tried).

        Returning ``None`` rather than the raw input on failure — unlike
        :meth:`restore` — matters to :func:`restore_words_gemini`: it is how
        that function tells a genuine restoration apart from "gave up", so
        only genuine ones get cached and a rate-limited chunk is retried on
        the next run instead of being cached as a permanent no-op.

        Raises :class:`DailyQuotaExhausted` instead of returning ``None``
        when the key's *daily* quota (not per-minute) is exhausted — see
        that class's docstring for why this one failure mode does not get
        the same "fall back and keep going" treatment as every other one.
        """
        if self._client is None:
            self.load()
        if not text.strip():
            return text
        prompt = _PROMPT.format(text=text)
        retryable = _retryable_errors()
        for attempt in range(self.max_retries):
            try:
                self._rate_limiter.wait()
                response = self._client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config={
                        "temperature": 0.0,
                        "thinking_config": {"thinking_level": "minimal"},
                    },
                )
                restored = (response.text or "").strip()
                return restored or text
            except retryable as error:
                if _is_daily_quota_error(error):
                    raise DailyQuotaExhausted(str(error)) from error
                if not _is_retryable(error):
                    print(f"[gemini-clean] giving up, keeping raw: {error}", flush=True)
                    return None
                if attempt == self.max_retries - 1:
                    print(f"[gemini-clean] giving up after {self.max_retries} "
                          f"attempts, keeping raw: {error}", flush=True)
                    return None
                delay = _retry_delay_seconds(error)
                if delay is None:
                    delay = BASE_BACKOFF_SECONDS * (2 ** attempt)
                time.sleep(delay + random.uniform(0, 1.0))
            except Exception as error:  # unexpected shape from the SDK
                print(f"[gemini-clean] non-retryable error, keeping raw: {error}", flush=True)
                return None
        return None

    def restore(self, text: str) -> str:
        """:meth:`restore_or_none`, falling back to the input on failure."""
        restored = self.restore_or_none(text)
        return text if restored is None else restored

    def restore_many(self, texts: list[str]) -> list[str]:
        """Sequential convenience wrapper mirroring ``CapuModel.restore_many``.

        :func:`restore_words_gemini` does not call this — it drives
        :meth:`restore_or_none` directly under its own thread pool so
        concurrency is controlled in exactly one place.
        """
        return [self.restore(t) for t in texts]


def _cache_key(model_id: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()[:16]


def _read_cache(cache_dir: Path, key: str) -> str | None:
    path = cache_dir / f"{key}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


def _write_cache(cache_dir: Path, key: str, text: str) -> None:
    (cache_dir / f"{key}.txt").write_text(text, encoding="utf-8")


_NOT_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _align_key(token: str) -> str:
    """Diacritic- and punctuation-blind form of one token, for diffing only.

    Folding away diacritics matters because Gemini's own corrections
    sometimes only change diacritics (ASR drops them constantly); stripping
    punctuation matters because Gemini adds it and the raw side never had
    any. Two different words can still fold to the same key by coincidence,
    but :func:`difflib.SequenceMatcher` only needs "probably the same word",
    not a guarantee — this is a heuristic for placing a timestamp, not a
    correctness-critical comparison.
    """
    return _NOT_WORD.sub("", fold(token))


def _tokenize_cleaned_text(text: str) -> list[str]:
    """Split Gemini's prose into tokens, dropping stray markdown decoration.

    The prompt asks for plain prose, but a token that is pure punctuation or
    markdown marks left over from a slip (a lone ``*``, ``-``, ``--``) would
    otherwise become a zero-length :func:`_align_key` that spuriously
    "matches" every other punctuation-only token — dropped here instead.
    """
    return [token for token in text.split() if _align_key(token)]


def realign_words(raw_words: list[Word], cleaned_text: str) -> list[Word]:
    """Rebuild timed words from Gemini's free-form rewrite of ``raw_words``.

    Diffs the two word streams by :func:`_align_key` (folded, punctuation
    stripped) rather than raw text, since Gemini's whole job is to change
    case, add punctuation, and occasionally swap a word for the one that was
    actually spoken. Each ``equal`` block copies the matching raw word's
    ``offset``/``nll`` straight across; anything Gemini changed gets an
    offset linearly interpolated between the nearest unchanged neighbours on
    either side (or pinned to whichever neighbour exists, at either end of
    the chunk) and the worst ``nll`` of whatever raw span it replaced — the
    same "worst confidence of the run it replaces" convention
    :func:`scripts.audio.postprocess.normalize_numbers` already uses for
    collapsing several raw words into one.

    Falls back to ``raw_words`` unchanged — the same "discard rather than
    trust it" bargain :func:`scripts.audio.postprocess.apply_capu_output`
    strikes for capu — when the two streams share less than
    ``MIN_SIMILARITY_RATIO`` of their tokens, since a response that
    different from its input is far more likely broken than freely
    corrected.
    """
    if not raw_words:
        return []
    cleaned_tokens = _tokenize_cleaned_text(cleaned_text)
    if not cleaned_tokens:
        return list(raw_words)

    raw_keys = [_align_key(word.text) for word in raw_words]
    cleaned_keys = [_align_key(token) for token in cleaned_tokens]
    matcher = difflib.SequenceMatcher(None, raw_keys, cleaned_keys, autojunk=False)
    if matcher.ratio() < MIN_SIMILARITY_RATIO:
        return list(raw_words)

    result: list[Word] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset, cleaned_index in enumerate(range(j1, j2)):
                raw_word = raw_words[i1 + offset]
                result.append(Word(cleaned_tokens[cleaned_index], raw_word.offset, raw_word.nll))
            continue
        if tag == "delete":
            continue
        # "replace" or "insert": j1:j2 cleaned tokens have no exact raw match.
        before = raw_words[i1 - 1] if i1 > 0 else None
        after = raw_words[i2] if i2 < len(raw_words) else None
        replaced = raw_words[i1:i2]
        if replaced:
            gap_nll = max(word.nll for word in replaced)
        elif before is not None:
            gap_nll = before.nll
        elif after is not None:
            gap_nll = after.nll
        else:
            gap_nll = float("inf")
        span = j2 - j1
        for position, cleaned_index in enumerate(range(j1, j2)):
            if before is not None and after is not None:
                offset = before.offset + (after.offset - before.offset) * (position + 1) / (span + 1)
            elif before is not None:
                offset = before.offset
            elif after is not None:
                offset = after.offset
            else:
                offset = 0.0
            result.append(Word(cleaned_tokens[cleaned_index], offset, gap_nll))
    return result


def restore_words_gemini(words: list[Word], cleaner: GeminiCleaner, *,
                          chunk_words: int = CHUNK_WORDS) -> list[Word]:
    """Restore one video's words via Gemini, chunk by chunk, resumably.

    Each chunk is looked up in ``cleaner.cache_dir`` by a content hash of
    ``(model_id, chunk text)`` before any network call is made, so re-running
    after an interruption only pays for chunks that never got a *successful*
    response. Uncached chunks are dispatched across ``cleaner.concurrency``
    threads, and each chunk's cache file is written the moment its own
    request lands — a killed run leaves a valid, reusable cache behind
    rather than losing everything.

    Only a genuine restoration is cached — a chunk :meth:`GeminiCleaner.
    restore_or_none` gave up on (rate-limited, malformed output) is *not*
    written to cache, so it is retried on the next run instead of being
    permanently frozen as an unrestored no-op.

    Every chunk goes through :func:`realign_words` regardless: caching
    Gemini's raw prose response (not the post-alignment word list) means a
    chunk is re-aligned at full strength if the alignment logic ever
    changes, without re-billing the API call.

    Propagates :class:`DailyQuotaExhausted` rather than catching it — a
    worker thread that hits it re-raises out of ``future.result()``, which
    stops this video (and, left uncaught by every caller up to
    ``build_clean_transcripts.main``, the whole run) rather than quietly
    falling back to raw text for it and everything after it.
    """
    if not words:
        return []
    cache_dir = cleaner.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = [words[i:i + chunk_words] for i in range(0, len(words), chunk_words)]
    requests = [" ".join(w.text for w in chunk) for chunk in chunks]
    keys = [_cache_key(cleaner.model_id, text) for text in requests]

    restored_texts: list[str | None] = [None] * len(chunks)
    pending = []
    for index, key in enumerate(keys):
        cached = _read_cache(cache_dir, key)
        if cached is not None:
            restored_texts[index] = cached
        else:
            pending.append(index)

    def work(index: int) -> tuple[int, str | None, bool]:
        restored = cleaner.restore_or_none(requests[index])
        if restored is not None:
            _write_cache(cache_dir, keys[index], restored)
        return index, restored, restored is not None

    if pending:
        gave_up = 0
        with ThreadPoolExecutor(max_workers=cleaner.concurrency) as pool:
            futures = [pool.submit(work, i) for i in pending]
            for done, future in enumerate(as_completed(futures), 1):
                index, restored, ok = future.result()
                restored_texts[index] = restored
                gave_up += not ok
                if done % 20 == 0:
                    print(f"[gemini-clean] {done}/{len(pending)} chunks "
                          f"restored (this video)", flush=True)
        if gave_up:
            print(f"[gemini-clean] {gave_up}/{len(pending)} chunks kept raw "
                  f"this run (will retry next run)", flush=True)

    restored: list[Word] = []
    for chunk, text in zip(chunks, restored_texts):
        if text is None:
            restored.extend(chunk)
        else:
            restored.extend(realign_words(chunk, text))
    return restored
