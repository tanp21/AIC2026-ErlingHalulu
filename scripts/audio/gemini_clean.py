"""Vietnamese transcript cleaning via the Gemini API.

Restores capitalisation and punctuation over a timed word stream, and
conservatively fixes an obvious ASR mistranscription (a word that makes no
sense in context) along the way. The sole restoration path in
:class:`scripts.audio.pipeline.CleanConfig` — an earlier local-model
alternative (a Vietnamese capitalisation/punctuation restorer, word-index
aligned) was retired in favour of this one.

**Free-form prompting, not word-constrained JSON.** An earlier version of
this module asked Gemini for a JSON array of exactly one string per input
word, so restoration could be applied index-by-index by position. That
constraint measurably hurt quality — a model asked to fill exactly N
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

This is not a local model: every chunk is one Gemini API call, so a whole
corpus run is bounded by network latency and API rate limits rather than GPU
memory. Two things follow from that:

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
the network, so everything else stays testable offline.
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

#: Per-request HTTP timeout, in seconds. **Must be set explicitly** — the SDK
#: builds its httpx client with ``timeout=None`` (no limit at all), so without
#: this a stalled connection blocks forever and :data:`MAX_RETRIES` never gets
#: a chance to fire, because a call that never returns never raises. A whole
#: corpus run was killed by exactly that: one request hung for 17+ minutes with
#: no error and no progress.
#:
#: Sized for the *largest* chunk, not the average one: ``generate_content`` is
#: not streamed here, so the read timeout covers the entire generation, and
#: ``CHUNK_WORDS`` allows a 12,000-word Vietnamese chunk (the longest response
#: seen in practice is 10,606 words, on an S01 video). A tight bound would time
#: out the real long-tail chunks rather than the stalled ones it is meant to
#: catch.
REQUEST_TIMEOUT_SECONDS = 180.0

#: Below this fraction of matched (folded, punctuation-stripped) tokens
#: between the raw chunk and Gemini's response, the response is treated as
#: broken rather than trusted — see :func:`realign_words`. A real cleaning
#: pass, however heavily it corrects individual words, still shares most of
#: its vocabulary with the input; a response this different from the input
#: is far more likely a truncated, off-topic, or otherwise mangled answer.
MIN_SIMILARITY_RATIO = 0.5

#: Below this fraction of the raw chunk's word count, a response is rejected
#: and the raw words are kept instead. Similarity alone cannot catch this: a
#: response that drops a whole stretch of the input still matches closely
#: everywhere it does exist, scoring ~0.63-0.88 — comfortably above
#: :data:`MIN_SIMILARITY_RATIO` — while realign then treats the missing
#: stretch as a ``delete`` opcode and drops those raw words outright. That is
#: real transcript content disappearing, not merely text left uncleaned.
#:
#: Measured, not guessed. Chunks a human read and judged faithful come back at
#: 98.6-99.8% of input length; every chunk that had lost real content landed
#: at 50.0-89.0%. Re-fetching the worst ones (S01-V007, 50.0% and 57.8%)
#: returned 98%+ responses that covered the whole chunk, confirming those were
#: a transient bad generation rather than something inherent to the input.
MIN_LENGTH_RATIO = 0.9

#: How many words per 1000 a response may *drop* (a raw word with no
#: counterpart in the response at all) before it is rejected. This is the
#: guard that encodes the actual definition of cleaning this corpus wants:
#: repair what the recogniser got wrong, **remove nothing**.
#:
#: :data:`MIN_LENGTH_RATIO` alone is not enough, because it compares whole
#: responses: a long video can lose a 668-word stretch and still come in at
#: 98.6% overall, so a localized removal sails through while the ratio looks
#: perfectly healthy. Measuring deletions directly catches it wherever it sits.
#:
#: 20 is set from the corpus's own distribution rather than taste. Words
#: deleted per 1000 across 525 long chunks: median 2.4, p90 8.9, p99 94.4. The
#: bulk is genuine repair — ASR stutters dropped ("khu công nghiệp **nghiệp**
#: bình đường"), mishearings swapped ("dù **giảm như** làm phát" -> "Dù **lạm
#: phát**", which difflib renders as delete+insert rather than a replacement).
#: The tail is content removal: university ad reads, song lyrics, repeated
#: formulas, at 49-196 per 1000. 20 sits above the repair noise floor and
#: below every confirmed removal, with room for the word-merging that legitimate
#: correction sometimes does.
MAX_DELETED_PER_1000_WORDS = 20.0

#: Absolute slack on top of :data:`MIN_LENGTH_RATIO`, in words. A pure ratio
#: misjudges small inputs: a 3-word chunk losing one word is a 33% shortfall
#: and would be rejected, even though a single dropped word is ordinary
#: correction (Gemini is told it may fix an obvious mistranscription) and not
#: remotely what a cut-off generation looks like. Truncation always loses a
#: lot, so requiring *both* a ratio breach and a shortfall larger than this
#: keeps the guard aimed at the real thing. At production chunk sizes
#: (~12,000 words) this slack is negligible.
MIN_TRUNCATION_SHORTFALL_WORDS = 20

#: The instruction Gemini actually receives. Deliberately written as a short
#: list of permitted edits followed by an explicit list of forbidden ones:
#: an earlier version only said "do not remove any content" once, buried in a
#: paragraph, and Gemini went on removing university ad reads, song lyrics and
#: repeated formulas anyway. The corpus's definition of cleaning is narrow —
#: repair what the recogniser got wrong, drop nothing — so the wording spells
#: out each removal it was caught doing, including the two transformations
#: that look like tidying but are content changes (digits, formula notation).
#:
#: Changing this string invalidates every cached response automatically, via
#: :func:`_cache_key` — otherwise a prompt change would silently keep serving
#: answers produced under the old one.
_PROMPT = (
    "You are correcting a Vietnamese speech-to-text transcript for a search "
    "index. The raw transcript has no punctuation, no capitalisation, and "
    "occasional speech-recognition mistakes.\n\n"
    "The ONLY changes you may make are:\n"
    "1. Add punctuation and paragraph breaks.\n"
    "2. Fix capitalisation.\n"
    "3. Correct a word the recogniser got wrong — wrong spelling, wrong tone "
    "mark, or an obvious mishearing. If you are not confident what was "
    "actually said, leave that word exactly as it is.\n\n"
    "You must NOT do any of the following, even when it looks like an "
    "improvement:\n"
    "- Do not delete, drop, skip, shorten or merge any word, phrase or "
    "sentence.\n"
    "- Do not remove repeated words, stutters, filler, false starts, "
    "advertisements, sponsor reads, song lyrics, or anything else you judge "
    "unimportant.\n"
    "- Do not merge passages that repeat or resemble each other.\n"
    "- Do not summarise, paraphrase, condense or reorder.\n"
    "- Do not convert spelled-out numbers into digits — write \"hai ngàn "
    "không trăm hai mươi tư\", never \"2024\".\n"
    "- Do not convert spoken words into symbols, abbreviations or formulas — "
    "write \"e mũ trừ lam nhân t\", never \"e^(-λt)\".\n"
    "- Do not add any information that is not in the input.\n\n"
    "Every word of the input must appear in your output, in the same order. "
    "The output must be at least as long as the input. If you are unsure "
    "whether a change is allowed, make no change.\n\n"
    "Output plain prose only: no markdown formatting (no asterisks, no "
    "bullet points, no headings), and no speaker labels.\n\n"
    "Raw transcript:\n{text}"
)


def _retryable_errors() -> tuple[type[BaseException], ...]:
    """Resolved lazily so importing this module never requires the SDK.

    ``google.genai.errors.APIError`` is the base for both ``ClientError``
    (4xx, including 429 rate limits) and ``ServerError`` (5xx) and carries a
    ``.code``, so catching just it and filtering in :func:`_is_retryable`
    covers every HTTP-shaped failure without needing the subclasses too.

    ``httpx``'s own transport errors are listed separately because they are
    **not** ``APIError`` and — checked, not assumed — ``httpx.ReadTimeout``
    inherits from neither the builtin ``TimeoutError`` nor ``ConnectionError``
    (its MRO is ``TimeoutException`` -> ``TransportError`` -> ``HTTPError`` ->
    ``Exception``). Without this a stalled connection would fall through to
    the bare ``except Exception`` below and be misread as a permanent,
    non-retryable failure — leaving the chunk raw for the rest of the run.
    """
    errors: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        import httpx

        errors.append(httpx.TransportError)
    except ImportError:
        pass
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
    as cleaned prose — see the module docstring: this is not word-for-word
    aligned to the input, so callers go through :func:`realign_words` rather
    than applying the result by word index.

    Never raises — an unrecoverable failure (exhausted retries, empty
    response) returns the input unchanged, which :func:`realign_words` turns
    into a safe no-op (aligning text against itself is exact).
    """

    def __init__(self, model_id: str = DEFAULT_MODEL, *, api_key: str | None = None,
                 cache_dir: Path | None = None, concurrency: int = CONCURRENCY,
                 max_retries: int = MAX_RETRIES, timeout: float = REQUEST_TIMEOUT_SECONDS,
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
        # http_options.timeout is in **milliseconds** (checked against the SDK,
        # not assumed). It has to be set here rather than per request:
        # generate_content in this SDK version takes no timeout argument, and
        # the client's own default is no timeout at all — see
        # REQUEST_TIMEOUT_SECONDS.
        self._client = genai.Client(
            api_key=key,
            http_options={"timeout": int(self.timeout * 1000)},
        )
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


def prompt_fingerprint() -> str:
    """Short hash of the instruction, so a prompt change invalidates the cache.

    Cache entries are keyed by content, and the prompt is as much a part of
    that content as the transcript is: the same chunk answered under a
    different instruction is a different response, and serving the old one
    would quietly defeat the change. Folding this into :func:`_cache_key`
    makes that automatic rather than something a future edit has to remember.
    """
    return hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()[:8]


def _cache_key(model_id: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(prompt_fingerprint().encode("utf-8"))
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


def _is_truncated(raw_word_count: int, cleaned_tokens: list[str]) -> bool:
    """Whether a response came back too short to be a real restoration.

    Both conditions must hold — see :data:`MIN_TRUNCATION_SHORTFALL_WORDS` for
    why the ratio alone is not enough on small inputs.
    """
    shortfall = raw_word_count - len(cleaned_tokens)
    if shortfall <= MIN_TRUNCATION_SHORTFALL_WORDS:
        return False
    return len(cleaned_tokens) < raw_word_count * MIN_LENGTH_RATIO


def _deleted_per_1000(matcher: difflib.SequenceMatcher, raw_word_count: int) -> float:
    """Words the response drops outright, per 1000 input words.

    Counts only ``delete`` opcodes — a raw word with no counterpart anywhere in
    the response. A corrected word shows up as ``replace`` (or as
    delete+insert, if the surrounding alignment shifted), and an added word as
    ``insert``; neither is a removal.
    """
    if not raw_word_count:
        return 0.0
    deleted = sum(
        i2 - i1 for tag, i1, i2, _j1, _j2 in matcher.get_opcodes() if tag == "delete"
    )
    return deleted / raw_word_count * 1000.0


def _rejection_reason(raw_words: list[Word], cleaned_text: str):
    """Why this response must be discarded, or ``None`` when it is acceptable.

    Returns ``(reason, cleaned_tokens, matcher)``; the matcher is handed back so
    a caller that accepts the response does not have to diff it a second time.

    Shared by :func:`realign_words` (which falls back to the raw words) and
    :func:`restore_words_gemini` (which refuses to cache and re-asks later), so
    the two can never disagree about which responses are usable.
    """
    cleaned_tokens = _tokenize_cleaned_text(cleaned_text)
    if not cleaned_tokens:
        return "response was empty", cleaned_tokens, None
    matcher = difflib.SequenceMatcher(
        None,
        [_align_key(word.text) for word in raw_words],
        [_align_key(token) for token in cleaned_tokens],
        autojunk=False,
    )
    if _is_truncated(len(raw_words), cleaned_tokens):
        return "response was cut off before the end of the chunk", cleaned_tokens, matcher
    if matcher.ratio() < MIN_SIMILARITY_RATIO:
        return "response is not recognisably a rewrite of the input", cleaned_tokens, matcher
    rate = _deleted_per_1000(matcher, len(raw_words))
    if rate > MAX_DELETED_PER_1000_WORDS:
        return f"response drops {rate:.0f} words per 1000, over the {MAX_DELETED_PER_1000_WORDS:.0f} allowed", cleaned_tokens, matcher
    return None, cleaned_tokens, matcher


def realign_words(raw_words: list[Word], cleaned_text: str) -> list[Word]:
    """Rebuild timed words from Gemini's free-form rewrite of ``raw_words``.

    Diffs the two word streams by :func:`_align_key` (folded, punctuation
    stripped) rather than raw text, since Gemini's whole job is to change
    case, add punctuation, and occasionally swap a word for the one that was
    actually spoken. Each ``equal`` block copies the matching raw word's
    ``offset``/``nll`` straight across; anything Gemini changed gets an
    offset linearly interpolated between the nearest unchanged neighbours on
    either side (or pinned to whichever neighbour exists, at either end of
    the chunk) and the worst ``nll`` of whatever raw span it replaced.

    Discards the response — falling back to ``raw_words`` unchanged — whenever
    :func:`_rejection_reason` objects: the response was empty, cut off before
    the end of the chunk, is not recognisably a rewrite of the input, or drops
    more words than cleaning is allowed to (``MAX_DELETED_PER_1000_WORDS``).
    The last of those is checked **before** the diff is applied: otherwise the
    dropped span becomes a ``delete`` opcode and those raw words disappear
    outright, silently deleting real transcript content.
    """
    if not raw_words:
        return []
    reason, cleaned_tokens, matcher = _rejection_reason(raw_words, cleaned_text)
    if reason is not None:
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
    permanently frozen as an unrestored no-op. A **truncated** response
    counts as not genuine for this purpose (see ``MIN_LENGTH_RATIO``), for
    the same reason and with a useful side effect: a cache entry written
    before that guard existed is treated as a miss on the next run, so it is
    re-requested and overwritten with a good response rather than rejected
    again and again for the rest of the corpus's life.

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

    def usable(index: int, text: str | None) -> bool:
        """A response worth trusting and worth keeping."""
        if text is None:
            return False
        return _rejection_reason(chunks[index], text)[0] is None

    restored_texts: list[str | None] = [None] * len(chunks)
    pending = []
    for index, key in enumerate(keys):
        cached = _read_cache(cache_dir, key)
        if usable(index, cached):
            restored_texts[index] = cached
        else:
            pending.append(index)

    def work(index: int) -> tuple[int, str | None, bool]:
        restored = cleaner.restore_or_none(requests[index])
        reason, _tokens, _matcher = _rejection_reason(chunks[index], restored or "")
        if restored is not None and reason is None:
            _write_cache(cache_dir, keys[index], restored)
            return index, restored, True
        if restored is not None:
            print(f"[gemini-clean] rejected a response for chunk {index}: {reason}",
                  flush=True)
        return index, None, False

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
