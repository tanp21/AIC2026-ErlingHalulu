"""Turn raw ASR token output into clean, searchable transcript text.

The 2026 transcripts arrive as ``(audio_key, offset, token, log_prob)`` rows:
BPE pieces, each stamped with the time it was decoded at. That timing is the
only link between text and video, so **every transform here works on a word
stream that carries offsets** and hands back rows in the same shape. Passage
windowing (``backend.audio_search.make_passages``) and subtitle cues
(``backend.audio_search.build_transcript_cues``) then keep working untouched.

``log_prob`` is misnamed in the source data. Its values run from 0 to 15.55 with
a median of 0.0097, and the ASR placeholder token ``<`` averages 2.89 against
0.11 for everything else — it is a *negative* log likelihood, so **lower is more
confident**. It is called ``nll`` throughout this module to stop that being
re-learned every time.

No I/O and no model imports, so all of it is unit-testable without a GPU. This
mirrors :mod:`scripts.ocr.postprocess`, which plays the same role for OCR.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import re

from scripts.audio import numbers
from scripts.ocr.scoring import fold

#: Highest NLL that still fits in a float column without tripping up consumers
#: that aggregate it. Anything at or above this is "no confidence recorded".
NLL_CEILING = 1000.0

#: Phrase width for boilerplate detection, in words. Long enough that ordinary
#: Vietnamese does not collide by chance, short enough to catch a sponsor read
#: that is topped and tailed differently in each video.
PHRASE_WORDS = 8

#: How many *distinct* videos a phrase must appear in before it counts as
#: boilerplate rather than content.
MIN_BOILERPLATE_VIDEOS = 5

_WHITESPACE = re.compile(r"(\s+)")


@lru_cache(maxsize=200_000)
def _fold_word(word: str) -> str:
    """Diacritic-blind form of a single word, memoised across the corpus.

    The corpus is 1.4M words but only a fraction of that is distinct, and
    :func:`scripts.ocr.scoring.fold` runs two Unicode normalisation passes, so
    caching turns corpus-wide detection from minutes into seconds.
    """
    return fold(word)


@dataclass(frozen=True)
class Word:
    """One whitespace-delimited word, with the time it started being spoken.

    ``offset`` is the offset of the token piece that *began* the word, and
    ``nll`` is the worst (highest) NLL among the pieces that formed it — merging
    must not launder a garbage piece into a confident-looking word.
    """

    text: str
    offset: float
    nll: float


def _usable(row) -> tuple[float, str, float] | None:
    """Coerce a raw row, or return ``None`` if it cannot be trusted."""
    token = row.get("token")
    if not isinstance(token, str) or not token:
        return None
    try:
        offset = float(row.get("offset"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(offset) or offset < 0:
        return None
    try:
        nll = float(row.get("log_prob"))
    except (TypeError, ValueError):
        nll = math.inf
    if math.isnan(nll):
        nll = math.inf
    return offset, token, nll


def merge_tokens_to_words(rows) -> list[Word]:
    """Collapse BPE token rows into words.

    A word ends wherever whitespace appears, whether that whitespace opens the
    next token (``" dự"``) or sits inside one (``" xin chào"``). Pieces with no
    leading space (``"ng"``) extend the word in progress, which is how
    ``" bằ" + "ng"`` becomes ``bằng``.
    """
    valid = sorted(
        (parsed for parsed in map(_usable, rows) if parsed is not None),
        key=lambda parsed: parsed[0],
    )
    words: list[Word] = []
    buffer: list[str] = []
    start = 0.0
    worst = math.inf

    def flush() -> None:
        nonlocal buffer, worst
        if buffer:
            words.append(Word("".join(buffer), start, worst))
            buffer = []
            worst = math.inf

    for offset, token, nll in valid:
        for part in _WHITESPACE.split(token):
            if not part:
                continue
            if part.isspace():
                flush()
                continue
            if not buffer:
                start = offset
                worst = nll
            else:
                worst = max(worst, nll)
            buffer.append(part)
    flush()
    return words


def _phrases(words, phrase_words: int):
    """Every folded ``phrase_words``-long window of ``words``."""
    folded = [_fold_word(word.text) for word in words]
    for index in range(len(folded) - phrase_words + 1):
        yield tuple(folded[index:index + phrase_words])


def boilerplate_document_frequency(
    words_by_video, *, phrase_words: int = PHRASE_WORDS
) -> Counter:
    """Count, per phrase, how many *distinct* videos contain it.

    Split out from :func:`find_boilerplate` because it is by far the expensive
    half — 1.4M windows over the 2026 corpus — while the threshold applied to it
    is free. A sweep over ``min_videos`` computes this once and filters it many
    times.
    """
    document_frequency = Counter()
    for words in words_by_video.values():
        document_frequency.update(set(_phrases(words, phrase_words)))
    return document_frequency


def phrases_above(document_frequency, min_videos: int) -> frozenset[tuple[str, ...]]:
    """Phrases seen in at least ``min_videos`` videos."""
    return frozenset(
        phrase
        for phrase, videos in document_frequency.items()
        if videos >= min_videos
    )


def find_boilerplate(
    words_by_video,
    *,
    phrase_words: int = PHRASE_WORDS,
    min_videos: int = MIN_BOILERPLATE_VIDEOS,
) -> frozenset[tuple[str, ...]]:
    """Phrases that recur across videos: sponsor reads, outros, channel stings.

    606 of the 873 videos in the 2026 corpus carry one of these, and
    ``"chương trình được tài trợ bởi công ty"`` alone appears in 403. Left in,
    they hand BM25 a large block of text that is identical across hundreds of
    documents and says nothing about any of them.

    Counted per *distinct video*, never per occurrence: a cooking show that
    repeats its own catchphrase ten times is still that show's content. Matching
    is diacritic-blind so an ASR run that drops a tone mark in one video still
    lines up with the others.
    """
    return phrases_above(
        boilerplate_document_frequency(words_by_video, phrase_words=phrase_words),
        min_videos,
    )


def strip_boilerplate(
    words, phrases, *, phrase_words: int = PHRASE_WORDS
) -> list[Word]:
    """Delete every span of ``words`` matching a known boilerplate phrase.

    Matches are marked, not cut one at a time, so consecutive windows overlap
    and a sponsor read longer than ``phrase_words`` disappears in full rather
    than leaving its first and last few words behind.
    """
    if not phrases:
        return list(words)
    doomed = [False] * len(words)
    for index, phrase in enumerate(_phrases(words, phrase_words)):
        if phrase in phrases:
            doomed[index:index + phrase_words] = [True] * phrase_words
    return [word for word, cut in zip(words, doomed) if not cut]


#: Words after which a lone number word is certainly a number rather than an
#: article. Vietnamese uses ``một`` as "a/an" constantly — "một người đàn ông" is
#: "a man", not "1 man" — so a single number word is only converted in a
#: date/time/ordinal frame like "ngày một tháng tám".
#:
#: ``năm`` is deliberately absent even though "năm 2024" is the commonest frame
#: of all: a year is always two or more number words, so it already qualifies on
#: length, while admitting ``năm`` as a trigger would turn the counting sequence
#: "hai ba bốn năm sáu" into "... năm 6".
STRONG_NUMBER_TRIGGERS = {
    "ngày", "tháng", "giờ", "phút", "giây",
    "thứ", "lớp", "khóa", "khoá", "hạng", "tuổi",
}


def _is_numeral(words, index: int) -> bool:
    """Whether ``words[index]`` is being used as a number here.

    Several Vietnamese number words are also very common ordinary words, and
    getting this wrong corrupts text rather than merely failing to improve it:

    ``năm``   five, but far more often "year" — needs a scale word after it
    ``không`` zero, but overwhelmingly the negation "not" — needs ``trăm`` after
    ``tư``    four, but also "private/thought" — needs a tens word before
    ``lẻ``    zero-filler, but also "odd/retail" — needs ``trăm`` before
    """
    word = words[index].text.lower()
    previous = words[index - 1].text.lower() if index else None
    following = words[index + 1].text.lower() if index + 1 < len(words) else None

    if word == "năm":
        return following in numbers.SCALE_FOLLOWERS or previous in numbers.FILLERS
    if word == "không":
        return following == "trăm"
    if word in numbers.UNIT_VARIANTS:
        return previous in {"mươi", "mười"} or previous in numbers.DIGITS
    if word in numbers.FILLERS:
        return previous == "trăm"
    if word in numbers.SCALE_FOLLOWERS or word == "mười":
        return True
    return word in numbers.DIGITS


def normalize_numbers(words) -> list[Word]:
    """Rewrite spelled-out numbers as digits, conservatively.

    A run is only replaced when it parses completely *and* is either at least
    two words long or sits directly after a date/time trigger. A trailing
    ``phần trăm`` is folded into a ``%`` sign, since 633 mentions of it in the
    corpus are otherwise unreachable from a query written "20%".

    The replacement word inherits the offset of the first word in the run, so a
    six-word year collapses to one token anchored where it started being spoken.
    """
    result: list[Word] = []
    index = 0
    while index < len(words):
        # A run has to *start* on a digit or "mười"; a scale word or a filler on
        # its own ("trăm", "lẻ") is never the head of a number.
        if not (_is_numeral(words, index)
                and words[index].text.lower() in set(numbers.DIGITS) | {"mười"}):
            result.append(words[index])
            index += 1
            continue

        end = index
        while end < len(words) and _is_numeral(words, end):
            end += 1
        run = words[index:end]
        value = numbers.parse_number([word.text.lower() for word in run])

        previous = result[-1].text.lower() if result else None
        allowed = len(run) >= 2 or previous in STRONG_NUMBER_TRIGGERS
        if value is None or not allowed:
            result.append(words[index])
            index += 1
            continue

        consumed = end
        text = numbers.format_number(value)
        if [w.text.lower() for w in words[end:end + 2]] == ["phần", "trăm"]:
            text = f"{value}%"
            consumed = end + 2
        result.append(Word(text, run[0].offset,
                           max(word.nll for word in words[index:consumed])))
        index = consumed
    return result


def drop_low_confidence(words, *, max_nll: float) -> list[Word]:
    """Keep only words the recogniser was confident about.

    ``max_nll`` is an *upper* bound because the score is a loss: the ``<``
    placeholder averages 2.89 where real speech averages 0.11.
    """
    return [word for word in words if word.nll <= max_nll]


def apply_capu_output(words, restored_text: str) -> list[Word]:
    """Overlay a capitalisation/punctuation model's output onto timed words.

    Restoration only changes the case of a word and appends punctuation to it,
    so the word *count* is invariant and alignment is by index. When the model
    breaks that invariant — merging, splitting, or paraphrasing — the whole
    chunk is rejected and the original words are returned unchanged.

    That guard is what makes the step safe to enable by default. It is the same
    bargain :func:`scripts.ocr.postcorrect.accept_correction` strikes with the
    vision-LLM: a model that rewrites instead of correcting costs nothing,
    because its output is discarded rather than trusted.
    """
    restored = restored_text.split()
    if len(restored) != len(words):
        return list(words)
    return [
        Word(text, word.offset, word.nll)
        for word, text in zip(words, restored)
    ]


def words_to_tokens(audio_key: str, words) -> list[dict]:
    """Render words back into the row shape ``step_data.parquet`` uses.

    Each word becomes a single space-prefixed token, so ``"".join(token)`` still
    reconstructs the transcript and ``.split()`` still recovers the words. Rows
    that went in as many pieces come back as one; that is intended, because a
    normalised span like ``2024`` has no meaningful sub-word timing.
    """
    return [
        {
            "audio_key": audio_key,
            "offset": word.offset,
            "token": f" {word.text}",
            "log_prob": min(word.nll, NLL_CEILING),
        }
        for word in words
    ]
