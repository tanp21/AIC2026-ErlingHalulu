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

Merging BPE pieces into words is the only transform left here — restoration
(capitalisation, punctuation, and the occasional ASR-mistranscription fix) now
runs entirely through Gemini (:mod:`scripts.audio.gemini_clean`), which works
on free-form prose rather than this module's per-word representation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

#: Highest NLL that still fits in a float column without tripping up consumers
#: that aggregate it. Anything at or above this is "no confidence recorded".
NLL_CEILING = 1000.0

_WHITESPACE = re.compile(r"(\s+)")


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
