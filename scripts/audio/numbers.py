"""Read Vietnamese number words as integers.

Not one of the 873 transcripts contains a digit — the recogniser spells every
number out, so ``2024`` is on the page as ``hai ngàn không trăm hai mươi tư``.
That is 5.5% of the corpus (77,361 words), including 1,088 year phrases, 1,609
mentions of ``tháng`` and 633 of ``phần trăm``, none of which a query written as
"2024" can currently match.

Parsing is deliberately strict: :func:`parse_number` returns ``None`` for
anything it cannot account for completely, and the caller leaves such spans
alone. A missed conversion costs one unmatched query; a wrong one corrupts the
transcript.

Pure functions, no I/O.
"""
from __future__ import annotations

#: Plain digits. ``năm`` is here but is heavily guarded by the caller: it is
#: also the word for "year", which is how it is used most often in narration.
DIGITS = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4,
    "năm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9,
}

#: Forms that only ever appear in the units place after a tens word:
#: ``hai mươi mốt`` is 21, ``hai mươi tư`` 24, ``mười lăm`` 15.
UNIT_VARIANTS = {"mốt": 1, "tư": 4, "lăm": 5, "nhăm": 5}

#: Scale words above one hundred, largest first — the parser splits on the
#: largest present so ``một ngàn chín trăm bốn mươi lăm`` recurses correctly.
BIG_SCALES = (("tỷ", 10**9), ("tỉ", 10**9), ("triệu", 10**6),
              ("nghìn", 1000), ("ngàn", 1000))

#: Zero-filler in the tens place: ``một trăm linh năm`` is 105.
FILLERS = {"linh", "lẻ"}

#: Spoken alternative to ``mươi`` for whole tens: ``hai chục`` is 20. Unlike
#: ``mươi`` it also takes quantifiers — ``mấy chục``, ``vài chục``, "a few
#: dozen" — which carry no number and are left alone by :func:`_unit`.
TENS_WORDS = {"mươi", "chục"}

#: Words that make a preceding ``năm``/``không`` unambiguously a numeral.
SCALE_FOLLOWERS = {"mươi", "chục", "trăm", "nghìn", "ngàn", "triệu", "tỷ", "tỉ"}

NUMBER_WORDS = set(DIGITS) | set(UNIT_VARIANTS) | FILLERS | SCALE_FOLLOWERS | {"mười"}


def _unit(words) -> int | None:
    """A single units-place word, or a colloquial ``ba mốt`` style pair."""
    if len(words) == 1:
        word = words[0]
        if word in UNIT_VARIANTS:
            return UNIT_VARIANTS[word]
        return DIGITS.get(word)
    # "ba mốt" is how people say "ba mươi mốt": tens digit, then a unit variant.
    if len(words) == 2 and words[0] in DIGITS and words[1] in UNIT_VARIANTS:
        return DIGITS[words[0]] * 10 + UNIT_VARIANTS[words[1]]
    return None


def _parse_below_thousand(words) -> int | None:
    value = 0
    words = list(words)
    after_hundreds = False

    if "trăm" in words:
        index = words.index("trăm")
        hundreds = _unit(words[:index])
        if hundreds is None:
            return None
        value += hundreds * 100
        words = words[index + 1:]
        after_hundreds = True

    had_filler = any(word in FILLERS for word in words)
    words = [word for word in words if word not in FILLERS]
    if not words:
        return value

    # "một trăm hai" is 120, not 102: a bare digit straight after the hundreds
    # is the tens place. Saying 102 out loud needs the filler — "một trăm linh
    # hai" — which is exactly what `had_filler` distinguishes.
    if after_hundreds and not had_filler and len(words) == 1:
        shorthand = _unit(words)
        if shorthand is not None:
            return value + shorthand * 10

    tens_word = next((w for w in words if w in TENS_WORDS), None)
    if tens_word is not None:
        index = words.index(tens_word)
        tens = _unit(words[:index])
        if tens is None:
            return None
        value += tens * 10
        words = words[index + 1:]
    elif words[0] == "mười":
        value += 10
        words = words[1:]

    if not words:
        return value
    units = _unit(words)
    if units is None:
        return None
    return value + units


def parse_number(words) -> int | None:
    """Integer value of a run of number words, or ``None`` if it does not parse.

    ``None`` is the common case for anything unusual, and it is the signal the
    caller uses to leave the original wording untouched.
    """
    words = list(words)
    if not words:
        return None
    for name, scale in BIG_SCALES:
        if name in words:
            index = words.index(name)
            # "ngàn chín trăm" with nothing before it means one thousand.
            remainder = words[index + 1:]
            # The same shorthand applies above a hundred — "một nghìn hai" is
            # 1200 — but unlike the hundreds case there is no filler form to
            # disambiguate it from 1002. Refuse rather than guess.
            if len(remainder) == 1 and remainder[0] in DIGITS:
                return None
            left = parse_number(words[:index]) if words[:index] else 1
            right = parse_number(remainder) if remainder else 0
            if left is None or right is None:
                return None
            return left * scale + right
    return _parse_below_thousand(words)


def format_number(value: int) -> str:
    """Render a parsed value the way a Vietnamese query would write it.

    Round millions and billions keep their scale word, because "100 triệu đồng"
    is what a person types and ``100000000`` is not. Everything else, years
    included, becomes plain digits.
    """
    if value >= 10**9 and value % 10**9 == 0:
        return f"{value // 10**9} tỷ"
    if value >= 10**6 and value % 10**6 == 0:
        return f"{value // 10**6} triệu"
    return str(value)
