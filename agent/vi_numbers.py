"""Spell digits the way the Vietnamese ASR writes them.

The recogniser spells every number out (see `scripts/audio/numbers.py`), so a
transcript query containing "211" or "88%" only matches once it reads
"hai trăm mười một" / "tám mươi tám phần trăm". The inverse of that module.
"""
from __future__ import annotations

import re

DIGITS = ("không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín")
SCALES = ("", "ngàn", "triệu", "tỷ")
NUMBER = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)(\s*%)?")


def _below_thousand(value: int, inner: bool) -> list[str]:
    """`inner`: this group follows a larger one, so "không trăm"/"linh" are spoken."""
    hundreds, rest = divmod(value, 100)
    tens, units = divmod(rest, 10)
    words = [DIGITS[hundreds], "trăm"] if hundreds or inner else []
    if tens == 0:
        if units:
            words += (["linh"] if words else []) + [DIGITS[units]]
    elif tens == 1:
        words.append("mười")
        if units:
            words.append("lăm" if units == 5 else DIGITS[units])
    else:
        words += [DIGITS[tens], "mươi"]
        if units:
            words.append({1: "mốt", 4: "tư", 5: "lăm"}.get(units, DIGITS[units]))
    return words


def spell_integer(value: int) -> str:
    if value == 0:
        return "không"
    groups = []
    while value:
        value, group = divmod(value, 1000)
        groups.append(group)
    if len(groups) > len(SCALES):
        raise ValueError("number too large")
    words = []
    for position in reversed(range(len(groups))):
        if groups[position] == 0:
            continue
        words += _below_thousand(groups[position], inner=position < len(groups) - 1)
        if SCALES[position]:
            words.append(SCALES[position])
    return " ".join(words)


def _spell_token(token: str) -> str:
    parts = re.split(r"[.,]", token)
    # "1.000" / "12,500,000": every group after the first is exactly three digits.
    if len(parts) > 1 and all(len(part) == 3 for part in parts[1:]):
        return spell_integer(int("".join(parts)))
    if len(parts) == 2:
        return f"{spell_integer(int(parts[0]))} phẩy {spell_integer(int(parts[1]))}"
    return " ".join(spell_integer(int(part)) for part in parts)


def spell_numbers(text: str) -> str:
    """Replace every digit group in `text` with spoken words; other text is kept."""
    def replace(match):
        try:
            spoken = _spell_token(match.group(1))
        except ValueError:
            return match.group(0)
        # "211kg" -> "hai trăm mười một kg": keep a unit glued to the digits separate.
        following = match.string[match.end():match.end() + 1]
        return spoken + (" phần trăm" if match.group(2) else "") + (" " if following.isalpha() else "")

    return " ".join(NUMBER.sub(replace, text).split())
