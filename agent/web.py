"""Web-search helpers: which queries get one, and reading the tagged reply lines.

Cooking queries are the corpus's biggest single genre (15 of the 55 round-1/2
2026 queries), mostly episodes of "Món ngon mỗi ngày", whose recipes are
published online with the dish name and ingredient quantities. A dish name is
what the transcript channel needs, and a published recipe cross-checks a QA
reading ("300g", "cá sòng").
"""
from __future__ import annotations

from difflib import SequenceMatcher
import re
import unicodedata

from .prompts import COOKING_LOOKUP_PROMPT, FACT_PROMPT, RECIPE_PROMPT

COOKING = re.compile(
    r"nấu|món|nguyên liệu|đầu bếp|chế biến|công thức|ẩm thực|tạp dề|topping|gia vị|"
    r"xào|chiên|hấp|luộc|nướng|ướp|chảo|nồi|bếp",
    re.IGNORECASE,
)
UNKNOWN = {"", "khong ro", "khong biet", "khong xac dinh", "unknown", "n/a"}


def is_cooking(text: str) -> bool:
    return bool(COOKING.search(text or ""))


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFD", text).replace("đ", "d").replace("Đ", "D")
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn").lower()


def tagged(text: str, tag: str) -> str | None:
    """The value of the last `TAG: value` line (accent- and case-insensitive), if any."""
    wanted = _fold(tag)
    value = None
    for line in (text or "").splitlines():
        head, sep, rest = line.strip().strip("*# ").partition(":")
        if sep and _fold(head.strip("* ")) == wanted:
            value = rest.strip().strip("*").strip()
    if value is None or _fold(value).strip(" .") in UNKNOWN:
        return None
    return value[:100]


def cooking_lookup(query: str, question: str | None) -> str:
    return COOKING_LOOKUP_PROMPT.format(
        query=query, question=f"Câu hỏi cần trả lời: {question}\n" if question else "")


def recipe_lookup(query: str, question: str, opening: str, moment: str) -> str:
    return RECIPE_PROMPT.format(query=query, question=question, opening=opening, moment=moment)


def fact_lookup(query: str, question: str, moment: str) -> str:
    return FACT_PROMPT.format(query=query, question=question, moment=moment)


def similar(a: str, b: str) -> float:
    """0-1 closeness of two short answers, ignoring accents and case."""
    return SequenceMatcher(None, _fold(a or ""), _fold(b or "")).ratio()


#: Where a lookup's dish name shows up. The prompts ask for a final
#: `TỪ KHÓA: <tên món>; <nguyên liệu>` line, but a grounded search does not always give
#: one -- a real reply wrote "**Tên món ăn:** Cá hồi áp chảo xốt tiêu (hoặc …)" with no
#: TỪ KHÓA line at all, and the run then searched nothing. The name is the one thing in
#: a lookup worth searching for: measured on the round-3 soy-sauce query, an audio query
#: with the name ("Cá hồi áp chảo xốt tiêu") ranks its video **first**, where the
#: query's own terms ranked it 30th of 84. The transcript says the dish out loud at
#: 48 s, so the name is a key the corpus has and the description does not.
DISH_PATTERNS = (
    re.compile(r"^\s*T[ỪU] KH[ÓO]A\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"t[êe]n m[óo]n(?:\s*[ăâ]n)?\s*:?\s*\**\s*(.+)$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"(?:l[àa]|m[óo]n)\s+\*\*(.+?)\*\*", re.IGNORECASE),
)
#: A reply that offers alternatives ("… xốt tiêu (hoặc … sốt tiêu xanh)") or a list
#: should be searched by its first name only.
NAME_TAIL = re.compile(r"\s*[(\[;,.–—]|\s+(?:hoặc|hay|hoac)\s")
#: Answers the model marks as unknown; "không rõ" is what the prompts ask for.
UNKNOWN_ANSWER = re.compile(r"^(kh[ôo]ng r[õo]|khong ro|unknown|n/?a)$", re.IGNORECASE)


def dish_name(text: str | None) -> str | None:
    """The dish name a lookup named, or None if it named none.

    The first field only when the reply lists ingredients after it -- the ingredients
    are what the query already described, so searching them adds nothing.
    """
    for pattern in DISH_PATTERNS:
        match = pattern.search(text or "")
        if not match:
            continue
        name = " ".join(match.group(1).split())
        name = name.split(";")[0]
        name = NAME_TAIL.split(name)[0].strip(" .,-*_")
        if len(name) > 2:
            return name
    return None
