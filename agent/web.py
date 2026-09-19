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
