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

from .prompts import COOKING_LOOKUP_PROMPT, FACT_PROMPT, RECIPE_PROMPT, TOPIC_LOOKUP_PROMPT

COOKING = re.compile(
    r"nấu|món|nguyên liệu|đầu bếp|chế biến|công thức|ẩm thực|tạp dề|topping|gia vị|"
    r"xào|chiên|hấp|luộc|nướng|ướp|chảo|nồi|bếp",
    re.IGNORECASE,
)
UNKNOWN = {"", "khong ro", "khong biet", "khong xac dinh", "unknown", "n/a"}


def is_cooking(text: str) -> bool:
    return bool(COOKING.search(text or ""))


#: Words that already say, before any planning, that the web knows what a query is about. Only
#: words that name a KIND of thing the corpus would call by its name: "lễ hội" is a festival
#: with a name the narrator says, "tỉnh" alone is every news report. Checked in this order; the
#: first match decides the lookup (`topic_lookup`), and the planner's own `topic` and
#: `lookups` add to it after planning.
TOPIC_PATTERNS = (
    ("festival", re.compile(
        r"lễ hội|hội làng|hội xuân|lễ rước|rước kiệu|rước đèn|lễ cúng|lễ tế|lễ dâng|tết nguyên đán|"
        r"trung thu|rằm tháng|phong tục|tập tục|nghi lễ|nghi thức|múa lân|múa rồng|múa sư tử|"
        r"đua thuyền|đua ghe|đua bò|chọi trâu|kéo co|quan họ|hát xoan|hát then|đờn ca tài tử",
        re.IGNORECASE)),
    ("event", re.compile(
        r"hội nghị|hội thảo|lễ khai mạc|lễ bế mạc|khánh thành|khai trương|lễ kỷ niệm|diễu binh|"
        r"diễu hành|triển lãm|hội chợ|giải đấu|chung kết|lễ trao|lễ phát động|cơn bão|lũ quét|"
        r"sạt lở|vụ cháy|hỏa hoạn|tai nạn|động đất", re.IGNORECASE)),
    ("place", re.compile(
        r"chùa|đền|đình làng|miếu|nhà thờ|thánh đường|lăng|di tích|di sản|danh lam|thắng cảnh|"
        r"phố cổ|làng nghề|khu du lịch|bảo tàng|thành cổ|cột cờ|cây cầu|con đèo|ngọn núi|hang|"
        r"thác|vịnh|hòn đảo|bãi biển|địa danh|địa điểm", re.IGNORECASE)),
    ("person", re.compile(
        r"nghệ nhân|nghệ sĩ|ca sĩ|danh hiệu|chủ tịch|bộ trưởng|thủ tướng|giáo sư|vận động viên|"
        r"cầu thủ|đạo diễn|nhà văn|anh hùng|người sáng lập", re.IGNORECASE)),
)
#: What each topic's lookup is after, for `TOPIC_LOOKUP_PROMPT`: (what the query is about, the noun).
TOPIC_GOALS = {
    "festival": ("một lễ hội, phong tục hoặc nghi lễ", "lễ hội"),
    "event": ("một sự kiện (tin tức, lễ, hội nghị, thi đấu, thiên tai)", "sự kiện"),
    "place": ("một địa danh (công trình, di tích, thắng cảnh, làng nghề)", "địa danh"),
    "person": ("một nhân vật", "nhân vật"),
}


def detect_topic(text: str) -> str | None:
    """The first topic in `TOPIC_PATTERNS` whose words the query uses, or None."""
    for topic, pattern in TOPIC_PATTERNS:
        if pattern.search(text or ""):
            return topic
    return None


def topic_lookup(query: str, topic: str) -> str:
    what, kind = TOPIC_GOALS[topic]
    return TOPIC_LOOKUP_PROMPT.format(query=query, what=what, kind=kind)


#: How long a keyword may be to be searched: a name or a short phrase, not a sentence.
KEYWORD_MAX_WORDS = 8


def keywords(text: str | None) -> list[str]:
    """The names a lookup's `TỪ KHÓA:` line lists, alternatives and asides dropped."""
    line = tagged(text or "", "TỪ KHÓA")
    if not line:
        return []
    found = []
    for part in re.split(r"[;,]", line):
        name = NAME_TAIL.split(part.strip())[0].strip(" .-*_\"'“”")
        if (len(name) > 2 and len(name.split()) <= KEYWORD_MAX_WORDS and not is_unknown(name)
                and name.casefold() not in (f.casefold() for f in found)):
            found.append(name)
    return found


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
#: A reading that says the moment does not answer the question, in the words models use for it
#: ("không rõ", "Không xác định được", "không đọc được số", "not visible", ...). Measured on the
#: first Qwen run: three readings of "không rõ" agreed with each other and outvoted nothing, so
#: the run submitted "không rõ" as its answer.
NON_ANSWER = re.compile(
    r"^(kh[oô]ng\s+(r[oõ]|bi[eế]t|x[aá]c\s+[dđ][iị]nh|[dđ][oọ]c\s+[dđ][uư][oợ]c|th[aấ]y|c[oó]\s+th[oô]ng\s+tin|"
    r"nh[iì]n\s+th[aấ]y|[dđ][uủ]\s+(d[uữ]|th[oô]ng)|th[eể])|unknown|n/?a|none|not\s+(visible|shown|known|determinable)|"
    r"cannot|unclear|kh[oô]ng$)", re.IGNORECASE)


DENIAL = re.compile(r"^(kh[oô]ng|ch[uư]a|there (is|are) no|no |none |the (image|frame)s? (does|do) not)",
                    re.IGNORECASE)


def is_unknown(answer: str | None) -> bool:
    """True for an empty answer and for one that only says there is no answer."""
    text = " ".join(str(answer or "").split()).strip(" .!\"'")
    if not text or NON_ANSWER.match(text) or _fold(text) in UNKNOWN:
        return True
    # A sentence that opens with a denial is an explanation, not an answer: "Không có hình ảnh
    # cân hay con số hiển thị trên cân trong các khung hình" was read as the answer to "what
    # does the scale show?". A real answer is a name, a number or a few words.
    return bool(DENIAL.match(text)) and len(text.split()) > 4


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
