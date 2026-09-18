"""Pure validation for holistic-caption paraphrases (no Gemini import).

A paraphrase may change wording, syntax, word order and synonyms only. Hard checks
reject anything that could change what PE-Core is asked to find; soft flags only
annotate. See docs/reports/visual-holistic-paraphrase-benchmark-2026.md.

Hard: tokens (> PE-Core limit), identity (same text after normalization), protected
(numbers, colours, quoted / ALL-CAPS text, mid-sentence proper nouns), framing (added
"photo"/"video"/... words), judge (LLM equivalence verdict, supplied by the caller).
Soft: over_target (tokens above the soft target), low_diversity (edit distance).
"""
import re
from dataclasses import dataclass, field

from backend.query_expansion import TOKEN_LIMIT, TOKEN_TARGET

DIVERSITY_VS_SOURCE = 0.25
DIVERSITY_VS_PREVIOUS = 0.15
MAX_RETRIES = 2
STRATEGIES = ("reorder", "synonym", "restructure", "combined")

# "one" is left out on purpose: "one man" <-> "a man" is a legitimate rewording and the
# LLM judge covers it; digits and two..twenty are unambiguous counts.
NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}
COLOR_CANONICAL = {
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green", "blue": "blue",
    "purple": "purple", "violet": "violet", "pink": "pink", "brown": "brown", "black": "black",
    "white": "white", "gray": "gray", "grey": "gray", "silver": "silver", "gold": "gold",
    "golden": "gold", "beige": "beige", "cyan": "cyan", "magenta": "magenta", "maroon": "maroon",
    "navy": "navy", "turquoise": "turquoise", "teal": "teal", "crimson": "crimson",
    "olive": "olive", "indigo": "indigo", "khaki": "khaki", "ivory": "ivory", "bronze": "bronze",
    "copper": "copper",
}
FRAMING_WORDS = {"photo", "photograph", "image", "picture", "video", "clip", "scene", "footage",
                 "shot", "frame", "still"}
NOT_PROPER = {"I"}

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’\-]*")
# Double quotes only: straight single quotes collide with apostrophes ("man's ... child's").
_QUOTED = re.compile(r"[\"“”]([^\"“”]{2,})[\"“”]")
_SENTENCE_END = re.compile(r"[.!?]\s*$")


def normalize_text(text):
    return " ".join(re.sub(r"[^a-z0-9\s]", " ", text.lower()).split())


def _words(text):
    return [m.group(0) for m in _WORD.finditer(text)]


def _framing_stem(word):
    word = word.lower()
    return word[:-1] if word.endswith("s") and word[:-1] in FRAMING_WORDS else word


def protected_tokens(text):
    """Categories of details that must survive paraphrasing unchanged."""
    numbers, colors, allcaps, proper = set(), set(), set(), set()
    quoted = {q.strip() for q in _QUOTED.findall(text)}
    sentence_start = True
    for match in _WORD.finditer(text):
        word = match.group(0)
        lower = word.lower()
        if word.isdigit():
            numbers.add(int(word))
        elif lower in NUMBER_WORDS:
            numbers.add(NUMBER_WORDS[lower])
        if lower in COLOR_CANONICAL:
            colors.add(COLOR_CANONICAL[lower])
        letters = [c for c in word if c.isalpha()]
        if len(word) >= 2 and letters and all(c.isupper() for c in letters):
            allcaps.add(word)
        elif word[0].isupper() and not sentence_start and word not in NOT_PROPER:
            proper.add(word)
        following = text[match.end():match.end() + 2]
        sentence_start = bool(_SENTENCE_END.match(following))
    return {"numbers": numbers, "colors": colors, "quoted": quoted,
            "allcaps": allcaps, "proper": proper}


def check_protected(source, paraphrase):
    """Missing or changed protected details, as human-readable failure strings."""
    src = protected_tokens(source)
    para = protected_tokens(paraphrase)
    para_words = set(_words(paraphrase))
    failures = []
    if src["numbers"] != para["numbers"]:
        failures.append(f"numbers {sorted(src['numbers'])} -> {sorted(para['numbers'])}")
    if src["colors"] != para["colors"]:
        failures.append(f"colors {sorted(src['colors'])} -> {sorted(para['colors'])}")
    for span in sorted(src["quoted"]):
        if span not in paraphrase:
            failures.append(f"quoted text missing: {span!r}")
    for token in sorted(src["allcaps"] | src["proper"]):
        if token not in para_words:
            failures.append(f"proper noun / on-screen text missing: {token!r}")
    return failures


def check_framing(source, paraphrase):
    src = {_framing_stem(w) for w in _words(source)}
    added = sorted({_framing_stem(w) for w in _words(paraphrase)} & FRAMING_WORDS - src)
    return [f"added framing word: {w!r}" for w in added]


def token_edit_distance(a, b):
    """Word-level Levenshtein distance on normalized text, divided by the longer length."""
    x, y = normalize_text(a).split(), normalize_text(b).split()
    if not x and not y:
        return 0.0
    previous = list(range(len(y) + 1))
    for i, xw in enumerate(x, 1):
        current = [i]
        for j, yw in enumerate(y, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (xw != yw)))
        previous = current
    return previous[-1] / max(len(x), len(y))


def judge_failures(verdict):
    failures = []
    if not verdict.get("equivalent", False):
        failures.append("judge: not equivalent")
    for key in ("added", "removed", "changed"):
        if verdict.get(key):
            failures.append(f"judge {key}: {verdict[key]}")
    return failures


@dataclass
class SlotChecks:
    tokens: int
    hard: list = field(default_factory=list)
    soft: list = field(default_factory=list)
    edit_dist_source: float = 0.0
    min_dist_prev: float | None = None


def deterministic_checks(source, paraphrase, previous, count_tokens):
    """All hard/soft checks except the LLM judge. previous = earlier accepted paraphrases."""
    tokens = count_tokens(paraphrase)
    checks = SlotChecks(tokens=tokens, edit_dist_source=token_edit_distance(source, paraphrase))
    if not paraphrase.strip():
        checks.hard.append("empty")
    if tokens > TOKEN_LIMIT:
        checks.hard.append(f"tokens {tokens} > {TOKEN_LIMIT}")
    elif tokens > TOKEN_TARGET:
        checks.soft.append("over_target")
    if normalize_text(paraphrase) == normalize_text(source):
        checks.hard.append("identical to source")
    for other in previous:
        if normalize_text(paraphrase) == normalize_text(other):
            checks.hard.append("identical to an earlier paraphrase")
            break
    checks.hard.extend(check_protected(source, paraphrase))
    checks.hard.extend(check_framing(source, paraphrase))
    if previous:
        checks.min_dist_prev = min(token_edit_distance(paraphrase, other) for other in previous)
    if (checks.edit_dist_source < DIVERSITY_VS_SOURCE
            or (checks.min_dist_prev is not None and checks.min_dist_prev < DIVERSITY_VS_PREVIOUS)):
        checks.soft.append("low_diversity")
    return checks


def run_validation_rounds(cases, *, regenerate_batch, judge_batch, count_tokens,
                          max_retries=MAX_RETRIES):
    """Validate every paraphrase slot, regenerating hard failures up to max_retries times.

    cases: {case_id: {"source": str, "slots": [{"strategy": str, "text": str}, ...]}}
    regenerate_batch(requests) -> {(case_id, slot_index): new_text}; each request is
        {id, slot, strategy, source, failed_text, failures, other_paraphrases}.
    judge_batch(pairs) -> {(case_id, slot_index): {equivalent, added, removed, changed}};
        each pair is {id, slot, source, paraphrase}. Every requested key must come back.

    Returns {case_id: {"source", "slots": [slot dict], "n_effective"}} where each slot dict
    has strategy, text, status ("ok" | "rejected"), attempts, tokens, edit_dist_source,
    min_dist_prev, soft flags, judge verdict, and history of failed attempts.
    """
    state = {}
    for case_id, case in cases.items():
        state[case_id] = {
            "source": case["source"],
            "slots": [{"strategy": slot["strategy"], "text": slot["text"], "attempts": 1,
                       "status": "pending", "history": []} for slot in case["slots"]],
        }
    judge_cache = {}

    for round_index in range(max_retries + 1):
        # 1) deterministic checks for every non-rejected slot, in slot order
        checks_by_key = {}
        for case_id, case in state.items():
            accepted_texts = []
            for i, slot in enumerate(case["slots"]):
                if slot["status"] == "rejected":
                    continue
                checks = deterministic_checks(case["source"], slot["text"], accepted_texts,
                                              count_tokens)
                checks_by_key[(case_id, i)] = checks
                if not checks.hard:
                    accepted_texts.append(slot["text"])

        # 2) judge whatever passed deterministic checks and isn't cached yet
        to_judge = [
            {"id": case_id, "slot": i, "source": state[case_id]["source"],
             "paraphrase": state[case_id]["slots"][i]["text"]}
            for (case_id, i), checks in checks_by_key.items()
            if not checks.hard and (state[case_id]["source"], state[case_id]["slots"][i]["text"])
            not in judge_cache
        ]
        if to_judge:
            verdicts = judge_batch(to_judge)
            for pair in to_judge:
                key = (pair["id"], pair["slot"])
                if key not in verdicts:
                    raise ValueError(f"judge output missing {key}")
                judge_cache[(pair["source"], pair["paraphrase"])] = verdicts[key]

        # 3) decide status; queue regenerations
        requests = []
        for (case_id, i), checks in checks_by_key.items():
            case = state[case_id]
            slot = case["slots"][i]
            failures = list(checks.hard)
            verdict = None
            if not failures:
                verdict = judge_cache[(case["source"], slot["text"])]
                failures = judge_failures(verdict)
            slot.update({"tokens": checks.tokens, "edit_dist_source": checks.edit_dist_source,
                         "min_dist_prev": checks.min_dist_prev, "flags": list(checks.soft),
                         "judge": verdict})
            if not failures:
                slot["status"] = "ok"
                continue
            slot["history"].append({"text": slot["text"], "failures": failures})
            if round_index < max_retries:
                slot["status"] = "pending"
                requests.append({
                    "id": case_id, "slot": i, "strategy": slot["strategy"],
                    "source": case["source"], "failed_text": slot["text"], "failures": failures,
                    "other_paraphrases": [s["text"] for j, s in enumerate(case["slots"])
                                          if j != i and s["status"] != "rejected"],
                })
            else:
                slot["status"] = "rejected"

        if not requests:
            break
        new_texts = regenerate_batch(requests)
        for request in requests:
            key = (request["id"], request["slot"])
            if key not in new_texts:
                raise ValueError(f"regeneration output missing {key}")
            slot = state[request["id"]]["slots"][request["slot"]]
            slot["text"] = new_texts[key]
            slot["attempts"] += 1

    for case in state.values():
        for slot in case["slots"]:
            if slot["status"] == "pending":  # loop ended right after a regeneration
                slot["status"] = "rejected"
        case["n_effective"] = sum(slot["status"] == "ok" for slot in case["slots"])
    return state
