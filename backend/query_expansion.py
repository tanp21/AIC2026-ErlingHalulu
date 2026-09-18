"""Gemini query expansion for PE-Core visual search: one holistic caption plus
2-5 single-facet aspect queries, all short enough to fit PE-Core's text tower
without silent truncation.

See docs/reports/visual-query-expansion-benchmark-2026.md for the benchmark
that validated (or didn't) shot-level fusion of these expansions, and the plan
this implements for the full scoring/fusion design.
"""
import json
import threading
from dataclasses import dataclass, field

PE_CORE_MODEL = "hf-hub:timm/PE-Core-bigG-14-448"
GEMINI_MODEL = "gemini-3.1-flash-lite"

TOKEN_TARGET = 50
TOKEN_LIMIT = 72  # PE-Core's context_length; content beyond this is silently truncated.

FACETS = ("people", "clothing", "object", "action", "setting", "text", "other")

_tokenizer = None
_tokenizer_lock = threading.Lock()


class ExpansionError(Exception):
    pass


def _tokenizer_instance():
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                import open_clip
                _tokenizer = open_clip.get_tokenizer(PE_CORE_MODEL)
    return _tokenizer


def count_tokens(text):
    """Untruncated token count including SOT/EOT, matching PE-Core's 72-slot context.

    tok.encode() returns the raw BPE length with no SOT/EOT and no truncation --
    the tokenized *tensor*'s non-zero count caps at 72 and can't reveal truncation,
    so counts must come from encode(), not from tokenizing into the fixed-size tensor.
    """
    return len(_tokenizer_instance().encode(text)) + 2


EXPAND_PROMPTS = {
    "v1": (
        "These are Vietnamese video-retrieval query prompts, written as instructions to a "
        "human competitor. For each one, produce:\n"
        "1. A \"holistic\" caption: one short English sentence, caption-style (like photo "
        "alt-text, not an instruction), describing the single most distinctive visual moment. "
        "Drop meta-commentary (\"find\", \"the video shows\", \"this is the moment when\").\n"
        "2. 2 to 5 \"aspects\": short English phrases, each describing ONE visual facet only "
        "(people/count, clothing, an object, an action, the setting, or on-screen text/proper "
        "nouns). Each aspect must be visually self-sufficient on its own (\"man in a black cap "
        "sitting next to a cardboard box\", not just \"black cap\"). Aspects must not paraphrase "
        "the holistic caption or each other. Use more aspects for prompts with more distinct "
        "visual detail, fewer for simple prompts. Drop non-visual content (a question asked in "
        "a QA prompt, sounds, narration).\n"
        "Keep every item under about 30 English words. Do not answer questions, explain, or add "
        "information not present in the original. Keep each id unchanged.\n\n"
    ),
    "v2": (
        "These are Vietnamese video-retrieval query prompts, written as instructions to a "
        "human competitor. For each one, produce:\n"
        "1. A \"holistic\" caption: one short English sentence, caption-style (like photo "
        "alt-text, not an instruction), describing the single most distinctive visual moment. "
        "Drop meta-commentary (\"find\", \"the video shows\", \"this is the moment when\").\n"
        "2. 1 to 5 \"aspects\": short English phrases. Each aspect must name a RARE, "
        "DISTINCTIVE visual detail that would help tell this shot apart from other similar "
        "shots -- an unusual color/pattern, a specific count, on-screen text, a proper noun, or "
        "an uncommon object/action. Do NOT write a generic facet (a plain \"setting\" or "
        "\"clothing\" description with no distinctive element) just to fill a quota -- if the "
        "prompt has only 1 distinctive extra detail beyond the holistic caption, return only 1 "
        "aspect; if it has none, return 0 aspects. Each aspect must still be visually "
        "self-sufficient on its own (\"man in a black cap sitting next to a cardboard box\", "
        "not just \"black cap\"). Aspects must not paraphrase the holistic caption or each "
        "other, and must not restate generic scene context already implied by the holistic "
        "caption (e.g. don't add \"outdoor setting\" if the holistic caption already describes "
        "an outdoor scene). Drop non-visual content (a question asked in a QA prompt, sounds, "
        "narration).\n"
        "Keep every item under about 30 English words. Do not answer questions, explain, or add "
        "information not present in the original. Keep each id unchanged.\n\n"
    ),
}

SHORTEN_PROMPT = (
    "This English CLIP-retrieval query is still too long for the model's token budget. "
    "Shorten it to at most 20 English words, keeping only the single most distinctive, rare "
    "visual detail (a specific color, count, object, on-screen text, or proper noun) plus its "
    "immediate context. Drop everything generic. Keep the id unchanged.\n\n"
)


@dataclass
class TokenItem:
    text: str
    tokens_before: int
    tokens_after: int
    was_shortened: bool = False
    over_target: bool = False
    text_original: str | None = None
    dropped: str | None = None


@dataclass
class Expansion:
    holistic: TokenItem
    aspects: list[TokenItem] = field(default_factory=list)
    aspect_count_generated: int = 0
    aspect_count_used: int = 0
    prompt_version: str = "v1"


def _apply_token_policy(text, *, shorten_fn, is_holistic):
    """Implements the plan's token policy. shorten_fn(text) -> shortened text | None."""
    tokens_before = count_tokens(text)
    if tokens_before <= TOKEN_TARGET:
        return TokenItem(text=text, tokens_before=tokens_before, tokens_after=tokens_before)

    if tokens_before <= TOKEN_LIMIT:
        shortened = shorten_fn(text)
        if shortened:
            tokens_after = count_tokens(shortened)
            if tokens_after <= TOKEN_TARGET:
                return TokenItem(text=shortened, tokens_before=tokens_before,
                                 tokens_after=tokens_after, was_shortened=True,
                                 text_original=text)
        # Shortening failed or wasn't attempted usefully: keep the original, over budget.
        return TokenItem(text=text, tokens_before=tokens_before, tokens_after=tokens_before,
                         over_target=True)

    # Over the hard limit: up to 2 shorten passes, then drop (aspect) or raise (holistic).
    current = text
    for _ in range(2):
        shortened = shorten_fn(current)
        if not shortened:
            break
        current = shortened
        tokens_after = count_tokens(current)
        if tokens_after <= TOKEN_LIMIT:
            return TokenItem(text=current, tokens_before=tokens_before, tokens_after=tokens_after,
                             was_shortened=True, over_target=tokens_after > TOKEN_TARGET,
                             text_original=text)
    if is_holistic:
        raise ExpansionError(f"holistic caption exceeds {TOKEN_LIMIT} tokens after shortening: {text!r}")
    return TokenItem(text=text, tokens_before=tokens_before, tokens_after=count_tokens(current),
                     dropped="exceeds_context", text_original=text)


def _dedup_aspects(aspects):
    seen = set()
    kept = []
    for aspect in aspects:
        key = " ".join(aspect.text.lower().split())
        if key in seen:
            continue
        seen.add(key)
        kept.append(aspect)
    return kept


def _call_gemini(client, prompt, payload, schema, timeout_s):
    config = {"temperature": 0.0, "thinking_config": {"thinking_level": "minimal"},
              "response_mime_type": "application/json", "response_json_schema": schema}
    if timeout_s is not None:
        config["http_options"] = {"timeout": int(timeout_s * 1000)}
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=prompt + json.dumps(payload, ensure_ascii=False),
        config=config,
    )
    return json.loads(response.text)


_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "holistic": {"type": "string"},
        "aspects": {
            "type": "array", "minItems": 0, "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {"text": {"type": "string"}, "facet": {"type": "string", "enum": list(FACETS)}},
                "required": ["text", "facet"], "additionalProperties": False,
            },
        },
    },
    "required": ["holistic", "aspects"], "additionalProperties": False,
}


def expand(text, client, *, prompt_version="v1", timeout_s=8.0):
    """Expand one Vietnamese (or English) prompt into a holistic caption + aspects."""
    prompt = EXPAND_PROMPTS[prompt_version]
    raw = _call_gemini(client, prompt, [{"id": "q", "text": text}], _SINGLE_SCHEMA, timeout_s)
    return _finalize(raw, prompt_version)


def _shorten(client, text, timeout_s):
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}
    try:
        raw = _call_gemini(client, SHORTEN_PROMPT, {"text": text}, schema, timeout_s)
        return raw.get("text") or None
    except Exception:
        return None


def _finalize(raw, prompt_version, client=None, timeout_s=8.0):
    def shorten_fn(t):
        return _shorten(client, t, timeout_s) if client is not None else None

    holistic = _apply_token_policy(raw["holistic"], shorten_fn=shorten_fn, is_holistic=True)
    aspect_items = []
    for aspect in raw["aspects"]:
        item = _apply_token_policy(aspect["text"], shorten_fn=shorten_fn, is_holistic=False)
        item.facet = aspect["facet"]  # type: ignore[attr-defined]
        aspect_items.append(item)
    aspect_items = _dedup_aspects(aspect_items)
    used = [a for a in aspect_items if a.dropped is None]
    return Expansion(holistic=holistic, aspects=aspect_items,
                     aspect_count_generated=len(raw["aspects"]), aspect_count_used=len(used),
                     prompt_version=prompt_version)


_BATCH_SCHEMA_TEMPLATE = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "holistic": {"type": "string"},
            "aspects": {
                "type": "array", "minItems": 0, "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}, "facet": {"type": "string", "enum": list(FACETS)}},
                    "required": ["text", "facet"], "additionalProperties": False,
                },
            },
        },
        "required": ["id", "holistic", "aspects"], "additionalProperties": False,
    },
}


def expand_batch(cases, client, *, prompt_version="v1", batch_size=12, timeout_s=30.0):
    """Offline batch variant. cases: list of {id, text}. Returns {id: Expansion}."""
    prompt = EXPAND_PROMPTS[prompt_version]
    results = {}
    for start in range(0, len(cases), batch_size):
        batch = cases[start:start + batch_size]
        payload = [{"id": case["id"], "text": case["text"]} for case in batch]
        rows = _call_gemini(client, prompt, payload, _BATCH_SCHEMA_TEMPLATE, timeout_s)
        by_id = {row["id"]: row for row in rows}
        if set(by_id) != {case["id"] for case in batch}:
            raise ExpansionError(f"batch starting at {start}: ids changed/missing")
        for case in batch:
            results[case["id"]] = _finalize(by_id[case["id"]], prompt_version, client, timeout_s)
    return results
