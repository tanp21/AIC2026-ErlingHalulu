"""The retrieval agent's DeepSeek planner, offered to a person searching by hand.

`agent/plan.py` already turns one query into per-channel search text: English
captions for PE-Core, Vietnamese phrases for the transcript index, literal
strings for OCR. That step is otherwise reachable only by running the whole
agent, so this module runs it alone and hands the result to the Search tab as
suggestions -- nothing here issues a search or edits a request.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agent.llm import LLMError
from agent.plan import analysis_messages, parse_plan

try:
    from .query_expansion import count_tokens
except Exception:  # pragma: no cover - only missing on a split-host teammate machine
    count_tokens = None

ATTEMPTS = 2


@dataclass
class Candidate:
    text: str
    #: PE-Core token count, or None when the tokenizer is unavailable (see `_tokens`).
    tokens: int | None = None
    #: False for another scene's or an alternate reading's caption: still visually
    #: accurate, but describing a less distinctive moment than the target scene, so
    #: in isolation it tends to match many other videos (measured: a target-scene
    #: caption ranked the gold video #1 on a real query; a same-plan secondary
    #: caption for a generic moment in the same clip ranked it below #45).
    primary: bool = True


@dataclass
class Refinement:
    summary_vi: str = ""
    visual: list[Candidate] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    ocr: list[str] = field(default_factory=list)
    events: list[Candidate] = field(default_factory=list)
    lookups: list[str] = field(default_factory=list)


def _tokens(text):
    """Advisory only: a count the UI can warn on, never a reason to drop a candidate.

    The tokenizer needs the PE-Core files in the Hugging Face cache, which a
    teammate running against a remote compute plane (AIC_CORE_URL) does not have.
    """
    if count_tokens is None:
        return None
    try:
        return count_tokens(text)
    except Exception:
        return None


def _dedup(texts):
    seen, result = set(), []
    for text in texts:
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result


def _from_plan(plan) -> Refinement:
    target = plan.scenes[plan.target_scene]
    primary_texts = _dedup(target.visual_en)
    primary_seen = {text.casefold() for text in primary_texts}

    secondary_raw = []
    for index, scene in enumerate(plan.scenes):
        if index != plan.target_scene and scene.visual_en:
            secondary_raw.append(scene.visual_en[0])
    secondary_raw.extend(plan.readings_en)
    secondary_texts = [text for text in _dedup(secondary_raw) if text.casefold() not in primary_seen]

    visual = [Candidate(text, _tokens(text), primary=True) for text in primary_texts]
    visual += [Candidate(text, _tokens(text), primary=False) for text in secondary_texts]
    return Refinement(
        summary_vi=plan.summary_vi,
        visual=visual,
        transcript=list(plan.transcript_vi),
        ocr=list(plan.ocr),
        events=[Candidate(text, _tokens(text)) for text in plan.events_en],
        lookups=list(plan.lookups),
    )


def refine(task, client) -> Refinement:
    """One DeepSeek planning call for `task`, as per-channel query suggestions.

    Thinking stays off: planning is 5-10x faster without it and loses little
    (see `deepseek_plan_thinking` in agent/config.py), and a person is waiting.
    """
    last_error = None
    for _ in range(ATTEMPTS):
        try:
            data = client.chat_json(analysis_messages(task), thinking=False)
            return _from_plan(parse_plan(data, task))
        except (LLMError, ValueError) as exc:
            last_error = exc
    raise LLMError(f"planning failed: {last_error}")
