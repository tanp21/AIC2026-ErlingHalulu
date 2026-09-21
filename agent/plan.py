"""The first reasoning step: one query -> a search plan -> the round-1 batch.

DeepSeek writes the plan; everything after that is deterministic so round 1 is
fast and testable. The planner is untrusted input here: every field is coerced,
trimmed and capped before it becomes a server request.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re

from .prompts import ANALYZE_SYSTEM
from .vi_numbers import spell_numbers

MAX_SCENES = 4
MAX_VARIANTS = 5  # caption + up to 4 paraphrases
MAX_PHRASES = 3
MAX_CAPTIONS = 2
MAX_READINGS = 2
MAX_TEXT = 300
BATCH_LIMIT = 12
TRAKE_TIGHT_GAP = 45
ANSWER_SOURCES = {"on_screen_text", "visual_count", "visual_attribute", "speech", "world_knowledge"}

#: A question that asks for an *amount* is answered by a sentence that states one, so
#: the terms which find it are the ones such a sentence must contain: the substance and
#: a measure. Measured on the round-3 soy-sauce query, whose answer line reads
#: "…hai muỗng cà phê nước tương Phú Sĩ…": BM25 over the transcript passages ranks the
#: video 30th of 84 for `nước tương` alone, 21st for `lượng nước tương`, 7th for
#: `lượng nước tương muỗng`, and 9th for `muỗng cà phê nước tương`. The bare term fails
#: because of boilerplate -- "nước tương Phú Sĩ" is the show's sponsor and is said in
#: every episode, so the one term that should discriminate ranks the ad reads. A
#: seasoning *verb* from the question does not help (it is not what the video says).
QUANTITY_MARKERS = ("lượng", "bao nhiêu", "số lượng", "mấy")


@dataclass
class Scene:
    description_vi: str
    visual_en: list[str]


@dataclass
class Plan:
    summary_vi: str
    scenes: list[Scene]
    target_scene: int = 0
    # Other plausible pictures of an ambiguous target scene, each searched on its own.
    readings_en: list[str] = field(default_factory=list)
    events_en: list[str] = field(default_factory=list)
    transcript_vi: list[str] = field(default_factory=list)
    caption_en: list[str] = field(default_factory=list)
    ocr: list[str] = field(default_factory=list)
    lookups: list[str] = field(default_factory=list)
    question_vi: str | None = None
    answer_source: str | None = None

    def to_dict(self):
        return asdict(self)


def _text(value, limit=MAX_TEXT):
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:limit]


def _texts(values, limit):
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    seen, result = set(), []
    for value in values:
        text = _text(value)
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            result.append(text)
    return result[:limit]


def analysis_messages(task):
    expected = ""
    if task.kind == "trake" and task.event_count:
        expected = f"\nThis trake query lists {task.event_count} events; events_en must have exactly {task.event_count} entries."
    return [
        {"role": "system", "content": ANALYZE_SYSTEM},
        {"role": "user", "content": (
            f"Task type: {task.kind}\nQuery id: {task.query_id}{expected}\n"
            f"Query:\n{task.text}\n\nReturn the JSON plan."
        )},
    ]


def parse_plan(data, task) -> Plan:
    if not isinstance(data, dict):
        raise ValueError("plan must be a JSON object")
    scenes = []
    for raw in data.get("scenes") or []:
        if not isinstance(raw, dict):
            continue
        visual = _texts(raw.get("visual_en"), MAX_VARIANTS)
        if visual:
            scenes.append(Scene(_text(raw.get("description_vi")), visual))
    scenes = scenes[:MAX_SCENES]
    events = _texts(data.get("events_en"), 6) if task.kind == "trake" else []
    if not scenes and events:
        scenes = [Scene("", [event]) for event in events[:MAX_SCENES]]
    if not scenes:
        raise ValueError("plan has no visual scene")
    try:
        target = int(data.get("target_scene") or 0)
    except (TypeError, ValueError):
        target = 0
    answer_source = data.get("answer_source")
    return Plan(
        summary_vi=_text(data.get("summary_vi")) or task.text[:MAX_TEXT],
        scenes=scenes,
        target_scene=min(max(target, 0), len(scenes) - 1),
        readings_en=_texts(data.get("readings_en"), MAX_READINGS) if task.kind != "trake" else [],
        events_en=events,
        transcript_vi=_texts(data.get("transcript_vi"), MAX_PHRASES),
        caption_en=_texts(data.get("caption_en"), MAX_CAPTIONS),
        # BM25 on a 1-2 character string ("kg") matches half the corpus.
        ocr=[text for text in _texts(data.get("ocr"), MAX_PHRASES)
             if len(re.sub(r"\W", "", text)) >= 3],
        lookups=_texts(data.get("lookups"), 2),
        question_vi=(_text(data.get("question_vi"), 500) or None) if task.kind == "qa" else None,
        answer_source=answer_source if task.kind == "qa" and answer_source in ANSWER_SOURCES else None,
    )


def transcript_text(phrase: str, spoken_numbers: bool = False) -> str:
    """A transcript query as the index spells numbers.

    The deployed index is built from cleaned transcripts, which write numbers as
    digits ("130.000" matches, "một trăm ba mươi ngàn" does not). Raw ASR output
    spells them out; `spoken_numbers` is for a server indexing that instead.
    """
    return spell_numbers(phrase) if spoken_numbers else phrase


def quantity_queries(plan: Plan) -> list[dict]:
    """Audio phrasings for a question whose answer states an amount, else nothing.

    The plan's `transcript_vi` already names the substance ("nước tương"); asking for
    it alone is what finds the sponsor reads instead of the recipe. Two shapes, both
    measured on the soy-sauce query: the question's own quantity word with the bare
    measure, and the spelled-out unit the answer sentence uses.

    Only when the plan says the answer is *spoken*. An amount read off a slide, a
    chart or a scale ("vùng này gồm bao nhiêu tỉnh?") is not in the transcript, and
    the round-1 batch is capped at `BATCH_LIMIT` and truncated in order, so these two
    queries would displace the OCR ones that do answer those.
    """
    if plan.answer_source != "speech":
        return []
    question = (plan.question_vi or "").lower()
    marker = next((word for word in QUANTITY_MARKERS if word in question), None)
    if marker is None or not plan.transcript_vi:
        return []
    substance = plan.transcript_vi[0]
    shapes = (f"{marker} {substance} muỗng", f"muỗng cà phê {substance}")
    return [
        {"kind": "audio", "id": f"r1.q{index}", "text": text, "method": "keyword", "limit": 100}
        for index, text in enumerate(shapes)
    ]


def initial_queries(plan: Plan, task, spoken_numbers: bool = False) -> list[dict]:
    """Round-1 batch: every channel the plan names, capped at the server's batch size.

    Built in priority order and truncated: the target scene first, then temporal
    structure, then transcript and OCR, then the remaining scenes. Each scene is one
    visual query whose paraphrases the server fuses into the caption's vector.
    """
    def visual(index):
        caption, *paraphrases = plan.scenes[index].visual_en
        query = {"kind": "visual", "id": f"r1.s{index}", "text": caption, "limit": 200}
        if paraphrases:
            query["paraphrases"] = paraphrases[:4]
        return query

    first = [visual(plan.target_scene)]
    # A reading is a different picture of the same scene, so it is its own query: fusing
    # it into the caption's vector would blur both (the paraphrase benchmark fuses only
    # faithful rewordings).
    readings = [
        {"kind": "visual", "id": f"r1.r{index}", "text": text, "limit": 200}
        for index, text in enumerate(plan.readings_en)
    ]
    rest = [visual(index) for index in range(len(plan.scenes)) if index != plan.target_scene]
    # Computed here because the spoken block below gives up its semantic twin for it.
    quantity = quantity_queries(plan)

    temporal = []
    events = plan.events_en if task.kind == "trake" else [
        scene.visual_en[0] for scene in plan.scenes
    ]
    if len(events) >= 2:
        base = {"kind": "temporal", "events": events[:6], "limit": 30}
        if task.kind == "trake":
            # Every TRAKE key so far (rounds 1 and 2) puts consecutive events within
            # ~15 s; a 300 s gap let the DP take one event from far away in the
            # video (p2-8's last event at +5 min). The loose path stays as a fallback.
            temporal = [{**base, "id": "r1.t", "max_gap_seconds": TRAKE_TIGHT_GAP},
                        {**base, "id": "r1.tw", "max_gap_seconds": 300}]
            if plan.transcript_vi:
                spoken_text = transcript_text(plan.transcript_vi[0], spoken_numbers)
                for query in temporal:
                    query["audio_text"] = spoken_text
        else:
            temporal = [{**base, "id": "r1.t", "max_gap_seconds": 180}]

    spoken = []
    for index, phrase in enumerate(plan.transcript_vi):
        variant = transcript_text(phrase, spoken_numbers)
        spoken.append({"kind": "audio", "id": f"r1.a{index}", "text": variant,
                       "method": "keyword", "limit": 100})
        # The semantic twin of the first phrase gives up its slot to the quantity
        # queries: it asks the same phrase again over a channel that measured worse
        # than keyword on exactly this kind of question ("semantic: nước tương" did
        # not return the target at all, keyword ranked it 30th and the quantity
        # phrasings 7th and 9th), and the batch is full, so a query added is a query
        # displaced -- three runs of it dropped the OCR query and a secondary scene.
        if index == 0 and not quantity:
            spoken.append({"kind": "audio", "id": "r1.a0s", "text": variant,
                           "method": "semantic", "limit": 100})
    written = [
        {"kind": "ocr", "id": f"r1.o{index}", "text": text, "limit": 50}
        for index, text in enumerate(plan.ocr)
    ]
    described = [
        {"kind": "caption", "id": f"r1.c{index}", "text": text, "limit": 100}
        for index, text in enumerate(plan.caption_en)
    ]
    # Ahead of the captions and the other scenes: when the question asks for an amount,
    # the channel that answers it is speech, and the batch is truncated in this order.
    return (first + readings + temporal + quantity
            + described + spoken + written + rest)[:BATCH_LIMIT]


def expected_events(plan: Plan, task) -> int:
    return task.event_count or len(plan.events_en) or 2


def describe_scenes(plan: Plan) -> str:
    lines = [f"  scene {index}: {scene.description_vi or scene.visual_en[0]}"
             for index, scene in enumerate(plan.scenes)]
    return "Scenes:\n" + "\n".join(lines) + "\n"
