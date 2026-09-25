"""Text-only evidence for a person judging candidates: what the words around a keyframe say.

The person is the judge, so nothing here looks at a picture. A moment has three kinds of
words around it -- what is said (transcript), what is written on screen (OCR), and what a
captioning model wrote about each shot -- and a query is a handful of details that each
either show up in them or do not. The checklist marks each detail *supported* (with the
quote and where), *contradicted* or *not mentioned*. It is a hint for a human to skim, not a
verdict: "not mentioned" is the common case (people do not say what is on screen) and is
never evidence against a candidate.

Two things this module is careful about, both learned on a real query:

- **A span, not a shot.** A query describes consecutive scenes and a candidate is one frame
  of one of them, so the apron, the flowers and the plate of a cooking query are in the shots
  *before* the frame that shows a clam being lifted. Evidence is read from `BEFORE_SECONDS`
  before to `AFTER_SECONDS` after the moment.
- **Excerpts, not paragraphs.** A shot caption runs ~3,400 characters and the detail is
  often past character 2,000, so handing DeepSeek the raw span (37k characters for three
  candidates) made it answer "nothing found" for the gold moment, and cutting each caption
  short cut the detail off. Instead each detail is turned into search terms once
  (`query_details`), sentences that contain them are selected per candidate and per detail,
  and DeepSeek only judges those few sentences.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
import math
import re
import threading
from time import monotonic
import unicodedata

from .llm import LLMError
from .prompts import DETAILS_SYSTEM, MATCH_CAPTION_SYSTEM, MATCH_SYSTEM, SUGGEST_SYSTEM, VISION_MATCH_SYSTEM

logger = logging.getLogger("agent")

#: How much of the video around a candidate its checklist may draw on. Measured on a
#: query whose gold video showed its details in the four shots before the asked-for frame.
BEFORE_SECONDS = 45.0
AFTER_SECONDS = 25.0
#: Candidates of one video further apart than this need separate calls.
CLUSTER_GAP_SECONDS = 70.0
MAX_CLUSTER = 8
#: Sentences kept per candidate and detail, and per call in total.
EXCERPTS_PER_DETAIL = 4
MAX_EXCERPTS = 120
MAX_SENTENCE_CHARS = 320
MIN_SENTENCE_CHARS = 12
RARE_SHARE = 0.25
SUGGEST_EXCERPTS = 12
STATUSES = ("supported", "partial", "contradicted", "not_mentioned")
VERDICTS = {"exact": "supported", "partial": "partial", "conflict": "contradicted"}
VISION_MATCH_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"},
        "details": {"type": "array", "items": {"type": "object", "properties": {
            "detail": {"type": "integer"}, "why": {"type": "string"},
            "verdict": {"type": "string", "enum": ["exact", "partial", "conflict", "unrelated"]},
            "note_vi": {"type": "string"},
        }, "required": ["detail", "why", "verdict"]}},
    }, "required": ["id", "details"]}}},
    "required": ["candidates"],
}
SOURCES = ("speech", "ocr", "caption")
MAX_DETAILS = 8
#: How many candidates one vision call judges at once, and how wide each frame is fetched. A judge
#: sees more of a small frame than a caption ever wrote down, and is asked at the pixel level, so it
#: does not need the full resolution `answer` reads text at. One candidate per call: a candidate now
#: carries the keyframes of the shots around it (up to `VISION_CONTEXT_MAX` a side, so a dozen-plus
#: images), and a call that mixes two candidates' stories into one judgement was measured worse than
#: the same images judged alone.
VISION_CHUNK = 1
VISION_IMAGE_WIDTH = 480
#: A query is often a short story of several shots ("five people gather, one carries something away,
#: a man wakes"; "the clip starts with X, ends with Y"), and no one frame shows all of it -- a
#: caption's checklist already reads a span around a moment for the same reason
#: (`BEFORE_SECONDS`/`AFTER_SECONDS`); vision reads the story around it, shot by shot, out to
#: `VISION_CONTEXT_SECONDS`. Two things were reported wrong and both are measured fixes:
#: - **A shot needs more than one frame.** On a p2-1 candidate, "the man wakes up"/"he shakes the
#:   lion" happen in the video's shots 4 and 5; one frame per shot had the judge answer "unrelated" in
#:   every run, where a shot's own keyframes (usually three) had it answer `exact` every time -- and
#:   still `unrelated` for the same detail on a different video, which is the property that matters.
#: - **A shot COUNT cap silently loses reach on a densely-cut video.** A p2-2 candidate's story spans
#:   the clip's start and end, 19 s apart in TIME but seven 1-2 s shots apart on that video; a cap of
#:   four shots per side stopped three shots short of the second beat while `VISION_CONTEXT_SECONDS`
#:   had 26 s of budget left unused. So the cap is on FRAMES, not shots: every shot within
#:   `VISION_CONTEXT_SECONDS` is walked outward from the moment, the nearest `VISION_CONTEXT_DENSE`
#:   shots each side keep every one of their keyframes (for a beat that is a change), the rest keep one
#:   (to reach as far as the frame budget allows), and `VISION_CONTEXT_MAX` frames a side is the actual
#:   limit -- so a video cut into many short shots still reaches minutes out, and one cut into few long
#:   ones still reads each one closely. Measured on the reported p2-2 candidate (three monkeys graffiti
#:   19 s / seven shots later): `exact` in every run at `VISION_CONTEXT_MAX = 16`, and a frame 4.5
#:   minutes from either story beat still came back `unrelated` for both, not a false positive from the
#:   wider reach. Where a server has no shot boundaries, the keyframe neighbourhood is used instead.
VISION_CONTEXT_RADIUS = 10
VISION_CONTEXT_SECONDS = 180.0
VISION_CONTEXT_DENSE = 1
VISION_CONTEXT_MAX = 16
VISION_CONTEXT_FRAMES = 4
#: Priority a status is replaced at: vision looks at the exact candidate, so it may upgrade a check a
#: caption's guess understated, but never on a lower-priority verdict than what is already there.
STATUS_RANK = {"supported": 3, "partial": 2, "contradicted": 1, "not_mentioned": 0}
MAX_TERMS = 8


@dataclass
class QueryDetails:
    """The query broken into checkable details, each with the words that would mention it."""
    details: list[dict] = field(default_factory=list)   # {"vi", "terms_en", "terms_vi"}
    question: dict | None = None                          # {"terms_en", "terms_vi"} for a Q&A query

    def __bool__(self):
        return bool(self.details)

    @property
    def texts(self) -> list[str]:
        return [detail["vi"] for detail in self.details]


class EvidenceStore:
    """Per-video transcript, OCR and caption windows, each fetched once; a missing one is empty.

    Thread-safe: candidates of one video are summarised from several threads, and a
    video costs three requests however many of its moments are looked at.

    `maxsize` bounds it least-recently-used. An agent run touches a few dozen videos and
    then throws the store away, so the bound never bites there; the Search tab's checker
    keeps one store for the life of the server process, where an unbounded cache would
    eventually hold every video's transcript, OCR and captions at once.
    """

    #: The store's kinds and the client method each is fetched with.
    METHODS = {"transcript": "transcript", "ocr": "ocr", "captions": "caption_windows"}

    def __init__(self, search, maxsize: int = 192):
        self.search = search
        self.maxsize = max(1, int(maxsize))
        self._lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], list] = OrderedDict()

    def _get(self, kind: str, video_id: str) -> list:
        key = (kind, video_id)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        fetch = getattr(self.search, self.METHODS[kind], None)
        try:
            value = list(fetch(video_id)) if fetch else []
        except Exception:  # words are context; losing them must never cost the run
            logger.warning("evidence: no %s for %s", kind, video_id)
            value = []
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)
        return value

    def transcript(self, video_id: str) -> list:
        return self._get("transcript", video_id)

    def ocr(self, video_id: str) -> list:
        return self._get("ocr", video_id)

    def captions(self, video_id: str) -> list:
        return self._get("captions", video_id)


# -- words ---------------------------------------------------------------------------------
def fold(text: str) -> str:
    """Lower-case and accent-free, like the `vi_folded` analyzer the searches use."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(char for char in text if unicodedata.category(char) != "Mn").lower()


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", fold(text))


def _word_matches(term_word: str, word: str) -> bool:
    """Equal, or `word` is `term_word` plus a short ending (clam/clams, hold/holding)."""
    return word == term_word or (len(term_word) >= 4 and word.startswith(term_word) and len(word) - len(term_word) <= 3)


def _term_words(terms: list[str]) -> list[list[str]]:
    return [found for found in (words(term) for term in terms) if found]


def _matched(term_sets: list[list[str]], sentence_words: set[str]) -> int:
    return sum(1 for term in term_sets if all(any(_word_matches(part, word) for word in sentence_words) for part in term))


def sentences_of(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[\"“A-ZÀ-Ỹ])", " ".join(text.split()))
    return [part[:MAX_SENTENCE_CHARS] for part in parts if len(part) >= MIN_SENTENCE_CHARS]


# Number words that are not also ordinary words: "sau" (after), "nam" (male), "bay" (fly) are left out.
NUMBER_WORDS = {
    "hai": 2, "ba": 3, "bon": 4, "muoi": 10,
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "pair": 2, "couple": 2, "dozen": 12,
}


def counts_in(text: str) -> set[int]:
    """The quantities of two or more a text states, in digits or in words (Vietnamese or English)."""
    found = set()
    for word in words(text):
        value = int(word) if word.isdigit() else NUMBER_WORDS.get(word)
        if value is not None and 2 <= value <= 20:
            found.add(value)
    return found


#: Words naming a graphic device the BROADCAST draws onto footage afterwards to point at something --
#: a circle, an arrow, a box, a highlight -- rather than a physical object, animal or person actually
#: present in a shot. Reported on p2-6: a query's "2 vòng tròn đỏ xuất hiện để khoanh vùng vị trí 2
#: thanh niên" (two red circles appear to mark the two men) was read like "4 con nghêu trên đĩa" --
#: the same circle drawn once around each rider, seconds apart, only ever shows one at a time in
#: whatever frames were sampled, so the guard kept calling the true candidate a conflict. How many
#: times a marker like this is used is a count across the WHOLE STORY, at different moments; it is
#: not something a still or a handful of frames can be asked to total, the way a plate of shellfish
#: can. `count_mismatch` does not apply the numeric guard to a detail that names one of these.
MARKER_WORDS = ("vong tron", "khoanh vung", "khoanh tron", "mui ten", "o vuong",
               "khung hinh chu nhat", "danh dau", "to dam", "highlight", "circle", "arrow")

#: Said on the detail's OWN line, where the judge reads it, on top of the general rule in
#: `VISION_MATCH_SYSTEM`: a rule six bullets up a long prompt was measured being ignored -- asked
#: about the very mark on p2-6's gold frame, the judge answered "conflict" three times out of three
#: with "only one red circle, not two", though one instance of the mark over the subject is exactly
#: what the detail means. So the detail carries the instruction itself.
MARKER_HINT = (" [a mark the broadcast drew onto the footage: seeing it ONCE over the subject is "
               "\"exact\" -- never count these, and never answer \"conflict\" for a lower count]")


def _is_marker_detail(detail: str) -> bool:
    folded = fold(detail)
    return any(marker in folded for marker in MARKER_WORDS)


def count_mismatch(detail: str, excerpt: str) -> str | None:
    """Why a sentence cannot fully support a detail that names a quantity, or None.

    A caption says "two shrimp" and a detail says "4 con": a model asked whether the sentence
    supports the detail will often say yes, because the plate, the shrimp and the hands are all
    there. The number is the one part that can be checked without a model, so it is -- for objects
    actually IN the shot; see `MARKER_WORDS` for what this deliberately does not gate.
    """
    if _is_marker_detail(detail):
        return None
    need = counts_in(detail)
    if not need:
        return None
    have = counts_in(excerpt)
    if need <= have:
        return None
    wanted = "/".join(str(n) for n in sorted(need))
    return (f"chi tiết cần {wanted}, đoạn nói {'/'.join(str(n) for n in sorted(have))}" if have
            else f"chi tiết cần {wanted}, đoạn không nêu số lượng")


# -- the span ------------------------------------------------------------------------------
def span_around(store: EvidenceStore, video_id: str, start: float, end: float) -> dict[str, list[dict]]:
    """The speech, on-screen text and caption windows overlapping `start..end`, in time order."""
    def overlapping(rows):
        return sorted(
            ({"start": float(row["start"]), "end": float(row["end"]), "text": " ".join(str(row.get("text") or "").split())}
             for row in rows if row["end"] >= start and row["start"] <= end and str(row.get("text") or "").strip()),
            key=lambda row: row["start"])

    return {"speech": overlapping(store.transcript(video_id)),
            "ocr": overlapping(store.ocr(video_id)),
            "captions": overlapping(store.captions(video_id))}


def clusters(items: list[dict]) -> list[list[dict]]:
    """Group candidates so each group shares one call about one video (at most `MAX_CLUSTER`).

    Groups come back best-ranked first (`items` are in rank order), so the candidates a
    person looks at first are the first to get a checklist.
    """
    groups: list[list[dict]] = []
    rank = {id(item): index for index, item in enumerate(items)}
    by_video: dict[str, list[dict]] = {}
    for item in sorted(items, key=lambda item: (item["video_id"], item["time"])):
        by_video.setdefault(item["video_id"], []).append(item)
    for video_items in by_video.values():
        current: list[dict] = []
        for item in video_items:
            if current and (item["time"] - current[-1]["time"] > CLUSTER_GAP_SECONDS or len(current) >= MAX_CLUSTER):
                groups.append(current)
                current = []
            current.append(item)
        if current:
            groups.append(current)
    return sorted(groups, key=lambda group: min(rank[id(item)] for item in group))


def _span_of(group: list[dict]) -> tuple[float, float]:
    return min(i["time"] for i in group) - BEFORE_SECONDS, max(i["time"] for i in group) + AFTER_SECONDS


def excerpts_of(span: dict[str, list[dict]], term_sets_by_detail: list[dict[str, list[list[str]]]]) -> list[dict]:
    """Every sentence of the span that holds a term of some detail, with its score per detail.

    `term_sets_by_detail[i]` gives the terms for detail `i` as `{"text": [...], "caption": [...]}`:
    speech and OCR are Vietnamese, a caption is English but also copies out what is written
    in the scene, so it answers to both.

    A term counts for what it is rare in: "yellow" and "next to" are in half the sentences of a
    lion-dance caption and say nothing, "pumpkin" and "lion" are in a few and say a lot. The score
    of a sentence for a detail is the weight of the rarest term it holds, `log(1 + N / (1 + df))`
    (N sentences in the span, df of them holding the term), plus a quarter of each other's. Counting
    terms alone let a generic sentence from the next shot outrank the one sentence that named the
    thing; summing weights still let two common words beat one specific one.
    """
    rows = []
    for source, sentences in (("speech", span["speech"]), ("ocr", span["ocr"]), ("caption", span["captions"])):
        for row in sentences:
            for sentence in (sentences_of(row["text"]) if source == "caption" else [row["text"][:MAX_SENTENCE_CHARS]]):
                rows.append({"source": source, "at": row["start"], "end": row["end"], "text": sentence,
                             "words": set(words(sentence)), "kind": "caption" if source == "caption" else "text"})
    total = len(rows)
    weights: dict[tuple[int, str], list[float]] = {}
    for index, terms in enumerate(term_sets_by_detail):
        for kind in ("text", "caption"):
            document_frequency = [sum(1 for row in rows if row["kind"] == kind and _matched([term], row["words"]))
                                  for term in terms[kind]]
            weights[(index, kind)] = [math.log(1 + total / (1 + df)) for df in document_frequency]
    found = []
    for row in rows:
        scores = {}
        for index, terms in enumerate(term_sets_by_detail):
            held = [weight for term, weight in zip(terms[row["kind"]], weights[(index, row["kind"])])
                    if _matched([term], row["words"])]
            if held:
                # The rarest word it holds, and a quarter of each other: three common words are still
                # not one specific one.
                scores[index] = round(max(held) + RARE_SHARE * (sum(held) - max(held)), 3)
        if scores:
            found.append({"source": row["source"], "at": row["at"], "end": row["end"], "text": row["text"], "scores": scores})
    return found


def select_excerpts(excerpts: list[dict], group: list[dict], detail_count: int, per_detail: int,
                    *, before: float = BEFORE_SECONDS, after: float = AFTER_SECONDS) -> list[dict]:
    """For each candidate and detail its best `per_detail` sentences within reach, numbered `e1`...

    Chosen per candidate so a moment with little around it is not crowded out by one in the
    middle of a rich stretch of the same video; a sentence two candidates share is kept once.

    `before`/`after` are how far "within reach" is. The agent keeps the measured defaults; the
    Search tab's checker widens them to the window the person asked it to check, and leans on
    `proximity` and the excerpt's own timestamp to say how far away the evidence was.
    """
    chosen: dict[int, dict] = {}
    for item in group:
        low, high = item["time"] - before, item["time"] + after
        near = [(index, e) for index, e in enumerate(excerpts) if e["end"] >= low and e["at"] <= high]
        for detail in range(detail_count):
            best = sorted((pair for pair in near if detail in pair[1]["scores"]),
                          key=lambda pair: (-pair[1]["scores"][detail], abs(pair[1]["at"] - item["time"])))[:per_detail]
            for index, e in best:
                chosen[index] = e
    ordered = sorted(chosen.items(), key=lambda pair: (pair[1]["at"], pair[0]))[:MAX_EXCERPTS]
    return [{**e, "id": f"e{number}"} for number, (_, e) in enumerate(ordered, 1)]


def _stamp(excerpt: dict) -> str:
    return f"{excerpt['source']} {excerpt['at']:.0f}s"


# -- the model calls -----------------------------------------------------------------------
def _clean_terms(raw) -> list[str]:
    return list(dict.fromkeys(" ".join(str(term).split()) for term in raw or [] if str(term).strip()))[:MAX_TERMS]


def _plan_reading(plan) -> str:
    """How the search plan read the query, in English: it is where "a yellow animal" became "a lion".

    The searches already found lion-dance videos for a query that never says lion, and the
    checklist has to look for the same thing or it will hunt for "animal" and read a yellow
    flower as the contradiction.
    """
    if plan is None:
        return ""
    lines = [scene.visual_en[0] for scene in plan.scenes if getattr(scene, "visual_en", None)]
    lines += list(getattr(plan, "caption_en", []) or []) + list(getattr(plan, "readings_en", []) or [])
    lines = [" ".join(str(line).split())[:220] for line in dict.fromkeys(lines) if str(line).strip()]
    return ("\n\nCách kế hoạch tìm kiếm hiểu truy vấn (tiếng Anh):\n" + "\n".join(f"- {line}" for line in lines[:8])) if lines else ""


#: Where a detail's weight starts before any candidate has been read, from how much it should tell
#: candidates apart (`distinctiveness`, 0-1 from DeepSeek). "a yellow animal" in a lion-dance video is
#: guessable from context and true of nearly every candidate; "carries the pumpkin away to hide it" is
#: true of one moment in the story. Reading candidates only refines this (`detail_weights`'s empirical
#: half): the prior already tells the checklist which details are worth more before it has read six.
PRIOR_MIN = 0.2
PRIOR_MAX = 1.0


def _distinctiveness(raw) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 1.0  # unscored: treat as distinctive rather than silently discounting it
    return min(max(value, 0.0), 1.0)


def _prior_weight(entry: dict) -> float:
    return PRIOR_MIN + (PRIOR_MAX - PRIOR_MIN) * entry.get("distinctiveness", 1.0)


def query_details(client, task, plan=None, *, system: str = DETAILS_SYSTEM) -> QueryDetails:
    """The query as 3-8 short details with the words that would mention each; empty if the model fails.

    `client` is anything with DeepSeek's `chat_json` -- the agent hands it its own DeepSeek
    client, the Search tab's checker hands it an `OpenRouterClient`, which accepts and
    ignores `thinking` for exactly this reason.

    `system` defaults to `DETAILS_SYSTEM`, the agent's own prompt. The checker passes
    `agent.prompts.CHECK_DETAILS_SYSTEM` instead, which adds a note about `task.text` possibly
    being labelled by search channel (`agent.checker.CheckQuery.description`) -- the agent's own
    `task.text` is never labelled, so its calls are unaffected by that note either way.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Loại: {task.kind}\nTruy vấn:\n{task.text}{_plan_reading(plan)}"},
    ]
    try:
        data = client.chat_json(messages, thinking=False, max_tokens=1500, temperature=0)
    except LLMError as exc:
        logger.warning("evidence: query details failed: %s", exc)
        return QueryDetails()
    if not isinstance(data, dict):
        return QueryDetails()
    details, seen = [], set()
    for raw in data.get("details") or []:
        entry = raw if isinstance(raw, dict) else {"vi": raw}
        text = " ".join(str(entry.get("vi") or "").split())[:120]
        if not text or text in seen:
            continue
        seen.add(text)
        details.append({"vi": text, "terms_en": _clean_terms(entry.get("terms_en")),
                        "terms_vi": _clean_terms(entry.get("terms_vi")),
                        "meaning_en": " ".join(str(entry.get("meaning_en") or "").split())[:240],
                        "same_en": _clean_terms(entry.get("same_en"))[:5],
                        "different_en": _clean_terms(entry.get("different_en"))[:5],
                        "distinctiveness": _distinctiveness(entry.get("distinctiveness"))})
    question = data.get("question")
    question = ({"terms_en": _clean_terms(question.get("terms_en")), "terms_vi": _clean_terms(question.get("terms_vi"))}
                if isinstance(question, dict) else None)
    return QueryDetails(details[:MAX_DETAILS], question if question and (question["terms_en"] or question["terms_vi"]) else None)


def _term_sets(entry: dict) -> dict[str, list[list[str]]]:
    return {"text": _term_words(entry["terms_vi"]), "caption": _term_words(entry["terms_en"] + entry["terms_vi"])}


def _ask(deepseek, system: str, entries: list[dict], excerpts: list[dict], *, english: bool) -> dict[int, dict[str, tuple[str, str]]] | None:
    """One judge call over `excerpts` of one language: detail index -> excerpt id -> (status, note).

    A caption is English and is judged in English, against the detail's English meaning and its
    same-thing / look-alike lists; speech and OCR are Vietnamese and are judged in Vietnamese,
    against the detail as the query wrote it. Asking an English caption to be compared with a
    Vietnamese phrase made the model translate and match in one step, and it got both wrong
    ("hoa riềng tía" read as purple). None when the call failed, `{}` when there was nothing to ask.
    """
    lines, shown = [], {}
    for index, entry in enumerate(entries):
        mine = [e for e in excerpts if index in e["scores"]]
        if not mine:
            continue
        lines.append(f"{index}. {(entry.get('meaning_en') or entry['vi']) if english else entry['vi']}")
        if english:
            lines += _meaning_lines(entry)
        lines += [f"   {e['id']} [{e['at']:.0f}s] {e['text']}" if english else f"   {e['id']} [{_stamp(e)}] {e['text']}" for e in mine]
        shown[index] = {e["id"] for e in mine}
    if not lines:
        return {}
    header = "Details of the query and their excerpts:\n" if english else "Chi tiết của truy vấn và các đoạn trích:\n"
    try:
        data = deepseek.chat_json([{"role": "system", "content": system}, {"role": "user", "content": header + "\n".join(lines)}],
                                  thinking=False, max_tokens=8000, temperature=0, salvage=True)
    except LLMError as exc:
        logger.warning("evidence: %s judge failed: %s", "caption" if english else "text", exc)
        return None
    judged: dict[int, dict[str, tuple[str, str]]] = {}
    for entry in (data.get("details") if isinstance(data, dict) else None) or []:
        index = entry.get("detail") if isinstance(entry, dict) else None
        if not isinstance(index, int) or isinstance(index, bool) or index not in shown:
            continue
        for verdict in entry.get("judgements") or []:
            if not isinstance(verdict, dict):
                continue
            excerpt_id, status = str(verdict.get("id") or "").strip(), VERDICTS.get(str(verdict.get("verdict") or "").strip())
            if excerpt_id not in shown[index] or status is None:
                continue
            # The reviewer reads Vietnamese: a caption is judged in English and explains itself in both.
            note = " ".join(str(verdict.get("note_vi") or verdict.get("why") or "").split())[:120]
            judged.setdefault(index, {})[excerpt_id] = (status, note)
    return judged


def _spread_nearest(frames: list[dict], count: int) -> list[dict]:
    """`count` frames of a side, nearest first, keeping the nearest and the one furthest into the window."""
    if count <= 0:
        return []
    if len(frames) <= count:
        return frames
    if count == 1:
        return frames[:1]
    return [frames[round(index * (len(frames) - 1) / (count - 1))] for index in range(count)]


#: Guards the per-client shot cache below; each video then has a lock of its own.
_SHOTS_LOCK = threading.Lock()
#: A video whose shots could not be read is not asked again for this long: the fall-back (the
#: keyframe neighbourhood) is good enough, and a host that just timed out is still slow.
SHOTS_RETRY_SECONDS = 60.0
#: Videos whose shots one client remembers. The Search tab's checker keeps one client for the
#: life of the server, so this is what bounds it; an agent run touches a few dozen videos.
SHOTS_CACHE_MAX = 128


def _shots_of(search, video_id: str) -> tuple:
    """A video's shots with the keyframes inside each, read once per video per run; `()` if there are none.

    Cached on the client object itself, so a run's server is asked once per video however many of its
    moments are checked, and a server that cannot answer (an older deployment, a video with no shot
    boundaries) simply leaves `()` behind -- the caller falls back to the keyframe neighbourhood. A
    shot with an EMPTY keyframe list is kept, not dropped: on a reported p2-6 candidate the shot right
    before the query's "two red circles" beat was 37 seconds long and had no indexed keyframe at all
    (nothing in it stood out enough for the offline embedding pass to sample), and dropping it here
    made that whole stretch structurally invisible to the checklist -- `_shot_frames` is what decides
    what a keyframe-less shot contributes.
    """
    fetch = getattr(search, "shots", None)
    if fetch is None:
        return ()
    with _SHOTS_LOCK:
        cache = getattr(search, "_evidence_shots", None)
        if cache is None:
            cache = OrderedDict()
            try:
                search._evidence_shots = cache
            except AttributeError:  # a client that takes no attributes is simply asked each time
                pass
        entry = cache.setdefault(video_id, {"lock": threading.Lock(), "rows": None, "failed_at": None})
        cache.move_to_end(video_id)
        while len(cache) > SHOTS_CACHE_MAX:
            cache.popitem(last=False)
    # One request per video, however many threads judge its moments at once: measured on a slow
    # compute host, sixteen candidates of two videos each waited out their own 8 s timeout.
    with entry["lock"]:
        if entry["rows"] is not None:
            return entry["rows"]
        if entry["failed_at"] is not None and monotonic() - entry["failed_at"] < SHOTS_RETRY_SECONDS:
            return ()
        try:
            entry["rows"] = tuple(fetch(video_id) or [])
        except Exception as exc:
            logger.warning("evidence: no shots for %s (%s)", video_id, exc)
            entry["failed_at"] = monotonic()
            return ()
        return entry["rows"]


def _shot_frames(search, video_id: str, frame_idx: int, time: float,
                 seconds: float, dense: int, most: int) -> list[dict] | None:
    """The keyframes of the shots around a moment, walked outward, or None without shot data.

    Every shot within `seconds` either side is a candidate, nearest first -- never capped by how MANY
    shots that is, only by how many FRAMES the side sends (`most`): a video cut into many short shots
    must still be able to reach the far end of the window, which capping the shot count could not do
    (see the module docstring). The nearest `dense` shots of a side keep every keyframe they have (a
    beat that is a change needs more than one look), the rest keep their first -- reach over density,
    once a beat is known to be a whole shot away rather than a blur inside one. A shot with NO indexed
    keyframe (nothing in it stood out enough offline to sample) still gets looked at: its own midpoint
    frame, fetched directly rather than picked from an index, so a long uneventful-looking shot is not
    silently skipped -- it counts as one frame for `dense`/`most` like any other shot's first keyframe.

    The candidate's OWN shot's OTHER keyframes are context too, ahead of every other shot's: on the
    reported p2-6 candidate, the second of the query's two red-circle beats sat 3.5 s later in that
    exact shot -- its own second keyframe -- which `vision_summarize` never asks for on its own (it
    sends only the candidate's own exact frame) and the old code here skipped outright
    (`shot is own: continue`), so it was invisible however wide the window around the shot was. Those
    keyframes are unconditionally dense: they are not a neighbouring shot's guess at what changed,
    they are the same shot the candidate's own moment is already in.
    """
    shots = _shots_of(search, video_id)
    if not shots:
        return None
    own = next((shot for shot in shots
                if shot["start_frame"] <= frame_idx <= shot["end_frame"]), None)
    if own is None:
        own = min(shots, key=lambda shot: abs(shot["start"] - time))
    own_keys = sorted((key for key in own["keyframes"] if key.get("frame_idx") != frame_idx),
                      key=lambda key: key["time"])
    own_earlier = [key for key in own_keys if key["time"] < time][::-1]     # nearest first
    own_later = [key for key in own_keys if key["time"] > time]
    earlier, later = [], []
    for shot in shots:
        if shot is own or shot["end"] < time - seconds or shot["start"] > time + seconds:
            continue
        if shot["keyframes"]:
            keys = [key for key in shot["keyframes"] if abs(key["time"] - time) <= seconds]
            if not keys:                             # the shot reaches into the window; its frames may not
                continue
        else:
            midpoint = (shot["start"] + shot["end"]) / 2
            if abs(midpoint - time) > seconds:
                continue
            keys = [{"video_id": video_id, "frame_idx": (shot["start_frame"] + shot["end_frame"]) // 2,
                    "time": midpoint}]
        middle = (shot["start"] + shot["end"]) / 2
        (earlier if middle < time else later).append((middle, sorted(keys, key=lambda key: key["time"])))
    kept: list[dict] = []
    for own_side, side in ((own_earlier, sorted(earlier, key=lambda pair: -pair[0])),
                           (own_later, sorted(later, key=lambda pair: pair[0]))):
        taken: list[dict] = list(own_side)   # the candidate's own shot first, ahead of any other shot's
        for position, (_, keys) in enumerate(side):
            use = keys if position < dense else keys[:1]
            if len(taken) + len(use) > most:
                break
            taken += use
        kept += taken
    return sorted(kept, key=lambda frame: frame["time"])


def _context_frames(search, video_id: str, frame_idx: int, time: float, *,
                    seconds: float = VISION_CONTEXT_SECONDS,
                    dense: int = VISION_CONTEXT_DENSE, most: int = VISION_CONTEXT_MAX,
                    count: int = VISION_CONTEXT_FRAMES, radius: int = VISION_CONTEXT_RADIUS) -> list[dict]:
    """The shots around a candidate -- several keyframes of the nearest, one of the rest -- out to `seconds`.

    Per-shot first (`_shot_frames`): a story's beats live in whole shots, and sampling by time can step
    over one entirely (a densely-cut video's shots are seconds long; a sparsely-cut one's are long
    enough that a handful of shots already covers `seconds`). Without shot boundaries the indexed
    keyframes around the moment are used instead, also nearest shot first. Missing route, no
    neighbours or a frame that is its own only neighbour is `[]`, never an error: context is a hint,
    and vision still has the candidate's own frame without it.
    """
    try:
        per_shot = _shot_frames(search, video_id, frame_idx, time, seconds, dense, most)
    except Exception as exc:  # a shot list in an unexpected shape must cost the checklist nothing
        logger.warning("evidence: shot context failed for %s (%s)", video_id, exc)
        per_shot = None
    if per_shot is not None:
        return per_shot
    try:
        frames = search.neighbours(video_id, frame_idx, radius)
    except Exception:
        return []

    def at(frame: dict) -> float:
        return float(frame.get("time") or 0.0)

    others = [frame for frame in frames if frame.get("frame_idx") != frame_idx]
    earlier = sorted((f for f in others if at(f) < time), key=lambda f: -at(f))
    later = sorted((f for f in others if at(f) > time), key=lambda f: at(f))
    kept = (_spread_nearest([f for f in earlier if time - at(f) <= seconds], count)
            + _spread_nearest([f for f in later if at(f) - time <= seconds], count))
    return sorted(kept, key=at)


def vision_summarize(vision, search, details: QueryDetails | list[dict], group: list[dict]) -> dict[str, dict[int, tuple[str, str]]]:
    """Judge candidates from their pixels -- their own keyframe and the shots around it -- in chunks.

    A caption is one model's guess at what a frame shows, made once, offline, without the query in
    mind -- it can misname a colour or miscount, and cannot be asked again. Looking at the actual
    pixels is ground truth the excerpt-based checks never have, and a report of a query being wrong
    showed why one frame is not enough: the query was a story ("five men beside the lion, one carries
    something away, a man wakes and shakes the lion"), and the moments that say so are the shots
    before and after the frame the query's own story leads to. So each candidate carries its own
    moment PLUS the shots around it (`_context_frames`), labelled by candidate, and the model judges
    the details across the whole set -- details that are right for the candidate because a shot next
    to it shows them count as checked by vision (see `summarize`, where a vision verdict is stamped at
    the candidate's own time). Returns `{candidate_id: {detail_index: (status, note)}}`; `{}` when
    `vision` is None, nothing could be judged, or every call failed -- the caller falls back to the
    text-only checklist, which is what this run has been doing all along.
    """
    entries = details.details if isinstance(details, QueryDetails) else details
    if vision is None or not entries or not group:
        return {}
    lines = []
    for index, entry in enumerate(entries):
        hint = MARKER_HINT if _is_marker_detail(str(entry.get("vi") or "")) else ""
        lines.append(f"{index}. {entry.get('meaning_en') or entry['vi']}{hint}")
        lines += _meaning_lines(entry)
    prompt = f"{VISION_MATCH_SYSTEM}\n\nDetails:\n" + "\n".join(lines)
    chunks = [group[i:i + VISION_CHUNK] for i in range(0, len(group), VISION_CHUNK)]

    def judge(chunk):
        """The reply for one chunk, with the chunk itself: its images are what the reply is about,
        whatever id the model writes into it (see `attribute`, and the one-candidate note below)."""
        labelled = []
        for candidate in chunk:
            try:
                own = search.image(candidate["video_id"], candidate["frame_idx"], VISION_IMAGE_WIDTH)
            except Exception:
                own = None
            if own:
                labelled.append((f"[{candidate['id']}] this candidate's own moment", own))
            for context in _context_frames(search, candidate["video_id"], candidate["frame_idx"], candidate["time"]):
                try:
                    data = search.image(context["video_id"], context["frame_idx"], VISION_IMAGE_WIDTH)
                except Exception:
                    continue
                if data:
                    offset = float(context["time"]) - candidate["time"]
                    when = f"{abs(offset):.0f}s earlier" if offset < 0 else f"{offset:.0f}s later"
                    labelled.append((f"[{candidate['id']}] shot {when} in the same story", data))
        if not labelled:
            return chunk, None
        # Said HERE, in the call about these images, and not only in the shared system prompt: asked
        # only by that prompt, a call carrying ONE candidate's images has answered with a dozen
        # invented ones (C1, C10, ...), run past the token cap, and been thrown away whole -- its
        # candidate then kept a text-only checklist with nothing saying so. Measured on p2-6's own
        # gold frame: three of eight calls ended that way, and the invented entries did not even name
        # the candidates they were about.
        ids = ", ".join(f'"{candidate["id"]}"' for candidate in chunk)
        said = (f"\n\nThis call carries {len(chunk)} candidate(s), no others: {ids}. Return exactly "
                f"{len(chunk)} entr{'y' if len(chunk) == 1 else 'ies'}, id {ids}, every detail listed.")
        try:
            return chunk, vision.judge(labelled, prompt + said, VISION_MATCH_SCHEMA, max_output_tokens=3000,
                                       job="checklist", salvage=True)
        except Exception as exc:  # a vision backend that cannot do this must not cost the checklist
            logger.warning("evidence: vision judge failed: %s", exc)
            return chunk, None

    found: dict[str, dict[int, tuple[str, str]]] = {}
    known = {candidate["id"] for candidate in group}

    def attribute(chunk, entries):
        """Which candidate each reply entry is about: by the label, not by the id it typed back.

        `VISION_CHUNK` is 1, so a call's images carry exactly one candidate's id and its reply is
        about that candidate whatever id the model types into it -- and it does get it wrong: a live
        call whose label was `[GOLD]` answered `{"id": "C1"}`, which `known` did not contain, so every
        check it had just made was thrown away and the candidate fell back to a text-only checklist
        with nothing saying so. The label is the truth here; the id is only what tells SEVERAL
        candidates of one call apart, which is why it is still what a multi-candidate chunk uses.
        """
        if len(chunk) == 1:
            label = chunk[0]["id"]
            named = [entry for entry in entries if str(entry.get("id", "")).strip("[] ") == label]
            return [(label, entry) for entry in (named or entries[:1])]
        return [(str(entry.get("id", "")).strip("[] "), entry) for entry in entries]

    with ThreadPoolExecutor(min(3, len(chunks))) as pool:
        for chunk, reply in pool.map(judge, chunks):
            for cid, entry in attribute(chunk, (reply.get("candidates")
                                                if isinstance(reply, dict) else None) or []):
                if cid not in known:
                    continue
                checks: dict[int, tuple[str, str]] = {}
                for item in entry.get("details") or []:
                    index = item.get("detail")
                    status = VERDICTS.get(str(item.get("verdict") or "").strip())
                    if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(entries) or status is None:
                        continue
                    note = " ".join(str(item.get("note_vi") or item.get("why") or "").split())[:120]
                    # The numeric guard the excerpt judges get applies here too: a detail that names a
                    # quantity is only as supported as the count the judge says it saw, so a judge that
                    # answers "exact" while its own words say "two shellfish" for a detail of four is
                    # capped at partial. Only when the judge's words DO state a count -- unlike a
                    # transcript sentence, a note is not everything the model saw, so silence about the
                    # number is not the contradiction the text path treats it as. (A live p2-30 run marked
                    # "places 4 X on a white plate" supported from a frame showing two.)
                    if status == "supported" and counts_in(f"{item.get('why') or ''} {note}") and (mismatch := count_mismatch(
                            str(entries[index].get("vi") or ""), f"{item.get('why') or ''} {note}")):
                        status, note = "partial", f"{mismatch}. {note}"[:120]
                    checks[index] = (status, note)
                found[cid] = checks
    return found


def summarize(deepseek, details: QueryDetails | list[dict], store: EvidenceStore, group: list[dict], *,
             vision=None, search=None) -> dict[str, dict]:
    """Checklist and one-line summary per candidate id of `group` (one video).

    Each check is `supported` (the words say the whole detail), `partial` (right subject, a part
    missing or generic: "flowers" for a torch-ginger vase, no count), `contradicted` (a part is
    stated differently) or `not_mentioned`. Speech and OCR are judged by their own call (`_ask`,
    `MATCH_SYSTEM`) -- real evidence a frame cannot give, kept always. The caption channel is
    SUSPENDED whenever `vision` is available: a caption is one model's guess about a frame, made
    once, offline, without the query in mind, and vision looking at the exact candidate's own pixels
    is strictly better evidence for the same question, so asking a second model to re-describe what
    the first one already got wrong is not worth the call. Only when `vision` is None (the feature
    turned off, or no backend for it) does the caption call run, as it always used to. A candidate
    no call answered for is simply absent: a summary is a hint and a missing one must not delay the
    person.
    """
    entries = details.details if isinstance(details, QueryDetails) else details
    if not entries or not group:
        return {}
    video_id = group[0]["video_id"]
    span = span_around(store, video_id, *_span_of(group))
    if not any(span.values()) and vision is None:
        return {}  # nothing to check this video's candidates against: no words, and no pixels either
    picked = (select_excerpts(excerpts_of(span, [_term_sets(entry) for entry in entries]), group, len(entries), EXCERPTS_PER_DETAIL)
             if any(span.values()) else [])
    by_id = {e["id"]: e for e in picked}
    heard_or_read = [e for e in picked if e["source"] != "caption"]
    with ThreadPoolExecutor(3) as pool:
        text_future = (None if vision is not None else
                       pool.submit(_ask, deepseek, MATCH_CAPTION_SYSTEM, entries, [e for e in picked if e["source"] == "caption"], english=True))
        speech_future = pool.submit(_ask, deepseek, MATCH_SYSTEM, entries, heard_or_read, english=False)
        vision_future = pool.submit(vision_summarize, vision, search, entries, group)
        answers = [text_future.result() if text_future else {}, speech_future.result()]
        seen = vision_future.result()
    # A call that was attempted returns None on failure, `{}` when it simply had nothing to ask (see
    # `_ask`) or when it did not run at all (caption, suspended). If nothing came back from any
    # source that was actually attempted, and at least one of those genuinely failed, there is
    # nothing to build a checklist from.
    if any(answer is None for answer in answers) and all(not answer for answer in answers) and not seen:
        return {}
    #: detail index -> excerpt id -> (status, note)
    judged: dict[int, dict[str, tuple[str, str]]] = {}
    for answer in answers:
        for index, verdicts in (answer or {}).items():
            for excerpt_id, (status, note) in verdicts.items():
                if status == "supported" and (mismatch := count_mismatch(entries[index]["vi"], by_id[excerpt_id]["text"])):
                    status, note = "partial", f"{mismatch}. {note}"[:120]
                judged.setdefault(index, {})[excerpt_id] = (status, note)

    order = {"supported": 0, "partial": 1, "contradicted": 2}
    found: dict[str, dict] = {}
    for item in group:
        low, high = item["time"] - BEFORE_SECONDS, item["time"] + AFTER_SECONDS
        checks, names = [], {"supported": [], "partial": [], "contradicted": []}
        for index, entry in enumerate(entries):
            near = [(order[status], -by_id[i]["scores"].get(index, 0), abs(by_id[i]["at"] - item["time"]), i, status, why)
                    for i, (status, why) in judged.get(index, {}).items()
                    if by_id[i]["end"] >= low and by_id[i]["at"] <= high]
            if near:
                # The best verdict wins; among equals, the sentence holding most of the detail's words, then the nearest.
                _, _, _, excerpt_id, status, why = min(near)
                e = by_id[excerpt_id]
                check = {"detail": index, "status": status, "source": e["source"], "at": round(e["at"], 1),
                        "quote": e["text"][:MAX_SENTENCE_CHARS], "note": why}
            else:
                check = {"detail": index, "status": "not_mentioned", "source": None, "at": None, "quote": "", "note": ""}
            # Vision looked at the candidate's own moment and the shots around it, so it is stamped at the
            # candidate's own time and takes full proximity -- the neighbouring shots are the same moment's
            # story, and a person judging wants them counted where they are asked about. It only ever
            # upgrades, never loses to a stale excerpt, and it is the only source that can turn
            # "not_mentioned" into a verdict at distance zero. Its status is weighted by the same
            # `detail_weights` every other source's is (distinctiveness prior × empirical share).
            seen_status, seen_note = seen.get(item["id"], {}).get(index, (None, None))
            if seen_status is not None and STATUS_RANK[seen_status] >= STATUS_RANK[check["status"]]:
                check = {"detail": index, "status": seen_status, "source": "vision", "at": round(item["time"], 1),
                        "quote": "", "note": seen_note}
            checks.append(check)
            if check["status"] != "not_mentioned":
                names[check["status"]].append(entry["vi"][:32])
        found[item["id"]] = {"checks": checks, "summary": _summary(names, len(entries))}
    return found


def _meaning_lines(entry: dict) -> list[str]:
    """What still counts as the same thing and what only looks like it, for the English caption judge."""
    lines = []
    if entry.get("same_en"):
        lines.append(f"   Same: {'; '.join(entry['same_en'])}")
    if entry.get("different_en"):
        lines.append(f"   Different: {'; '.join(entry['different_en'])}")
    return lines


# -- what the checklist says about a candidate, as a number ----------------------------------
#: A detail nearly every read candidate satisfies tells the moment apart from nothing.
COMMON_SHARE = 0.6
#: Shares over fewer candidates than this are not trusted, so every detail weighs the same.
ENOUGH_TO_JUDGE = 6
MIN_WEIGHT = 0.15
#: What a check is worth. Only what the words SUPPORT counts: a contradiction is shown to the reviewer
#: and lowers nothing. On ten round-2 queries with a known answer, ranking a contradicted candidate below
#: one nobody read put the right moment 12th where retrieval had it 1st (MRR .36 against retrieval's .50,
#: .53 with contradictions ignored): captions get colours, counts and identities wrong ("a black strap"
#: for the white string the query gives), so a conflict is as often the caption's mistake as the
#: candidate's. A partial says the subject is right and the part that tells candidates apart is missing,
#: so it is worth little.
STATUS_VALUE = {"supported": 1.0, "partial": 0.25}
#: An excerpt counts for a candidate by how near its shot is: a frame is judged by what its own shot and
#: its neighbours say, not by a shot half a minute away. Weight 1 at distance 0, 1/2 at this many seconds.
PROXIMITY_SECONDS = 20.0
#: A moment's score is half what is said around it, half what the whole video says (see `video_coverage`).
OWN_WEIGHT = 0.5
VIDEO_WEIGHT = 0.5
#: Scores closer than this are a tie, and retrieval order settles it: evidence this coarse says which
#: candidates are supported, not which of two similarly supported ones is the better moment. Chosen on
#: 36 runs of 21 round-2 queries whose answer was among the 80 shown (three collections): a band of .10 beat
#: retrieval's MRR in all three (+.05, +.07, +.04); .25 was better on the first and worse on the third.
TIE_BAND = 0.10


def detail_weights(summaries, count: int, entries: list[dict] | None = None) -> tuple[list[float], list[bool]]:
    """How much each detail counts, and which are "common" (satisfied by most candidates read).

    Two factors, multiplied: a *prior* from `entries[i]["distinctiveness"]` (1.0 for every detail
    when `entries` is not given), which is what the checklist has before any candidate is read, and an
    *empirical* factor, 1 minus the share of read candidates that say the detail in full once
    `ENOUGH_TO_JUDGE` have been -- not below `MIN_WEIGHT`. The prior is what keeps a guessable detail
    ("a yellow animal" in a lion dance) from outweighing a story-specific one before enough candidates
    have been read to measure it empirically; the empirical factor catches a detail the prior missed.
    """
    priors = [_prior_weight(entry) for entry in entries] if entries else [1.0] * count
    read = [summary for summary in summaries if summary]
    weights, common = [], []
    for index in range(count):
        said = sum(1 for summary in read
                   if any(check["detail"] == index and check["status"] == "supported" for check in summary["checks"]))
        share = said / len(read) if len(read) >= ENOUGH_TO_JUDGE else 0.0
        empirical = max(MIN_WEIGHT, 1.0 - share)
        weights.append(round(priors[index] * empirical, 4))
        common.append(share >= COMMON_SHARE)
    return weights, common


def proximity(distance: float) -> float:
    return 1.0 / (1.0 + max(0.0, distance) / PROXIMITY_SECONDS)


def _own(summary: dict, weights: list[float], time: float) -> float:
    total = sum(weights) or 1.0
    earned = 0.0
    for check in summary["checks"]:
        value = STATUS_VALUE.get(check["status"], 0.0)
        if value and check["detail"] < len(weights):
            near = proximity(abs(check["at"] - time)) if check.get("at") is not None else 1.0
            earned += weights[check["detail"]] * value * near
    return earned / total


def video_coverage(read: list[tuple[str, dict]], weights: list[float]) -> dict[str, float]:
    """For each video, the weighted share of the query's details its read candidates say anywhere.

    A story query ("five men beside a yellow lion, one hides a pumpkin, the man wakes and shakes the
    lion") is spread over shots, and a frame shows one of them: no single moment says it all, but the
    right video says most of it somewhere. This is what lets a moment be ranked by its video's story.
    """
    total = sum(weights) or 1.0
    best: dict[str, dict[int, float]] = {}
    for video_id, summary in read:
        for check in summary["checks"]:
            value = STATUS_VALUE.get(check["status"], 0.0)
            if value and check["detail"] < len(weights):
                best.setdefault(video_id, {})[check["detail"]] = max(best.get(video_id, {}).get(check["detail"], 0.0), value)
    return {video_id: sum(weights[d] * v for d, v in details.items()) / total for video_id, details in best.items()}


def evidence_of(summary: dict | None, weights: list[float], time: float = 0.0, coverage: float = 0.0) -> dict | None:
    """`{"score", "share", "video", "tier"}` for one candidate's checklist, or None when it has none.

    `share` is what the words around the moment itself say (the bar's fill); `video` is what the video
    says anywhere; `score` is their mix, and `tier` is what the ordering uses: 0 when the words support
    the candidate, 1 when they say nothing -- the same as a candidate nobody has read.
    """
    if summary is None or not weights:
        return None
    own = _own(summary, weights, time)
    score = OWN_WEIGHT * own + VIDEO_WEIGHT * coverage
    return {"score": round(score, 4), "share": round(own, 4), "video": round(coverage, 4), "tier": 0 if score > 0 else 1}


def _summary(names: dict[str, list[str]], total: int) -> str:
    supported, partial, contradicted = names["supported"], names["partial"], names["contradicted"]
    if not (supported or partial or contradicted):
        return "Chưa thấy chi tiết nào trong lời thoại, chữ trên màn hình hay mô tả cảnh quanh đây."
    text = f"Khớp hẳn {len(supported)}/{total} chi tiết" + (": " + "; ".join(supported[:3]) if supported else "")
    if partial:
        text += f". Gần đúng: {'; '.join(partial[:2])}"
    if contradicted:
        text += f". Mâu thuẫn: {'; '.join(contradicted[:2])}"
    return text


def suggest_answers(deepseek, question: str, details: QueryDetails, store: EvidenceStore, items: list[dict]) -> list[dict]:
    """Answers the words around these moments give to a Q&A question, best first."""
    if details.question:
        entries = [details.question]
    else:  # no terms for the question itself: whatever the query's own details would mention
        entries = [{"terms_en": [t for d in details.details for t in d["terms_en"]],
                    "terms_vi": [t for d in details.details for t in d["terms_vi"]]}]
    found = []
    for group in clusters(items):
        video_id = group[0]["video_id"]
        span = span_around(store, video_id, *_span_of(group))
        if not any(span.values()):
            continue
        picked = select_excerpts(excerpts_of(span, [_term_sets(e) for e in entries]), group, 1, SUGGEST_EXCERPTS)
        if not picked:
            continue
        messages = [
            {"role": "system", "content": SUGGEST_SYSTEM},
            {"role": "user", "content": f"Câu hỏi: {question}\n\nCác đoạn trích:\n" + "\n".join(
                f"{e['id']} [{_stamp(e)}] {e['text']}" for e in picked)},
        ]
        try:
            data = deepseek.chat_json(messages, thinking=False, max_tokens=800, temperature=0)
        except LLMError as exc:
            logger.warning("evidence: answer suggestion failed: %s", exc)
            continue
        by_id = {e["id"]: e for e in picked}
        for entry in (data.get("answers") if isinstance(data, dict) else None) or []:
            answer = " ".join(str((entry or {}).get("answer") or "").split())[:100]
            e = by_id.get(str((entry or {}).get("excerpt") or "").strip())
            if not answer or e is None:
                continue
            try:
                confidence = float(entry.get("confidence") or 0)
            except (TypeError, ValueError):
                confidence = 0.0
            nearest = min(group, key=lambda item: abs(item["time"] - e["at"]))
            found.append({"answer": answer, "cid": nearest["id"], "confidence": min(max(confidence, 0.0), 10.0),
                          "source": e["source"], "quote": e["text"][:160]})
    return sorted(found, key=lambda item: -item["confidence"])


def summarize_all(deepseek, details: QueryDetails | list[dict], store: EvidenceStore, items: list[dict], *,
                  workers: int = 3, on_batch=None, vision=None, search=None) -> dict[str, dict]:
    """`summarize` over each video of `items`, side by side; `on_batch(found)` as each finishes."""
    groups = clusters(items)
    found: dict[str, dict] = {}
    if not groups:
        return found
    with ThreadPoolExecutor(min(workers, len(groups))) as pool:
        for result in pool.map(lambda group: summarize(deepseek, details, store, group, vision=vision, search=search), groups):
            found.update(result)
            if on_batch is not None and result:
                on_batch(result)
    return found
