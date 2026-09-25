"""One moment, checked against one query, on demand -- the Search tab's "Check by AI".

The retrieval agent already verifies candidates by looking at them: it splits the query into
details (`evidence.query_details`), shows a model the candidate's keyframe plus the keyframes of
the shots around it (`evidence.vision_summarize`), and merges the verdicts. That only happens
inside a whole agent run. A person working the Search tab has the same question about one frame
they are looking at right now, and wants it answered in a few seconds without starting a run.

So this is that pipeline, cut down to one moment and one press of a button:

    details (cached per query)  ──►  ONE vision call  ──►  per-detail verdicts
                                ╲                          + an overall 0-10
                                 ╲                         + the Q&A answer, if it is a question
                                  ╲
                                   ─►  ONE text call over the speech/OCR sentences in the window

Three things differ from the agent's checklist, all because there is exactly one candidate:

- **No id attribution problem.** `vision_summarize` has to work out which candidate a reply is
  about and has been measured getting it wrong; here the reply is about the only moment there is.
- **The answer rides along.** The images are already assembled and paid for, so a Q&A answer comes
  back from the same call instead of the separate reading the agent does (`Inspector.answer`).
- **The weights are the prior only.** `evidence.detail_weights`'s empirical half needs a
  population of read candidates to know which details are common; one moment has no population,
  so a detail is worth its `distinctiveness` prior and nothing else.

Everything else -- the judging rules, the count guard, the broadcast-marker exemption, the four
verdicts, the shot walk -- is `agent.evidence` and `agent.prompts`, unchanged.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging

from . import evidence
from .evidence import (
    EXCERPTS_PER_DETAIL, MARKER_HINT, MAX_SENTENCE_CHARS, STATUS_RANK, STATUS_VALUE, VERDICTS,
    QueryDetails, _ask, _is_marker_detail, _meaning_lines, _prior_weight, _term_sets,
    count_mismatch, counts_in, excerpts_of, proximity, select_excerpts, span_around,
)
from .llm import LLMError
from .prompts import CHECK_ANSWER_RULE, CHECK_SYSTEM, MATCH_SYSTEM

logger = logging.getLogger("agent")

#: The window around the moment the check draws on, in seconds either side. ±5 minutes is what
#: the Search tab asks for by default: a person pressing the button wants "is this segment the
#: one?", and a story query's beats are shots, sometimes minutes, from the frame that ranked.
CHECK_WINDOW_SECONDS = 300.0
CHECK_QUICK_SECONDS = 60.0
CHECK_MIN_SECONDS = 30.0
CHECK_MAX_SECONDS = 600.0
#: Same width the agent's checklist judges at: a judge sees more of a small frame than a caption
#: ever wrote down, and this is not the job that has to read text off the screen.
CHECK_IMAGE_WIDTH = 480
#: `DENSE=0`: every shot keeps exactly ONE keyframe, never a shot's whole set. A person is
#: waiting on one call here (unlike the agent's own multi-round verification, where a beat a
#: single frame could blur past is worth another look), so speed wins over that extra depth --
#: `evidence._one_of` picks it as the MIDDLE of however many the shot has (`pick="middle"` below),
#: not the first, which can still be mid-transition from the previous cut. `MAX` is the real
#: limit and it is on FRAMES, not shots -- see the long note on `evidence.VISION_CONTEXT_MAX` for
#: why that distinction was measured to matter. Fourteen a side, against the agent's sixteen:
#: this window is 5 minutes rather than 3, so the budget buys reach at the edges instead of depth.
CHECK_CONTEXT_DENSE = 0
CHECK_CONTEXT_MAX = 14
CHECK_CONTEXT_FRAMES = 4
#: At most 29 images at 480 px -- about 17k input tokens, well inside one call.
CHECK_IMAGE_WORKERS = 6
#: Speech and on-screen text handed to the vision call so a Q&A answer can be read off the words
#: as well as the pixels, capped so a ten-minute window cannot crowd out the rules.
CHECK_SAID_CHARS = 2400
#: The share of that going to speech, with the rest left for on-screen text. Speech keeps its own
#: budget rather than competing for one pool because the narration is where a Vietnamese bulletin
#: puts names and facts, while on-screen text here is mostly broadcaster furniture -- see `_said`
#: for the measurement that made this necessary.
CHECK_SPEECH_SHARE = 0.6

CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "details": {"type": "array", "items": {"type": "object", "properties": {
            "detail": {"type": "integer"},
            "why": {"type": "string"},
            "verdict": {"type": "string", "enum": ["exact", "partial", "conflict", "unrelated"]},
            "note_vi": {"type": "string"},
        }, "required": ["detail", "why", "verdict"]}},
        "overall": {"type": "object", "properties": {
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "reason": {"type": "string"},
        }, "required": ["score", "reason"]},
        # "ocr" is deliberately absent: the answer is read from the keyframes or the speech, never
        # from on-screen text -- see `prompts.CHECK_ANSWER_RULE`. `_answer` still accepts it from a
        # reply that names it anyway, so a good answer mislabelled is not thrown away.
        "answer": {"type": "object", "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 10},
            "source": {"type": "string", "enum": ["frame", "speech", "unknown"]},
            "quote": {"type": "string"},
        }, "required": ["text", "confidence", "source"]},
    },
    "required": ["details", "overall"],
}


def check_schema(asked: bool) -> dict:
    """`CHECK_SCHEMA`, with `answer` promoted to the front and made required when a question was asked.

    Measured live: a reply came back with the whole answer object written INTO the `overall.reason`
    string -- `...nêu rõ trong lời bình (speech). "answer": {"text": "MEDELLIN", "confidence": 10,
    "source": "` -- which parses as a reply whose `answer` member is missing, so the person was shown
    "This moment does not answer the question" for a question the model had, in fact, answered. The
    answer came after the prose in property order and was not required, which is exactly where a
    weaker model loses track of which field it is inside. Committing to it before the `overall`
    prose is the mitigation; re-running the same query came back clean 6 times out of 6, so this is
    a hardening against a rare malformation rather than a measured rate change.
    """
    if not asked:
        return CHECK_SCHEMA
    return {
        **CHECK_SCHEMA,
        "properties": {"answer": CHECK_SCHEMA["properties"]["answer"], **CHECK_SCHEMA["properties"]},
        "required": ["details", "answer", "overall"],
    }


class CheckError(RuntimeError):
    """The check could not be made at all -- no model reply about the pixels."""


@dataclass(frozen=True)
class CheckTarget:
    video_id: str
    frame_idx: int
    time: float


#: What each channel means to the judge, in the order they are shown. The Search tab sends
#: `visual_text`/`audio_text`/`ocr_text` exactly like a real search request does
#: (`backend.app.SearchReq`) rather than one flattened sentence -- flattening lost the
#: difference between "describe what the frame shows" and "find these exact words", which
#: mattered most for OCR: a bare on-screen string ("Nà Ní") judged as a scene description makes
#: no sense, and an audio query is spoken words to match, not an action to see.
#: English, for `CHECK_SYSTEM` (the vision judge's prompt, which is in English).
CHANNEL_LABELS_EN = (
    ("visual_text", "Visual description (English)"),
    ("audio_text", "Spoken words to find (Vietnamese speech)"),
    ("ocr_text", "On-screen text to find"),
)
#: Vietnamese, for `DETAILS_SYSTEM` (the checklist step, a Vietnamese prompt shared with the
#: agent) -- worded to match `CHECK_DETAILS_NOTE`, which tells that step what each label means.
CHANNEL_LABELS_VI = (
    ("visual_text", "Mô tả hình ảnh (tiếng Anh)"),
    ("audio_text", "Lời thoại cần tìm"),
    ("ocr_text", "Chữ trên màn hình cần tìm"),
)


@dataclass(frozen=True)
class CheckQuery:
    """What the person searched with, kept in its own channels rather than merged into one
    free-text description -- see `CHANNEL_LABELS_EN`. At least one of the three is non-empty;
    the caller (`backend/checker.py`) enforces that before this is built.
    """
    visual_text: str = ""
    audio_text: str = ""
    ocr_text: str = ""
    kind: str = "kis"
    question: str | None = None

    def _channels(self, labels) -> list[tuple[str, str]]:
        return [(label, text.strip()) for field, label in labels
                if (text := getattr(self, field)).strip()]

    @property
    def channels(self) -> list[tuple[str, str]]:
        """`(label, text)` for each non-empty channel, English labels, for the vision prompt."""
        return self._channels(CHANNEL_LABELS_EN)

    @property
    def description(self) -> str:
        """One Vietnamese-labelled blob for `query_details` (`DETAILS_SYSTEM` is Vietnamese)."""
        return "\n".join(f"{label}: {text}" for label, text in self._channels(CHANNEL_LABELS_VI))

    @property
    def asked(self) -> str | None:
        """The Q&A question: the explicit one, or the query itself when the kind says it is one."""
        if self.question and self.question.strip():
            return self.question.strip()
        return self.description or None if self.kind == "qa" else None


def check_moment(client, search, store, *, query: CheckQuery, target: CheckTarget,
                 details: QueryDetails | list[dict],
                 window_seconds: float = CHECK_WINDOW_SECONDS, span: dict | None = None) -> dict:
    """Check one moment against one query; raises `CheckError` if the pixels could not be judged.

    `client` is any backend with `judge()` and `chat_json()` -- an `OpenRouterClient` in the
    deployment this was built for. `search` fetches images and shots, `store` the per-video words.

    The text half failing costs the check its speech/OCR evidence and nothing else; the vision
    half failing is the error, because without it there is no answer to the question that was
    asked. That asymmetry is the opposite of the agent's checklist, where vision is the optional
    extra on top of the words -- here the person pressed a button that means "look at this".

    `span` is the moment's transcript/OCR window, when the caller has already fetched it. It is
    independent of `details`, and on a cold check both cost seconds while neither can start until
    the other has finished -- so `backend.checker` fetches the two CONCURRENTLY and passes this in
    rather than letting the serial default run (see the timing note there).
    """
    entries = list(details.details if isinstance(details, QueryDetails) else details)
    window = max(CHECK_MIN_SECONDS, min(CHECK_MAX_SECONDS, float(window_seconds)))
    if span is None:
        span = span_around(store, target.video_id, target.time - window, target.time + window)
    with ThreadPoolExecutor(2) as pool:
        text_future = pool.submit(_text_checks, client, entries, span, target, window)
        vision_future = pool.submit(_vision_check, client, search, entries, span, query, target, window)
        text = text_future.result()
        seen, overall, answer, frames_seen = vision_future.result()

    checks = _merge(entries, text, seen, target)
    weights = [_prior_weight(entry) for entry in entries]
    support = _support(checks, weights, target)
    return {
        "video_id": target.video_id,
        "frame_idx": target.frame_idx,
        "time": round(target.time, 2),
        "window_seconds": window,
        "verdict": overall,
        "details": [{
            "text": entry["vi"],
            "meaning_en": entry.get("meaning_en") or "",
            "distinctiveness": round(float(entry.get("distinctiveness", 1.0)), 2),
            "weight": round(weights[index], 3),
        } for index, entry in enumerate(entries)],
        "checks": checks,
        "support": support,
        "answer": answer,
        "frames_seen": frames_seen,
    }


# -- the words ------------------------------------------------------------------------------
def _text_checks(client, entries: list[dict], span: dict, target: CheckTarget,
                 window: float) -> dict[int, dict]:
    """Speech and on-screen text in the window, judged per detail; `{}` if there is none or it failed.

    Captions are deliberately not asked. A caption is one model's guess about a frame, made once,
    offline, without the query in mind, and the same call that judges the pixels is looking at
    those very pixels with the query in hand -- `agent.evidence.summarize` suspends the caption
    channel for exactly this reason whenever vision is available, and vision is always available
    here.
    """
    if not entries:
        return {}
    try:
        heard_or_read = [row for row in excerpts_of(span, [_term_sets(entry) for entry in entries])
                         if row["source"] != "caption"]
        if not heard_or_read:
            return {}
        picked = select_excerpts(heard_or_read, [{"time": target.time}], len(entries),
                                 EXCERPTS_PER_DETAIL, before=window, after=window)
        by_id = {row["id"]: row for row in picked}
        judged = _ask(client, MATCH_SYSTEM, entries, picked, english=False)
    except Exception as exc:
        # The words are the optional half here -- the person pressed a button that means "look
        # at this", and the pixels are what answers it. `_ask` already turns an `LLMError` into
        # None; this is for everything else, so a malformed transcript row cannot cost the
        # check the answer it was actually asked for.
        logger.warning("check: text evidence failed: %s", type(exc).__name__)
        return {}
    if not judged:                       # None (the call failed) or {} (nothing to ask)
        return {}
    best: dict[int, dict] = {}
    order = {"supported": 0, "partial": 1, "contradicted": 2}
    for index, verdicts in judged.items():
        ranked = sorted(
            ((order[status], -by_id[eid]["scores"].get(index, 0),
              abs(by_id[eid]["at"] - target.time), eid, status, note)
             for eid, (status, note) in verdicts.items() if eid in by_id))
        if not ranked:
            continue
        _, _, _, eid, status, note = ranked[0]
        excerpt = by_id[eid]
        if status == "supported" and (mismatch := count_mismatch(entries[index]["vi"], excerpt["text"])):
            status, note = "partial", f"{mismatch}. {note}"[:120]
        best[index] = {"status": status, "source": excerpt["source"],
                       "at": round(excerpt["at"], 1),
                       "quote": excerpt["text"][:MAX_SENTENCE_CHARS], "note": note}
    return best


def _nearest(rows: list[tuple[float, str]], target_time: float, budget: int) -> list[tuple[float, str]]:
    """The lines closest to `target_time` that fit in `budget` characters, nearest first."""
    kept: list[tuple[float, str]] = []
    used = 0
    for start, line in sorted(rows, key=lambda row: abs(row[0] - target_time)):
        used += len(line) + 1
        if used > budget and kept:
            break
        kept.append((start, line))
    return kept


def _said(span: dict, target_time: float, limit: int = CHECK_SAID_CHARS) -> str:
    """The speech and on-screen text of the window, in time order, for the vision call to read.

    **Each channel gets its own budget** (`CHECK_SPEECH_SHARE` to speech, the rest to on-screen
    text), instead of both competing for one pool. That matters for a Q&A answer: the narration is
    where a Vietnamese bulletin puts a place, a person or a fact, while the on-screen text of this
    corpus is dominated by broadcaster furniture -- clocks, channel logos, tickers, "coming up
    next" banners -- which the corpus note itself calls noise. Measured on the Colombia flower
    query (L22_V012, the moment at 692 s): 169 OCR rows in the ±300 s window of which exactly one
    carried anything about the story, against 7 speech lines that held the answer ("...tại
    Medellin."). Sharing one budget let that furniture take 1598 of 2400 characters, and the model
    answered "which city" with "Silletero" -- a word off the caption's festival name -- in one run
    and "Medellín" in another. Speech now cannot be crowded out by it.

    Within each channel the lines CLOSEST TO THE MOMENT are kept first, rather than however far a
    blind head-truncation from the window's start reaches: the same ±300 s window ran to 11.5k
    characters, and the line naming the city sat at offset 5580 -- past a from-the-start cutoff --
    so the model once answered "unknown" to "which city" with the frame, the speech and the query
    all right.

    (Also fixes a latent ordering bug: the previous cut sorted the *strings* lexicographically,
    which reads "[speech 100s]" before "[speech 20s]" and put every "on screen" line before every
    "speech" line regardless of when either was said.)
    """
    def rows(label, entries):
        return [(float(row["start"]), f"[{label} {row['start']:.0f}s] {row['text']}")
                for row in entries if row["text"]]

    speech = _nearest(rows("speech", span["speech"]), target_time,
                      int(limit * CHECK_SPEECH_SHARE))
    spent = sum(len(line) + 1 for _, line in speech)
    onscreen = _nearest(rows("on screen", span["ocr"]), target_time, max(limit - spent, 0))
    kept = sorted(speech + onscreen, key=lambda item: item[0])
    return "\n".join(line for _, line in kept)[:limit]


# -- the pixels -----------------------------------------------------------------------------
def _images(search, target: CheckTarget, window: float) -> list[tuple[str, bytes]]:
    """The moment's own keyframe plus the keyframes of the shots around it, labelled and in order.

    `[M]` is the label on every one of them, so the shared judging rules -- which speak of a
    candidate and its bracketed id -- stay literally true with exactly one candidate present.
    An image that cannot be cut is skipped, never raised: a missing neighbour costs reach, and
    a missing own frame is caught by the caller as "nothing to look at".
    """
    wanted = [(target.frame_idx, target.time, "[M] this moment itself")]
    for context in evidence._context_frames(
            search, target.video_id, target.frame_idx, target.time,
            seconds=window, dense=CHECK_CONTEXT_DENSE, most=CHECK_CONTEXT_MAX,
            count=CHECK_CONTEXT_FRAMES, pick="middle"):
        offset = float(context["time"]) - target.time
        when = f"{abs(offset):.0f}s earlier" if offset < 0 else f"{offset:.0f}s later"
        wanted.append((int(context["frame_idx"]), float(context["time"]),
                       f"[M] shot {when} in the same story"))

    def fetch(item):
        frame_idx, _, label = item
        try:
            return label, search.image(target.video_id, frame_idx, CHECK_IMAGE_WIDTH)
        except Exception:
            return label, None

    with ThreadPoolExecutor(min(CHECK_IMAGE_WORKERS, len(wanted))) as pool:
        fetched = list(pool.map(fetch, wanted))
    return [(label, data) for label, data in fetched if data]


def _vision_check(client, search, entries: list[dict], span: dict, query: CheckQuery,
                  target: CheckTarget, window: float):
    """`(checks, verdict, answer, frames_seen)` from one call over the moment's images."""
    images = _images(search, target, window)
    if not images:
        raise CheckError("no keyframe of this moment could be read")
    asked = query.asked
    prompt = CHECK_SYSTEM + (CHECK_ANSWER_RULE if asked else "")
    lines = []
    for index, entry in enumerate(entries):
        hint = MARKER_HINT if _is_marker_detail(str(entry.get("vi") or "")) else ""
        lines.append(f"{index}. {entry.get('meaning_en') or entry['vi']}{hint}")
        lines += _meaning_lines(entry)
    channels = query.channels
    prompt += "\n\nWhat the person searched with:\n" + "\n".join(
        f"- {label}: {text[:2000]}" for label, text in channels)
    if lines:
        prompt += "\n\nDetails:\n" + "\n".join(lines)
    else:
        # No checklist (the details call failed, or the query is too short to split): the overall
        # verdict is still worth having, and is what the person actually pressed the button for.
        prompt += "\n\nThere is no detail list for this query: judge the query as a whole."
    if asked:
        prompt += f"\n\nThe question: {asked[:500]}"
    if said := _said(span, target.time):
        prompt += f"\n\nWhat is said and shown in words around this moment:\n{said}"

    try:
        reply = client.judge(images, prompt, check_schema(bool(asked)), max_output_tokens=3000,
                             job="check", query=query.description, question=asked, salvage=True)
    except LLMError as exc:
        raise CheckError(str(exc)) from exc
    if not isinstance(reply, dict):
        raise CheckError("the judge returned no usable answer")
    return (_vision_details(reply, entries), _verdict(reply),
            _answer(reply) if asked else None, len(images))


def _vision_details(reply: dict, entries: list[dict]) -> dict[int, dict]:
    found: dict[int, dict] = {}
    for item in reply.get("details") or []:
        if not isinstance(item, dict):
            continue
        index = item.get("detail")
        status = VERDICTS.get(str(item.get("verdict") or "").strip())
        if not isinstance(index, int) or isinstance(index, bool) \
                or not 0 <= index < len(entries) or status is None:
            continue
        why = str(item.get("why") or "")
        note = " ".join(str(item.get("note_vi") or why).split())[:120]
        # The same numeric guard the text judge gets: a judge that answers "exact" while its own
        # words say "two shellfish" for a detail naming four is capped at partial. Only when its
        # words DO state a count -- a note is not everything the model saw, so silence about the
        # number is not the contradiction a transcript sentence's silence would be.
        if status == "supported" and counts_in(f"{why} {note}") \
                and (mismatch := count_mismatch(str(entries[index].get("vi") or ""), f"{why} {note}")):
            status, note = "partial", f"{mismatch}. {note}"[:120]
        found[index] = {"status": status, "note": note}
    return found


def _verdict(reply: dict) -> dict:
    overall = reply.get("overall")
    overall = overall if isinstance(overall, dict) else {}
    try:
        score = float(overall.get("score", 0))
    except (TypeError, ValueError):
        score = 0.0
    return {"score": min(max(score, 0.0), 10.0),
            "reason": " ".join(str(overall.get("reason") or "").split())[:200]}


def _answer(reply: dict) -> dict | None:
    answer = reply.get("answer")
    if not isinstance(answer, dict):
        return None
    text = " ".join(str(answer.get("text") or "").split())[:100]
    source = str(answer.get("source") or "unknown").strip()
    try:
        confidence = float(answer.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not text or source == "unknown":
        # "this moment does not answer it" is an answer the person needs to see, and it is not
        # the same thing as the call having failed.
        return {"text": "", "confidence": 0.0, "source": "unknown", "quote": ""}
    # "ocr" is accepted though the schema no longer offers it: the source label is the model's own
    # report of where it read the answer, and it is measurably unreliable -- on the Colombia query
    # the right answer ("Medellín", which only the narration says) came back labelled "ocr" three
    # runs out of three. Dropping the text over a mislabelled source would lose a good answer, so
    # it is kept and shown for what it claims to be.
    return {"text": text, "confidence": min(max(confidence, 0.0), 10.0),
            "source": source if source in ("frame", "speech", "ocr") else "unknown",
            "quote": " ".join(str(answer.get("quote") or "").split())[:MAX_SENTENCE_CHARS]}


# -- putting the two halves together ---------------------------------------------------------
def _merge(entries: list[dict], text: dict[int, dict], seen: dict[int, dict],
           target: CheckTarget) -> list[dict]:
    """One check per detail: what the words said, upgraded by what the pixels showed.

    Vision only ever upgrades, exactly as in `evidence.summarize`: it looked at this moment and
    the shots around it, so it is stamped at the moment's own time and takes full proximity, but
    a detail it is silent about keeps whatever the speech or on-screen text said about it.
    """
    checks = []
    for index in range(len(entries)):
        found = text.get(index)
        check = ({"detail": index, **found} if found else
                 {"detail": index, "status": "not_mentioned", "source": None,
                  "at": None, "quote": "", "note": ""})
        pixels = seen.get(index)
        if pixels and STATUS_RANK[pixels["status"]] >= STATUS_RANK[check["status"]]:
            check = {"detail": index, "status": pixels["status"], "source": "vision",
                     "at": round(target.time, 1), "quote": "", "note": pixels["note"]}
        checks.append(check)
    return checks


def _support(checks: list[dict], weights: list[float], target: CheckTarget) -> dict:
    """How much of the query this moment's evidence says, weighted by each detail's prior.

    `detail_weights`'s empirical half -- dimming a detail nearly every candidate satisfies --
    needs six read candidates to mean anything, and one moment is not six. So a detail is worth
    its `distinctiveness` prior alone.

    A check counts for what it is worth WHERE IT IS: `evidence.proximity` discounts a sentence
    by how far its shot is from the moment. The window here is minutes wide rather than the
    agent's 45 seconds, so this matters more, not less -- a line spoken four minutes away is
    shown to the person with its timestamp and counts for almost nothing. A vision verdict is at
    distance zero by construction: the shots around the moment ARE that moment's story.

    A contradiction lowers nothing. It is counted and shown (the person reads the reason and the
    frame themselves), but on ten round-2 queries with a known answer, letting contradictions
    demote put the right moment 12th where retrieval had it 1st: captions and judges get colours,
    counts and identities wrong often enough that a conflict is as often their mistake as the
    moment's.
    """
    tally = {"supported": 0, "partial": 0, "contradicted": 0, "not_mentioned": 0}
    earned = 0.0
    for check in checks:
        tally[check["status"]] = tally.get(check["status"], 0) + 1
        value = STATUS_VALUE.get(check["status"], 0.0)
        if not value:
            continue
        at = check.get("at")
        near = 1.0 if at is None else proximity(abs(float(at) - target.time))
        earned += weights[check["detail"]] * value * near
    total = sum(weights) or 1.0
    return {"share": round(min(earned / total, 1.0), 4),
            "supported": tally["supported"], "partial": tally["partial"],
            "contradicted": tally["contradicted"], "total": len(checks)}
