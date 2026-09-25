"""Everything that looks at pixels: Gemini judging, frame picking and QA reading."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

from .llm import LLMError
from .plan import describe_scenes
from .pool import Verdict
from .prompts import (
    FALSIFY_PROMPT, PICK_FRAME_PROMPT, QA_ANSWER_RULE, QA_PROMPT, SCENE_RULE, SEQUENCE_PROMPT,
    VERIFY_PROMPT,
)

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 10},
                    "scene": {"type": "integer"},
                    "answer": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "reason"],
            },
        }
    },
    "required": ["candidates"],
}
PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "reason": {"type": "string"},
    },
    "required": ["id", "confidence"],
}
SEQUENCE_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "score": {"type": "integer", "minimum": 0, "maximum": 10},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score"],
            },
        }
    },
    "required": ["events"],
}
FALSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "detail": {"type": "string"},
                    "status": {"type": "string", "enum": ["confirmed", "contradicted", "not_visible"]},
                },
                "required": ["detail", "status"],
            },
        },
        "verdict": {"type": "string", "enum": ["accept", "reject"]},
        "reason": {"type": "string"},
    },
    "required": ["details", "verdict"],
}
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 10},
        "evidence": {"type": "string"},
    },
    "required": ["answer", "confidence"],
}


#: How much of a shot caption reaches a judge. The corpus averages ~3400 characters;
#: a verify chunk of five candidates would spend ~4k tokens of context on prose that
#: is only a hint. The head is the right end to keep because the caption describes the
#: window the frame sits in, and that window is at most 20 s.
MAX_CAPTION_CHARS = 600


def label_context(candidate, caption: str | None = None) -> dict[str, str]:
    """The retrieval context Gemini is shown beside a keyframe, trimmed as it sees it.

    Shared with the UI's review view, so a teammate checking candidates in
    parallel reads exactly what the model is judging, not a longer or shorter
    version of it -- which is why both sources below go through one limit.

    `caption` is the one fetched for the frames being judged, which the route
    resolves from a frame index; left unset, the caption that rode in on a
    candidate the caption channel retrieved is used instead. Same CapRL text
    either way, and the fetched one is only absent when the server has none.
    """
    def first(kinds, limit):
        text = next((e.text for e in candidate.evidence.values() if e.kind in kinds and e.text), None)
        return text[:limit] if text else None

    described = caption or first(("caption",), MAX_CAPTION_CHARS)
    context = {"speech": first(("audio",), 200), "on_screen": first(("ocr", "ocr_exact"), 150),
               "caption": described[:MAX_CAPTION_CHARS] if described else None}
    return {key: value for key, value in context.items() if value}


def candidate_label(candidate, caption: str | None = None) -> str:
    parts = [f"[{candidate.cid}] {candidate.video_id} t={candidate.time:.1f}s"]
    context = label_context(candidate, caption)
    if "speech" in context:
        parts.append(f'speech nearby: "{context["speech"]}"')
    if "on_screen" in context:
        parts.append(f'on-screen text in this shot: "{context["on_screen"]}"')
    if "caption" in context:
        # Quoted and disclaimed on purpose: this comes from another model looking at
        # the same frames, so it can be confidently wrong, and the judge must weight
        # what it can see over what it is told.
        parts.append(f'shot caption (auto, may be wrong): "{context["caption"]}"')
    return " | ".join(parts)


def caption_at(captions, time_, default="(none)") -> str:
    """The caption of the shot window containing `time_`, else the nearest one."""
    if not captions:
        return default
    inside = [c for c in captions if c["start"] <= time_ <= c["end"]]
    nearest = inside or sorted(captions, key=lambda c: abs((c["start"] + c["end"]) / 2 - time_))[:1]
    return " ".join(str(nearest[0].get("text", "")).split()) or default


def transcript_window(cues, center, seconds=40.0, empty="(no speech)") -> str:
    """Timed lines of whatever was said -- or, with another timed stream, written -- nearby."""
    lines = [
        f"[{cue['start']:.0f}s] {cue['text']}" for cue in cues
        if cue["end"] >= center - seconds and cue["start"] <= center + seconds
    ]
    return "\n".join(lines) or empty


class Inspector:
    def __init__(self, vision, search, *, chunk_size=8, workers=4, image_width=640,
                 observer=None):
        # `vision` is whichever backend `AIC_AGENT_VISION` selected — Gemini, a
        # local model, or one reached over HTTP. They differ only inside `judge`.
        self.vision = vision
        self.search = search
        self.chunk_size = chunk_size
        self.image_width = image_width
        self._workers = workers
        self.observer = observer
        self._images = {}
        self._captions: dict[tuple[str, int], str] = {}
        self._lock = threading.Lock()

    def _judge(self, job, prompt, schema, images, *, max_output_tokens=8192, query=None,
               scenes="", question=None):
        """The one place any of the five jobs reaches a vision model.

        `observer` is for the benchmark harness, which records the exact inputs a
        real run produced so the same calls can be replayed through other models.
        It is None in every production path.

        `query` is the original Vietnamese question — or, for `sequence`, one per
        frame — and `scenes`/`question` are the caller's own renderings of the scene
        list and the Q&A question. Only the local backends use them, to build a short
        prompt in place of the long English one Gemini is given.
        """
        reply = self.vision.judge(
            images, prompt, schema, max_output_tokens=max_output_tokens, job=job,
            query=query, scenes=scenes, question=question,
        )
        if self.observer is not None:
            self.observer(job=job, prompt=prompt, schema=schema, images=images, reply=reply)
        return reply

    def _image(self, video_id, frame_idx, width):
        key = (video_id, int(frame_idx), width)
        with self._lock:
            if key in self._images:
                return self._images[key]
        data = self.search.image(video_id, frame_idx, width)
        with self._lock:
            self._images[key] = data
        return data

    def images(self, frames, width=None):
        """JPEG bytes per (video, frame), or None for a frame the server could not cut."""
        width = self.image_width if width is None else width

        def fetch(frame):
            try:
                return self._image(frame[0], frame[1], width)
            except Exception:
                return None

        with ThreadPoolExecutor(self._workers) as pool:
            return list(pool.map(fetch, frames))

    def captions(self, frames) -> dict[tuple[str, int], str]:
        """Shot captions per `(video, frame)`, for the frames that have one.

        One request per video, so a chunk of five neighbours costs one round trip
        rather than five. Returns `{}` against a server without the route, and for
        2025 videos, which were never captioned.
        """
        if not getattr(self.search, "captions_for_frames", None):
            return {}
        hits, want = {}, {}
        for video_id, frame_idx in frames:
            key = (video_id, int(frame_idx))
            with self._lock:
                if key in self._captions:
                    hits[key] = self._captions[key]
                    continue
            want.setdefault(video_id, []).append(int(frame_idx))

        def fetch(item):
            video_id, indices = item
            try:
                return video_id, self.search.captions_for_frames(video_id, indices)
            except Exception:  # a caption is context; never let it cost the round
                return video_id, {}

        with ThreadPoolExecutor(self._workers) as pool:
            for video_id, found in pool.map(fetch, list(want.items())):
                for frame_idx, value in (found or {}).items():
                    text = (value or {}).get("caption") or ""
                    if text:
                        hits[(video_id, frame_idx)] = text
        with self._lock:
            self._captions.update(hits)
        return hits

    def caption_for(self, video_id, frame_idx) -> str | None:
        """The caption this run already fetched for one frame, if it fetched one.

        The UI's review view reads it so a reviewer sees the caption the judge
        read. Without it a caption the verify step fetched reaches the model and
        not the person checking its work, which is the one thing
        `label_context`'s shared trimming exists to prevent.
        """
        with self._lock:
            return self._captions.get((video_id, int(frame_idx)))

    def verify(self, task, plan, candidates, question=None) -> tuple[dict[str, Verdict], list[str]]:
        """Judge candidates in chunks; returns verdicts by id and any chunk errors."""
        if not candidates:
            return {}, []
        question = question or plan.question_vi
        # Only rendered when there is more than one scene: with one, `pool.py`'s
        # scene bonus cannot move anything, and asking for the index would spend
        # tokens on a field nothing reads.
        scenes = (describe_scenes(plan) + SCENE_RULE) if len(plan.scenes) > 1 else ""
        prompt = VERIFY_PROMPT.format(
            query=task.text,
            scenes=scenes,
            question=f"Question to answer: {question}\n" if question else "",
            answer_rule=QA_ANSWER_RULE if question else "",
        )
        chunks = [candidates[i:i + self.chunk_size]
                  for i in range(0, len(candidates), self.chunk_size)]

        def judge(chunk):
            frames = [(c.video_id, c.frame_idx) for c in chunk]
            data = self.images(frames)
            captions = self.captions(frames)
            # A candidate without an image stays unverified instead of failing its chunk.
            labelled = [(candidate_label(c, captions.get((c.video_id, c.frame_idx))), image)
                        for c, image in zip(chunk, data) if image]
            if not labelled:
                raise RuntimeError("no candidate image could be fetched")
            return self._judge("verify", prompt, VERIFY_SCHEMA, labelled, query=task.text,
                               scenes=scenes, question=question)

        verdicts, errors = {}, []
        known = {candidate.cid for candidate in candidates}
        with ThreadPoolExecutor(min(self._workers, len(chunks))) as pool:
            futures = [pool.submit(judge, chunk) for chunk in chunks]
            for future in futures:
                try:
                    reply = future.result()
                except Exception as exc:  # one failed chunk must not sink the round
                    errors.append(str(exc))
                    continue
                for item in (reply or {}).get("candidates", []):
                    cid = str(item.get("id", "")).strip("[] ")
                    if cid not in known:
                        continue
                    verdicts[cid] = Verdict(
                        score=float(min(max(item.get("score", 0), 0), 10)),
                        reason=str(item.get("reason", ""))[:200],
                        answer=(str(item.get("answer") or "").strip()[:100] or None),
                        scene=item.get("scene") if isinstance(item.get("scene"), int) else None,
                    )
        return verdicts, errors

    def pick_frame(self, query, event, frames):
        """First frame (of time-ordered `frames`) where `event` shows, or None."""
        if not frames:
            return None, 0, "no frames"
        labels = [f"F{index}" for index in range(len(frames))]
        data = self.images([(f["video_id"], f["frame_idx"]) for f in frames], width=320)
        labelled = [(f"[{label}] t={frame['time']:.1f}s", image)
                    for label, frame, image in zip(labels, frames, data) if image]
        if not labelled:
            return None, 0, "no frame image could be fetched"
        try:
            reply = self._judge(
                "pick", PICK_FRAME_PROMPT.format(query=query, event=event), PICK_SCHEMA,
                labelled, max_output_tokens=4096,
            )
        except LLMError as exc:
            return None, 0, str(exc)
        label = str(reply.get("id", "")).strip("[] ")
        if label not in labels:
            return None, int(reply.get("confidence", 0) or 0), str(reply.get("reason", ""))
        return frames[labels.index(label)], int(reply.get("confidence", 0) or 0), str(reply.get("reason", ""))

    def falsify(self, task, candidate, neighbours):
        """A skeptical second look before committing: (rejected, reason, contradicted).

        Uses the neighbouring keyframes too, since one keyframe can miss a detail the
        next one shows.
        """
        frames = sorted({(f["video_id"], f["frame_idx"]) for f in neighbours}
                        | {(candidate.video_id, candidate.frame_idx)}, key=lambda item: item[1])
        data = self.images(frames)
        labelled = [
            ("[CANDIDATE]" if frame == candidate.frame_idx else f"[frame {frame}]", image)
            for (_, frame), image in zip(frames, data) if image
        ]
        reply = self._judge("falsify", FALSIFY_PROMPT.format(query=task.text), FALSIFY_SCHEMA,
                            labelled, max_output_tokens=4096)
        contradicted = [str(item.get("detail", "")) for item in reply.get("details", [])
                        if item.get("status") == "contradicted"]
        rejected = reply.get("verdict") == "reject" or bool(contradicted)
        return rejected, str(reply.get("reason", ""))[:200], contradicted

    def check_sequence(self, task, events, video_id, frames):
        """Per-event scores (0-10, one per frame) for one TRAKE path.

        Each frame is judged against its own event, unlike `verify`, which scores a
        keyframe against the whole query -- where an intro shot the query mentions
        for context outscores the events themselves.
        """
        data = self.images([(video_id, frame) for frame in frames])
        if not any(data):
            raise RuntimeError("no event frame image could be fetched")
        # An event whose frame the server could not cut is scored 5 (unknown) below
        # rather than failing the whole path.
        labelled = [(f"[E{index + 1}] {video_id} frame {frame}", image)
                    for index, (frame, image) in enumerate(zip(frames, data)) if image]
        listing = "\n".join(f"E{index + 1}: {event}" for index, event in enumerate(events))
        reply = self._judge(
            "sequence", SEQUENCE_PROMPT.format(query=task.text, events=listing), SEQUENCE_SCHEMA,
            labelled, max_output_tokens=4096,
            # One query per frame: each is scored against its own event, so handing
            # the compact prompt the joined event list would ask whether one frame
            # starts all of them at once.
            query=[str(event) for event in events],
        )
        scores = {str(item.get("id", "")).strip("[] "): float(item.get("score", 0) or 0)
                  for item in reply.get("events", [])}
        return [min(max(scores.get(f"E{index + 1}", 0.0), 0.0), 10.0) if image else 5.0
                for index, image in enumerate(data)]

    def answer(self, task, question, candidate, neighbours, cues, hints="", captions=(),
               on_screen=""):
        """Read a QA answer from one moment: full frame, neighbours, speech, on-screen text and caption."""
        frames = [(candidate.video_id, candidate.frame_idx)]
        frames += [(f["video_id"], f["frame_idx"]) for f in neighbours
                   if f["frame_idx"] != candidate.frame_idx][:6]
        data = [self._image(candidate.video_id, candidate.frame_idx, 0)]
        data += self.images(frames[1:])
        labelled = [(f"[{video}:{frame}]", image)
                    for (video, frame), image in zip(frames, data) if image]
        # The on-screen block is left out entirely when the caller has no OCR for the moment:
        # `_read_answer` (the model-judged path) never passes one, and its prompt must stay
        # the prompt this reader was tuned on.
        prompt = QA_PROMPT.format(
            query=task.text, question=question,
            transcript=transcript_window(cues, candidate.time),
            on_screen=(f"\nText read off the screen in this shot, timed "
                       f"(OCR of the keyframe; the reading, not what was written, is what can be "
                       f"wrong here):\n{on_screen}\n" if on_screen else ""),
            caption=caption_at(captions, candidate.time),
            hints=hints or "(none)",
        )
        return self._judge("answer", prompt, ANSWER_SCHEMA, labelled, max_output_tokens=4096)
