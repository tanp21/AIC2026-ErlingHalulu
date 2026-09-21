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


def label_context(candidate) -> dict[str, str]:
    """The retrieval context Gemini is shown beside a keyframe, trimmed as it sees it.

    Shared with the UI's review view, so a teammate checking candidates in
    parallel reads exactly what the model is judging, not a longer or shorter
    version of it.
    """
    def first(kinds, limit):
        text = next((e.text for e in candidate.evidence.values() if e.kind in kinds and e.text), None)
        return text[:limit] if text else None

    context = {"speech": first(("audio",), 200), "on_screen": first(("ocr", "ocr_exact"), 150),
               "caption": first(("caption",), 300)}
    return {key: value for key, value in context.items() if value}


def candidate_label(candidate) -> str:
    parts = [f"[{candidate.cid}] {candidate.video_id} t={candidate.time:.1f}s"]
    context = label_context(candidate)
    if "speech" in context:
        parts.append(f'speech nearby: "{context["speech"]}"')
    if "on_screen" in context:
        parts.append(f'on-screen text in this shot: "{context["on_screen"]}"')
    if "caption" in context:
        parts.append(f'shot caption: "{context["caption"]}"')
    return " | ".join(parts)


def caption_at(captions, time_, default="(none)") -> str:
    """The caption of the shot window containing `time_`, else the nearest one."""
    if not captions:
        return default
    inside = [c for c in captions if c["start"] <= time_ <= c["end"]]
    nearest = inside or sorted(captions, key=lambda c: abs((c["start"] + c["end"]) / 2 - time_))[:1]
    return " ".join(str(nearest[0].get("text", "")).split()) or default


def transcript_window(cues, center, seconds=40.0) -> str:
    lines = [
        f"[{cue['start']:.0f}s] {cue['text']}" for cue in cues
        if cue["end"] >= center - seconds and cue["start"] <= center + seconds
    ]
    return "\n".join(lines) or "(no speech)"


class Inspector:
    def __init__(self, gemini, search, *, chunk_size=8, workers=4, image_width=640):
        self.gemini = gemini
        self.search = search
        self.chunk_size = chunk_size
        self.image_width = image_width
        self._workers = workers
        self._images = {}
        self._lock = threading.Lock()

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

    def verify(self, task, plan, candidates, question=None) -> tuple[dict[str, Verdict], list[str]]:
        """Judge candidates in chunks; returns verdicts by id and any chunk errors."""
        if not candidates:
            return {}, []
        question = question or plan.question_vi
        prompt = VERIFY_PROMPT.format(
            query=task.text,
            scenes=(describe_scenes(plan) + SCENE_RULE) if len(plan.scenes) > 1 else "",
            question=f"Question to answer: {question}\n" if question else "",
            answer_rule=QA_ANSWER_RULE if question else "",
        )
        chunks = [candidates[i:i + self.chunk_size]
                  for i in range(0, len(candidates), self.chunk_size)]

        def judge(chunk):
            data = self.images([(c.video_id, c.frame_idx) for c in chunk])
            # A candidate without an image stays unverified instead of failing its chunk.
            labelled = [(candidate_label(c), image) for c, image in zip(chunk, data) if image]
            if not labelled:
                raise RuntimeError("no candidate image could be fetched")
            return self.gemini.judge(labelled, prompt, VERIFY_SCHEMA)

        verdicts, errors = {}, []
        known = {candidate.cid for candidate in candidates}
        with ThreadPoolExecutor(min(4, len(chunks))) as pool:
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
            reply = self.gemini.judge(
                labelled, PICK_FRAME_PROMPT.format(query=query, event=event), PICK_SCHEMA,
                max_output_tokens=4096,
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
        reply = self.gemini.judge(labelled, FALSIFY_PROMPT.format(query=task.text), FALSIFY_SCHEMA,
                                  max_output_tokens=4096)
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
        reply = self.gemini.judge(
            labelled, SEQUENCE_PROMPT.format(query=task.text, events=listing), SEQUENCE_SCHEMA,
            max_output_tokens=4096,
        )
        scores = {str(item.get("id", "")).strip("[] "): float(item.get("score", 0) or 0)
                  for item in reply.get("events", [])}
        return [min(max(scores.get(f"E{index + 1}", 0.0), 0.0), 10.0) if image else 5.0
                for index, image in enumerate(data)]

    def answer(self, task, question, candidate, neighbours, cues, hints="", captions=()):
        """Read a QA answer from one moment: full frame, neighbours, speech and caption."""
        frames = [(candidate.video_id, candidate.frame_idx)]
        frames += [(f["video_id"], f["frame_idx"]) for f in neighbours
                   if f["frame_idx"] != candidate.frame_idx][:6]
        data = [self._image(candidate.video_id, candidate.frame_idx, 0)]
        data += self.images(frames[1:])
        labelled = [(f"[{video}:{frame}]", image)
                    for (video, frame), image in zip(frames, data) if image]
        prompt = QA_PROMPT.format(
            query=task.text, question=question,
            transcript=transcript_window(cues, candidate.time),
            caption=caption_at(captions, candidate.time),
            hints=hints or "(none)",
        )
        return self.gemini.judge(labelled, prompt, ANSWER_SCHEMA, max_output_tokens=4096)
