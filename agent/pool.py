"""The candidate moments one agent run has found so far, and how they rank.

Every search result is folded into *moments*: one (video, keyframe) per distinct
point in time. Point hits (a keyframe from visual/similar search) anchor moments;
span hits (a transcript passage, an OCR'd shot) back every moment inside their span
and only create a moment of their own when nothing is there yet. Ranking is
reciprocal-rank fusion over every query that touched a moment, plus a video-level
term so a moment in a video many queries agree on outranks an isolated one.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

RRF_K = 60
POINT_WINDOW_SECONDS = 4.0
SPAN_PAD_SECONDS = 2.0
VIDEO_WEIGHT = 0.5
KIND_WEIGHTS = {
    "visual": 1.0, "similar": 0.8, "temporal": 1.0, "audio": 0.8,
    "ocr": 0.8, "ocr_exact": 1.2, "neighbour": 0.0,
}
POINT_KINDS = {"visual", "similar", "temporal", "neighbour"}
VERIFIED_MATCH = 6.0
VERIFIED_REJECT = 3.0


@dataclass
class Evidence:
    source: str
    kind: str
    rank: int
    score: float
    text: str | None = None
    start: float | None = None
    end: float | None = None


@dataclass
class Verdict:
    score: float
    reason: str
    answer: str | None = None
    scene: int | None = None


@dataclass
class Candidate:
    cid: str
    video_id: str
    frame_idx: int
    time: float
    evidence: dict[str, Evidence] = field(default_factory=dict)
    verdict: Verdict | None = None
    fused: float = 0.0

    def add(self, evidence: Evidence) -> None:
        current = self.evidence.get(evidence.source)
        if current is None or evidence.rank < current.rank:
            self.evidence[evidence.source] = evidence


@dataclass
class Sequence:
    """One temporal-search path: an ordered keyframe per event in one video."""
    source: str
    rank: int
    video_id: str
    frames: list[int]
    times: list[float]
    score: float
    candidate_ids: list[str]
    refined: bool = False


def _weight(evidence: Evidence) -> float:
    return KIND_WEIGHTS.get(evidence.kind, 1.0)


class CandidatePool:
    def __init__(self):
        self.candidates: dict[str, Candidate] = {}
        self.sequences: list[Sequence] = []
        self._by_video: dict[str, list[Candidate]] = defaultdict(list)
        self._by_frame: dict[tuple[str, int], Candidate] = {}

    def __len__(self):
        return len(self.candidates)

    def get(self, cid: str) -> Candidate | None:
        return self.candidates.get(cid)

    def _new(self, video_id, frame_idx, time):
        candidate = Candidate(f"C{len(self.candidates) + 1}", video_id, int(frame_idx), float(time))
        self.candidates[candidate.cid] = candidate
        self._by_video[video_id].append(candidate)
        self._by_frame[(video_id, int(frame_idx))] = candidate
        return candidate

    def _point(self, video_id, frame_idx, time, evidence, merge=True):
        candidate = self._by_frame.get((video_id, int(frame_idx)))
        if candidate is None and not merge:
            candidate = self._new(video_id, frame_idx, time)
        elif candidate is None:
            nearby = [
                existing for existing in self._by_video.get(video_id, ())
                if abs(existing.time - time) <= POINT_WINDOW_SECONDS
            ]
            candidate = (min(nearby, key=lambda existing: abs(existing.time - time))
                         if nearby else self._new(video_id, frame_idx, time))
        candidate.add(evidence)
        return candidate

    def _span(self, hit, evidence):
        start = hit.get("start", hit["time"]) - SPAN_PAD_SECONDS
        end = hit.get("end", hit["time"]) + SPAN_PAD_SECONDS
        inside = [c for c in self._by_video.get(hit["video_id"], ()) if start <= c.time <= end]
        if not inside:
            inside = [self._point(hit["video_id"], hit["frame_idx"], hit["time"], evidence)]
        for candidate in inside:
            candidate.add(evidence)
        return inside

    def add_frame(self, video_id, frame_idx, time, source="manual", kind="neighbour"):
        """Register one specific keyframe (e.g. a neighbour worth inspecting).

        Never merged into a nearby moment: the point is to look at this exact frame.
        """
        return self._point(video_id, frame_idx, time, Evidence(source, kind, 1, 0.0), merge=False)

    def add_results(self, results: dict[str, dict]) -> list[str]:
        """Fold one `/agent/search` batch in; returns ids of candidates it created."""
        before = set(self.candidates)
        # Point hits first, by rank across queries, so the strongest keyframes anchor
        # moments before transcript/OCR spans attach to them.
        points, spans = [], []
        for source, result in results.items():
            kind = result.get("kind")
            if kind == "temporal":
                self._add_temporal(source, result.get("hits", []))
                continue
            for hit in result.get("hits", []):
                (points if kind in POINT_KINDS else spans).append((hit["rank"], source, kind, hit))
        for rank, source, kind, hit in sorted(points, key=lambda item: item[0]):
            self._point(hit["video_id"], hit["frame_idx"], hit["time"],
                        Evidence(source, kind, rank, hit.get("score", 0.0)))
        for rank, source, kind, hit in sorted(spans, key=lambda item: item[0]):
            if kind == "ocr" and hit.get("exact"):
                kind = "ocr_exact"
            self._span(hit, Evidence(
                source, kind, rank, hit.get("score", 0.0), hit.get("text"),
                hit.get("start"), hit.get("end"),
            ))
        return [cid for cid in self.candidates if cid not in before]

    def _add_temporal(self, source, videos):
        # Only each video's best path: the server already ranks videos by it, and
        # alternates within one video mostly re-pick neighbouring keyframes.
        for video in videos:
            if not video.get("sequences"):
                continue
            sequence = video["sequences"][0]
            ids = []
            for event_index, event in enumerate(sequence["events"]):
                candidate = self._point(
                    video["video_id"], event["frame_idx"], event["time"],
                    Evidence(f"{source}.e{event_index + 1}", "temporal", video["rank"],
                             event.get("score", 0.0)),
                )
                ids.append(candidate.cid)
            self.sequences.append(Sequence(
                source, video["rank"], video["video_id"],
                [event["frame_idx"] for event in sequence["events"]],
                [event["time"] for event in sequence["events"]],
                float(sequence.get("score", video.get("score", 0.0))), ids,
            ))

    def rescore(self) -> None:
        video_best: dict[str, dict[str, float]] = defaultdict(dict)
        for candidate in self.candidates.values():
            candidate.fused = 0.0
            for evidence in candidate.evidence.values():
                contribution = _weight(evidence) / (RRF_K + evidence.rank)
                candidate.fused += contribution
                best = video_best[candidate.video_id]
                best[evidence.source] = max(best.get(evidence.source, 0.0), contribution)
        video_scores = {video: sum(best.values()) for video, best in video_best.items()}
        for candidate in self.candidates.values():
            candidate.fused += VIDEO_WEIGHT * video_scores.get(candidate.video_id, 0.0)

    def ordered(self) -> list[Candidate]:
        """Verified matches first, then unverified by fusion, rejected last."""
        self.rescore()

        def key(candidate):
            verdict = candidate.verdict
            if verdict is None:
                return (1, 0.0, -candidate.fused)
            if verdict.score >= VERIFIED_MATCH:
                return (0, -verdict.score, -candidate.fused)
            if verdict.score <= VERIFIED_REJECT:
                return (2, -verdict.score, -candidate.fused)
            return (1, -verdict.score / 10, -candidate.fused)

        return sorted(self.candidates.values(), key=key)

    def unverified(self, limit: int) -> list[Candidate]:
        return [candidate for candidate in self.ordered() if candidate.verdict is None][:limit]

    def video_ranking(self) -> list[str]:
        seen = []
        for candidate in self.ordered():
            if candidate.video_id not in seen:
                seen.append(candidate.video_id)
        return seen
