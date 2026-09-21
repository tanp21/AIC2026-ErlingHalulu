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
    "ocr": 0.8, "ocr_exact": 1.2, "caption": 1.0, "neighbour": 0.0,
}
POINT_KINDS = {"visual", "similar", "temporal", "neighbour"}
SEQUENCE_RRF_K = 10
SCENE_BONUS = 1.0
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
    #: "gemini", or "human" for a teammate's decision in the UI, which always wins.
    by: str = "gemini"


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
    # Gemini's per-event 0-10 check of `frames` (before any refinement), if run.
    event_scores: list[float] | None = None


def local_path(event_hits: list[list[dict]], max_gap: float, span_cost: float = 0.001):
    """Best in-order keyframe per event within one video, or None.

    `event_hits[e]` are that event's scoped visual hits (`time`, `frame_idx`,
    `score`). Consecutive events must be strictly later and at most `max_gap`
    seconds apart; among valid paths the highest summed score wins, a little
    less per second of total span.
    """
    if not event_hits or not all(event_hits):
        return None
    # best[e][j] = (value, back-pointer) for a path ending at hit j of event e
    best = [[(hit.get("score", 0.0), None) for hit in event_hits[0]]]
    for index in range(1, len(event_hits)):
        row = []
        for hit in event_hits[index]:
            options = [
                (value - span_cost * (hit["time"] - previous["time"]) + hit.get("score", 0.0), j)
                for j, ((value, _), previous) in enumerate(zip(best[-1], event_hits[index - 1]))
                if value is not None and 0 < hit["time"] - previous["time"] <= max_gap
            ]
            row.append(max(options) if options else (None, None))
        best.append(row)
    ends = [(value, j) for j, (value, _) in enumerate(best[-1]) if value is not None]
    if not ends:
        return None
    value, j = max(ends)
    path = []
    for index in range(len(event_hits) - 1, -1, -1):
        path.append(event_hits[index][j])
        j = best[index][j][1]
    return value, path[::-1]


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
        for candidate in self.candidates.values():
            candidate.fused = sum(_weight(e) / (RRF_K + e.rank) for e in candidate.evidence.values())
        video_scores = self._video_support()
        for candidate in self.candidates.values():
            candidate.fused += VIDEO_WEIGHT * video_scores.get(candidate.video_id, 0.0)

    def ordered(self) -> list[Candidate]:
        """Verified matches first, then unverified by fusion, rejected last.

        Among verified matches, a video in which Gemini confirmed several different
        scenes of a multi-scene query gains SCENE_BONUS per extra scene: each
        keyframe shows one scene, so on p1-17 a 10/10 motorbike shot from another
        video outranked the gold video whose mudslide, road marker and motorbike
        scenes were each matched.
        """
        self.rescore()
        scenes: dict[str, set] = defaultdict(set)
        for candidate in self.candidates.values():
            verdict = candidate.verdict
            if verdict is not None and verdict.scene is not None and verdict.score >= VERIFIED_MATCH:
                scenes[candidate.video_id].add(verdict.scene)

        def key(candidate):
            verdict = candidate.verdict
            if verdict is None:
                return (1, 0.0, -candidate.fused)
            if verdict.score >= VERIFIED_MATCH:
                if verdict.by == "human":  # a teammate's acceptance outranks any model
                    return (-1, -verdict.score, -candidate.fused)
                bonus = SCENE_BONUS * max(0, len(scenes.get(candidate.video_id, ())) - 1)
                return (0, -(verdict.score + bonus), -candidate.fused)
            if verdict.score <= VERIFIED_REJECT:
                return (2, -verdict.score, -candidate.fused)
            return (1, -verdict.score / 10, -candidate.fused)

        return sorted(self.candidates.values(), key=key)

    def unverified(self, limit: int) -> list[Candidate]:
        return [candidate for candidate in self.ordered() if candidate.verdict is None][:limit]

    def to_verify(self, limit: int, quota_share: float = 0.6) -> list[Candidate]:
        """Unverified candidates worth a Gemini look: each query's best hits, then by fusion.

        Fusion alone favours moments many weak hits agree on (a transcript span, an
        OCR'd shot, a temporal event), and on round 2 it left a caption's rank-1
        keyframe -- the gold frame of p2-3 -- unverified at fused rank 32. Up to
        `quota_share` of the slots therefore go round-robin to every query's best
        unverified hits, visual queries first.
        """
        pending = [candidate for candidate in self.ordered() if candidate.verdict is None]
        per_source: dict[str, list[tuple[int, Candidate]]] = defaultdict(list)
        for candidate in pending:
            for evidence in candidate.evidence.values():
                if evidence.kind != "neighbour":
                    per_source[evidence.source].append((evidence.rank, candidate))
        sources = sorted(per_source, key=lambda source: (
            -_weight(next(iter(per_source[source]))[1].evidence[source]), source))
        queues = [[c for _, c in sorted(per_source[source], key=lambda item: item[0])]
                  for source in sources]
        chosen: list[Candidate] = []
        quota = min(limit, round(limit * quota_share))
        while len(chosen) < quota and any(queues):
            for queue in queues:
                while queue and queue[0] in chosen:
                    queue.pop(0)
                if queue and len(chosen) < quota:
                    chosen.append(queue.pop(0))
        chosen += [candidate for candidate in pending if candidate not in chosen][:limit - len(chosen)]
        return chosen

    def ranked_sequences(self, count: int | None = None) -> list[Sequence]:
        """Unique temporal paths, best first, fused over every temporal query and the video.

        A path's own rank is backed by other temporal queries that found the same
        video and by how strongly the non-temporal channels (visual, speech, OCR)
        point at that video: on p1-16 the gold video was path 8 but video 2 overall,
        its intro shot being in the query rather than in the events.
        """
        unique: dict[tuple, Sequence] = {}
        best_rank: dict[tuple[str, str], int] = {}
        for sequence in self.sequences:
            source_key = (sequence.video_id, sequence.source)
            best_rank[source_key] = min(best_rank.get(source_key, sequence.rank), sequence.rank)
            if count and len(sequence.frames) != count:
                continue
            key = (sequence.video_id, tuple(sequence.frames))
            if key not in unique or sequence.rank < unique[key].rank:
                unique[key] = sequence
        support = self._video_support(exclude={"temporal", "neighbour"})
        video_rank = {video: rank for rank, video in enumerate(
            sorted(support, key=support.get, reverse=True), 1)}

        def score(sequence):
            value = sum(1.0 / (SEQUENCE_RRF_K + rank)
                        for (video, _), rank in best_rank.items() if video == sequence.video_id)
            if sequence.video_id in video_rank:
                value += 1.0 / (SEQUENCE_RRF_K + video_rank[sequence.video_id])
            return value

        return sorted(unique.values(), key=lambda s: (-score(s), s.rank))

    def _video_support(self, exclude=frozenset()) -> dict[str, float]:
        best: dict[str, dict[str, float]] = defaultdict(dict)
        for candidate in self.candidates.values():
            for evidence in candidate.evidence.values():
                if evidence.kind in exclude:
                    continue
                contribution = _weight(evidence) / (RRF_K + evidence.rank)
                sources = best[candidate.video_id]
                sources[evidence.source] = max(sources.get(evidence.source, 0.0), contribution)
        return {video: sum(sources.values()) for video, sources in best.items()}

    def video_ranking(self) -> list[str]:
        seen = []
        for candidate in self.ordered():
            if candidate.video_id not in seen:
                seen.append(candidate.video_id)
        return seen
