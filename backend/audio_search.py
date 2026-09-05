"""Transcript passage construction and Elasticsearch-backed retrieval."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import json
import os
from pathlib import Path
import math
from typing import Callable, Hashable, Iterable, Sequence


AUDIO_INDEX = os.environ.get("AIC_AUDIO_INDEX", "aic_audio_v1")
EMBEDDING_DIM = 2560
QUERY_INSTRUCTION = (
    "Given a Vietnamese or English description, retrieve the video transcript "
    "passage that best matches the described spoken content."
)
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
ROOT = Path(__file__).resolve().parent.parent
_AUDIO_DIR = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data")) / "2026" / "audio"
RAW_TRANSCRIPT_PATH = _AUDIO_DIR / "step_data.parquet"
AUDIO_FLAT_STEM = Path(os.environ.get(
    "AIC_AUDIO_FLAT_PATH",
    Path(os.environ.get("AIC_INDEX_DIR", ROOT / "indexes")) / "audio_qwen3_4b",
))


def transcript_path() -> Path:
    """Which transcript parquet the runtime reads. Raw ASR output by default.

    Cleaned transcripts are **opt-in** via ``AIC_TRANSCRIPT_PATH``. An earlier
    version of this function preferred ``step_data_clean.parquet`` whenever the
    file happened to exist, which meant an offline cleaning run silently changed
    what a running server served — the subtitles moved without anyone deploying
    anything. Generating a file and serving it are separate decisions, so
    switching now takes an explicit environment variable.

    ``scripts.audio.build_clean_transcripts`` writes its output in this same
    schema, so any parquet named here is interchangeable with the raw one.
    """
    override = os.environ.get("AIC_TRANSCRIPT_PATH")
    return Path(override) if override else RAW_TRANSCRIPT_PATH


#: Resolved at import for backwards compatibility; call ``transcript_path()``
#: to honour an ``AIC_TRANSCRIPT_PATH`` set after import.
TRANSCRIPT_PATH = transcript_path()

_client = None
_model = None


@dataclass(frozen=True)
class Passage:
    passage_id: str
    video_id: str
    start_sec: float
    end_sec: float
    anchor_sec: float
    text: str


@dataclass(frozen=True)
class AudioHit:
    video_id: str
    start_sec: float
    end_sec: float
    anchor_sec: float
    text: str
    score: float
    bm25_rank: int | None = None
    vector_rank: int | None = None
    passage_id: str | None = None
    matched_keywords: tuple[str, ...] = ()


def _clean_text(text: str) -> str:
    return " ".join(text.replace("<", " ").replace(">", " ").split())


def build_transcript_cues(rows: Sequence[dict]) -> list[dict]:
    valid = []
    for row in rows:
        token = row.get("token")
        if not isinstance(token, str) or not token:
            continue
        try:
            offset = float(row.get("offset"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(offset) and offset >= 0:
            valid.append({"offset": offset, "token": token})
    valid.sort(key=lambda row: row["offset"])
    groups = []
    tokens: list[str] = []
    start = last = 0.0
    characters = 0
    for row in valid:
        if tokens and row["offset"] - start >= 8:
            groups.append((start, last, _clean_text("".join(tokens))))
            tokens = []
            characters = 0
        if not tokens:
            start = row["offset"]
        tokens.append(row["token"])
        characters += len(row["token"])
        last = row["offset"]
        ends_sentence = row["token"].rstrip().endswith((".", "!", "?", "…"))
        if ends_sentence or characters >= 160:
            groups.append((start, last, _clean_text("".join(tokens))))
            tokens = []
            characters = 0
    if tokens:
        groups.append((start, last, _clean_text("".join(tokens))))

    cues = []
    for index, (cue_start, cue_last, text) in enumerate(groups):
        next_start = groups[index + 1][0] if index + 1 < len(groups) else cue_last + 2
        cues.append({
            "start": round(cue_start, 3),
            "end": round(min(cue_last + 2, next_start), 3),
            "text": text,
        })
    return cues


@lru_cache(maxsize=32)
def load_transcript_cues(video_id: str) -> list[dict]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        transcript_path(),
        columns=["offset", "token"],
        filters=[("audio_key", "=", video_id)],
    )
    columns = table.to_pydict()
    return build_transcript_cues([
        {"offset": offset, "token": token}
        for offset, token in zip(columns["offset"], columns["token"])
    ])


def build_index_definition() -> dict:
    return {
        "settings": {
            "analysis": {
                "analyzer": {
                    "vi_folded": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    }
                }
            }
        },
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "passage_id": {"type": "keyword"},
                "video_id": {"type": "keyword"},
                "group": {"type": "keyword"},
                "year": {"type": "keyword"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "anchor_sec": {"type": "float"},
                "text": {
                    "type": "text",
                    "fields": {"folded": {"type": "text", "analyzer": "vi_folded"}},
                },
                "embedding": {
                    "type": "dense_vector",
                    "dims": EMBEDDING_DIM,
                    "index": True,
                    "similarity": "cosine",
                },
            },
        },
    }


def build_bm25_request(text: str, k: int = 100) -> dict:
    candidate_count = max(100, k)
    return {
        "size": candidate_count,
        "_source": {"excludes": ["embedding"]},
        "highlight": {
            "pre_tags": ["<aic-hit>"], "post_tags": ["</aic-hit>"],
            "fields": {"text": {"number_of_fragments": 0}},
        },
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": text,
                            "fields": ["text^2", "text.folded"],
                            "type": "best_fields",
                        }
                    },
                    {
                        "multi_match": {
                            "query": text,
                            "fields": ["text^4", "text.folded^2"],
                            "type": "phrase",
                            "slop": 2,
                        }
                    },
                    {
                        "multi_match": {
                            "query": text,
                            "fields": ["text", "text.folded"],
                            "type": "best_fields",
                            "fuzziness": "AUTO",
                            "prefix_length": 1,
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        },
    }


def build_knn_request(query_vector: Sequence[float], k: int = 100) -> dict:
    candidate_count = max(100, k)
    return {
        "size": candidate_count,
        "_source": {"excludes": ["embedding"]},
        "knn": {
            "field": "embedding",
            "query_vector": list(query_vector),
            "k": candidate_count,
            "num_candidates": min(10_000, max(200, candidate_count * 2)),
        },
    }


def make_passages(
    video_id: str,
    rows: Sequence[dict],
    window_seconds: int = 60,
    stride_seconds: int = 15,
) -> list[Passage]:
    if not rows:
        return []
    end = int(float(rows[-1]["offset"])) + 1
    passages = []
    for start in range(0, end + 1, stride_seconds):
        text = _clean_text(
            "".join(
                str(row["token"])
                for row in rows
                if start <= float(row["offset"]) < start + window_seconds
            )
        )
        if not text:
            continue
        passages.append(
            Passage(
                passage_id=f"{video_id}:{start * 1000}",
                video_id=video_id,
                start_sec=float(start),
                end_sec=float(start + window_seconds),
                anchor_sec=float(start + window_seconds / 2),
                text=text,
            )
        )
    return passages


def _merge_cue_fragments_into_sentences(cues: list[dict]) -> list[dict]:
    """Rejoin ``build_transcript_cues``' fragments into whole sentences.

    Cues split on an 8-second gap or a 160-character cap whether or not that
    lands mid-sentence — a reasonable subtitle line either way, but a
    misleading unit to judge topic similarity from: a lone trailing word
    like ``"Nam."`` embeds nothing like the sentence it was cut from. A
    fragment that doesn't end in sentence punctuation is joined onto the
    next one until one does, or the cues run out.
    """
    sentences: list[dict] = []
    buffer: list[dict] = []
    for cue in cues:
        buffer.append(cue)
        if cue["text"].rstrip().endswith((".", "!", "?", "…")):
            sentences.append({
                "start": buffer[0]["start"],
                "end": buffer[-1]["end"],
                "text": _clean_text(" ".join(c["text"] for c in buffer)),
            })
            buffer = []
    if buffer:
        sentences.append({
            "start": buffer[0]["start"],
            "end": buffer[-1]["end"],
            "text": _clean_text(" ".join(c["text"] for c in buffer)),
        })
    return sentences


def make_semantic_passages(
    video_id: str,
    rows: Sequence[dict],
    embed_cues: Callable[[list[str]], Sequence[Sequence[float]]],
    *,
    similarity_threshold: float = 0.5,
) -> list[Passage]:
    """Merge ``build_transcript_cues``' sentence-like spans into topic-coherent
    passages, greedily, on a similarity drop against a running centroid.

    No length bounds by design: a chunk ends purely where the topic does — a
    single-cue outlier or a fifteen-cue monologue are both legitimate output
    shapes. ``embed_cues`` is injected rather than hardcoded to
    ``load_embedding_model()`` so this stays unit-testable without Qwen and a
    GPU, matching this module's pattern of only importing heavy dependencies
    inside the functions that need them (``get_client``,
    ``load_embedding_model``).

    Running centroid, not just the immediately previous cue, for stability
    against one noisy sentence flipping the topic decision. The embedding
    actually written to the search index is computed later from each
    passage's *final merged text* (see ``build_audio_index.build``'s
    ``flush``), not derived from this centroid — the centroid here only
    decides where a boundary falls.
    """
    import numpy as np

    cues = _merge_cue_fragments_into_sentences(build_transcript_cues(rows))
    if not cues:
        return []

    vectors = np.asarray(embed_cues([cue["text"] for cue in cues]), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms  # defensive; embed_cues is expected to already
                                # pass normalize_embeddings=True

    passages: list[Passage] = []
    chunk_cues = [cues[0]]
    member_sum = vectors[0].copy()
    centroid = vectors[0].copy()

    def flush() -> None:
        start, end = chunk_cues[0]["start"], chunk_cues[-1]["end"]
        text = _clean_text(" ".join(cue["text"] for cue in chunk_cues))
        if text:
            passages.append(Passage(
                passage_id=f"{video_id}:{int(round(start * 1000))}",
                video_id=video_id,
                start_sec=start,
                end_sec=end,
                anchor_sec=(start + end) / 2,
                text=text,
            ))

    for index in range(1, len(cues)):
        similarity = float(vectors[index] @ centroid)
        if similarity >= similarity_threshold:
            chunk_cues.append(cues[index])
            member_sum = member_sum + vectors[index]
            norm = float(np.linalg.norm(member_sum))
            centroid = member_sum / norm if norm else member_sum
        else:
            flush()
            chunk_cues = [cues[index]]
            member_sum = vectors[index].copy()
            centroid = vectors[index].copy()
    flush()
    return passages


@dataclass(frozen=True)
class SemanticChunkConfig:
    """Hyperparameters for :func:`make_bounded_semantic_passages`.

    Phase 1 baseline: a 3-sentence context window either side of a candidate
    boundary, local-max breakpoints within the top ``breakpoint_percentile``
    of a video's own change-score distribution (distribution-relative, not an
    absolute similarity threshold — see ``_breakpoint_candidates``), 8/25/40s
    min/target/max duration, and 1-sentence overlap. Pause- and shot-aware
    boosting are staged for a later phase and are not read by the assembler
    yet, so no fields for them exist here.
    """

    context_window_sentences: int = 3
    breakpoint_percentile: float = 15.0
    #: Below this many boundary scores, a video's own percentile is too noisy
    #: to trust (e.g. one score from two sentences is either 0th or 100th
    #: percentile by construction). ``_breakpoint_candidates`` returns no
    #: candidates at all in that case, which makes the assembler fall back to
    #: pure duration-bounded, sentence-aware chunking for that video.
    min_scores_for_percentile: int = 4
    min_duration_seconds: float = 8.0
    target_duration_seconds: float = 25.0
    max_duration_seconds: float = 40.0
    overlap_sentences: int = 1
    #: Phase 2, experimental — off by default in production. A pause (silence
    #: gap) at a boundary that is long relative to this video's own pause
    #: distribution (>= its own ``pause_relative_percentile``) can *promote* a
    #: borderline semantic candidate (see ``pause_borderline_semantic_percentile``
    #: below) into the argmax pool in ``[target_duration, max_duration]`` — see
    #: ``_pause_boosted_indices`` and ``_assemble_bounded_chunks``. It never
    #: forces a split on its own, never applies in the growth zone below
    #: ``target_duration``, and — as of the gated design below — can never
    #: make an otherwise-weak boundary eligible by itself. A benchmark found
    #: this gated form clearly improves manual boundary quality (see
    #: docs/audio-search.md) but not retrieval by a large enough margin on a
    #: 44-query benchmark to justify enabling it by default; kept as a fully
    #: tested optional feature for future re-benchmarking.
    pause_boost_enabled: bool = False
    pause_relative_percentile: float = 75.0
    #: A pause-ablation benchmark (see docs/audio-search.md) found that an
    #: ungated pause boost lets pauses drive the large majority of
    #: boundaries -- ~86% at the 75th percentile -- and roughly half of the
    #: pause-only boundaries read as false topic transitions on manual
    #: review, concentrated in interview/commentary-style content. Pause is
    #: now only allowed to *promote* a boundary that already clears this
    #: wider, still-local-max ``breakpoint_candidates`` percentile -- it can
    #: never make an otherwise-weak boundary eligible on its own. Must stay
    #: looser (larger) than ``breakpoint_percentile`` for "borderline" to
    #: mean anything.
    pause_borderline_semantic_percentile: float = 30.0

    def to_json(self) -> dict:
        return {
            "context_window_sentences": self.context_window_sentences,
            "breakpoint_percentile": self.breakpoint_percentile,
            "min_scores_for_percentile": self.min_scores_for_percentile,
            "min_duration_seconds": self.min_duration_seconds,
            "target_duration_seconds": self.target_duration_seconds,
            "max_duration_seconds": self.max_duration_seconds,
            "overlap_sentences": self.overlap_sentences,
            "pause_boost_enabled": self.pause_boost_enabled,
            "pause_relative_percentile": self.pause_relative_percentile,
            "pause_borderline_semantic_percentile": self.pause_borderline_semantic_percentile,
        }

    @classmethod
    def from_json(cls, data: dict) -> "SemanticChunkConfig":
        defaults = cls()
        return cls(
            context_window_sentences=data.get(
                "context_window_sentences", defaults.context_window_sentences
            ),
            breakpoint_percentile=data.get(
                "breakpoint_percentile", defaults.breakpoint_percentile
            ),
            min_scores_for_percentile=data.get(
                "min_scores_for_percentile", defaults.min_scores_for_percentile
            ),
            min_duration_seconds=data.get(
                "min_duration_seconds", defaults.min_duration_seconds
            ),
            target_duration_seconds=data.get(
                "target_duration_seconds", defaults.target_duration_seconds
            ),
            max_duration_seconds=data.get(
                "max_duration_seconds", defaults.max_duration_seconds
            ),
            overlap_sentences=data.get("overlap_sentences", defaults.overlap_sentences),
            pause_boost_enabled=data.get(
                "pause_boost_enabled", defaults.pause_boost_enabled
            ),
            pause_relative_percentile=data.get(
                "pause_relative_percentile", defaults.pause_relative_percentile
            ),
            pause_borderline_semantic_percentile=data.get(
                "pause_borderline_semantic_percentile",
                defaults.pause_borderline_semantic_percentile,
            ),
        )


def _change_scores(vectors, window_sentences: int = 3) -> list[float]:
    """Windowed semantic-change score between each pair of adjacent sentences.

    ``scores[i]`` is the change between sentence ``i`` and ``i+1``, computed
    from the mean of up to ``window_sentences`` normalized vectors on each
    side rather than the single adjacent sentence — a lone noisy sentence
    embedding is smoothed out by its neighbours. The window shrinks (down to
    one sentence) near either end of the video rather than padding with
    nothing.
    """
    import numpy as np

    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)
    if n < 2:
        return []
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = vectors / norms
    scores = []
    for i in range(n - 1):
        lo = max(0, i - window_sentences + 1)
        hi = min(n, i + 1 + window_sentences)
        left = normalized[lo:i + 1].mean(axis=0)
        right = normalized[i + 1:hi].mean(axis=0)
        left_norm = float(np.linalg.norm(left)) or 1.0
        right_norm = float(np.linalg.norm(right)) or 1.0
        similarity = float((left / left_norm) @ (right / right_norm))
        scores.append(1.0 - similarity)
    return scores


def _breakpoint_candidates(
    scores: Sequence[float], *, percentile: float = 15.0, min_scores: int = 4
) -> set[int]:
    """Candidate boundary indices: a local maximum within the top ``percentile``
    of this video's own change-score distribution.

    Distribution-relative rather than an absolute cutoff, since how similar
    consecutive sentences are varies by video and topic density. Below
    ``min_scores`` data points a percentile is not a meaningful statistic (one
    score is trivially its own 0th/100th percentile), so this returns no
    candidates at all — the assembler then falls back to duration-bounded,
    sentence-aware chunking with no semantic signal for that video.
    """
    if len(scores) < min_scores:
        return set()
    import numpy as np

    threshold = np.percentile(scores, 100 - percentile)
    candidates = set()
    for i, score in enumerate(scores):
        if score < threshold:
            continue
        left_ok = i == 0 or score >= scores[i - 1]
        right_ok = i == len(scores) - 1 or score >= scores[i + 1]
        if left_ok and right_ok:
            candidates.add(i)
    return candidates


def _pause_lengths(sentences: list[dict]) -> list[float]:
    """Silence gap after each sentence, aligned with ``_change_scores``' and
    ``_breakpoint_candidates``' boundary indexing: ``pauses[i]`` is the gap
    between sentence ``i`` and sentence ``i + 1``.

    Built from the same cue-end heuristic ``build_transcript_cues`` already
    uses (``min(last_token_offset + 2, next_cue_start)``) since the source
    data carries no true end-of-speech timestamp — this is an approximation,
    not a measured silence.
    """
    return [
        sentences[i + 1]["start"] - sentences[i]["end"]
        for i in range(len(sentences) - 1)
    ]


def _pause_boosted_indices(
    pauses: Sequence[float], *, percentile: float = 75.0, min_scores: int = 4
) -> set[int]:
    """Boundary indices whose pause is long relative to *this video's own*
    pause distribution (>= its own ``percentile``).

    Distribution-relative for the same reason as ``_breakpoint_candidates``:
    typical pause length varies by narration style, so an absolute threshold
    (e.g. "> 1.5s") would not travel across videos. Below ``min_scores``
    pauses the percentile is not meaningful, mirroring
    ``_breakpoint_candidates``'s own guard — returns no boosted indices
    rather than a noisy one computed from too few data points. This is a
    *soft* signal: see ``_assemble_bounded_chunks`` for how it only ever
    widens the argmax pool in ``[target_duration, max_duration]``, never
    forces a split and never applies in the growth zone below
    ``target_duration``.
    """
    if len(pauses) < min_scores:
        return set()
    import numpy as np

    threshold = np.percentile(pauses, percentile)
    return {i for i, pause in enumerate(pauses) if pause >= threshold}


def _assemble_bounded_chunks(
    sentences: list[dict], scores: Sequence[float], candidates: set[int], config: "SemanticChunkConfig",
    *, pause_boosted: set[int] | None = None, borderline_candidates: set[int] | None = None,
    debug: list[dict] | None = None,
) -> list[list[dict]]:
    """Group sentences into duration-bounded, semantically-cut chunks.

    ``debug``, if given a list, gets one dict appended per emitted chunk
    (before the trailing-tail merge, which then folds its entry into the
    chunk it merges into): ``{"reason": "semantic" | "forced" | "end_of_video",
    "boundary_score": float | None, "duration": float}``. This is purely for
    benchmark/analysis instrumentation — production callers never pass it, so
    it costs nothing there, and it does not appear on :class:`Passage` or in
    the Elasticsearch index.

    ``[min_duration, target_duration)`` is a pure growth zone: no candidate,
    however strong, ends a chunk there — otherwise a strong-but-early valley
    would produce undersized chunks and defeat the point of having a target
    duration. Only once a chunk's duration reaches ``target_duration`` does
    the assembler start considering candidates, and it takes the single
    strongest one (``argmax`` of the change score) anywhere up to
    ``max_duration`` — looking further ahead costs nothing since the chunk
    would keep growing anyway, so there is no distance-from-target tradeoff
    to weigh and no extra hyperparameter for it.

    If no candidate falls in ``[target_duration, max_duration]``, the chunk is
    force-split at the last sentence boundary that does not exceed
    ``max_duration`` — except for a single sentence whose own duration already
    exceeds ``max_duration``, which is emitted alone rather than split
    mid-sentence or left to swallow the rest of the video.

    ``pause_boosted`` (from ``_pause_boosted_indices``, gated by
    ``config.pause_boost_enabled``), if given, can widen which boundaries are
    eligible for that same argmax pool. It never adds a new way to *force* a
    split (``force_idx`` tracking is unaffected) and never applies below
    ``target_duration`` — same growth-zone rule as the semantic candidates it
    augments. What it takes to actually become eligible depends on
    ``borderline_candidates``:

    - ``borderline_candidates is None`` (the raw, ungated signal — kept only
      for the Phase 2 ablation benchmark, never called this way from
      :func:`make_bounded_semantic_passages`): any pause-boosted index is
      eligible on its own, even with no semantic support at all. A benchmark
      found this lets pause dominate the large majority of boundaries and
      manually reads as a false topic transition roughly half the time on
      interview/commentary-style content.
    - ``borderline_candidates`` given (from a wider, still-local-max
      ``_breakpoint_candidates`` call, e.g. ``pause_borderline_semantic_percentile``):
      a pause-boosted index is eligible *only if it is also a borderline
      semantic candidate* — pause promotes a boundary the semantic signal was
      already unsure about, it cannot manufacture one on its own. A boundary
      already in ``candidates`` (strong semantic) is always eligible
      regardless of pause or borderline status.
    """
    chunks: list[list[dict]] = []
    start = 0
    n = len(sentences)
    while start < n:
        force_idx = start  # always at least one sentence, even if it alone exceeds max
        window: list[int] = []
        hit_max = False
        for j in range(start, n):
            cum = sentences[j]["end"] - sentences[start]["start"]
            if j > start and cum > config.max_duration_seconds:
                hit_max = True
                break
            force_idx = j
            strong = j in candidates
            pause_hit = pause_boosted is not None and j in pause_boosted
            if pause_hit and borderline_candidates is not None:
                pause_hit = j in borderline_candidates
            eligible = strong or pause_hit
            if (
                cum >= config.target_duration_seconds
                and j < len(scores)
                and eligible
            ):
                window.append(j)
        if window:
            end = max(window, key=lambda idx: scores[idx])
            reason = "semantic"
        elif hit_max:
            end = force_idx
            reason = "forced"
        else:
            end = force_idx
            reason = "end_of_video"
        chunks.append(sentences[start:end + 1])
        if debug is not None:
            in_candidates = end in candidates
            in_pause_boosted = bool(pause_boosted and end in pause_boosted)
            in_borderline = bool(borderline_candidates is not None and end in borderline_candidates)
            debug.append({
                "reason": reason,
                "boundary_score": scores[end] if end < len(scores) else None,
                "duration": sentences[end]["end"] - sentences[start]["start"],
                "pause_boosted": bool(in_pause_boosted and not in_candidates),
                "semantic_candidate": in_candidates,
                "pause_eligible": in_pause_boosted,
                "borderline_semantic": in_borderline,
            })
        if end >= n - 1:
            break  # reached the last sentence; an overlap carry-over would
                    # otherwise re-emit it alone as a degenerate trailing chunk
        next_start = end + 1 - config.overlap_sentences
        start = max(start + 1, next_start)
    if len(chunks) > 1:
        tail = chunks[-1]
        if tail[-1]["end"] - tail[0]["start"] < config.min_duration_seconds:
            chunks[-2] = chunks[-2] + tail
            chunks.pop()
            if debug is not None:
                merged = debug.pop()
                debug[-1] = {
                    **debug[-1],
                    "reason": f"{debug[-1]['reason']}+merged_tail({merged['reason']})",
                    "duration": chunks[-1][-1]["end"] - chunks[-1][0]["start"],
                }
    return chunks


def _sentences_to_passage(video_id: str, sentences: list[dict]) -> "Passage":
    start = sentences[0]["start"]
    end = sentences[-1]["end"]
    text = _clean_text(" ".join(sentence["text"] for sentence in sentences))
    return Passage(
        passage_id=f"{video_id}:{int(round(start * 1000))}",
        video_id=video_id,
        start_sec=start,
        end_sec=end,
        anchor_sec=(start + end) / 2,
        text=text,
    )


def make_bounded_semantic_passages(
    video_id: str,
    rows: Sequence[dict],
    embed_cues: Callable[[list[str]], Sequence[Sequence[float]]],
    *,
    config: SemanticChunkConfig | None = None,
    debug: list[dict] | None = None,
) -> list[Passage]:
    """Duration-bounded semantic chunking: sentence-aware, context-window
    boundary detection with hard min/target/max duration constraints and
    sentence overlap.

    Builds on the same sentence-preparation and ``embed_cues``-injection seam
    as :func:`make_semantic_passages` (rejoin cue fragments into sentences,
    embed each one, no model hardcoded here), but replaces that function's
    unbounded running-centroid/absolute-threshold merge with windowed
    change-score boundary detection (:func:`_change_scores`), distribution-
    relative candidate selection (:func:`_breakpoint_candidates`), and the
    duration-constrained assembler (:func:`_assemble_bounded_chunks`).
    :func:`make_semantic_passages` is left untouched — it remains useful as
    the "pure semantic, no duration bounds" baseline to benchmark against.

    ``config.pause_boost_enabled`` (off by default) additionally lets a long
    pause between sentences *promote* a borderline semantic candidate (one
    that clears ``config.pause_borderline_semantic_percentile`` but not the
    stricter ``config.breakpoint_percentile``) into the eligible pool. A
    pause can never make an otherwise-weak boundary eligible on its own —
    see :func:`_pause_boosted_indices` and
    :func:`_assemble_bounded_chunks`'s docstring for why (an earlier,
    ungated version of this let pauses dominate boundary choice and produced
    false topic transitions on manual review).

    ``debug``, if given a list, is passed straight to
    :func:`_assemble_bounded_chunks` — see its docstring for the shape. It is
    benchmark instrumentation, not part of this function's production return
    value.
    """
    config = config or SemanticChunkConfig()
    sentences = _merge_cue_fragments_into_sentences(build_transcript_cues(rows))
    if not sentences:
        return []
    if len(sentences) == 1:
        if debug is not None:
            debug.append({
                "reason": "end_of_video",
                "boundary_score": None,
                "duration": sentences[0]["end"] - sentences[0]["start"],
            })
        return [_sentences_to_passage(video_id, sentences)]

    vectors = embed_cues([sentence["text"] for sentence in sentences])
    scores = _change_scores(vectors, config.context_window_sentences)
    candidates = _breakpoint_candidates(
        scores,
        percentile=config.breakpoint_percentile,
        min_scores=config.min_scores_for_percentile,
    )
    pause_boosted = None
    borderline_candidates = None
    if config.pause_boost_enabled:
        pauses = _pause_lengths(sentences)
        pause_boosted = _pause_boosted_indices(
            pauses,
            percentile=config.pause_relative_percentile,
            min_scores=config.min_scores_for_percentile,
        )
        borderline_candidates = _breakpoint_candidates(
            scores,
            percentile=config.pause_borderline_semantic_percentile,
            min_scores=config.min_scores_for_percentile,
        )
    groups = _assemble_bounded_chunks(
        sentences, scores, candidates, config, pause_boosted=pause_boosted,
        borderline_candidates=borderline_candidates, debug=debug
    )
    return [_sentences_to_passage(video_id, group) for group in groups]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[tuple[Hashable, float]]], rank_constant: int = 60
) -> list[tuple[Hashable, float]]:
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking in rankings:
        for rank, (key, _) in enumerate(ranking, 1):
            first_seen.setdefault(key, order)
            order += 1
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank_constant + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], first_seen[item[0]]))


def merge_temporal_hits(hits: Iterable[AudioHit]) -> list[AudioHit]:
    clusters: list[AudioHit] = []
    for hit in sorted(hits, key=lambda h: (h.video_id, h.start_sec, -h.score)):
        if clusters and clusters[-1].video_id == hit.video_id and hit.start_sec < clusters[-1].end_sec:
            current = clusters[-1]
            best = hit if hit.score > current.score else current
            clusters[-1] = replace(
                best,
                start_sec=min(current.start_sec, hit.start_sec),
                end_sec=max(current.end_sec, hit.end_sec),
            )
        else:
            clusters.append(hit)
    return sorted(clusters, key=lambda h: -h.score)


def parse_search_response(response: dict) -> list[AudioHit]:
    hits = []
    for raw in response.get("hits", {}).get("hits", []):
        source = raw.get("_source", {})
        required = {"video_id", "start_sec", "end_sec", "anchor_sec", "text"}
        if not required.issubset(source):
            continue
        highlighted = " ".join(raw.get("highlight", {}).get("text", []))
        matched = tuple(dict.fromkeys(
            part.strip() for part in highlighted.split("<aic-hit>")[1:]
            for part in [part.split("</aic-hit>", 1)[0]] if part.strip()
        ))
        hits.append(
            AudioHit(
                video_id=source["video_id"],
                start_sec=float(source["start_sec"]),
                end_sec=float(source["end_sec"]),
                anchor_sec=float(source["anchor_sec"]),
                text=source["text"],
                score=float(raw.get("_score") or 0.0),
                passage_id=source.get("passage_id") or raw.get("_id"),
                matched_keywords=matched,
            )
        )
    return hits


def get_client():
    global _client
    if _client is None:
        from elasticsearch import Elasticsearch

        _client = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200"))
    return _client


def load_embedding_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        device = os.environ.get("AUDIO_EMBED_DEVICE", "cuda")
        import torch
        _model = SentenceTransformer(
            MODEL_NAME, device=device, model_kwargs={"torch_dtype": torch.float16},
        )
    return _model


def encode_query(text: str) -> list[float]:
    vector = load_embedding_model().encode(
        text,
        prompt=f"Instruct: {QUERY_INSTRUCTION}\nQuery:",
        normalize_embeddings=True,
    )
    return vector.astype("float32").tolist()


@lru_cache(maxsize=1)
def load_audio_flat():
    index_path = AUDIO_FLAT_STEM.with_suffix(".flat.ip")
    metadata_path = AUDIO_FLAT_STEM.with_suffix(".json")
    if not index_path.exists() or not metadata_path.exists():
        return None
    import faiss

    index = faiss.read_index(str(index_path))
    metadata = json.loads(metadata_path.read_text())
    if index.d != EMBEDDING_DIM or index.ntotal != len(metadata):
        raise RuntimeError("audio FlatIP index and metadata do not match")
    return index, metadata


def search_audio_flat(query_vector: Sequence[float], k: int) -> list[AudioHit] | None:
    loaded = load_audio_flat()
    if loaded is None:
        return None
    import numpy as np

    index, metadata = loaded
    count = min(max(100, k), index.ntotal)
    scores, ids = index.search(np.asarray([query_vector], dtype="float32"), count)
    hits = []
    for rank, (score, position) in enumerate(zip(scores[0], ids[0]), 1):
        if position < 0:
            continue
        source = metadata[int(position)]
        hits.append(AudioHit(
            video_id=source["video_id"], start_sec=float(source["start_sec"]),
            end_sec=float(source["end_sec"]), anchor_sec=float(source["anchor_sec"]),
            text=source["text"], score=float(score), vector_rank=rank,
            passage_id=source.get("passage_id"),
        ))
    return merge_temporal_hits(hits)[:k]


def _ranked_hits(response: dict, *, rank_field: str) -> list[AudioHit]:
    ranked = []
    for rank, hit in enumerate(parse_search_response(response), 1):
        ranked.append(replace(hit, **{rank_field: rank}))
    return merge_temporal_hits(ranked)


def search_audio_keyword(
    text: str, k: int = 100, *, client=None, index: str = AUDIO_INDEX,
) -> list[AudioHit]:
    """BM25-only transcript search.  This intentionally never loads Qwen."""
    client = client or get_client()
    return _ranked_hits(client.search(index=index, body=build_bm25_request(text, k)),
                        rank_field="bm25_rank")[:k]


def search_audio_semantic(
    text: str, k: int = 100, *, client=None, embed=None, index: str = AUDIO_INDEX,
) -> list[AudioHit]:
    """Dense-only transcript search.  This intentionally never runs BM25."""
    query_vector = (embed or encode_query)(text)
    flat_hits = search_audio_flat(query_vector, k)
    if flat_hits is not None:
        return flat_hits
    client = client or get_client()
    return _ranked_hits(client.search(index=index, body=build_knn_request(query_vector, k)),
                        rank_field="vector_rank")[:k]


def search_audio(
    text: str,
    k: int = 100,
    *,
    client=None,
    embed=None,
    index: str = AUDIO_INDEX,
) -> list[AudioHit]:
    """Legacy RRF entry point retained for callers outside the search API."""
    client = client or get_client()
    embed = embed or encode_query
    query_vector = embed(text)
    responses = [
        client.search(index=index, body=build_bm25_request(text, k)),
        client.search(index=index, body=build_knn_request(query_vector, k)),
    ]
    rankings = []
    by_id = {}
    rank_maps = []
    for response in responses:
        ranking = []
        ranks = {}
        for rank, hit in enumerate(parse_search_response(response), 1):
            passage_id = hit.passage_id or f"{hit.video_id}:{int(hit.start_sec * 1000)}"
            ranking.append((passage_id, hit.score))
            ranks[passage_id] = rank
            by_id.setdefault(passage_id, hit)
        rankings.append(ranking)
        rank_maps.append(ranks)
    fused = []
    for passage_id, score in reciprocal_rank_fusion(rankings):
        hit = by_id[passage_id]
        fused.append(
            replace(
                hit,
                score=score,
                bm25_rank=rank_maps[0].get(passage_id),
                vector_rank=rank_maps[1].get(passage_id),
                passage_id=passage_id,
            )
        )
    return merge_temporal_hits(fused)[:k]
