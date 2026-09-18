"""Pure fusion of several ranked query hit lists into one ranked list of
canonical groups (typically shots).

- `fuse()`: holistic query + aspect queries (docs/reports/visual-query-expansion-benchmark-2026.md).
- `weighted_shot_rrf()` / `weighted_shot_score()`: paraphrases of one caption
  (docs/reports/visual-holistic-paraphrase-benchmark-2026.md).
- `combine_vectors()`: early fusion of query embeddings.

No I/O. Callers build `Hit` lists (and an `Adjacency` map for `fuse()`) from FAISS
search results and shot boundaries.
"""
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable

import numpy as np


@dataclass(frozen=True)
class Hit:
    """One frame hit for one query, already ranked (list order = rank)."""
    group_id: Hashable
    payload: Any = None
    time: float = 0.0
    score: float | None = None


@dataclass
class Adjacency:
    """group_id -> adjacent group_id (or None) in the same video, for neighbor evidence."""
    prev: dict = field(default_factory=dict)
    next: dict = field(default_factory=dict)


@dataclass
class FusedGroup:
    group_id: Hashable
    score: float
    rep_hit: Hit
    holistic_rank: int | None
    aspect_ranks: list  # list[int | None], one per aspect, in input order
    support: float
    used_neighbor: dict  # {"holistic": bool, 0: bool, 1: bool, ...}


def dedup_hits(hits, window_seconds, video_of):
    """Drop a hit if a kept hit in the same video already fell within window_seconds.

    Mirrors backend/temporal_search.py's deduplicate_candidates, generalized to
    operate on group_id-derived video identity instead of a literal video_id field.
    """
    if window_seconds <= 0:
        return list(hits)
    kept = []
    kept_times_by_video: dict = {}
    for hit in hits:
        video = video_of(hit.group_id)
        times = kept_times_by_video.setdefault(video, [])
        if any(abs(hit.time - t) <= window_seconds for t in times):
            continue
        times.append(hit.time)
        kept.append(hit)
    return kept


def shot_ranks(hits):
    """Collapse a query's ranked hit list to {group_id: (rank, hit)} by first appearance."""
    ranks = {}
    rank = 0
    for hit in hits:
        if hit.group_id in ranks:
            continue
        rank += 1
        ranks[hit.group_id] = (rank, hit)
    return ranks


def _direct(rank_map, group_id, k):
    entry = rank_map.get(group_id)
    return 1.0 / (k + entry[0]) if entry else 0.0


def _evidence(rank_map, group_id, k, gamma, adjacency):
    """e_q(g): direct evidence, or gamma-discounted best neighbor evidence, never both."""
    direct = _direct(rank_map, group_id, k)
    if gamma <= 0:
        return direct
    neighbor = 0.0
    prev_id = adjacency.prev.get(group_id)
    next_id = adjacency.next.get(group_id)
    if prev_id is not None:
        neighbor = max(neighbor, _direct(rank_map, prev_id, k))
    if next_id is not None:
        neighbor = max(neighbor, _direct(rank_map, next_id, k))
    return max(direct, gamma * neighbor)


def _aggregate(values, agg, m):
    if not values:
        return 0.0
    n = len(values)
    if agg == "mean":
        return sum(values) / n
    if agg == "top_m_mean":
        top = sorted(values, reverse=True)[:min(m, n)]
        return sum(top) / len(top)
    raise ValueError(f"unknown agg: {agg!r}")


def _best_direct_hit(rank_maps, group_id):
    """Best (lowest-rank) direct hit for group_id across a list of rank maps, or None."""
    best = None
    for rank_map in rank_maps:
        entry = rank_map.get(group_id)
        if entry and (best is None or entry[0] < best[0]):
            best = entry
    return best[1] if best else None


def fuse(holistic_hits, aspect_hits_list, adjacency: Adjacency, *, lam: float, k: float = 60,
        agg: str = "mean", m: int = 2, gamma: float = 0.0, neighbor_scope: str = "aspects",
        T_h: float = float("inf"), beta: float = 1.0, dedup_seconds: float = 0.0,
        video_of: Callable[[Hashable], Any] = lambda g: g[0]):
    """Fuse one holistic + N aspect ranked hit lists into a scored group list.

    Implements the scoring definition exactly:
      final(g) = lam * h(g) + (1-lam) * support(g) * a(g)
    h(g) = holistic evidence (direct, plus gamma-discounted neighbor evidence only when
    neighbor_scope == "all"). a(g) = agg of aspect evidence (direct, plus gamma-discounted
    neighbor evidence). support(g) = 1 if the *direct* holistic rank is <= T_h, else beta.
    Candidates are groups with >= 1 direct hit from any query; neighbor-only groups never
    become candidates (they would have no representative frame).
    """
    if not (0.0 <= gamma <= 1.0):
        raise ValueError("gamma must be in [0, 1]")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1]")
    if neighbor_scope not in ("aspects", "all"):
        raise ValueError("neighbor_scope must be 'aspects' or 'all'")

    holistic_hits = dedup_hits(holistic_hits, dedup_seconds, video_of)
    aspect_hits_list = [dedup_hits(hits, dedup_seconds, video_of) for hits in aspect_hits_list]

    h_ranks = shot_ranks(holistic_hits)
    a_ranks_list = [shot_ranks(hits) for hits in aspect_hits_list]

    candidates = set(h_ranks)
    for a_ranks in a_ranks_list:
        candidates |= set(a_ranks)

    h_gamma = gamma if neighbor_scope == "all" else 0.0

    results = []
    for group_id in candidates:
        h_direct = _direct(h_ranks, group_id, k)
        h_val = _evidence(h_ranks, group_id, k, h_gamma, adjacency)
        rank_h_direct = h_ranks[group_id][0] if group_id in h_ranks else None
        used_neighbor = {"holistic": h_val > h_direct + 1e-15}

        aspect_evidences = []
        aspect_ranks_out = []
        for i, a_ranks in enumerate(a_ranks_list):
            a_direct = _direct(a_ranks, group_id, k)
            a_val = _evidence(a_ranks, group_id, k, gamma, adjacency)
            aspect_evidences.append(a_val)
            aspect_ranks_out.append(a_ranks[group_id][0] if group_id in a_ranks else None)
            used_neighbor[i] = a_val > a_direct + 1e-15
        a_agg = _aggregate(aspect_evidences, agg, m)

        support = 1.0 if (rank_h_direct is not None and rank_h_direct <= T_h) else beta
        final = lam * h_val + (1 - lam) * support * a_agg
        if final <= 0:
            continue

        rep_hit = h_ranks[group_id][1] if group_id in h_ranks else \
            _best_direct_hit(a_ranks_list, group_id)

        results.append(FusedGroup(
            group_id=group_id, score=final, rep_hit=rep_hit, holistic_rank=rank_h_direct,
            aspect_ranks=aspect_ranks_out, support=support, used_neighbor=used_neighbor,
        ))

    results.sort(key=lambda g: (
        -g.score,
        g.holistic_rank is None,
        g.holistic_rank if g.holistic_rank is not None else float("inf"),
        min((r for r in g.aspect_ranks if r is not None), default=float("inf")),
        str(g.group_id),
    ))
    return results


# ---------------------------------------------------------------------------
# Paraphrase fusion: several wordings of the same caption, weighted.
# ---------------------------------------------------------------------------

@dataclass
class WeightedGroup:
    group_id: Hashable
    score: float
    rep_hit: Hit
    ranks: list  # list[int | None], shot-level rank per input list, in input order


def _check_weights(hit_lists, weights):
    if len(hit_lists) != len(weights):
        raise ValueError("hit_lists and weights must have the same length")
    if any(w < 0 or not math.isfinite(w) for w in weights):
        raise ValueError("weights must be finite and non-negative")


def _weighted_fuse(hit_lists, weights, contribution, *, dedup_seconds, video_of):
    """Shared core. contribution(i, rank_map, group_id) -> this list's term before weighting.

    Only positive-weight lists propose candidates, so a zero-weight list changes nothing.
    Sums use math.fsum (correctly rounded) so reordering the lists cannot flip exact ties.
    """
    _check_weights(hit_lists, weights)
    rank_maps = [shot_ranks(dedup_hits(hits, dedup_seconds, video_of)) for hits in hit_lists]
    active = [i for i, w in enumerate(weights) if w > 0]

    candidates = set()
    for i in active:
        candidates |= set(rank_maps[i])

    results = []
    for group_id in candidates:
        ranks = [rank_maps[i][group_id][0] if group_id in rank_maps[i] else None
                 for i in range(len(hit_lists))]
        score = math.fsum(weights[i] * contribution(i, rank_maps[i], group_id) for i in active)
        present = [i for i in active if ranks[i] is not None]
        # Representative: best shot-level rank, then heavier list, then earlier list.
        best = min(present, key=lambda i: (ranks[i], -weights[i], i))
        results.append(WeightedGroup(group_id=group_id, score=score,
                                     rep_hit=rank_maps[best][group_id][1], ranks=ranks))

    def sort_key(group):
        present = [i for i in active if group.ranks[i] is not None]
        return (
            -group.score,
            -max(weights[i] for i in present),
            min(group.ranks[i] for i in present),
            str(group.group_id),
        )

    results.sort(key=sort_key)
    return results


def weighted_shot_rrf(hit_lists, weights, *, k=60, dedup_seconds=0.0,
                      video_of: Callable[[Hashable], Any] = lambda g: g[0]):
    """score(g) = sum_i w_i / (k + r_i(g)), where r_i(g) is g's shot-level rank in list i
    and an absent group contributes 0. Exact ties go to the group retrieved by the heaviest
    list, then the best rank, then group id."""
    if k <= 0:
        raise ValueError("k must be positive")

    def contribution(i, rank_map, group_id):
        entry = rank_map.get(group_id)
        return 1.0 / (k + entry[0]) if entry else 0.0

    return _weighted_fuse(hit_lists, weights, contribution, dedup_seconds=dedup_seconds,
                          video_of=video_of)


def weighted_shot_score(hit_lists, weights, *, dedup_seconds=0.0,
                        video_of: Callable[[Hashable], Any] = lambda g: g[0]):
    """score(g) = sum_i w_i * s_i(g), where s_i(g) is the score of g's first-appearance hit
    in list i (its maximum, since lists are score-sorted). If g is absent from list i, s_i(g)
    is that list's lowest retrieved score (the K-th score), so absence never beats presence."""
    fills = []
    for hits in hit_lists:
        if any(hit.score is None for hit in hits):
            raise ValueError("weighted_shot_score needs Hit.score on every hit")
        fills.append(min((hit.score for hit in hits), default=0.0))

    def contribution(i, rank_map, group_id):
        entry = rank_map.get(group_id)
        return entry[1].score if entry else fills[i]

    return _weighted_fuse(hit_lists, weights, contribution, dedup_seconds=dedup_seconds,
                          video_of=video_of)


def combine_vectors(vectors, weights):
    """Early fusion: normalize(sum_i w_i * normalize(v_i)), returned as float32."""
    matrix = np.asarray(vectors, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or len(w) != len(matrix):
        raise ValueError("vectors must be (n, d) with one weight per vector")
    if np.any(w < 0) or not np.all(np.isfinite(w)) or w.sum() <= 0:
        raise ValueError("weights must be non-negative with a positive sum")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero vector cannot be normalized")
    mixed = (w[:, None] * (matrix / norms[:, None])).sum(axis=0)
    mixed_norm = np.linalg.norm(mixed)
    if mixed_norm == 0:
        raise ValueError("weighted vectors cancel out")
    return (mixed / mixed_norm).astype(np.float32)
