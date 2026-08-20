"""Transcript passage construction and Elasticsearch-backed retrieval."""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Hashable, Iterable, Sequence


AUDIO_INDEX = "aic_audio_v1"
EMBEDDING_DIM = 1024
QUERY_INSTRUCTION = (
    "Given a Vietnamese or English description, retrieve the video transcript "
    "passage that best matches the described spoken content."
)
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

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


def _clean_text(text: str) -> str:
    return " ".join(text.replace("<", " ").replace(">", " ").split())


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
        "query": {
            "multi_match": {
                "query": text,
                "fields": ["text^2", "text.folded"],
                "type": "best_fields",
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
            "num_candidates": max(200, candidate_count * 2),
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
        hits.append(
            AudioHit(
                video_id=source["video_id"],
                start_sec=float(source["start_sec"]),
                end_sec=float(source["end_sec"]),
                anchor_sec=float(source["anchor_sec"]),
                text=source["text"],
                score=float(raw.get("_score") or 0.0),
                passage_id=source.get("passage_id") or raw.get("_id"),
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
        _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def encode_query(text: str) -> list[float]:
    vector = load_embedding_model().encode(
        text,
        prompt=f"Instruct: {QUERY_INSTRUCTION}\nQuery:",
        normalize_embeddings=True,
    )
    return vector.astype("float32").tolist()


def search_audio(
    text: str,
    k: int = 100,
    *,
    client=None,
    embed=None,
    index: str = AUDIO_INDEX,
) -> list[AudioHit]:
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
