"""Shot-caption index definition and Elasticsearch-backed retrieval.

Every 2026 shot is captioned offline by CapRL-Video-4B (see
``docs/caprl-shot-captioning.md``): one long, chronological English caption per
shot window of at most 20 s, transcribing whatever text is legible in the filmed
scene. That is a different kind of text from the two Elasticsearch channels that
already exist, and it reaches what neither can:

- a transcript passage says what was *said*, a caption says what was *shown*;
- OCR reads short burned-in overlays from one keyframe, a caption reads the
  slide, the exam paper and the packaging, and counts what it sees;
- a CLIP embedding compares a 72-token prompt against one keyframe, so it cannot
  express "the third truck", "after the woman seals the box", or a full sentence
  of a lecture slide -- the failure types left in
  ``docs/reports/visual-search-recall-failure-analysis-2026.md``.

BM25 only, like ``ocr_search``: the index ships as soon as the captions are
written, with no embedding pass over 103k long documents. The mapping and
``build_knn_request`` are already here so ``--dense`` can be turned on later
without reindexing decisions being reopened; ``search_captions_semantic`` uses
the same Qwen3 encoder as the transcript channel, so a dense caption index must
be built with ``scripts.indexing.build_caption_index --dense``, which embeds
through that same model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from typing import Sequence

CAPTION_INDEX = os.environ.get("AIC_CAPTION_INDEX", "aic_captions_2026_v1")
#: Qwen3-Embedding, as used by the transcript channel (``audio_search``).
EMBEDDING_DIM = 2560
QUERY_INSTRUCTION = (
    "Given a Vietnamese or English description of a video moment, retrieve the "
    "caption that describes what is shown in that shot."
)

_client = None


@dataclass(frozen=True)
class CaptionHit:
    video_id: str
    shot_idx: int
    window_idx: int
    frame_idx: int
    start_sec: float
    end_sec: float
    anchor_sec: float
    text: str
    score: float
    rank: int | None = None
    matched_keywords: tuple[str, ...] = ()


def build_index_definition(dense: bool = False) -> dict:
    """The index mapping; `dense` adds the (optional) caption embedding field.

    Three analyzers over one caption: `text` (standard) for ordinary English
    words, `text.english` (stemmed) so "spinning" finds "spins", and
    `text.folded` (ASCII-folded) for the Vietnamese names and on-screen strings
    the model transcribes into an otherwise English caption, which a query may
    write with or without diacritics.
    """
    properties = {
        "unit_id": {"type": "keyword"},
        "video_id": {"type": "keyword"},
        "group": {"type": "keyword"},
        "year": {"type": "keyword"},
        "shot_idx": {"type": "integer"},
        "window_idx": {"type": "integer"},
        "frame_idx": {"type": "integer"},
        "start_sec": {"type": "float"},
        "end_sec": {"type": "float"},
        "anchor_sec": {"type": "float"},
        "text": {
            "type": "text",
            "fields": {
                "english": {"type": "text", "analyzer": "english"},
                "folded": {"type": "text", "analyzer": "vi_folded"},
            },
        },
    }
    if dense:
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": EMBEDDING_DIM,
            "index": True,
            "similarity": "cosine",
        }
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
        "mappings": {"dynamic": "strict", "properties": properties},
    }


def build_bm25_request(text: str, k: int = 100) -> dict:
    """Ranked caption search.

    A caption is a paragraph, so a phrase match ("wooden pillars", "bánh khọt
    mould") is far stronger evidence than the same words scattered through it,
    and gets the heavier boost. The stemmed field carries ordinary description
    wording; the folded field carries names and transcribed on-screen text.
    """
    candidate_count = max(100, k)
    return {
        "size": candidate_count,
        "_source": {"excludes": ["embedding"]},
        "highlight": {
            "pre_tags": ["<aic-hit>"], "post_tags": ["</aic-hit>"],
            "fields": {"text": {"number_of_fragments": 3, "fragment_size": 160}},
        },
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": text,
                            "fields": ["text^2", "text.english^2", "text.folded"],
                            "type": "best_fields",
                        }
                    },
                    {
                        "multi_match": {
                            "query": text,
                            "fields": ["text^4", "text.english^3", "text.folded^2"],
                            "type": "phrase",
                            "slop": 4,
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


def parse_search_response(response: dict) -> list[CaptionHit]:
    hits = []
    for raw in response.get("hits", {}).get("hits", []):
        source = raw.get("_source", {})
        required = {"video_id", "shot_idx", "frame_idx", "start_sec", "end_sec", "anchor_sec", "text"}
        if not required.issubset(source):
            continue
        highlighted = " ".join(raw.get("highlight", {}).get("text", []))
        matched = tuple(dict.fromkeys(
            piece.strip() for piece in highlighted.split("<aic-hit>")[1:]
            for piece in [piece.split("</aic-hit>", 1)[0]] if piece.strip()
        ))
        hits.append(CaptionHit(
            video_id=source["video_id"],
            shot_idx=int(source["shot_idx"]),
            window_idx=int(source.get("window_idx", 0)),
            frame_idx=int(source["frame_idx"]),
            start_sec=float(source["start_sec"]),
            end_sec=float(source["end_sec"]),
            anchor_sec=float(source["anchor_sec"]),
            text=source["text"],
            score=float(raw.get("_score") or 0.0),
            matched_keywords=matched,
        ))
    return hits


def reciprocal_rank_fusion(rankings: Sequence[Sequence[CaptionHit]], k: int = 60) -> list[CaptionHit]:
    """Merge keyword and semantic rankings; the score becomes the RRF score."""
    scores: dict[tuple[str, int, int], float] = {}
    best: dict[tuple[str, int, int], CaptionHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, 1):
            key = (hit.video_id, hit.shot_idx, hit.window_idx)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best:
                best[key] = hit
    merged = [replace(best[key], score=score) for key, score in scores.items()]
    merged.sort(key=lambda hit: -hit.score)
    return [replace(hit, rank=rank) for rank, hit in enumerate(merged, 1)]


def get_client():
    from . import search_core

    if search_core.remote():
        return search_core.remote_elasticsearch()
    global _client
    if _client is None:
        from elasticsearch import Elasticsearch

        _client = Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://127.0.0.1:9200"))
    return _client


def search_captions_keyword(
    text: str, k: int = 100, *, client=None, index: str = CAPTION_INDEX,
) -> list[CaptionHit]:
    """BM25 caption search."""
    client = client or get_client()
    response = client.search(index=index, body=build_bm25_request(text, k))
    ranked = [replace(hit, rank=rank)
              for rank, hit in enumerate(parse_search_response(response), 1)]
    return ranked[:k]


def search_captions_semantic(
    text: str, k: int = 100, *, client=None, index: str = CAPTION_INDEX,
) -> list[CaptionHit]:
    """Dense caption search; only works against an index built with `--dense`."""
    from . import audio_search

    client = client or get_client()
    vector = audio_search.encode_query(f"{QUERY_INSTRUCTION}\n{text}")
    response = client.search(index=index, body=build_knn_request(vector, k))
    ranked = [replace(hit, rank=rank)
              for rank, hit in enumerate(parse_search_response(response), 1)]
    return ranked[:k]


def search_captions_hybrid(
    text: str, k: int = 100, *, client=None, index: str = CAPTION_INDEX,
) -> list[CaptionHit]:
    """Keyword and dense rankings merged by RRF, as the transcript channel does."""
    client = client or get_client()
    keyword = search_captions_keyword(text, k, client=client, index=index)
    semantic = search_captions_semantic(text, k, client=client, index=index)
    return reciprocal_rank_fusion([keyword, semantic])[:k]


def captions_for_video(
    video_id: str, k: int = 500, *, client=None, index: str = CAPTION_INDEX,
) -> list[CaptionHit]:
    """Every caption window of one video, in time order, untruncated.

    A ranked hit carries only a snippet, but a caption is where a count, a slide
    or an exam question is written out in full, so the agent reads the whole
    caption of the windows it is about to judge.
    """
    client = client or get_client()
    response = client.search(index=index, body={
        "size": k,
        "_source": {"excludes": ["embedding"]},
        "query": {"term": {"video_id": video_id}},
        "sort": [{"start_sec": "asc"}],
    })
    return parse_search_response(response)


def index_exists(index: str = CAPTION_INDEX, *, client=None) -> bool:
    """Whether captions are searchable on this deployment.

    The caption channel is optional: a server whose Elasticsearch has no caption
    index answers the agent's caption sub-query with an error instead of failing
    the batch, and the agent drops the channel for the rest of the run.
    """
    try:
        return bool((client or get_client()).indices.exists(index=index))
    except Exception:
        return False
