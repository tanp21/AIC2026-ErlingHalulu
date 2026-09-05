"""On-screen text (OCR) index definition and Elasticsearch-backed retrieval.

One representative keyframe per shot is OCR'd offline (see
``scripts/indexing/build_ocr_index.py`` and ``docs/reports/ocr-shot-audit.md``
for why a single frame, not all three, and why the accuracy bar is
"good enough for ASCII-folded keyword search", not exact diacritics). This
module mirrors ``audio_search``'s BM25 half — no dense vector, since OCR text
is short on-screen overlays (names, timestamps, captions) meant to be found
by literal keyword match, not semantic similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import os
import re
from typing import Sequence
import unicodedata


OCR_INDEX = os.environ.get("AIC_OCR_INDEX", "aic_ocr_2026_v1")

_client = None


@dataclass(frozen=True)
class OcrHit:
    video_id: str
    shot_idx: int
    frame_idx: int
    start_sec: float
    end_sec: float
    anchor_sec: float
    text: str
    score: float
    bm25_rank: int | None = None
    matched_keywords: tuple[str, ...] = ()


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
                "video_id": {"type": "keyword"},
                "group": {"type": "keyword"},
                "shot_idx": {"type": "integer"},
                "frame_idx": {"type": "integer"},
                "start_sec": {"type": "float"},
                "end_sec": {"type": "float"},
                "anchor_sec": {"type": "float"},
                "text": {
                    "type": "text",
                    "fields": {"folded": {"type": "text", "analyzer": "vi_folded"}},
                },
            },
        },
    }


def build_bm25_request(text: str, k: int = 100, *, strict: bool = False) -> dict:
    candidate_count = max(100, k)
    if strict:
        query = {
            "multi_match": {
                "query": text,
                "fields": ["text^2", "text.folded"],
                "type": "phrase",
                "slop": 0,
            }
        }
    else:
        query = {
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
        }
    return {
        "size": candidate_count,
        "highlight": {
            "pre_tags": ["<aic-hit>"], "post_tags": ["</aic-hit>"],
            "fields": {"text": {"number_of_fragments": 0}},
        },
        "query": query,
    }


def _folded_tokens(value: str) -> list[str]:
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.findall(r"\w+", folded)


def build_filter_request(text: str, k: int = 100) -> dict:
    """A typo-tolerant, proximity-bound query for gating another mode's
    results by on-screen text.

    ``build_bm25_request``'s non-strict query is a "should" soup tuned for a
    *ranked* results list, where a term matching anywhere in a long, noisy
    OCR blob (one shot's overlay text can run to a full news ticker) is
    harmless -- it just sorts low. Reused as a binary pass/fail filter,
    matching each query term independently -- even with `operator: "and"`
    requiring all of them -- lets two unrelated words that each happen to
    occur *somewhere* in that blob (e.g. a fuzzy hit on one word plus an
    unrelated exact hit on a common word like "Nam" many words away) pass
    together. A `span_near` of fuzzy term matches instead requires the
    terms to actually sit near each other, like a real (if slightly
    misspelled) phrase -- and the boosted phrase clause still lets a
    literal, correctly-ordered match outscore a fuzzy one.
    """
    candidate_count = max(100, k)
    terms = _folded_tokens(text)
    should = [
        {
            "multi_match": {
                "query": text,
                "fields": ["text^4", "text.folded^2"],
                "type": "phrase",
                "slop": 2,
            }
        }
    ]
    if terms:
        should.append({
            "span_near": {
                "clauses": [
                    {
                        "span_multi": {
                            "match": {
                                "fuzzy": {
                                    "text.folded": {
                                        "value": term, "fuzziness": "AUTO", "prefix_length": 1,
                                    }
                                }
                            }
                        }
                    }
                    for term in terms
                ],
                "slop": 3,
                "in_order": True,
            }
        })
    return {
        "size": candidate_count,
        "highlight": {
            "pre_tags": ["<aic-hit>"], "post_tags": ["</aic-hit>"],
            "fields": {"text": {"number_of_fragments": 0}},
        },
        "query": {"bool": {"should": should, "minimum_should_match": 1}},
    }


def contains_folded_phrase(query: str, text: str) -> bool:
    def tokens(value):
        return _folded_tokens(value)

    query_tokens = tokens(query)
    text_tokens = tokens(text)
    return bool(query_tokens) and any(
        text_tokens[index:index + len(query_tokens)] == query_tokens
        for index in range(len(text_tokens) - len(query_tokens) + 1)
    )


def parse_search_response(response: dict) -> list[OcrHit]:
    hits = []
    for raw in response.get("hits", {}).get("hits", []):
        source = raw.get("_source", {})
        required = {"video_id", "shot_idx", "frame_idx", "start_sec", "end_sec", "anchor_sec", "text"}
        if not required.issubset(source):
            continue
        highlighted = " ".join(raw.get("highlight", {}).get("text", []))
        matched = tuple(dict.fromkeys(
            part.strip() for part in highlighted.split("<aic-hit>")[1:]
            for part in [part.split("</aic-hit>", 1)[0]] if part.strip()
        ))
        hits.append(
            OcrHit(
                video_id=source["video_id"],
                shot_idx=int(source["shot_idx"]),
                frame_idx=int(source["frame_idx"]),
                start_sec=float(source["start_sec"]),
                end_sec=float(source["end_sec"]),
                anchor_sec=float(source["anchor_sec"]),
                text=source["text"],
                score=float(raw.get("_score") or 0.0),
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


def search_ocr_keyword(
    text: str, k: int = 100, *, client=None, index: str = OCR_INDEX,
    strict: bool = False,
) -> list[OcrHit]:
    """BM25-only on-screen-text search."""
    client = client or get_client()
    response = client.search(index=index, body=build_bm25_request(text, k, strict=strict))
    ranked = [
        replace(hit, bm25_rank=rank)
        for rank, hit in enumerate(parse_search_response(response), 1)
    ]
    return ranked[:k]


def search_ocr_filter(text: str, k: int = 100, *, client=None, index: str = OCR_INDEX) -> list[OcrHit]:
    """Typo-tolerant AND-match search for gating another search mode's results."""
    client = client or get_client()
    response = client.search(index=index, body=build_filter_request(text, k))
    ranked = [
        replace(hit, bm25_rank=rank)
        for rank, hit in enumerate(parse_search_response(response), 1)
    ]
    return ranked[:k]
