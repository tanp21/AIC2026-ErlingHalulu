"""`/agent/search` batches over the public search routes, for servers without it.

The public routes are slower (each pools its full 10k-result snapshot) and run one
at a time under the caller's own login, so a teammate searching in the UI with that
login can supersede them (the client retries 409s). Results are mapped to the
batch endpoint's compact hit format, so the rest of the agent cannot tell the
difference.
"""
from __future__ import annotations

import json
import re
import unicodedata

PAGE_SIZE = 100
MAX_SCOPED_PAGES = 20


def _fold(text):
    folded = "".join(
        character for character in unicodedata.normalize("NFKD", (text or "").casefold())
        if not unicodedata.combining(character)
    ).replace("đ", "d")
    return re.findall(r"\w+", folded)


def contains_phrase(query, text):
    """Same test as `backend.ocr_search.contains_folded_phrase`."""
    needle, haystack = _fold(query), _fold(text)
    return bool(needle) and any(
        haystack[i:i + len(needle)] == needle for i in range(len(haystack) - len(needle) + 1)
    )


def _point(rank, item):
    return {"rank": rank, "video_id": item["video_id"], "frame_idx": item["frame_idx"],
            "time": item["time"], "score": item.get("score", 0.0)}


def _collect(request, page, limit, video_ids, convert):
    """Walk result pages until `limit` hits (inside `video_ids`, if given)."""
    wanted = set(video_ids or ())
    max_pages = MAX_SCOPED_PAGES if wanted else -(-limit // PAGE_SIZE)
    hits, pages = [], 1
    while True:
        for item in page.get("results", []):
            if wanted and item["video_id"] not in wanted:
                continue
            hits.append(convert(len(hits) + 1, item))
            if len(hits) == limit:
                return hits
        cursor = page.get("next_cursor")
        if not cursor or pages >= max_pages:
            return hits
        page = request("POST", "/search/next", json={"cursor": cursor}).json()
        pages += 1


def _audio(rank, item):
    return {**_point(rank, item), "start": item.get("transcript_start"),
            "end": item.get("transcript_end"), "text": item.get("transcript_snippet"),
            "matched_keywords": item.get("matched_keywords", [])}


def run_query(request, query):
    kind = query["kind"]
    limit = query.get("limit", 100)
    if kind == "visual":
        # /search takes one text, so paraphrases (fused server-side by /agent/search)
        # are dropped rather than rank-fused: rank fusion of paraphrases measured
        # worse than the caption alone.
        page = request("POST", "/search", json={"visual_text": query["text"], "mode": "visual"}).json()
        return _collect(request, page, limit, query.get("video_ids"), _point)
    if kind == "similar":
        payload = {"mode": "visual", "image_weight": 1.0,
                   "reference": {"video_id": query["video_id"], "frame_idx": query["frame_idx"]}}
        page = request("POST", "/search/image", data={"payload": json.dumps(payload)}).json()
        return _collect(request, page, limit, query.get("video_ids"), _point)
    if kind == "audio":
        page = request("POST", "/search", json={
            "audio_text": query["text"], "mode": "audio",
            "audio_method": query.get("method", "keyword"),
        }).json()
        return _collect(request, page, limit, None, _audio)
    if kind == "ocr":
        page = request("POST", "/search", json={"ocr_text": query["text"], "mode": "ocr"}).json()

        def ocr(rank, item):
            return {**_point(rank, item), "start": item.get("ocr_start"), "end": item.get("ocr_end"),
                    "text": item.get("ocr_snippet"),
                    "exact": contains_phrase(query["text"], item.get("ocr_snippet", "")),
                    "matched_keywords": item.get("matched_keywords", [])}

        return _collect(request, page, limit, None, ocr)
    if kind == "temporal":
        body = {"events": query["events"], "max_gap_seconds": query.get("max_gap_seconds", 120)}
        if query.get("audio_text"):
            body["audio_text"] = query["audio_text"]
        videos = request("POST", "/search/temporal", json=body).json()["videos"]
        return [
            {
                "rank": rank, "video_id": video["video_id"], "score": video["best_score"],
                "sequences": [
                    {"score": sequence["score"], "events": [
                        {"frame_idx": e["frame_idx"], "time": e["time"], "score": e.get("score", 0.0)}
                        for e in sequence["events"]
                    ]}
                    for sequence in sorted(video["sequences"], key=lambda s: s["rank"])[:3]
                ],
            }
            for rank, video in enumerate(videos[:limit], 1)
        ]
    raise ValueError(f"unknown query kind {kind}")


def run_batch(request, queries, error_type):
    results = {}
    for query in queries:
        try:
            results[query["id"]] = {"kind": query["kind"], "hits": run_query(request, query)}
        except error_type as exc:
            results[query["id"]] = {"kind": query["kind"], "error": str(exc)}
    return results
