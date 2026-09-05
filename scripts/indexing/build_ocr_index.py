#!/usr/bin/env python3
"""Build the Elasticsearch on-screen-text (OCR) index for 2026 L21-L30 videos.

Reads the merged per-keyframe OCR output (every pre-extracted keyframe, not
just one per shot -- see ``scripts/ocr/run_corpus.py`` and
``docs/reports/ocr-shot-audit.md``), drops any known fixed station-bug/ticker
regions for the video's group (``GROUP_IGNORE_REGIONS``), then suppresses
remaining burned-in furniture (logos, clocks, and any other scrolling ticker)
per video via ``scripts.ocr.postprocess.suppress_static_overlays``, resolves
each shot's time window via the same shot-boundary + frame-map machinery the
rest of the app uses (never fps arithmetic), and bulk-indexes one document per
shot into Elasticsearch alongside the audio index.

Suppression needs a whole video's keyframes at once (it tells a ticker apart
from real text by how often a screen position repeats across many frames), so
this reads and groups the full per-keyframe source before resolving shots --
unlike the old per-shot source, which could be streamed one entry at a time.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from backend import frame_maps
from backend import shots
from backend.ocr_search import OCR_INDEX, build_index_definition, get_client
from scripts.ocr.postprocess import (
    drop_regions, harmonize_diacritics, render_text, reading_order,
    suppress_static_overlays, to_lines,
)

DEFAULT_SOURCE = ROOT / "data" / "2026" / "ocr" / "ocr_2026_v2.json"

#: Fixed per-group masks for burned-in station furniture, as normalised
#: ``(x0, y0, x1, y1)`` boxes -- a deterministic backstop for the statistical
#: `suppress_static_overlays` rule below: it needs no history (works even on a
#: video with fewer than that rule's `min_frames` keyframes) and isn't fooled
#: by furniture the recogniser reads inconsistently frame to frame (measured:
#: the same corner bug came back as "HTV"/"HTM"/"HIV"/"HTL" across different
#: frames of the same video -- no single spelling clears the frequency rule's
#: threshold on its own, but the region always does).
#:
#: Each group's box was measured from one representative video, sampling
#: keyframes spread across its full length, then checked for a text/box
#: combination that recurs at (about) the same spot across most of the
#: sample -- the profile of a corner bug or a full-width ticker, not a
#: shot-specific graphic. Not exhaustively verified against every video in
#: the group; revisit if a group's videos turn out not to share one template.
#:
#: - L21/L22 ("60 giây" newscasts): a full-width scrolling ticker across the
#:   bottom of the frame (see the ticker band below) *and* a station
#:   bug/clock in the top-right corner. Ticker line centres measured at
#:   y=0.93-0.94 of frame height on two flagged videos (L22_V029 frame 24798,
#:   L22_V003 frame 7112); the lowest real story caption/banner in the same
#:   two videos centred at y<=0.881 -- a comfortable gap. The corner bug
#:   ("HD"/clock digits) spanned roughly x=0.82-0.93, y=0.08-0.17.
#: - L23/L24: an "HTV / THỂ THAO" (sports) bug in the top-right corner,
#:   x=0.86-0.97, y=0.03-0.16. L23 also carries a live race-telemetry HUD
#:   (lap counter, clock, distance) in the top-left -- left alone here since
#:   it's dynamic broadcast data tied to the event, not static furniture.
#: - L26-L29: an "HTV / Online" bug in the top-left corner, x=0.0-0.11,
#:   y=0.0-0.14.
#: - L30: a corner bug the recogniser could not read consistently enough to
#:   transcribe (garbled every sample), top-right, x=0.83-0.99, y=0.03-0.12.
#: - L25 is deliberately **not** masked: it's a full-screen article-reading
#:   video (a "THANHNIEN"/"thanhnien.vn" watermark) where the watermark's
#:   footprint overlaps almost the entire body-text area frame to frame, so
#:   any rectangle wide enough to catch it would also delete the real
#:   article text sitting in the same space. Left to
#:   `suppress_static_overlays`'s constant-text rule, which is a safe fit
#:   here since the watermark string barely varies -- imperfect (a
#:   `THANKNIEN`/`THANHNIEN` misread can dodge the frequency threshold) but
#:   low-severity: a single brand token, not a sentence that dominates BM25
#:   the way a ticker crawl does.
GROUP_IGNORE_REGIONS: dict[str, list[tuple[float, float, float, float]]] = {
    "L21": [(0.0, 0.90, 1.0, 1.0), (0.80, 0.0, 1.0, 0.20)],
    "L22": [(0.0, 0.90, 1.0, 1.0), (0.80, 0.0, 1.0, 0.20)],
    "L23": [(0.83, 0.0, 1.0, 0.20)],
    "L24": [(0.83, 0.0, 1.0, 0.20)],
    "L26": [(0.0, 0.0, 0.14, 0.16)],
    "L27": [(0.0, 0.0, 0.14, 0.16)],
    "L28": [(0.0, 0.0, 0.14, 0.16)],
    "L29": [(0.0, 0.0, 0.14, 0.16)],
    "L30": [(0.80, 0.0, 1.0, 0.16)],
}


def _drop_regions_from_records(records: list[dict], regions) -> list[dict]:
    """Applies :func:`drop_regions` to every record's lines, same record shape in and out."""
    result = []
    for record in records:
        kept = drop_regions(to_lines(record.get("lines", [])), regions)
        result.append({
            **record,
            "lines": [
                {"text": line.text, "box": [list(p) for p in line.box], "conf": line.conf}
                for line in kept
            ],
        })
    return result


def _group_by_video(raw: dict) -> dict[tuple[str, str], list[dict]]:
    """``{(group, video_id): [records sorted by frame_idx]}``, one record per keyframe."""
    by_video: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for key, entry in raw.items():
        group, video_id, frame_idx_str = key.split("/")
        by_video[(group, video_id)].append({
            "frame_idx": int(frame_idx_str),
            "lines": entry.get("lines", []),
        })
    for records in by_video.values():
        records.sort(key=lambda r: r["frame_idx"])
    return by_video


def _pick_anchor(records: list[dict]) -> dict | None:
    """The keyframe with the most confidence-weighted surviving text.

    Mirrors ``postprocess.group_segments``'s anchor selection: the frame whose
    thumbnail best shows why the shot matched. Records whose lines were fully
    suppressed (a shot that was only ever the ticker/logo, no real content)
    score zero and lose to any keyframe with real text; if every keyframe in
    the shot lost everything, there is no anchor.
    """
    scored = [
        (record, sum(len(line["text"]) * line.get("conf", 1.0) for line in record["lines"]))
        for record in records
    ]
    scored = [(record, score) for record, score in scored if score > 0]
    if not scored:
        return None
    return max(scored, key=lambda item: item[1])[0]


def iter_documents(source: Path):
    """Yields (doc_id, source_dict) pairs, skipping shots with no surviving
    text (ticker/logo-only, or empty) and shots whose boundary JSON or frame
    map can't be resolved (logged, not fatal)."""
    raw = json.loads(source.read_text())
    by_video = _group_by_video(raw)
    shot_cache: dict[tuple[str, str], "shots.ShotIndex"] = {}
    frame_map_cache: dict[str, "frame_maps.FrameMap"] = {}
    skipped_empty = 0
    skipped_errors = 0

    for (group, video_id), records in by_video.items():
        regions = GROUP_IGNORE_REGIONS.get(group)
        if regions:
            records = _drop_regions_from_records(records, regions)
        suppressed = suppress_static_overlays(records)

        try:
            row = {"year": "2026", "group": group, "video_id": video_id}
            shot_index = shots.load(row)
            mapping = frame_maps.load(video_id)
        except (OSError, FileNotFoundError) as exc:
            print(f"[ocr-index] skipping video {group}/{video_id}: {exc}", file=sys.stderr)
            skipped_errors += len(records)
            continue
        shot_cache[(group, video_id)] = shot_index
        frame_map_cache[video_id] = mapping

        for shot_idx, (start_frame, end_frame) in enumerate(shot_index.bounds):
            in_shot = [
                record for record in suppressed
                if int(start_frame) <= record["frame_idx"] <= int(end_frame)
            ]
            anchor = _pick_anchor(in_shot)
            if anchor is None:
                skipped_empty += 1
                continue

            ordered = harmonize_diacritics(reading_order(to_lines(anchor["lines"])))
            text = render_text(ordered)
            if not text:
                skipped_empty += 1
                continue

            try:
                start_sec, end_sec = shots.shot_bounds_seconds(
                    mapping, int(start_frame), int(end_frame)
                )
                anchor_sec = mapping.frame_time(anchor["frame_idx"])
            except (IndexError, KeyError, ValueError) as exc:
                key = f"{group}/{video_id}/{shot_idx}"
                print(f"[ocr-index] skipping {key}: {exc}", file=sys.stderr)
                skipped_errors += 1
                continue

            yield f"{video_id}:{shot_idx}", {
                "video_id": video_id,
                "group": group,
                "shot_idx": shot_idx,
                "frame_idx": anchor["frame_idx"],
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "anchor_sec": round(anchor_sec, 3),
                "text": text,
            }

    print(f"[ocr-index] skipped {skipped_empty} empty/suppressed-away shots, "
          f"{skipped_errors} unresolvable shots", file=sys.stderr)


def build(source: Path, index: str, bulk_size: int, recreate: bool):
    from elasticsearch.helpers import bulk

    client = get_client()
    if client.indices.exists(index=index):
        if not recreate:
            raise SystemExit(f"index {index!r} already exists; pass --recreate")
        client.indices.delete(index=index)
    client.indices.create(index=index, body=build_index_definition())

    total = 0
    batch = []

    def flush():
        nonlocal total
        if not batch:
            return
        bulk(client.options(request_timeout=120), batch)
        total += len(batch)
        print(f"[ocr-index] indexed {total} shots", flush=True)
        batch.clear()

    for doc_id, source_doc in iter_documents(source):
        batch.append({"_index": index, "_id": doc_id, "_source": source_doc})
        if len(batch) >= bulk_size:
            flush()
    flush()
    client.indices.refresh(index=index)

    count = client.count(index=index)["count"]
    if count != total:
        raise RuntimeError(f"Elasticsearch count {count} != generated {total}")
    print(f"[ocr-index] DONE: {count} shots -> {index}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--index", default=OCR_INDEX)
    parser.add_argument("--bulk-size", type=int, default=1000)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    build(args.source, args.index, args.bulk_size, args.recreate)


if __name__ == "__main__":
    main()
