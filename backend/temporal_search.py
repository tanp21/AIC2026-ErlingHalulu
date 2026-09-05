"""Pure temporal sequence ranking over precomputed event candidates."""
from collections import defaultdict
from bisect import bisect_left, insort
import heapq


SCORING_METHODS = (
    "raw_mean",
    "raw_min",
    "shifted_harmonic",
    "minmax_mean",
    "percentile_mean",
)


def deduplicate_candidates(candidates, window_seconds, limit):
    """Greedily keep the highest-ranked hit in each video/time neighborhood."""
    times_by_video = defaultdict(list)
    kept = []
    for hit in candidates:
        times = times_by_video[hit["video_id"]]
        position = bisect_left(times, hit["time"])
        nearby = (
            position > 0 and hit["time"] - times[position - 1] <= window_seconds
        ) or (
            position < len(times) and times[position] - hit["time"] <= window_seconds
        )
        if nearby:
            continue
        row = dict(hit)
        row["event_rank"] = len(kept) + 1
        kept.append(row)
        insort(times, hit["time"])
        if len(kept) == limit:
            break
    return kept


def score_values(scores, normalized, percentiles, method):
    if method == "raw_mean":
        return sum(scores) / len(scores)
    if method == "raw_min":
        return min(scores)
    if method == "shifted_harmonic":
        shifted = [max((score + 1.0) / 2.0, 1e-6) for score in scores]
        return len(shifted) / sum(1.0 / score for score in shifted)
    if method == "minmax_mean":
        return sum(normalized) / len(normalized)
    if method == "percentile_mean":
        return sum(percentiles) / len(percentiles)
    raise ValueError(f"unknown temporal scoring method: {method}")


def _component(hit, method):
    if method in {"raw_mean", "raw_min"}:
        return hit["score"]
    if method == "shifted_harmonic":
        return max((hit["score"] + 1.0) / 2.0, 1e-6)
    if method == "minmax_mean":
        return hit["normalized_score"]
    if method == "percentile_mean":
        return hit["percentile_score"]
    raise ValueError(f"unknown temporal scoring method: {method}")


def _initial_value(component, method):
    return -1.0 / component if method == "shifted_harmonic" else component


def _extend_value(previous, component, method):
    if method == "raw_min":
        return min(previous, component)
    if method == "shifted_harmonic":
        return previous - 1.0 / component
    return previous + component


def _final_score(value, event_count, method):
    if method in {"raw_mean", "minmax_mean", "percentile_mean"}:
        return value / event_count
    if method == "shifted_harmonic":
        return event_count / -value
    return value


def _prepare_candidates(candidates_by_event, candidate_k):
    prepared = []
    denominator = max(candidate_k - 1, 1)
    for event_hits in candidates_by_event:
        scores = [hit["score"] for hit in event_hits]
        low = min(scores) if scores else 0.0
        high = max(scores) if scores else 0.0
        rows = []
        for hit in event_hits:
            row = dict(hit)
            row["normalized_score"] = (
                0.5 if high == low else (row["score"] - low) / (high - low)
            )
            row["percentile_score"] = max(
                0.0, 1.0 - (row["event_rank"] - 1) / denominator
            )
            rows.append(row)
        prepared.append(rows)
    return prepared


def build_sequences(candidates_by_event, max_gap_seconds, scoring_method, candidate_k,
                    beam_size=1):
    """Return the best path ending at every reachable final-event candidate."""
    if scoring_method not in SCORING_METHODS:
        raise ValueError(f"unknown temporal scoring method: {scoring_method}")
    if len(candidates_by_event) < 2 or any(not hits for hits in candidates_by_event):
        return []

    prepared = _prepare_candidates(candidates_by_event, candidate_k)
    by_event_video = []
    for event_hits in prepared:
        grouped = defaultdict(list)
        for row in event_hits:
            grouped[row["video_id"]].append(row)
        by_event_video.append(grouped)

    videos = set(by_event_video[0])
    for grouped in by_event_video[1:]:
        videos &= set(grouped)

    paths = []
    event_count = len(prepared)
    for video_id in videos:
        layers = [
            sorted(grouped[video_id], key=lambda row: (row["frame_idx"], row["time"]))
            for grouped in by_event_video
        ]
        states = [[{
                "value": _initial_value(_component(row, scoring_method), scoring_method),
                "previous": None,
                "start_time": row["time"],
                "frames": (row["frame_idx"],),
            }]
            for row in layers[0]
        ]
        state_layers = [states]

        for event_index in range(1, event_count):
            previous_rows = layers[event_index - 1]
            current_rows = layers[event_index]
            previous_state_groups = state_layers[-1]
            heap = []
            previous_index = 0
            current_states = []
            for row in current_rows:
                while (
                    previous_index < len(previous_rows)
                    and previous_rows[previous_index]["frame_idx"] < row["frame_idx"]
                ):
                    for state_index, state in enumerate(previous_state_groups[previous_index]):
                        heapq.heappush(heap, (
                            -state["value"], state["start_time"], state["frames"],
                            previous_index, state_index, previous_rows[previous_index]["time"],
                        ))
                    previous_index += 1
                cutoff = row["time"] - max_gap_seconds
                while heap and heap[0][5] < cutoff:
                    heapq.heappop(heap)
                selected = []
                while heap and len(selected) < beam_size:
                    item = heapq.heappop(heap)
                    if item[5] < cutoff:
                        continue
                    if item[5] <= row["time"]:
                        selected.append(item)
                for item in selected:
                    heapq.heappush(heap, item)
                group = []
                for _, _, _, best_index, best_state_index, _ in selected:
                    best = previous_state_groups[best_index][best_state_index]
                    group.append({
                        "value": _extend_value(
                            best["value"], _component(row, scoring_method), scoring_method
                        ),
                        "previous": (best_index, best_state_index),
                        "start_time": best["start_time"],
                        "frames": (*best["frames"], row["frame_idx"]),
                    })
                current_states.append(group)
            state_layers.append(current_states)

        for endpoint, final_states in enumerate(state_layers[-1]):
            for final_state_index, state in enumerate(final_states):
                indexes = [endpoint]
                cursor, state_index = endpoint, final_state_index
                for layer_index in range(event_count - 1, 0, -1):
                    cursor, state_index = state_layers[layer_index][cursor][state_index]["previous"]
                    indexes.append(cursor)
                indexes.reverse()
                events = [layers[index][candidate] for index, candidate in enumerate(indexes)]
                visual_score = _final_score(state["value"], event_count, scoring_method)
                paths.append({
                    "video_id": video_id,
                    "start_time": events[0]["time"],
                    "end_time": events[-1]["time"],
                    "events": events,
                    "visual_score": visual_score,
                    "score": visual_score,
                    "source": "visual",
                })

    paths.sort(key=lambda path: (
        -path["visual_score"], path["start_time"], path["video_id"],
        tuple(event["frame_idx"] for event in path["events"]),
    ))
    for rank, path in enumerate(paths, 1):
        path["visual_rank"] = rank
    return paths


def fuse_audio(paths, audio_hits, window_seconds=30, rrf_k=60):
    ranked_audio = defaultdict(list)
    for rank, hit in enumerate(audio_hits, 1):
        ranked_audio[hit["video_id"]].append((rank, hit))
    fused = []
    for fallback_rank, original in enumerate(paths, 1):
        path = dict(original)
        visual_rank = path.get("visual_rank", fallback_rank)
        matches = [
            (rank, hit) for rank, hit in ranked_audio.get(path["video_id"], [])
            if path["start_time"] - window_seconds
            <= hit["time"]
            <= path["end_time"] + window_seconds
        ]
        path["score"] = 1.0 / (rrf_k + visual_rank)
        path["source"] = "visual"
        if matches:
            audio_rank, hit = min(matches, key=lambda item: item[0])
            path["score"] += 1.0 / (rrf_k + audio_rank)
            path["source"] = "both"
            path["audio_evidence"] = {**hit, "audio_rank": audio_rank}
        fused.append(path)
    return fused


def temporal_nms(paths):
    ordered = sorted(paths, key=lambda path: (
        -path["score"], path["start_time"], path["video_id"],
        tuple(event["frame_idx"] for event in path.get("events", [])),
    ))
    accepted = []
    intervals = defaultdict(list)
    for path in ordered:
        overlaps = any(
            not (path["end_time"] < start or path["start_time"] > end)
            for start, end in intervals[path["video_id"]]
        )
        if overlaps:
            continue
        accepted.append(path)
        intervals[path["video_id"]].append((path["start_time"], path["end_time"]))
    return accepted
