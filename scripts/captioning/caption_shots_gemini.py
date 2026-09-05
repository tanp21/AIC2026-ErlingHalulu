#!/usr/bin/env python3
"""Caption selected shots from their HD keyframes with Gemini."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from google import genai
from google.genai import types
import numpy as np
import pyarrow as pa
import subprocess

from scripts.captioning import caption_shots
from scripts.audio.gemini_clean import _is_retryable, _retry_delay_seconds
from scripts.ocr.scoring import fold


MODEL_ID = "gemini-3.1-flash-lite"
PILOT_SHOTS = (
    ("L27_V010", 83),
    ("L30_V078", 20),
    ("L30_V078", 21),
    ("L30_V072", 19),
    ("K17_V003", 102),
    ("K19_V016", 158),
    ("L23_V013", 20),
    ("K07_V025", 225),
    ("K07_V025", 226),
    ("K07_V025", 227),
)
DESCRIPTION_PROMPT = """Produce a detailed English description optimized for high semantic-retrieval recall. The images are dense chronological frames from one video shot. Describe only visible evidence; do not infer names, locations, causes, occupations, identities, or intent that are not visually established.

Build a comprehensive static inventory: setting and terrain; foreground/background; lighting and weather; camera angle, distance, composition, and movement; every visible person with count, approximate age group, clothing, colors, accessories, pose, and position; every distinct animal, vehicle, food, tool, device, structure, container, material, and other object with count, color, shape, state, and relative position. Preserve fine distinctions, spatial order, repetitions, and unusual details instead of replacing them with broad categories.

Then build a chronological action inventory. For each visible change, state actor + precise action + affected object + direction/result. Include appearance/disappearance, entry/exit, transfer, placement, removal, contact/separation, rotation, opening/closing, changes in count or order, body/head orientation, and initial/final state. Use a precise common noun plus a more specific noun when visually justified. Do not collapse concrete actions into generic phrases such as "works on," "handles something," or "interacts with equipment." If densely sampled frames still cannot distinguish a fine action, describe the observable before-to-after change without guessing its purpose.

Before returning, check coverage explicitly: give an exact count whenever it is visually countable; distinguish every person by clothing color, role in the frame, and position; distinguish repeated objects by count and left/right/foreground/background placement; name the object an action affects; and state the first observable instant of each new action or rotation. For evidence-rich shots, write 400-600 words and always finish both required sections within the output limit. Do not shorten the caption by replacing this inventory with general summaries.

Merge uninterrupted repetition of the same motion into one action statement; do not narrate frame-by-frame continuation or restate an unchanged scene. Keep separate statements only for distinct actions, actors, objects, directions, contacts, transitions, or state changes. Favor high-information detail and broad semantic coverage over repetitive wording.

For prominent, permanent physical scene-text that identifies the setting or an object—such as a sign, memorial inscription, plaque, architectural motto, shop name, banner, race arch, or scale label—transcribe each complete readable phrase verbatim as part of the description, including text arranged vertically. Do not spend description tokens copying long documents, recipes, charts, or a complete OCR inventory: a separate OCR pass records every visible text item verbatim.

Completely ignore TV/broadcast graphics overlaid on the footage. Do not mention or transcribe channel/station watermarks and logos (for example HTV, HTV9, Tuoi Tre TV or tuoitreTV), show/program names (for example 60 Giay), clocks, replay badges, lower thirds, subtitles, news tickers, or scrolling red text at the bottom. This exclusion applies only to broadcast overlays; keep real text on physical objects in the scene.

These frames are the automatic sampling for one detected shot, but automatic shot-boundary detection occasionally misses a real cut and leaves two unrelated scenes inside one shot. If the frames clearly show two or more unrelated scenes, camera setups, or subjects rather than one continuous take, open the description with one short sentence flagging this (for example: "Frames span an apparent shot change partway through."), then describe each scene as its own self-contained block, in the order the frames appear, and do the same in the action inventory. Never merge subjects, objects, or actions from different scenes into a single statement, and never invent a transition, camera move, or relationship between them that is not directly visible. A shot that is genuinely one continuous take needs no such note. Splitting into scenes must never come at the cost of action precision: give the scene that has more frames the same exact-verb, exact-object action detail required above (rotation, contact, transfer, opening/closing, first observable instant of a new action, and so on) that a single continuous shot would get; do not fall back to vaguer phrasing such as "manipulates," "adjusts," or "works on" just because the frames are split across scenes.

Return exactly two plain-text sections: "Detailed visual description:" and "Chronological action inventory:". Do not include a text/OCR section; a separate pass handles it. Do not use markdown, timestamps, or references to frames or instructions."""

TEXT_PROMPT = """Perform an exhaustive OCR audit of text physically present in this video scene. Find every distinct readable physical text item. The images show chronological full frames and enlarged overlapping crops from the same shot. Work silently in three passes before answering: (1) map each distinct physical text region or surface, (2) transcribe each region in its natural layout and reading order, and (3) verify every character, Vietnamese diacritic, digit, decimal, unit, punctuation mark, and symbol against all available views.

A semicolon separates physical text regions, never individual words or characters. Keep every continuous line, sign, label, inscription, or phrase as one complete item. Reconstruct text printed one word or character per line: read vertical columns top-to-bottom and emit one complete phrase per column; read ordinary horizontal text left-to-right. Before answering, reject and repair any sequence of one-word items that actually belongs to the same sign, column, plaque, banner, arch, document line, table row, chart label, scale, gauge, package, garment, vehicle, or screen. Merge duplicate sightings of the same physical text across frames and crops, but preserve repetitions printed on distinct objects. Never infer obscured text; use [?] only for genuinely unreadable characters.

Search all physical scene surfaces, including distant background text. Enumerate every visible number and label on scales, lists, tables, diagrams, maps, and gauges in spatial reading order; never collapse values into a range or summary. Exclude broadcaster overlays: station/channel logos and watermarks, program names, clocks, replay markers, subtitles, lower thirds, news tickers, and scrolling text. Return only the verbatim visible text as one plain-text semicolon-separated inventory. Do not add headings, markdown, translations, object names, locations, descriptions, commentary, or words not visible in the scene. Return "None" only when no physical scene text is readable."""
COMPACT_DESCRIPTION_PROMPT = """Produce a self-contained 250-400 word English video-shot description optimized for semantic retrieval. Describe only visible evidence. Cover setting, foreground/background, camera view, lighting, every important person and object with distinguishing colors, clothing, state and relative position, then every distinct action and before-to-after change in chronological order. Use precise actor + action + object + direction/result wording, especially for first contact, separation, rotation, transfer, entry/exit, body orientation and initial/final state. Do not replace concrete actions with generic phrases, infer identity or intent, narrate repeated frames, transcribe text, or mention broadcaster overlays. Return plain text with the headings "Detailed visual description:" and "Chronological action inventory:" only."""
COMBINED_PROMPT = f"""Analyze this one video shot and return exactly one JSON object with string fields "caption" and "ocr". Do not use markdown.

For "caption":
{DESCRIPTION_PROMPT}

For "ocr":
{TEXT_PROMPT}
"""
PROMPT_SHA256 = hashlib.sha256(
    (DESCRIPTION_PROMPT + "\0" + TEXT_PROMPT).encode()
).hexdigest()
GEMINI_CAPTION_SCHEMA = pa.schema([
    *caption_shots.CAPTION_SCHEMA,
    ("ocr", pa.string()),
])
EXPERIMENT_SCHEMA = pa.schema([
    *GEMINI_CAPTION_SCHEMA,
    ("strategy", pa.string()),
    ("api_mode", pa.string()),
    ("input_tokens", pa.int64()),
    ("thinking_tokens", pa.int64()),
    ("truncated", pa.bool_()),
    ("request_count", pa.int32()),
    ("image_count", pa.int32()),
    ("upload_bytes", pa.int64()),
])
BROADCAST_OVERLAY = re.compile(
    r"(?i)\b(?:HTV\d*|tuoi\s*tre\s*tv|tuoitretv|60\s*giay|replay|lower[- ]third|ticker|watermark)\b"
)
CLOCK_OVERLAY = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
PILOT_REQUIRED_TEXT = {
    ("L27_V010", 83): ("HỎA HỒNG NHẬT TẢO", "KIẾM BẠT KIÊN GIANG"),
    ("L30_V078", 20): ("Nhân bánh cuốn", "200 gr", "Cách xào nhân"),
    ("L30_V078", 21): ("Nhân bánh cuốn", "200 gr", "Cách xào nhân"),
    ("L30_V072", 19): ("Giang Ly", "Khánh Vĩnh", "Khánh Hòa", "04.08.2024"),
    ("K17_V003", 102): ("Mộc Châu", "HOÀNG SA", "TRƯỜNG SA"),
    ("K19_V016", 158): ("20", "18", "16", "14", "12", "10", "8"),
    ("L23_V013", 20): ("Tam Kỳ", "Quảng Nam", "TON DONG A"),
}

STRATEGIES = (
    "frames-fixed",
    "video-default",
    "video-low",
    "video-targeted-ocr",
    "video-targeted-ocr-compact",
)
FULL_2026_SHOT_COUNT = 96_624
BATCH_INPUT_USD_PER_MILLION = 0.125
BATCH_OUTPUT_USD_PER_MILLION = 0.75
DESCRIPTION_MAX_OUTPUT_TOKENS = 1_536
COST_BENCHMARK_STRATEGIES = (
    "video-high+ocr-full-minimal",
    "video-high+ocr-target8-minimal",
    "video-high+ocr-target4-minimal",
    "video-low+ocr-full-minimal",
    "video-low+ocr-target8-minimal",
    "video-low+ocr-target4-minimal",
    "combined-video-high+ocr-target8-minimal",
)
PILOT_QUERIES = (
    {"query": "HỎA HỒNG NHẬT TẢO OANH THIÊN ĐỊA KIẾM BẠT KIÊN GIANG KHẤP QUỶ THẦN",
     "gold": (("L27_V010", 83),)},
    {"query": "Thẻ công thức Nhân bánh cuốn gồm 200 gr thịt nạc dăm và phần Cách xào nhân",
     "gold": (("L30_V078", 20), ("L30_V078", 21))},
    {"query": "Giang Ly ở Khánh Vĩnh, Khánh Hòa, ngày 04.08.2024",
     "gold": (("L30_V072", 19),)},
    {"query": "Biển du lịch Mộc Châu và bản đồ có chữ HOÀNG SA, TRƯỜNG SA",
     "gold": (("K17_V003", 102),)},
    {"query": "Cận cảnh cây cột hoặc thước đo có đầy đủ các số 20 18 16 14 12 10 8",
     "gold": (("K19_V016", 158),)},
    {"query": "Cảnh đua xe đạp tại vạch đích ở Tam Kỳ, Quảng Nam dưới cổng TÔN ĐÔNG Á",
     "gold": (("L23_V013", 20),)},
    {"query": "Một cánh tay rô-bốt hạ khung xe xuống cho tới khoảnh khắc khung chạm chiếc xe",
     "gold": (("K07_V025", 225),)},
    {"query": "Một công nhân bắt đầu quay tay xoay để xoay thiết bị trong xưởng lắp ráp",
     "gold": (("K07_V025", 226),)},
    {"query": "Công nhân mặc áo đen quay mặt về camera và bắt đầu gấp chiếc khăn cạnh cửa ô tô",
     "gold": (("K07_V025", 227),)},
)


def select_keyframes(record: caption_shots.ShotRecord, data_dir: Path, limit: int = 8):
    directory = data_dir / record.year / "keyframes" / record.group / record.video_id
    paths = [
        path for path in sorted(directory.glob("*.png"))
        if path.stem.isdigit() and record.start_frame <= int(path.stem) <= record.end_frame
    ]
    if not paths:
        raise FileNotFoundError(f"no keyframes for {record.video_id}/{record.shot_index}")
    if len(paths) <= limit:
        return paths
    return [paths[round(index * (len(paths) - 1) / (limit - 1))] for index in range(limit)]


def sample_positions(start: int, end: int, limit: int):
    count = min(limit, end - start + 1)
    return np.rint(np.linspace(start, end, count)).astype(int).tolist()


def sample_video_frames(record: caption_shots.ShotRecord, limit: int):
    positions = sample_positions(0, record.end_frame - record.start_frame, limit)
    select = "+".join(f"eq(n\\,{position})" for position in positions)
    result = subprocess.run([
        "ffmpeg", "-loglevel", "error", "-ss", f"{record.start_sec:.6f}",
        "-i", record.video_path, "-t", f"{record.duration_sec:.6f}",
        "-vf", f"select={select}", "-fps_mode", "vfr", "-frames:v", str(len(positions)),
        "-an", "-c:v", "mjpeg", "-q:v", "2", "-f", "image2pipe", "pipe:1",
    ], check=True, capture_output=True, timeout=120)
    frames, offset = [], 0
    while (start := result.stdout.find(b"\xff\xd8", offset)) >= 0:
        end = result.stdout.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        frames.append(result.stdout[start:end + 2])
        offset = end + 2
    if len(frames) != len(positions):
        raise OSError(
            f"decoded {len(frames)}/{len(positions)} frames for "
            f"{record.video_id}/{record.shot_index}"
        )
    return frames


def text_audit_images(frames):
    representatives = [frames[0], frames[len(frames) // 2], frames[-1]]
    images = list(representatives)
    for encoded in representatives:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = frame.shape[:2]
        crops = (
            frame[:int(height * .62), :int(width * .62)],
            frame[:int(height * .62), int(width * .38):],
            frame[int(height * .38):, :int(width * .62)],
            frame[int(height * .38):, int(width * .38):],
            frame[int(height * .2):int(height * .8), int(width * .2):int(width * .8)],
        )
        for crop in crops:
            ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError("cannot encode text-audit crop")
            images.append(jpeg.tobytes())
    return images


def video_sampling_fps(record, target_frames=12):
    return min(24.0, record.fps, target_frames / record.duration_sec)


def image_parts(images, resolution="MEDIA_RESOLUTION_HIGH"):
    return [
        types.Part.from_bytes(
            data=image,
            mime_type="image/jpeg",
            media_resolution={"level": resolution},
        )
        for image in images
    ]


def video_part(uploaded_file, record, resolution="MEDIA_RESOLUTION_HIGH", target_frames=24):
    return types.Part(
        file_data=types.FileData(
            file_uri=uploaded_file.uri,
            mime_type=uploaded_file.mime_type,
        ),
        video_metadata=types.VideoMetadata(
            start_offset=f"{record.start_sec:.6f}s",
            end_offset=f"{record.end_sec:.6f}s",
            fps=video_sampling_fps(record, target_frames),
        ),
        media_resolution={"level": resolution},
    )


def _load_ocr_detector():
    site_packages = ROOT / ".venv-ocr/lib/python3.12/site-packages"
    if site_packages.exists() and str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
    from scripts.ocr.engines import OcrConfig, OcrPipeline

    return OcrPipeline(OcrConfig())


def targeted_text_audit_images(frames, detector=None, max_blocks_per_frame=8):
    """One full-frame fallback plus detector-focused blocks from three frames."""
    from scripts.ocr.reader import text_blocks

    detector = detector or _load_ocr_detector()
    representatives = [frames[0], frames[len(frames) // 2], frames[-1]]
    images = [representatives[len(representatives) // 2]]
    for encoded in representatives:
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        blocks = text_blocks(frame, detector.detect_boxes(frame))[:max_blocks_per_frame]
        for _, crop in blocks:
            ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError("cannot encode detected text block")
            images.append(jpeg.tobytes())
    return images


def generation_config(*, resolution, max_output_tokens, thinking_level="minimal"):
    return types.GenerateContentConfig(
        temperature=0.0,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
        media_resolution=resolution,
        max_output_tokens=max_output_tokens,
    )


def make_request(parts, prompt, *, resolution, max_output_tokens, metadata,
                 thinking_level="minimal"):
    return types.InlinedRequest(
        contents=[types.Content(
            role="user", parts=[*parts, types.Part(text=prompt)],
        )],
        config=generation_config(
            resolution=resolution, max_output_tokens=max_output_tokens,
            thinking_level=thinking_level,
        ),
        metadata=metadata,
    )


def request_key(strategy, pass_name, record):
    return f"{strategy}|{pass_name}|{record.video_id}|{record.shot_index}"


def request_json(request):
    # A JSONL line contains a GenerateContentRequest, not the SDK-only
    # InlinedRequest wrapper. Keep the API's documented JSON field name.
    value = request.model_dump(mode="json", by_alias=False, exclude_none=True)
    value["generation_config"] = value.pop("config")
    value.pop("metadata", None)
    if not isinstance(value["contents"], list):
        value["contents"] = [value["contents"]]
    return value


def response_result(response, *, allow_truncated=False):
    if not response.candidates:
        raise RuntimeError("Gemini response has no candidates")
    finish_reason = getattr(response.candidates[0], "finish_reason", None)
    truncated = getattr(finish_reason, "name", finish_reason) == "MAX_TOKENS"
    if truncated and not allow_truncated:
        raise RuntimeError("truncated Gemini response")
    value = " ".join((response.text or "").split())
    if not value:
        raise RuntimeError("empty Gemini response")
    usage = response.usage_metadata
    return {
        "text": value,
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "thinking_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
        "cached_tokens": int(getattr(usage, "cached_content_token_count", 0) or 0),
        "truncated": truncated,
    }


def parse_batch_response(value, *, allow_truncated=False):
    # Batch can add usage fields before the installed SDK publishes them.
    return response_result(
        types.GenerateContentResponse.model_validate(value, extra="ignore"),
        allow_truncated=allow_truncated,
    )


def wait_for_file(client, uploaded, timeout=1800):
    deadline = time.monotonic() + timeout
    while True:
        current = client.files.get(name=uploaded.name)
        state = getattr(current.state, "name", current.state)
        if state == "ACTIVE":
            return current
        if state == "FAILED":
            raise RuntimeError(f"file processing failed: {current.error}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"file did not become ACTIVE: {uploaded.name}")
        time.sleep(5)


def upload_videos(client, records):
    uploaded = {}
    for record in records:
        if record.video_id in uploaded:
            continue
        file = client.files.upload(
            file=record.video_path,
            config={"mime_type": "video/mp4", "display_name": record.video_id},
        )
        uploaded[record.video_id] = wait_for_file(client, file)
        print(f"uploaded {record.video_id}", flush=True)
    return uploaded


def _batch_state_name(job):
    return getattr(job.state, "name", job.state)


def verify_batch_access(client, model_id):
    """Fail before expensive preprocessing/uploads when the key is free-tier."""
    try:
        job = client.batches.create(
            model=model_id,
            src=[{"contents": [{"role": "user", "parts": [{"text": "Reply OK"}]}]}],
            config={"display_name": "caption-batch-access-check"},
        )
    except Exception as error:
        if "FAILED_PRECONDITION" in str(error):
            raise RuntimeError(
                "Gemini Batch API is unavailable for this project. Enable a paid "
                "billing tier (and positive prepay balance when required) in AI Studio."
            ) from error
        if "PERMISSION_DENIED" in str(error):
            raise RuntimeError(
                "Gemini denied this project access. Check AI Studio Projects/Billing "
                "for an Unavailable or appeal banner, then contact Cloud Billing support."
            ) from error
        raise
    client.batches.cancel(name=job.name)


def run_batch(client, model_id, keyed_requests, work_dir, *, poll_seconds=20,
              allow_truncated=False):
    """Submit JSONL Batch work, wait, parse responses, and remove remote files."""
    work_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        json.dumps(
            [(key, request_json(request)) for key, request in keyed_requests],
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    state_path = work_dir / f"batch-{digest}.json"
    result_path = work_dir / f"batch-{digest}-results.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    input_file = None
    output_file = None
    completed = False
    try:
        if state.get("job_name"):
            job = client.batches.get(name=state["job_name"])
        else:
            with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8") as stream:
                for key, request in keyed_requests:
                    stream.write(json.dumps(
                        {"key": key, "request": request_json(request)},
                        ensure_ascii=False,
                    ) + "\n")
                stream.flush()
                input_file = client.files.upload(
                    file=stream.name,
                    config={"mime_type": "jsonl", "display_name": f"caption-{digest}"},
                )
                input_file = wait_for_file(client, input_file)
            job = client.batches.create(
                model=model_id,
                src=input_file.name,
                config={"display_name": f"caption-{digest}"},
            )
            caption_shots.atomic_write_json({
                "request_sha256": digest,
                "job_name": job.name,
                "input_file": input_file.name,
                "keys": [key for key, _ in keyed_requests],
            }, state_path)
        while _batch_state_name(job) in {"JOB_STATE_QUEUED", "JOB_STATE_PENDING", "JOB_STATE_RUNNING"}:
            print(f"batch {job.name}: {_batch_state_name(job)}", flush=True)
            time.sleep(poll_seconds)
            job = client.batches.get(name=job.name)
        if _batch_state_name(job) not in {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"}:
            raise RuntimeError(f"batch {_batch_state_name(job)}: {job.error}")
        if not job.dest or not job.dest.file_name:
            raise RuntimeError("batch completed without an output file")
        output_file = client.files.get(name=job.dest.file_name)
        payload = client.files.download(file=output_file)
        lines = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
        results = {}
        expected_keys = [key for key, _ in keyed_requests]
        for index, line in enumerate(lines):
            key = line.get("key", expected_keys[index])
            if line.get("error"):
                raise RuntimeError(f"batch request {key} failed: {line['error']}")
            results[key] = parse_batch_response(
                line["response"], allow_truncated=allow_truncated,
            )
        missing = set(expected_keys) - set(results)
        if missing:
            raise RuntimeError(f"batch missing responses: {sorted(missing)}")
        caption_shots.atomic_write_json(results, result_path)
        completed = True
        return results
    finally:
        names = {
            state.get("input_file"),
            getattr(input_file, "name", None),
            getattr(output_file, "name", None),
        }
        for name in names:
            if not name:
                continue
            try:
                client.files.delete(name=name)
            except Exception:
                pass
        if not completed and state_path.exists():
            state_path.unlink()


def generate(client, model_id, images, prompt, *, thinking_level, max_output_tokens):
    contents = [
        *[types.Part.from_bytes(data=image, mime_type="image/jpeg") for image in images],
        prompt,
    ]
    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config={
                    "temperature": 0.0,
                    "thinking_config": {"thinking_level": thinking_level},
                    "media_resolution": "MEDIA_RESOLUTION_HIGH",
                    "max_output_tokens": max_output_tokens,
                },
            )
            break
        except Exception as error:
            if attempt == 4 or not _is_retryable(error):
                raise
            delay = _retry_delay_seconds(error)
            time.sleep(delay if delay is not None else 2 ** attempt)
    finish_reason = getattr(response.candidates[0], "finish_reason", None)
    if getattr(finish_reason, "name", finish_reason) == "MAX_TOKENS":
        raise RuntimeError("truncated Gemini response")
    text = " ".join((response.text or "").split())
    if not text:
        raise RuntimeError("empty Gemini response")
    usage = response.usage_metadata
    tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
    return text, tokens


def clean_ocr(text):
    return "; ".join(
        item.strip() for item in text.split(";")
        if item.strip()
        and not BROADCAST_OVERLAY.fullmatch(fold(item.strip()))
        and not CLOCK_OVERLAY.fullmatch(item.strip())
    )


def parse_combined_response(text):
    value = json.loads(text)
    if set(value) != {"caption", "ocr"}:
        raise ValueError("combined response must contain only caption and ocr")
    caption = " ".join(str(value["caption"]).split())
    ocr = clean_ocr(" ".join(str(value["ocr"]).split()))
    if not caption or not ocr:
        raise ValueError("combined response has an empty caption or ocr")
    return caption, ocr


def caption_one(client, record, data_dir, model_id, max_frames):
    frames = sample_video_frames(record, max_frames)
    started = time.monotonic()
    description, description_tokens = generate(
        client, model_id, frames, DESCRIPTION_PROMPT,
        thinking_level="minimal", max_output_tokens=DESCRIPTION_MAX_OUTPUT_TOKENS,
    )
    visible_text, text_tokens = generate(
        client, model_id, text_audit_images(frames), TEXT_PROMPT,
        thinking_level="low", max_output_tokens=4096,
    )
    visible_text = clean_ocr(visible_text)
    caption = f"{description} Visible scene text: {visible_text}"
    return {
        "year": record.year,
        "group": record.group,
        "video_id": record.video_id,
        "shot_index": record.shot_index,
        "start_frame": record.start_frame,
        "end_frame": record.end_frame,
        "start_sec": record.start_sec,
        "end_sec": record.end_sec,
        "duration_sec": record.duration_sec,
        "caption": caption,
        "ocr": visible_text,
        "model_id": model_id,
        "prompt_sha256": PROMPT_SHA256,
        "output_tokens": description_tokens + text_tokens,
        "elapsed_sec": time.monotonic() - started,
    }


def required_text_failures(rows):
    failures = []
    for row in rows:
        required = PILOT_REQUIRED_TEXT.get((row["video_id"], row["shot_index"]), ())
        haystack = fold(row["ocr"])
        missing = [text for text in required if fold(text) not in haystack]
        if missing:
            failures.append((row["video_id"], row["shot_index"], missing))
    return failures


def validate_required_text(rows):
    failures = required_text_failures(rows)
    if failures:
        raise RuntimeError(f"pilot required text missing: {failures}")


def pilot_records(data_dir: Path):
    wanted = set(PILOT_SHOTS)
    records = [
        record for year in ("2025", "2026")
        for record in caption_shots.load_shot_records(data_dir, year)
        if (record.video_id, record.shot_index) in wanted
    ]
    found = {(record.video_id, record.shot_index) for record in records}
    if found != wanted:
        raise ValueError(f"missing pilot shots: {sorted(wanted - found)}")
    return sorted(records, key=lambda row: PILOT_SHOTS.index((row.video_id, row.shot_index)))


def strategy_prompt_hash(strategy):
    description = COMPACT_DESCRIPTION_PROMPT if strategy.endswith("compact") else DESCRIPTION_PROMPT
    return hashlib.sha256((strategy + "\0" + description + "\0" + TEXT_PROMPT).encode()).hexdigest()


def build_experiment_requests(records, uploaded, frames_by_shot, targeted_by_shot):
    description_requests = []
    ocr_requests = []
    for record in records:
        key = (record.video_id, record.shot_index)
        frames = frames_by_shot[key]
        metadata = {"video_id": record.video_id, "shot_index": str(record.shot_index)}
        description_requests.append((
            request_key("frames-fixed", "description", record),
            make_request(
                image_parts(frames), DESCRIPTION_PROMPT,
                resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=DESCRIPTION_MAX_OUTPUT_TOKENS,
                metadata={**metadata, "strategy": "frames-fixed", "pass": "description"},
            ),
        ))
        for strategy, resolution, prompt in (
            ("video-default", "MEDIA_RESOLUTION_HIGH", DESCRIPTION_PROMPT),
            ("video-low", "MEDIA_RESOLUTION_LOW", DESCRIPTION_PROMPT),
            ("video-targeted-ocr-compact", "MEDIA_RESOLUTION_HIGH", COMPACT_DESCRIPTION_PROMPT),
        ):
            description_requests.append((
                request_key(strategy, "description", record),
                make_request(
                    [video_part(uploaded[record.video_id], record, resolution)], prompt,
                    resolution=resolution, max_output_tokens=DESCRIPTION_MAX_OUTPUT_TOKENS,
                    metadata={**metadata, "strategy": strategy, "pass": "description"},
                ),
            ))
        ocr_requests.extend((
            (
                request_key("shared", "ocr-fixed", record),
                make_request(
                    image_parts(text_audit_images(frames)), TEXT_PROMPT,
                    resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=4096,
                    thinking_level="low",
                    metadata={**metadata, "strategy": "shared", "pass": "ocr-fixed"},
                ),
            ),
            (
                request_key("shared", "ocr-targeted", record),
                make_request(
                    image_parts(targeted_by_shot[key]), TEXT_PROMPT,
                    resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=4096,
                    thinking_level="low",
                    metadata={**metadata, "strategy": "shared", "pass": "ocr-targeted"},
                ),
            ),
        ))
    return description_requests, ocr_requests


def _request_media_bytes(request):
    total = 0
    contents = request.contents if isinstance(request.contents, list) else [request.contents]
    for content in contents:
        for part in content.parts or []:
            if part.inline_data and part.inline_data.data:
                total += len(part.inline_data.data)
    return total


def experiment_rows(records, descriptions, ocr_results, request_lookup, uploaded):
    rows = {strategy: [] for strategy in STRATEGIES}
    shots_per_video = {}
    for record in records:
        shots_per_video[record.video_id] = shots_per_video.get(record.video_id, 0) + 1
    for record in records:
        for strategy in STRATEGIES:
            description_strategy = (
                "video-default" if strategy == "video-targeted-ocr" else strategy
            )
            description_key = request_key(description_strategy, "description", record)
            ocr_pass = "ocr-targeted" if "targeted-ocr" in strategy else "ocr-fixed"
            ocr_key = request_key("shared", ocr_pass, record)
            description = descriptions[description_key]
            ocr = ocr_results[ocr_key]
            description_text = description["text"]
            visible_text = clean_ocr(ocr["text"])
            caption = f"{description_text} Visible scene text: {visible_text}"
            requests = (request_lookup[description_key], request_lookup[ocr_key])
            upload_bytes = sum(_request_media_bytes(request) for request in requests)
            if strategy != "frames-fixed":
                upload_bytes += int(
                    (uploaded[record.video_id].size_bytes or 0) / shots_per_video[record.video_id]
                )
            rows[strategy].append({
                "year": record.year,
                "group": record.group,
                "video_id": record.video_id,
                "shot_index": record.shot_index,
                "start_frame": record.start_frame,
                "end_frame": record.end_frame,
                "start_sec": record.start_sec,
                "end_sec": record.end_sec,
                "duration_sec": record.duration_sec,
                "caption": caption,
                "ocr": visible_text,
                "model_id": MODEL_ID,
                "prompt_sha256": strategy_prompt_hash(strategy),
                "output_tokens": description["output_tokens"] + ocr["output_tokens"],
                "elapsed_sec": 0.0,
                "strategy": strategy,
                "api_mode": "batch",
                "input_tokens": description["input_tokens"] + ocr["input_tokens"],
                "thinking_tokens": description["thinking_tokens"] + ocr["thinking_tokens"],
                "truncated": description.get("truncated", False) or ocr.get("truncated", False),
                "request_count": 2,
                "image_count": sum(
                    1 for request in requests
                    for content in (
                        request.contents if isinstance(request.contents, list)
                        else [request.contents]
                    )
                    for part in content.parts or [] if part.inline_data
                ),
                "upload_bytes": upload_bytes,
            })
    return rows


def _load_retrieval_encoder(model_name="Qwen/Qwen3-Embedding-0.6B"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        model_name, device="cpu", local_files_only=True,
    )


def broadcast_overlay_leaks(rows):
    return [
        f"{row['video_id']}/{row['shot_index']}"
        for row in rows
        if BROADCAST_OVERLAY.search(fold(row["caption"]))
    ]


def temporal_action_failures(rows):
    requirements = {
        ("K07_V025", 225): (("contact", "touch"), ("car", "vehicle", "chassis")),
        ("K07_V025", 226): (("rotate", "turn", "spin"), ("wheel", "crank", "handle")),
        ("K07_V025", 227): (("fold",), ("cloth", "towel", "rag"), ("black",)),
    }
    failures = []
    for row in rows:
        groups = requirements.get((row["video_id"], row["shot_index"]))
        text = row["caption"].casefold()
        if groups and not all(any(word in text for word in group) for group in groups):
            failures.append(f"{row['video_id']}/{row['shot_index']}")
    return failures


def evaluate_strategies(rows_by_strategy, encoder=None):
    """Compare retrieval/OCR quality and Batch cost against fixed frames."""
    encoder = encoder or _load_retrieval_encoder()
    queries = [item["query"] for item in PILOT_QUERIES]
    query_vectors = np.asarray(encoder.encode(
        queries, prompt_name="query", normalize_embeddings=True,
    ))
    reports = {}
    for strategy, rows in rows_by_strategy.items():
        document_vectors = np.asarray(encoder.encode(
            [row["caption"] for row in rows], normalize_embeddings=True,
        ))
        scores = query_vectors @ document_vectors.T
        keys = [(row["video_id"], row["shot_index"]) for row in rows]
        ranks = []
        per_query = []
        for query, query_scores in zip(PILOT_QUERIES, scores):
            order = np.argsort(-query_scores)
            rank = min(
                int(np.where(order == keys.index(gold))[0][0]) + 1
                for gold in query["gold"]
            )
            ranks.append(rank)
            per_query.append({
                "query": query["query"],
                "gold": [f"{video_id}/{shot}" for video_id, shot in query["gold"]],
                "rank": rank,
                "top": f"{keys[int(order[0])][0]}/{keys[int(order[0])][1]}",
                "gold_score": float(max(
                    query_scores[keys.index(gold)] for gold in query["gold"]
                )),
            })
        required_count = matched_count = 0
        for row in rows:
            required = PILOT_REQUIRED_TEXT.get(
                (row["video_id"], row["shot_index"]), (),
            )
            required_count += len(required)
            haystack = fold(row["ocr"])
            matched_count += sum(fold(text) in haystack for text in required)
        input_tokens = sum(row["input_tokens"] for row in rows)
        output_tokens = sum(
            row["output_tokens"] + row["thinking_tokens"] for row in rows
        )
        pilot_cost = (
            input_tokens * BATCH_INPUT_USD_PER_MILLION
            + output_tokens * BATCH_OUTPUT_USD_PER_MILLION
        ) / 1_000_000
        reports[strategy] = {
            "recall_at_1": sum(rank <= 1 for rank in ranks) / len(ranks),
            "recall_at_3": sum(rank <= 3 for rank in ranks) / len(ranks),
            "mrr": sum(1 / rank for rank in ranks) / len(ranks),
            "per_query": per_query,
            "ocr_required_recall": matched_count / required_count,
            "ocr_required_matched": matched_count,
            "ocr_required_total": required_count,
            "broadcast_overlay_leaks": broadcast_overlay_leaks(rows),
            "mean_description_words": sum(
                len(row["caption"].partition("Visible scene text:")[0].split())
                for row in rows
            ) / len(rows),
            "input_tokens": input_tokens,
            "billed_output_tokens": output_tokens,
            "pilot_batch_cost_usd": pilot_cost,
            "projected_2026_batch_cost_usd": pilot_cost / len(rows) * FULL_2026_SHOT_COUNT,
            "mean_upload_mb_per_shot": sum(row["upload_bytes"] for row in rows)
            / len(rows) / 1_000_000,
            "temporal_action_failures": temporal_action_failures(rows),
            "truncated_shots": [
                f"{row['video_id']}/{row['shot_index']}"
                for row in rows if row.get("truncated", False)
            ],
        }
    baseline = reports["frames-fixed"]
    for report in reports.values():
        report["passes_quality_gate"] = (
            report["recall_at_1"] >= baseline["recall_at_1"]
            and report["recall_at_3"] >= baseline["recall_at_3"]
            and report["mrr"] >= baseline["mrr"]
            and report["ocr_required_recall"] == 1.0
            and report["mean_description_words"] >= .9 * baseline["mean_description_words"]
            and not report["temporal_action_failures"]
            and not report["truncated_shots"]
        )
        report["cost_saving_vs_baseline"] = 1 - (
            report["pilot_batch_cost_usd"] / baseline["pilot_batch_cost_usd"]
        )
    return reports


def run_strategy_benchmark(client, records, output_dir, max_frames=12):
    started = time.monotonic()
    verify_batch_access(client, MODEL_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_by_shot = {
        (record.video_id, record.shot_index): sample_video_frames(record, max_frames)
        for record in records
    }
    detector = _load_ocr_detector()
    targeted_by_shot = {}
    for record in records:
        key = (record.video_id, record.shot_index)
        targeted_by_shot[key] = targeted_text_audit_images(
            frames_by_shot[key], detector=detector,
        )
        print(f"detected OCR blocks {record.video_id}/{record.shot_index}: "
              f"{len(targeted_by_shot[key]) - 1}", flush=True)
    uploaded = upload_videos(client, records)
    try:
        description_requests, ocr_requests = build_experiment_requests(
            records, uploaded, frames_by_shot, targeted_by_shot,
        )
        descriptions = run_batch(
            client, MODEL_ID, description_requests, output_dir / "batch",
        )
        ocr_results = run_batch(
            client, MODEL_ID, ocr_requests, output_dir / "batch",
        )
        lookup = dict(description_requests + ocr_requests)
        rows_by_strategy = experiment_rows(
            records, descriptions, ocr_results, lookup, uploaded,
        )
        for strategy, rows in rows_by_strategy.items():
            path = output_dir / f"{strategy}.parquet"
            caption_shots.atomic_write_table(rows, EXPERIMENT_SCHEMA, path)
            caption_shots.atomic_write_json(rows, path.with_suffix(".json"))
        report = {
            "pilot_elapsed_sec": time.monotonic() - started,
            "shot_count": len(records),
            "strategies": evaluate_strategies(rows_by_strategy),
        }
        caption_shots.atomic_write_json(report, output_dir / "comparison.json")
        return rows_by_strategy
    finally:
        for file in uploaded.values():
            try:
                client.files.delete(name=file.name)
            except Exception:
                pass


def run_cost_benchmark(client, records, output_dir, baseline_path, max_frames=12,
                       embedding_model="Qwen/Qwen3-Embedding-0.6B"):
    """Find the cheapest video/OCR configuration that clears the pilot gates."""
    started = time.monotonic()
    verify_batch_access(client, MODEL_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(baseline_path.read_text())
    if {(row["video_id"], row["shot_index"]) for row in baseline} != set(PILOT_SHOTS):
        raise ValueError("baseline must contain exactly the ten pilot shots")
    for row in baseline:
        row["strategy"] = "frames-fixed"
        row.setdefault("truncated", False)

    frames_by_shot = {
        (record.video_id, record.shot_index): sample_video_frames(record, max_frames)
        for record in records
    }
    detector = _load_ocr_detector()
    target8, target4 = {}, {}
    for record in records:
        key = (record.video_id, record.shot_index)
        target8[key] = targeted_text_audit_images(
            frames_by_shot[key], detector=detector, max_blocks_per_frame=8,
        )
        target4[key] = targeted_text_audit_images(
            frames_by_shot[key], detector=detector, max_blocks_per_frame=4,
        )
        print(f"OCR crops {record.video_id}/{record.shot_index}: "
              f"target8={len(target8[key]) - 1} target4={len(target4[key]) - 1}", flush=True)

    uploaded = upload_videos(client, records)
    try:
        descriptions, ocr_requests, combined_requests = [], [], []
        request_lookup = {}
        for record in records:
            key = (record.video_id, record.shot_index)
            metadata = {"video_id": record.video_id, "shot_index": str(record.shot_index)}
            for name, resolution in (("video-high", "MEDIA_RESOLUTION_HIGH"),
                                     ("video-low", "MEDIA_RESOLUTION_LOW")):
                item = (
                    request_key(name, "description", record),
                    make_request(
                        [video_part(uploaded[record.video_id], record, resolution)],
                        DESCRIPTION_PROMPT, resolution=resolution,
                        max_output_tokens=DESCRIPTION_MAX_OUTPUT_TOKENS,
                        metadata={**metadata, "strategy": name, "pass": "description"},
                    ),
                )
                descriptions.append(item)
                request_lookup[item[0]] = item[1]
            for name, images in (("ocr-full-minimal", text_audit_images(frames_by_shot[key])),
                                 ("ocr-target8-minimal", target8[key]),
                                 ("ocr-target4-minimal", target4[key])):
                item = (
                    request_key("shared", name, record),
                    make_request(
                        image_parts(images), TEXT_PROMPT,
                        resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=4096,
                        thinking_level="minimal",
                        metadata={**metadata, "strategy": "shared", "pass": name},
                    ),
                )
                ocr_requests.append(item)
                request_lookup[item[0]] = item[1]
            item = (
                request_key("combined-video-high+ocr-target8-minimal", "combined", record),
                make_request(
                    [video_part(uploaded[record.video_id], record), *image_parts(target8[key])],
                    COMBINED_PROMPT, resolution="MEDIA_RESOLUTION_HIGH",
                    max_output_tokens=2048, thinking_level="minimal",
                    metadata={**metadata, "strategy": "combined", "pass": "combined"},
                ),
            )
            combined_requests.append(item)
            request_lookup[item[0]] = item[1]

        descriptions_result = run_batch(
            client, MODEL_ID, descriptions, output_dir / "batch",
        )
        ocr_result = run_batch(
            client, MODEL_ID, ocr_requests, output_dir / "batch",
            allow_truncated=True,
        )
        combined_result = run_batch(
            client, MODEL_ID, combined_requests, output_dir / "batch",
            allow_truncated=True,
        )
        rows_by_strategy = {"frames-fixed": baseline}
        shots_per_video = {}
        for record in records:
            shots_per_video[record.video_id] = shots_per_video.get(record.video_id, 0) + 1

        def row_for(record, strategy, description, ocr, requests):
            upload_bytes = sum(_request_media_bytes(request) for request in requests.requests)
            upload_bytes += int(
                (uploaded[record.video_id].size_bytes or 0) / shots_per_video[record.video_id]
            )
            return {
                "year": record.year, "group": record.group,
                "video_id": record.video_id, "shot_index": record.shot_index,
                "start_frame": record.start_frame, "end_frame": record.end_frame,
                "start_sec": record.start_sec, "end_sec": record.end_sec,
                "duration_sec": record.duration_sec,
                "caption": f"{description['text']} Visible scene text: {ocr}",
                "ocr": ocr, "model_id": MODEL_ID,
                "prompt_sha256": strategy_prompt_hash(strategy),
                "output_tokens": sum(value["output_tokens"] for value in requests.results),
                "elapsed_sec": 0.0, "strategy": strategy, "api_mode": "batch",
                "input_tokens": sum(value["input_tokens"] for value in requests.results),
                "thinking_tokens": sum(value["thinking_tokens"] for value in requests.results),
                "truncated": any(value.get("truncated", False) for value in requests.results),
                "request_count": len(requests.requests),
                "image_count": sum(
                    1 for request in requests.requests
                    for content in request.contents for part in content.parts or []
                    if part.inline_data
                ),
                "upload_bytes": upload_bytes,
            }

        from types import SimpleNamespace
        for strategy in COST_BENCHMARK_STRATEGIES[:-1]:
            description_name, ocr_name = strategy.split("+")
            rows = []
            for record in records:
                description_key = request_key(description_name, "description", record)
                ocr_key = request_key("shared", ocr_name, record)
                description = descriptions_result[description_key]
                ocr = clean_ocr(ocr_result[ocr_key]["text"])
                rows.append(row_for(
                    record, strategy, description, ocr,
                    SimpleNamespace(
                        requests=[request_lookup[description_key], request_lookup[ocr_key]],
                        results=[description, ocr_result[ocr_key]],
                    ),
                ))
            rows_by_strategy[strategy] = rows

        strategy = COST_BENCHMARK_STRATEGIES[-1]
        rows = []
        for record in records:
            response_key = request_key(strategy, "combined", record)
            response = combined_result[response_key]
            if response.get("truncated", False):
                description_text, ocr = response["text"], "None"
            else:
                try:
                    description_text, ocr = parse_combined_response(response["text"])
                except (json.JSONDecodeError, TypeError, ValueError):
                    response["truncated"] = True
                    description_text, ocr = response["text"], "None"
            request = request_lookup[response_key]
            rows.append(row_for(
                record, strategy, {**response, "text": description_text}, ocr,
                SimpleNamespace(requests=[request], results=[response]),
            ))
        rows_by_strategy[strategy] = rows

        for strategy, rows in rows_by_strategy.items():
            path = output_dir / f"{strategy}.parquet"
            caption_shots.atomic_write_table(rows, EXPERIMENT_SCHEMA, path)
            caption_shots.atomic_write_json(rows, path.with_suffix(".json"))
        encoder = _load_retrieval_encoder(embedding_model)
        reports = evaluate_strategies(rows_by_strategy, encoder=encoder)
        passing = [
            name for name in COST_BENCHMARK_STRATEGIES
            if reports[name]["passes_quality_gate"]
        ]
        two_pass = [name for name in passing if not name.startswith("combined-")]
        eligible = list(two_pass)
        if two_pass and COST_BENCHMARK_STRATEGIES[-1] in passing:
            best_two_cost = min(reports[name]["pilot_batch_cost_usd"] for name in two_pass)
            combined_name = COST_BENCHMARK_STRATEGIES[-1]
            if reports[combined_name]["pilot_batch_cost_usd"] <= .95 * best_two_cost:
                eligible.append(combined_name)
        selected = min(
            eligible, key=lambda name: reports[name]["pilot_batch_cost_usd"],
            default=None,
        )
        report = {
            "pilot_elapsed_sec": time.monotonic() - started,
            "shot_count": len(records), "embedding_model": embedding_model,
            "selected_strategy": selected, "strategies": reports,
        }
        caption_shots.atomic_write_json(report, output_dir / "comparison.json")
        return report
    finally:
        for file in uploaded.values():
            try:
                client.files.delete(name=file.name)
            except Exception:
                pass


def run_frames_fixed_pilot(client, records, output_dir, max_frames=12):
    """Cheap quality check: fixed frames plus the separate exhaustive OCR pass."""
    verify_batch_access(client, MODEL_ID)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_by_shot = {
        (record.video_id, record.shot_index): sample_video_frames(record, max_frames)
        for record in records
    }
    descriptions, ocr_requests = [], []
    for record in records:
        key = (record.video_id, record.shot_index)
        metadata = {"video_id": record.video_id, "shot_index": str(record.shot_index)}
        descriptions.append((
            request_key("frames-fixed-1024", "description", record),
            make_request(
                image_parts(frames_by_shot[key]), DESCRIPTION_PROMPT,
                resolution="MEDIA_RESOLUTION_HIGH",
                max_output_tokens=DESCRIPTION_MAX_OUTPUT_TOKENS,
                metadata={**metadata, "strategy": "frames-fixed-1024", "pass": "description"},
            ),
        ))
        ocr_requests.append((
            request_key("shared", "ocr-fixed", record),
            make_request(
                image_parts(text_audit_images(frames_by_shot[key])), TEXT_PROMPT,
                resolution="MEDIA_RESOLUTION_HIGH", max_output_tokens=4096,
                thinking_level="low",
                metadata={**metadata, "strategy": "shared", "pass": "ocr-fixed"},
            ),
        ))
    description_results = run_batch(client, MODEL_ID, descriptions, output_dir / "batch")
    ocr_results = run_batch(client, MODEL_ID, ocr_requests, output_dir / "batch")
    lookup = dict(descriptions + ocr_requests)
    rows = []
    for record in records:
        description_key = request_key("frames-fixed-1024", "description", record)
        ocr_key = request_key("shared", "ocr-fixed", record)
        description, ocr = description_results[description_key], ocr_results[ocr_key]
        visible_text = clean_ocr(ocr["text"])
        requests = (lookup[description_key], lookup[ocr_key])
        rows.append({
            "year": record.year, "group": record.group,
            "video_id": record.video_id, "shot_index": record.shot_index,
            "start_frame": record.start_frame, "end_frame": record.end_frame,
            "start_sec": record.start_sec, "end_sec": record.end_sec,
            "duration_sec": record.duration_sec,
            "caption": f"{description['text']} Visible scene text: {visible_text}",
            "ocr": visible_text, "model_id": MODEL_ID,
            "prompt_sha256": strategy_prompt_hash("frames-fixed-1024"),
            "output_tokens": description["output_tokens"] + ocr["output_tokens"],
            "elapsed_sec": 0.0, "strategy": "frames-fixed-1024", "api_mode": "batch",
            "input_tokens": description["input_tokens"] + ocr["input_tokens"],
            "thinking_tokens": description["thinking_tokens"] + ocr["thinking_tokens"],
            "truncated": description.get("truncated", False) or ocr.get("truncated", False),
            "request_count": 2,
            "image_count": sum(
                1 for request in requests for content in request.contents
                for part in content.parts or [] if part.inline_data
            ),
            "upload_bytes": sum(_request_media_bytes(request) for request in requests),
        })
    caption_shots.atomic_write_table(rows, EXPERIMENT_SCHEMA, output_dir / "captions.parquet")
    caption_shots.atomic_write_json(rows, output_dir / "captions.json")
    caption_shots.atomic_write_json({
        "shot_count": len(rows),
        "mean_description_words": sum(
            len(row["caption"].partition("Visible scene text:")[0].split())
            for row in rows
        ) / len(rows),
        "input_tokens": sum(row["input_tokens"] for row in rows),
        "output_tokens": sum(row["output_tokens"] + row["thinking_tokens"] for row in rows),
        "required_text_failures": required_text_failures(rows),
    }, output_dir / "summary.json")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=caption_shots.shots.DATA)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "artifacts/captions/gemini-3.1-flash-lite/pilot_shots.parquet"
    ))
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--benchmark-strategies", action="store_true")
    parser.add_argument("--cost-benchmark", action="store_true")
    parser.add_argument("--frames-fixed-pilot", action="store_true")
    parser.add_argument("--strategy-output-dir", type=Path, default=(
        ROOT / "artifacts/captions/gemini-3.1-flash-lite/strategy-benchmark"
    ))
    parser.add_argument("--frames-fixed-output-dir", type=Path, default=(
        ROOT / "artifacts/captions/gemini-3.1-flash-lite/frames-fixed-1024"
    ))
    parser.add_argument("--cost-output-dir", type=Path, default=(
        ROOT / "artifacts/captions/gemini-3.1-flash-lite/cost-benchmark"
    ))
    parser.add_argument("--baseline-json", type=Path, default=(
        ROOT / "artifacts/captions/gemini-3.1-flash-lite/frames-fixed-1024/captions.json"
    ))
    parser.add_argument(
        "--embedding-model", default="Qwen/Qwen3-Embedding-0.6B",
        help="local SentenceTransformer used only for pilot retrieval scoring",
    )
    args = parser.parse_args(argv)
    if args.max_frames < 1 or args.concurrency < 1:
        parser.error("max-frames and concurrency must be positive")
    if not os.environ.get("GEMINI_API_KEY"):
        parser.error("GEMINI_API_KEY is not set")

    records = pilot_records(args.data_dir)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    if args.cost_benchmark:
        run_cost_benchmark(
            client, records, args.cost_output_dir, args.baseline_json,
            max_frames=args.max_frames, embedding_model=args.embedding_model,
        )
        print(args.cost_output_dir.resolve())
        return 0
    if args.benchmark_strategies:
        run_strategy_benchmark(
            client, records, args.strategy_output_dir, max_frames=args.max_frames,
        )
        print(args.strategy_output_dir.resolve())
        return 0
    if args.frames_fixed_pilot:
        run_frames_fixed_pilot(
            client, records, args.frames_fixed_output_dir, max_frames=args.max_frames,
        )
        print(args.frames_fixed_output_dir.resolve())
        return 0
    rows = []
    if args.output.exists():
        rows = [
            row for row in caption_shots.pq.read_table(args.output).to_pylist()
            if row["model_id"] == args.model and row["prompt_sha256"] == PROMPT_SHA256
            and (row["video_id"], row["shot_index"]) in PILOT_SHOTS
        ]
        for row in rows:
            row.setdefault("ocr", row["caption"].partition("Visible scene text: ")[2])
    completed = {(row["video_id"], row["shot_index"]) for row in rows}
    records = [
        record for record in records
        if (record.video_id, record.shot_index) not in completed
    ]
    order = {key: index for index, key in enumerate(PILOT_SHOTS)}
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        pending = {
            pool.submit(caption_one, client, record, args.data_dir, args.model, args.max_frames): record
            for record in records
        }
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(f"captioned {row['video_id']}/{row['shot_index']}", flush=True)
            rows.sort(key=lambda item: order[(item["video_id"], item["shot_index"])])
            caption_shots.atomic_write_table(rows, GEMINI_CAPTION_SCHEMA, args.output)
            caption_shots.atomic_write_json(rows, args.output.with_suffix(".json"))
    rows.sort(key=lambda row: order[(row["video_id"], row["shot_index"])])
    validate_required_text(rows)
    caption_shots.atomic_write_table(rows, GEMINI_CAPTION_SCHEMA, args.output)
    caption_shots.atomic_write_json(rows, args.output.with_suffix(".json"))
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
