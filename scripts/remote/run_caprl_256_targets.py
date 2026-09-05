#!/usr/bin/env python3
import json
import os
from pathlib import Path
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from scripts.captioning.caption_shots import (
    CAPTION_SCHEMA, PROMPT, PROMPT_SHA256, load_shot_records, strip_timestamps,
)

TARGETS = [
    ("L27_V010", 83), ("L30_V078", 20), ("L30_V078", 21),
    ("L30_V072", 19), ("K17_V003", 102), ("K19_V016", 158),
    ("L23_V013", 21),
]
root = Path("/workspace/aic")
max_frames = int(os.environ.get("CAPRL_MAX_FRAMES", "32"))
max_tokens = int(os.environ.get("CAPRL_MAX_NEW_TOKENS", "256"))
output_label = os.environ.get("CAPRL_OUTPUT_LABEL", "256-targets")
records = load_shot_records(root / "data", "2025") + load_shot_records(root / "data", "2026")
lookup = {(record.video_id, record.shot_index): record for record in records}
selected = [lookup[target] for target in TARGETS]

os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
model_id = "internlm/CapRL-Video-4B"
started = time.perf_counter()
processor = AutoProcessor.from_pretrained(model_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, dtype=torch.float16).cuda().eval()
cold_start = time.perf_counter() - started
rows = []

for record in selected:
    shot_started = time.perf_counter()
    messages = [{"role": "user", "content": [
        {"type": "video", "video": record.video_path,
         "video_start": record.input_start_sec, "video_end": record.input_end_sec,
         "max_frames": max_frames},
        {"type": "text", "text": PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True,
        return_video_metadata=True,
    )
    videos, video_metadata = zip(*videos)
    inputs = processor(
        text=text, images=images, videos=list(videos),
        video_metadata=list(video_metadata), return_tensors="pt",
        do_resize=False, **video_kwargs,
    ).to("cuda")
    generated = model.generate(**inputs, max_new_tokens=max_tokens)
    output_ids = generated[0, inputs.input_ids.shape[1]:]
    caption = strip_timestamps(processor.decode(output_ids, skip_special_tokens=True).strip())
    rows.append({
        "year": record.year, "group": record.group, "video_id": record.video_id,
        "shot_index": record.shot_index, "start_frame": record.start_frame,
        "end_frame": record.end_frame, "start_sec": record.start_sec,
        "end_sec": record.end_sec, "duration_sec": record.duration_sec,
        "caption": caption, "model_id": model_id, "prompt_sha256": PROMPT_SHA256,
        "output_tokens": len(output_ids),
        "elapsed_sec": time.perf_counter() - shot_started,
    })
    print(f"{record.video_id} shot {record.shot_index}: {len(output_ids)} tokens", flush=True)

output = root / "artifacts/captions" / output_label
output.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(rows, schema=CAPTION_SCHEMA), output / "captions.parquet")
(output / "results.json").write_text(json.dumps({
    "backend": "transformers", "max_tokens": max_tokens, "max_frames": max_frames,
    "cold_start_seconds": cold_start,
    "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() // 1024**2,
    "shots": rows,
}, ensure_ascii=False, indent=2) + "\n")
