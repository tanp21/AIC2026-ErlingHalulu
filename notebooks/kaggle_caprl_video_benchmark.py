"""Kaggle T4x2 benchmark for CapRL-Video-4B on Batch 1 (2026)."""
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile


OUTPUT = Path("/kaggle/working/caprl_benchmark.json")
MODEL = "internlm/CapRL-Video-4B"
MAX_FRAMES = 32
VISION_TOKEN_BUDGET = 24_576
SHOT_DATASET_URL = ""
SAMPLE_MANIFEST = []
POPULATION = {}
PROMPT = (
    "Describe this video clip in English in great detail and in strictly chronological "
    "order. Cover all visible people and their appearance, objects, actions and "
    "interactions, setting, lighting, camera changes, and readable on-screen text. "
    "Return only one self-contained caption without headings or timestamps."
)
BINS = (("le_2s", 0, 2), ("2_5s", 2, 5), ("5_15s", 5, 15),
        ("15_30s", 15, 30), ("30_120s", 30, 120),
        ("gt_120s", 120, float("inf")))
TIMESTAMP = re.compile(
    r"\[\s*\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[-–—]\s*\d{1,2}:\d{2}(?::\d{2})?)?\s*\]\s*"
)


def save(value):
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(OUTPUT)


def bucket(seconds):
    return next(name for name, low, high in BINS
                if (low < seconds <= high) or (low == 0 and seconds <= high))


def load_records():
    if SAMPLE_MANIFEST:
        directory = Path("/tmp/videos")
        directory.mkdir(parents=True, exist_ok=True)
        for item in SAMPLE_MANIFEST:
            path = directory / f"{item['video_id']}.mp4"
            if not path.exists():
                urllib.request.urlretrieve(item["url"], path)
            item["video_path"] = str(path)
        return SAMPLE_MANIFEST

    import kagglehub

    video_root = Path(kagglehub.dataset_download("tommr211/aic-batch1-videos"))
    try:
        shot_root = Path(kagglehub.dataset_download("tommr211/aic-shot-intervals"))
    except Exception:
        if not SHOT_DATASET_URL:
            raise
        archive, shot_root = Path("/tmp/shots.zip"), Path("/tmp/shots")
        urllib.request.urlretrieve(SHOT_DATASET_URL, archive)
        with zipfile.ZipFile(archive) as source:
            source.extractall(shot_root)
    videos = {
        path.stem: path
        for path in video_root.rglob("*.mp4")
    }
    records = []
    shot_files = [path for path in shot_root.rglob("*.json")
                  if "batch1_2026_shots" in path.parts]
    for path in sorted(shot_files):
        data = json.loads(path.read_text())
        video_id, fps = path.stem, float(data["fps"])
        if video_id not in videos:
            raise FileNotFoundError(f"missing video: {video_id}")
        for shot_index, (start, end) in enumerate(data["shots"]):
            records.append({
                "video_id": video_id,
                "shot_index": shot_index,
                "start_frame": int(start),
                "end_frame": int(end),
                "fps": fps,
                "duration_sec": (end - start + 1) / fps,
                "video_path": str(videos[video_id]),
            })
    if len(videos) != 873 or len(records) != 96_624:
        raise RuntimeError(
            f"unexpected corpus: {len(videos)} videos, {len(records)} shots; "
            f"roots={[str(video_root), str(shot_root)]}"
        )
    return records


def sample_records(records, count=60, seed=42):
    grouped = {name: [] for name, _, _ in BINS}
    for record in records:
        grouped[bucket(record["duration_sec"])].append(record)
    each, remainder = divmod(count, len(BINS))
    rng, selected = random.Random(seed), []
    for position, (name, _, _) in enumerate(BINS):
        selected += rng.sample(grouped[name], each + (position < remainder))
    return selected


def prepare(record, processor):
    from qwen_vl_utils import process_vision_info

    messages = [{"role": "user", "content": [{
        "type": "video",
        "video": record["video_path"],
        "video_start": max(0, (record["start_frame"] - .25) / record["fps"]),
        "video_end": (record["end_frame"] + .25) / record["fps"],
        "max_frames": MAX_FRAMES,
    }, {"type": "text", "text": PROMPT}]}]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images, videos, kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    multimodal = {}
    if images is not None:
        multimodal["image"] = images
    if videos is not None:
        multimodal["video"] = videos
    return {"prompt": prompt, "multi_modal_data": multimodal,
            "mm_processor_kwargs": kwargs}


def main():
    records = load_records()
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "decord==0.6.0", "qwen-vl-utils==0.0.14",
        "transformers==5.15.1", "vllm==0.28.0",
    ], check=True)
    os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"
    os.environ["HF_HOME"] = "/tmp/huggingface"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # qwen-vl-utils uses this budget to resize long videos before tokenization.
    os.environ["MODEL_SEQ_LEN"] = str(VISION_TOKEN_BUDGET)

    import torch
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    gpu = [{"name": torch.cuda.get_device_name(i),
            "vram_bytes": torch.cuda.get_device_properties(i).total_memory}
           for i in range(torch.cuda.device_count())]
    if len(gpu) != 2:
        raise RuntimeError(f"expected Kaggle T4x2, found {gpu}")

    selected = records if SAMPLE_MANIFEST else sample_records(records)
    population = POPULATION or {
        name: sum(bucket(r["duration_sec"]) == name for r in records)
        for name, _, _ in BINS
    }
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL)
    engine = LLM(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=2,
        max_model_len=32_768,
        gpu_memory_utilization=.80,
        limit_mm_per_prompt={"video": 1},
        seed=42,
    )
    load_seconds = time.perf_counter() - started
    sampling = SamplingParams(max_tokens=1024)

    rows = []
    for offset in range(0, len(selected)):
        batch = selected[offset:offset + 1]
        batch_started = time.perf_counter()
        outputs = engine.generate(
            [prepare(record, processor) for record in batch],
            sampling_params=sampling,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - batch_started
        for record, output in zip(batch, outputs):
            completion = output.outputs[0]
            rows.append({
                **{key: record[key] for key in
                   ("video_id", "shot_index", "duration_sec")},
                "bucket": bucket(record["duration_sec"]),
                "elapsed_sec": elapsed / len(batch),
                "output_tokens": len(completion.token_ids),
                "caption": " ".join(TIMESTAMP.sub("", completion.text).split()),
            })
        save({"status": "running", "gpu": gpu, "model_load_seconds": load_seconds,
              "completed_shots": len(rows), "sample_shots": len(selected), "rows": rows})
        print(f"captioned {len(rows)}/{len(selected)}", flush=True)

    stats, estimated_seconds = {}, 0.0
    for name, _, _ in BINS:
        values = [row for row in rows if row["bucket"] == name]
        if not values:
            continue
        mean = sum(row["elapsed_sec"] for row in values) / len(values)
        estimated_seconds += population[name] * mean
        stats[name] = {
            "population_shots": population[name],
            "sample_shots": len(values),
            "mean_seconds_per_shot": mean,
            "mean_output_tokens": sum(row["output_tokens"] for row in values) / len(values),
        }
    save({
        "status": "success", "model": MODEL, "gpu": gpu,
        "precision": "float16", "tensor_parallel_size": 2,
        "max_model_len": 32_768, "gpu_memory_utilization": .80,
        "max_frames": MAX_FRAMES, "vision_token_budget": VISION_TOKEN_BUDGET,
        "model_load_seconds": load_seconds,
        "sample_shots": len(rows), "population_shots": sum(population.values()),
        "estimated_total_seconds": estimated_seconds,
        "estimated_total_hours": estimated_seconds / 3600,
        "buckets": stats, "rows": rows,
    })


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        result = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
        result.update({"status": "failed", "error_type": type(error).__name__,
                       "error": str(error), "traceback": traceback.format_exc()})
        save(result)
        raise
