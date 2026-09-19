#!/usr/bin/env python3
"""Caption every 2026 shot with CapRL-Video-4B and map keyframes to shots.

The model reads a time slice of the original MP4 through qwen-vl-utils; no
temporary clip files are created. Outputs live under the gitignored artifacts
directory and are resumable one video at a time.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from backend import shots


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "internlm/CapRL-Video-4B"
DEFAULT_MAX_FRAMES = 32
DEFAULT_MAX_MODEL_LEN = 32_768
DEFAULT_VISION_TOKEN_BUDGET = 24_576
PROMPT = (
    "Describe only the filmed visual content of this video clip in English, in "
    "strictly chronological order. Be factual and conservative: state only what is "
    "directly visible, and do not guess identities, ages, relationships, locations, "
    "causes, or unreadable details. Cover people, actions, objects, interactions, "
    "setting, lighting, and camera changes. Transcribe all clearly legible text that "
    "is physically present in the filmed scene, such as signs, documents, labels, "
    "and product packaging; omit text that is blurred or uncertain. Never mention any "
    "broadcaster or program name, logo, watermark, or brandmark (including HTV, Tuổi "
    "Trẻ TV, and 60 Giây), even if it appears embedded in the filmed view. Also ignore "
    "lower-thirds, scrolling tickers, timestamps, scoreboards, and every other "
    "broadcast graphic or overlay. Return one concise, self-contained caption without "
    "headings or timestamps."
)
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode()).hexdigest()
TIMESTAMP = re.compile(
    r"\[\s*\d{1,2}:\d{2}(?::\d{2})?"
    r"(?:\s*[-–—]\s*\d{1,2}:\d{2}(?::\d{2})?)?\s*\]\s*"
)
DURATION_BINS = (
    ("le_2s", 0, 2),
    ("2_5s", 2, 5),
    ("5_15s", 5, 15),
    ("15_30s", 15, 30),
    ("30_120s", 30, 120),
    ("gt_120s", 120, float("inf")),
)

KEYFRAME_SCHEMA = pa.schema([
    ("year", pa.string()),
    ("group", pa.string()),
    ("video_id", pa.string()),
    ("frame_idx", pa.int32()),
    ("shot_index", pa.int32()),
])
CAPTION_SCHEMA = pa.schema([
    ("year", pa.string()),
    ("group", pa.string()),
    ("video_id", pa.string()),
    ("shot_index", pa.int32()),
    ("start_frame", pa.int32()),
    ("end_frame", pa.int32()),
    ("start_sec", pa.float64()),
    ("end_sec", pa.float64()),
    ("duration_sec", pa.float64()),
    ("caption", pa.string()),
    ("model_id", pa.string()),
    ("prompt_sha256", pa.string()),
    ("output_tokens", pa.int32()),
    ("elapsed_sec", pa.float64()),
])
# The window pipeline (scripts.captioning.caption_api) writes this instead. It is a
# superset rather than a replacement: CAPTION_SCHEMA is still the contract for the
# earlier in-process runner and for every parquet already under artifacts/captions.
# Primary key is (video_id, shot_index, window_index); an unsplit shot has
# window_index 0 and window_count 1, so its row means exactly what it meant before.
WINDOW_CAPTION_SCHEMA = pa.schema([
    ("year", pa.string()),
    ("group", pa.string()),
    ("video_id", pa.string()),
    ("shot_index", pa.int32()),
    ("window_index", pa.int32()),
    ("window_count", pa.int32()),
    ("start_frame", pa.int32()),
    ("end_frame", pa.int32()),
    ("start_sec", pa.float64()),
    ("end_sec", pa.float64()),
    ("duration_sec", pa.float64()),
    ("window_start_sec", pa.float64()),
    ("window_end_sec", pa.float64()),
    ("window_duration_sec", pa.float64()),
    ("frames_used", pa.int32()),
    ("frame_width", pa.int32()),
    ("summary", pa.string()),
    ("caption", pa.string()),
    ("onscreen_text", pa.string()),
    ("raw_output", pa.string()),
    ("model_id", pa.string()),
    ("prompt_sha256", pa.string()),
    ("kv_cache_dtype", pa.string()),
    ("max_tokens", pa.int32()),
    ("prompt_tokens", pa.int32()),
    ("output_tokens", pa.int32()),
    ("total_tokens", pa.int32()),
    ("finish_reason", pa.string()),
    ("elapsed_sec", pa.float64()),
])


@dataclass(frozen=True)
class ShotRecord:
    year: str
    group: str
    video_id: str
    shot_index: int
    start_frame: int
    end_frame: int
    fps: float
    video_path: str

    @property
    def start_sec(self):
        return self.start_frame / self.fps

    @property
    def input_start_sec(self):
        # Keep ceil(start * fps) stable across floating-point round-off.
        return max(0.0, (self.start_frame - 0.25) / self.fps)

    @property
    def end_sec(self):
        return (self.end_frame + 1) / self.fps

    @property
    def input_end_sec(self):
        """Inclusive end timestamp expected by qwen-vl-utils video readers."""
        return (self.end_frame + 0.25) / self.fps

    @property
    def duration_sec(self):
        return (self.end_frame - self.start_frame + 1) / self.fps


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def duration_bucket(seconds):
    for name, low, high in DURATION_BINS:
        if low < seconds <= high or (low == 0 and seconds <= high):
            return name
    raise AssertionError(f"duration outside bins: {seconds}")


def load_shot_records(data_dir, year="2026", groups=()):
    data_dir, year = Path(data_dir), str(year)
    selected = set(groups)
    records = []
    for path in sorted((data_dir / year / "shots").glob("*/*.json")):
        group, video_id = path.parent.name, path.stem
        if selected and group not in selected:
            continue
        index = shots.load_path(path)
        if index.fps <= 0:
            raise ValueError(f"invalid fps in {path}")
        video = data_dir / year / "videos" / group / f"{video_id}.mp4"
        if not video.exists():
            raise FileNotFoundError(f"video missing for {path}: {video}")
        records.extend(
            ShotRecord(
                year, group, video_id, shot_index, int(start), int(end),
                index.fps, str(video.resolve()),
            )
            for shot_index, (start, end) in enumerate(index.bounds)
        )
    if not records:
        raise ValueError(f"no shots found under {data_dir / year / 'shots'}")
    return records


def records_by_video(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record.video_id].append(record)
    return dict(grouped)


def select_shard(records, shard_count, shard_index):
    """Split whole videos across workers while balancing the shot count."""
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("shard index must be in [0, shard count)")
    grouped = records_by_video(records)
    loads = [0] * shard_count
    assignments = [set() for _ in range(shard_count)]
    for video_id, video_records in sorted(
        grouped.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        target = min(range(shard_count), key=lambda index: (loads[index], index))
        assignments[target].add(video_id)
        loads[target] += len(video_records)
    selected = assignments[shard_index]
    return [record for record in records if record.video_id in selected]


def build_keyframe_mapping(data_dir, year, records):
    """Return keyframe rows and validate that every selected shot is represented."""
    data_dir = Path(data_dir)
    rows, ignored_temporary = [], 0
    populated = Counter()
    for video_id, video_records in sorted(records_by_video(records).items()):
        first = video_records[0]
        directory = data_dir / str(year) / "keyframes" / first.group / video_id
        if not directory.is_dir():
            raise FileNotFoundError(f"keyframes missing: {directory}")
        bounds = np.asarray(
            [(record.start_frame, record.end_frame) for record in video_records],
            dtype=np.int64,
        )
        index = shots.ShotIndex(first.fps, bounds)
        for path in sorted(directory.iterdir()):
            if path.name.endswith(".tmp.png"):
                ignored_temporary += 1
                continue
            if path.suffix.lower() != ".png":
                continue
            try:
                frame_idx = int(path.stem)
            except ValueError as error:
                raise ValueError(f"invalid keyframe name: {path}") from error
            shot_index = index.shot_at(frame_idx)
            if shot_index is None:
                raise ValueError(f"keyframe is outside every shot: {path}")
            rows.append({
                "year": str(year), "group": first.group, "video_id": video_id,
                "frame_idx": frame_idx, "shot_index": shot_index,
            })
            populated[(video_id, shot_index)] += 1
    missing = [record for record in records
               if (record.video_id, record.shot_index) not in populated]
    if missing:
        first = missing[0]
        raise ValueError(
            f"{len(missing)} shots have no keyframes; first is "
            f"{first.video_id}/{first.shot_index}"
        )
    summary = {
        "shot_count": len(records),
        "keyframe_count": len(rows),
        "ignored_temporary": ignored_temporary,
        "keyframes_per_shot": dict(sorted(Counter(populated.values()).items())),
    }
    return rows, summary


def atomic_write_table(rows, schema, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary)
    os.replace(temporary, path)


def atomic_write_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def select_benchmark_shots(records, count=60, seed=42):
    grouped = defaultdict(list)
    for record in records:
        grouped[duration_bucket(record.duration_sec)].append(record)
    active = [item for item in DURATION_BINS if grouped[item[0]]]
    if count < len(active):
        raise ValueError(f"benchmark needs at least {len(active)} shots")
    each, remainder = divmod(count, len(active))
    rng, selected = random.Random(seed), []
    for position, (name, _, _) in enumerate(active):
        wanted = each + int(position < remainder)
        if len(grouped[name]) < wanted:
            raise ValueError(f"duration bucket {name} has only {len(grouped[name])} shots")
        selected.extend(rng.sample(grouped[name], wanted))
    return selected


def strip_timestamps(caption):
    return " ".join(TIMESTAMP.sub("", caption).split())


def prepare_vllm_input(record, processor, max_frames=DEFAULT_MAX_FRAMES):
    from qwen_vl_utils import process_vision_info

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video", "video": record.video_path,
                "video_start": record.input_start_sec,
                "video_end": record.input_end_sec,
                "max_frames": max_frames,
            },
            {"type": "text", "text": PROMPT},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    multimodal = {}
    if image_inputs is not None:
        multimodal["image"] = image_inputs
    if video_inputs is not None:
        multimodal["video"] = video_inputs
    return {
        "prompt": prompt,
        "multi_modal_data": multimodal,
        "mm_processor_kwargs": video_kwargs,
    }


def create_model(model_id, seed, max_tokens, max_model_len, dtype,
                 tensor_parallel_size, gpu_memory_utilization):
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    processor = AutoProcessor.from_pretrained(model_id)
    engine = LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"video": 1},
        seed=seed,
    )
    return processor, engine, SamplingParams(max_tokens=max_tokens)


def caption_records(records, processor, engine, sampling, batch_size, model_id,
                    max_frames=DEFAULT_MAX_FRAMES):
    completed = []
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        started = time.perf_counter()
        inputs = [prepare_vllm_input(record, processor, max_frames) for record in batch]
        outputs = engine.generate(inputs, sampling_params=sampling, use_tqdm=False)
        elapsed = time.perf_counter() - started
        for record, output in zip(batch, outputs):
            completion = output.outputs[0]
            caption = strip_timestamps(completion.text.strip())
            if not caption:
                raise RuntimeError(f"empty caption for {record.video_id}/{record.shot_index}")
            completed.append({
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
                "model_id": model_id,
                "prompt_sha256": PROMPT_SHA256,
                "output_tokens": len(completion.token_ids),
                # vLLM exposes batch wall time, not an honest per-request split.
                "elapsed_sec": elapsed / len(batch),
            })
        log(f"captioned {min(offset + len(batch), len(records))}/{len(records)} shots")
    return completed


def query_gpu_used_mb():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        )
        return max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class GpuMonitor:
    def __init__(self, interval=0.5):
        self.interval = interval
        self.peak_mb = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            used = query_gpu_used_mb()
            if used is not None:
                self.peak_mb = max(self.peak_mb, used)

    def __enter__(self):
        used = query_gpu_used_mb()
        self.peak_mb = used or 0
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=2)
        used = query_gpu_used_mb()
        self.peak_mb = max(self.peak_mb, used or 0)


def benchmark_summary(records, rows, peak_gpu_mb, cold_start_sec, model_id,
                      max_frames, max_model_len, vision_token_budget, dtype,
                      tensor_parallel_size, gpu_memory_utilization):
    population = Counter(duration_bucket(record.duration_sec) for record in records)
    measurements = defaultdict(list)
    tokens = defaultdict(list)
    for row in rows:
        bucket = duration_bucket(row["duration_sec"])
        measurements[bucket].append(row["elapsed_sec"])
        tokens[bucket].append(row["output_tokens"])
    buckets, estimated_seconds = {}, 0.0
    for name, _, _ in DURATION_BINS:
        if not population[name]:
            continue
        mean_seconds = float(np.mean(measurements[name]))
        estimated_seconds += population[name] * mean_seconds
        buckets[name] = {
            "population_shots": population[name],
            "sample_shots": len(measurements[name]),
            "mean_seconds_per_shot": mean_seconds,
            "p50_seconds_per_shot": float(np.median(measurements[name])),
            "mean_output_tokens": float(np.mean(tokens[name])),
        }
    return {
        "status": "success",
        "model_id": model_id,
        "prompt": PROMPT,
        "prompt_sha256": PROMPT_SHA256,
        "frame_sampling": {"fps": 2, "min_frames": 4, "max_frames": max_frames},
        "max_model_len": max_model_len,
        "vision_token_budget": vision_token_budget,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "cold_start_seconds": cold_start_sec,
        "peak_gpu_memory_mb": peak_gpu_mb,
        "sample_shots": len(rows),
        "population_shots": len(records),
        "estimated_total_seconds": estimated_seconds,
        "estimated_total_hours": estimated_seconds / 3600,
        "buckets": buckets,
    }


def run_benchmark(records, output_dir, model_id, count, seed, batch_size, max_tokens,
                  max_frames, max_model_len, vision_token_budget, dtype,
                  tensor_parallel_size, gpu_memory_utilization):
    selected = select_benchmark_shots(records, count, seed)
    median_duration = float(np.median([record.duration_sec for record in records]))
    warmup = min(records, key=lambda record: abs(record.duration_sec - median_duration))
    monitor = GpuMonitor()
    started = time.perf_counter()
    try:
        with monitor:
            processor, engine, sampling = create_model(
                model_id, seed, max_tokens, max_model_len, dtype,
                tensor_parallel_size, gpu_memory_utilization,
            )
            cold_start = time.perf_counter() - started
            log(f"model ready in {cold_start:.1f}s; warming up on {warmup.video_id}")
            caption_records(
                [warmup], processor, engine, sampling, 1, model_id, max_frames
            )
            rows = caption_records(
                selected, processor, engine, sampling, batch_size, model_id, max_frames
            )
    except Exception as error:
        summary = {
            "status": "failed",
            "model_id": model_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": monitor.peak_mb,
        }
        atomic_write_json(summary, Path(output_dir) / "benchmark.json")
        raise
    atomic_write_table(rows, CAPTION_SCHEMA, Path(output_dir) / "benchmark_shots.parquet")
    summary = benchmark_summary(
        records, rows, monitor.peak_mb, cold_start, model_id, max_frames,
        max_model_len, vision_token_budget, dtype, tensor_parallel_size,
        gpu_memory_utilization,
    )
    atomic_write_json(summary, Path(output_dir) / "benchmark.json")
    return summary


def part_is_complete(path, expected):
    if not path.exists():
        return False
    table = pq.read_table(path, columns=["video_id", "shot_index"])
    rows = table.to_pylist()
    return [(row["video_id"], row["shot_index"]) for row in rows] == [
        (record.video_id, record.shot_index) for record in expected
    ]


def run_all(records, output_dir, model_id, seed, batch_size, max_tokens,
            max_frames, max_model_len, dtype, tensor_parallel_size,
            gpu_memory_utilization):
    processor, engine, sampling = create_model(
        model_id, seed, max_tokens, max_model_len, dtype,
        tensor_parallel_size, gpu_memory_utilization,
    )
    parts = Path(output_dir) / "parts"
    for video_id, video_records in sorted(records_by_video(records).items()):
        path = parts / video_records[0].group / f"{video_id}.parquet"
        if part_is_complete(path, video_records):
            log(f"resume: {video_id} already complete")
            continue
        rows = caption_records(
            video_records, processor, engine, sampling, batch_size, model_id,
            max_frames,
        )
        atomic_write_table(rows, CAPTION_SCHEMA, path)
    tables = []
    for video_id, video_records in sorted(records_by_video(records).items()):
        path = parts / video_records[0].group / f"{video_id}.parquet"
        if not part_is_complete(path, video_records):
            raise RuntimeError(f"caption part incomplete: {path}")
        tables.append(pq.read_table(path, schema=CAPTION_SCHEMA))
    table = pa.concat_tables(tables)
    if table.num_rows != len(records):
        raise RuntimeError(f"expected {len(records)} captions, got {table.num_rows}")
    target = Path(output_dir) / "shot_captions.parquet"
    temporary = target.with_name(f".{target.name}.tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, target)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=shots.DATA)
    parser.add_argument("--year", default="2026")
    parser.add_argument("--groups", default="", help="comma-separated groups")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "artifacts" / "captions" / "2026",
    )
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument(
        "--vision-token-budget", type=int, default=DEFAULT_VISION_TOKEN_BUDGET,
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-shots", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument(
        "--skip-keyframe-mapping", action="store_true",
        help="skip keyframe artifacts when captioning on a video-only worker",
    )
    args = parser.parse_args(argv)
    if (args.batch_size < 1 or args.max_tokens < 1 or args.max_frames < 2
            or args.max_model_len < 1 or args.vision_token_budget < 1
            or args.tensor_parallel_size < 1 or args.benchmark_shots < 0):
        parser.error("batch size/max tokens must be positive; benchmark shots cannot be negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("gpu memory utilization must be in (0, 1]")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")
    return args


def main(argv=None):
    args = parse_args(argv)
    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    records = load_shot_records(args.data_dir, args.year, groups)
    if args.shard_count > 1:
        records = select_shard(records, args.shard_count, args.shard_index)
        args.output_dir /= f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
        log(f"selected shard {args.shard_index}/{args.shard_count}: {len(records):,} shots")
    if not args.skip_keyframe_mapping:
        mapping, mapping_summary = build_keyframe_mapping(args.data_dir, args.year, records)
        atomic_write_table(mapping, KEYFRAME_SCHEMA, args.output_dir / "keyframe_shots.parquet")
        atomic_write_json(mapping_summary, args.output_dir / "mapping_summary.json")
        log(f"mapped {len(mapping):,} keyframes to {len(records):,} shots")
    if args.map_only:
        return 0

    # Decord seeks directly to shot ranges; torchvision would decode each range eagerly.
    os.environ.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    os.environ.setdefault("MODEL_SEQ_LEN", str(args.vision_token_budget))
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.benchmark_shots:
        summary = run_benchmark(
            records, args.output_dir, args.model, args.benchmark_shots,
            args.seed, args.batch_size, args.max_tokens,
            args.max_frames, args.max_model_len, args.vision_token_budget,
            args.dtype, args.tensor_parallel_size, args.gpu_memory_utilization,
        )
        log(f"estimated full run: {summary['estimated_total_hours']:.1f} hours")
    else:
        run_all(
            records, args.output_dir, args.model, args.seed, args.batch_size,
            args.max_tokens, args.max_frames, args.max_model_len, args.dtype,
            args.tensor_parallel_size, args.gpu_memory_utilization,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
