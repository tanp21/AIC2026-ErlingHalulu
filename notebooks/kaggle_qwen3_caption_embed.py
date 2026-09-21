#!/usr/bin/env python3
"""Embed every 2026 shot caption with Qwen3-Embedding on Kaggle T4x2.

Self-contained script notebook, same contract as ``notebooks/kaggle_siglip2_embed.py``:
configuration comes from environment variables, the caption parquet is mounted
from ``/kaggle/input``, and output shards land in ``/kaggle/working/embeddings/<run>/``
as ``emb_*.npy`` + ``meta.parquet`` + ``info.json`` + ``checksums.json``.

Runs (AIC_RUN):

    qwen3-8b-fp16   Qwen3-Embedding-8B, fp16, both T4s (accelerate device_map,
                    falling back to a layer split if accelerate misdispatches)
    qwen3-4b-fp16   Qwen3-Embedding-4B, fp16, one model per T4 (2 workers)
    qwen3-4b-int8   Qwen3-Embedding-4B, bitsandbytes int8, one per T4 (2 workers)

Captions are documents: they are encoded with **no instruction prefix**. Queries
(the 137 official cases, when AIC_EVAL=1) are encoded with ``INSTRUCTION`` in the
Qwen3 ``Instruct: ...\\nQuery: `` format; both conventions are recorded in info.json.

Modes (flag or AIC_MODE):

    --self-test   pure logic test, no GPU, no model download
    --dry-run     token statistics and batch plan for the run, no GPU
    --smoke       encode 8 captions with AIC_SMOKE_MODEL (default 0.6B) through
                  the single-device and layer-split paths and compare them

Env: AIC_RUN, AIC_CANARY=1 (first 512 units only), AIC_EVAL=1, AIC_TOKEN_BUDGET,
AIC_BATCH_CAP, AIC_WORKERS (override the run's worker count), AIC_SHARD_INDEX,
AIC_SHARD_COUNT, AIC_RESUME=1, AIC_ALLOW_LOCAL=1, AIC_CAPTIONS_PATH,
AIC_QUERIES_PATH, AIC_ANSWERS_PATH, AIC_FPS_PATH, AIC_WORK_DIR, AIC_MODE.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import numpy as np


MAX_SEQ_LEN = 2048
PART_SHARD_SIZE = 20_000
OUTPUT_SHARD_SIZE = 50_000
EXPECTED_ROWS = 103_086
EXPECTED_VIDEOS = 873
TOLERANCE_SECONDS = 10
EVAL_K = 1000
INSTRUCTION = (
    "Given an English description of a moment inside a video, retrieve the shot "
    "caption that best describes it."
)
RUNS = {
    "qwen3-8b-fp16": {
        "model": "Qwen/Qwen3-Embedding-8B", "dim": 4096, "workers": 1,
        "quant": None, "device_mode": "shard", "token_budget": 4096,
    },
    "qwen3-4b-fp16": {
        "model": "Qwen/Qwen3-Embedding-4B", "dim": 2560, "workers": 2,
        "quant": None, "device_mode": "single", "token_budget": 8192,
    },
    "qwen3-4b-int8": {
        "model": "Qwen/Qwen3-Embedding-4B", "dim": 2560, "workers": 2,
        "quant": "int8", "device_mode": "single", "token_budget": 8192,
    },
}
META_COLUMNS = ("group", "video_id", "shot_index", "window_index", "window_count",
                "window_start_sec", "window_end_sec")


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_duration(seconds):
    seconds = int(max(seconds, 0))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_tokens(count):
    return f"{count / 1e6:.1f}M" if count >= 1e6 else f"{count / 1e3:.1f}k"


def on_kaggle():
    return Path("/kaggle/input").exists() or Path("/kaggle/working").exists()


def version_tuple(text):
    parts = []
    for piece in str(text).split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def ensure_dependencies(spec):
    import importlib.util

    import transformers

    missing = []
    if version_tuple(transformers.__version__) < (4, 51):
        missing.append("transformers>=4.51,<6")
    # No CUDA call here: the parent process must stay CUDA-free until mp.spawn,
    # otherwise forked GPU workers fail to re-initialize the context.
    needs_accelerate = bool(spec["quant"]) or spec["device_mode"] == "shard"
    if needs_accelerate and importlib.util.find_spec("accelerate") is None:
        missing.append("accelerate>=0.30")
    if spec["quant"] and importlib.util.find_spec("bitsandbytes") is None:
        missing.append("bitsandbytes>=0.45")
    if missing and importlib.util.find_spec("pip") is not None:
        log(f"installing {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
        if any(package.startswith("transformers") for package in missing):
            raise SystemExit("transformers was upgraded; restart the session so the new "
                             "version is imported, then re-run")
    elif missing:
        log(f"warning: {missing} unavailable and pip is missing; continuing without them")
    log(f"transformers {transformers.__version__} ready")


def ensure_model_cached(model_id):
    """Download the model once in the parent so both workers read the same cache."""
    from huggingface_hub import snapshot_download

    log(f"pre-downloading {model_id} into the Hugging Face cache")
    path = snapshot_download(model_id)
    log(f"model cached at {path}")


def find_input(name, env_key, root=None):
    override = os.environ.get(env_key)
    if override:
        return Path(override)
    # os.walk(followlinks=True) + resolve() dedup so a dataset exposed both as
    # /kaggle/input/<slug> and /kaggle/input/datasets/<owner>/<slug> counts once.
    root = Path(root or os.environ.get("AIC_INPUT_ROOT", "/kaggle/input"))
    seen = {}
    for current, _, files in os.walk(root, followlinks=True):
        if name in files:
            path = Path(current) / name
            seen.setdefault(path.resolve(), path)
    matches = sorted(seen.values())
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, found {matches}")
    return matches[0]


def load_rows(captions_path):
    import pyarrow.parquet as pq

    columns = list(META_COLUMNS) + ["caption"]
    rows = pq.read_table(captions_path, columns=columns).to_pylist()
    for index, row in enumerate(rows):
        row["index"] = index
    return rows


def validate_rows(rows, strict=True):
    keys = {(row["video_id"], row["shot_index"], row["window_index"]) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("duplicate (video_id, shot_index, window_index) rows")
    if strict and len(rows) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} caption units, got {len(rows)}")
    videos = {row["video_id"] for row in rows}
    if strict and len(videos) != EXPECTED_VIDEOS:
        raise RuntimeError(f"expected {EXPECTED_VIDEOS} videos, got {len(videos)}")


def greedy_batches(order, lengths, token_budget, batch_cap):
    batches, current, current_max = [], [], 0
    for raw_index in order:
        index = int(raw_index)
        length = int(lengths[index])
        grown = max(current_max, length) * (len(current) + 1)
        if current and (grown > token_budget or len(current) >= batch_cap):
            batches.append(np.asarray(current, dtype=np.int64))
            current, current_max = [], 0
        current.append(index)
        current_max = max(current_max, length)
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


def partition_by_cost(costs, workers):
    total = float(sum(costs))
    bounds = [0]
    running, target = 0, 1
    for index, cost in enumerate(costs):
        running += cost
        if target < workers and running >= total * target / workers:
            bounds.append(index + 1)
            target += 1
    bounds.append(len(costs))
    return [(bounds[position], bounds[position + 1]) for position in range(workers)]


def shard_range(total, index, count):
    """Contiguous row range for one account in a sharded run; count slices cover all."""
    return total * index // count, total * (index + 1) // count


def window_rows(rows, canary, shard_index, shard_count):
    """The rows this process owns: the canary head, or one shard's contiguous slice.

    Every process must apply this window before partitioning between workers.
    Skipping it made sharded runs re-encode the corpus head: indices in each
    shard started at 0, so merge_parts could not place them.
    """
    if canary:
        return rows[:512]
    if shard_count > 1:
        start, end = shard_range(len(rows), shard_index, shard_count)
        return rows[start:end]
    return rows


class PartWriter:
    """Append-only sharded writer; resume skips indices already in meta.jsonl."""

    def __init__(self, root, dim, shard_size=PART_SHARD_SIZE, resume=False):
        self.root = Path(root)
        self.dim = dim
        self.shard_size = shard_size
        self.completed = set()
        self.shard = 0
        meta_path = self.root / "meta.jsonl"
        available = {int(path.stem.split("_")[1]) for path in self.root.glob("emb_*.npy")}
        if resume and meta_path.exists():
            # Only trust meta rows whose shard file survived the crash: a shard is
            # saved atomically at flush, so rows pointing at a missing shard were
            # still buffered and must be re-encoded.
            kept = [json.loads(line) for line in meta_path.read_text().splitlines()
                    if json.loads(line)["shard"] in available]
            self.completed = {record["index"] for record in kept}
            self.shard = max(available) + 1 if available else 0
            meta_path.write_text("".join(json.dumps(record) + "\n" for record in kept))
            self.meta = open(meta_path, "a", buffering=1)
        else:
            shutil.rmtree(self.root, ignore_errors=True)
            self.root.mkdir(parents=True, exist_ok=True)
            self.meta = open(meta_path, "w", buffering=1)
        self.buffer = np.empty((shard_size, dim), dtype=np.float32)
        self.fill = 0

    def add(self, vector, index):
        if index in self.completed:
            return
        self.buffer[self.fill] = vector
        self.meta.write(json.dumps({
            "index": int(index), "shard": self.shard, "row": int(self.fill),
        }) + "\n")
        self.fill += 1
        if self.fill == self.shard_size:
            self.flush()

    def flush(self):
        if not self.fill:
            return
        temporary = self.root / f".emb_{self.shard:06d}.tmp.npy"
        np.save(temporary, self.buffer[: self.fill])
        temporary.replace(self.root / f"emb_{self.shard:06d}.npy")
        self.shard += 1
        self.fill = 0

    def close(self):
        self.flush()
        self.meta.close()


def merge_parts(parts_dir, output_dir, dim, rows, rows_by_index):
    records = []
    for meta_path in sorted(Path(parts_dir).glob("part*/meta.jsonl")):
        for line in meta_path.read_text().splitlines():
            record = json.loads(line)
            record["part"] = meta_path.parent.name
            records.append(record)
    records.sort(key=lambda record: record["index"])
    indices = [record["index"] for record in records]
    if len(records) != len(rows) or indices != [row["index"] for row in rows]:
        raise RuntimeError(f"merged {len(records)} vectors for {len(rows)} rows; "
                           f"index coverage mismatch")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("emb_*.npy"):
        stale.unlink()
    buffer = np.empty((OUTPUT_SHARD_SIZE, dim), dtype=np.float32)
    fill = shard_index = 0
    cache_key, cache = None, None
    for record in records:
        key = (record["part"], record["shard"])
        if key != cache_key:
            cache = np.load(Path(parts_dir) / record["part"] / f"emb_{record['shard']:06d}.npy",
                            mmap_mode="r")
            cache_key = key
        buffer[fill] = cache[record["row"]]
        fill += 1
        if fill == OUTPUT_SHARD_SIZE:
            np.save(output_dir / f"emb_{shard_index:06d}.npy", buffer)
            shard_index += 1
            fill = 0
    if fill:
        np.save(output_dir / f"emb_{shard_index:06d}.npy", buffer[:fill])

    import pyarrow as pa
    import pyarrow.parquet as pq

    index = [record["index"] for record in records]
    meta = {"index": pa.array(index, type=pa.int32())}
    for column in META_COLUMNS:
        meta[column] = pa.array([rows_by_index[item][column] for item in index])
    pq.write_table(pa.table(meta), output_dir / "meta.parquet")

    summaries = []
    for path in sorted(Path(parts_dir).glob("part*/summary.json")):
        summaries.append(json.loads(path.read_text()))
    return summaries


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_checksums(output_dir):
    output_dir = Path(output_dir)
    checksums = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "checksums.json":
            checksums[path.name] = sha256_file(path)
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


def model_load_kwargs(spec):
    import torch
    import transformers
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(spec["model"])
    config.use_cache = False
    kwargs = {"config": config, "attn_implementation": "sdpa"}
    dtype_key = "dtype" if version_tuple(transformers.__version__)[0] >= 5 else "torch_dtype"
    kwargs[dtype_key] = torch.float16
    if spec["quant"] == "int8":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    return kwargs


def shard_layers(model, devices):
    """Layer-split fallback: wrap each decoder layer so it moves its inputs.

    ``Qwen3Model.forward`` then runs unmodified, which is why this survives
    transformers API drift better than rebuilding the forward by hand.
    """
    import torch

    class MoveToDevice(torch.nn.Module):
        def __init__(self, layer, device):
            super().__init__()
            self.layer = layer.to(device)
            self.device = device

        def forward(self, hidden_states, *args, **kwargs):
            hidden_states = hidden_states.to(self.device)
            moved = {}
            for key, value in kwargs.items():
                if torch.is_tensor(value):
                    moved[key] = value.to(self.device)
                elif isinstance(value, (tuple, list)):
                    moved[key] = type(value)(
                        item.to(self.device) if torch.is_tensor(item) else item
                        for item in value
                    )
                else:
                    moved[key] = value
            return self.layer(hidden_states, *args, **moved)

    layers = list(model.layers)
    wrapped = torch.nn.ModuleList()
    for index, layer in enumerate(layers):
        device = devices[min(index * len(devices) // len(layers), len(devices) - 1)]
        wrapped.append(MoveToDevice(layer, device))
    model.layers = wrapped
    model.embed_tokens.to(devices[0])
    rotary = getattr(model, "rotary_emb", None)
    if rotary is not None:
        rotary.to(devices[0])
    model.norm.to(devices[-1])
    return model


class HFEncoder:
    def __init__(self, spec, rank, token_budget, batch_cap, strategy=None,
                 devices=None, logger=log):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.spec = spec
        self.logger = logger
        self.token_budget = token_budget
        self.batch_cap = batch_cap
        self.dim = spec["dim"]
        self.tokenizer = AutoTokenizer.from_pretrained(spec["model"], padding_side="left")
        kwargs = model_load_kwargs(spec)
        strategy = strategy or spec["device_mode"]
        self.model = None
        self.strategy = strategy

        if spec["quant"]:
            device = os.environ.get("AIC_DEVICE") or f"cuda:{rank}"
            kwargs["device_map"] = {"": device}
            self.model = AutoModel.from_pretrained(spec["model"], **kwargs).eval()
            self.input_device = device
            self.strategy = f"single-{spec['quant']}"
        elif strategy == "single":
            # Load straight onto the GPU through accelerate when it is available:
            # a full CPU copy first doubles peak RAM per worker and both workers
            # start at once. Without accelerate (local dev), fall back to a copy.
            import importlib.util
            device = os.environ.get("AIC_DEVICE") or f"cuda:{rank}"
            if importlib.util.find_spec("accelerate") is not None:
                kwargs["device_map"] = {"": device}
                self.model = AutoModel.from_pretrained(spec["model"], **kwargs).eval()
            else:
                self.model = AutoModel.from_pretrained(spec["model"], **kwargs).eval()
                self.model = self.model.to(device)
            self.input_device = device
        elif torch.cuda.device_count() >= 2:
            try:
                kwargs["device_map"] = "auto"
                kwargs["max_memory"] = {index: "13GiB" for index in range(torch.cuda.device_count())}
                self.model = AutoModel.from_pretrained(spec["model"], **kwargs).eval()
                self.input_device = "cuda:0"
                if self._probe():
                    self.strategy = "accelerate"
                else:
                    raise RuntimeError("accelerate dispatch produced non-finite embeddings")
            except Exception as error:
                self.logger(f"accelerate sharding failed ({type(error).__name__}: {error}); "
                            f"falling back to layer split")
                self.model = None
                kwargs.pop("device_map")
                kwargs.pop("max_memory")
                gc.collect()
                torch.cuda.empty_cache()
        if self.model is None:
            devices = devices or [f"cuda:{index}" for index in range(torch.cuda.device_count())]
            if not devices:
                devices = ["cuda:0"]
            self.model = AutoModel.from_pretrained(spec["model"], **kwargs)
            self.model = shard_layers(self.model, devices).eval()
            self.input_device = devices[0]
            self.strategy = "layer-split"
            if not self._probe():
                raise RuntimeError("layer-split produced non-finite embeddings")

    def _forward(self, input_ids, attention_mask):
        import torch
        with torch.inference_mode():
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return output.last_hidden_state

    def _probe(self):
        try:
            vectors = self.encode(["a probe caption", "another probe caption"])
            return bool(np.isfinite(vectors).all()
                        and np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3))
        except Exception as error:
            self.logger(f"probe failed: {type(error).__name__}: {error}")
            return False

    def _lengths(self, texts):
        encoded = self.tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN,
                                 add_special_tokens=True)["input_ids"]
        return np.asarray([len(ids) for ids in encoded], dtype=np.int64)

    def encode(self, texts, prompt=""):
        import torch

        if prompt:
            texts = [prompt + text for text in texts]
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        lengths = self._lengths(texts)
        order = np.argsort(lengths, kind="stable")
        vectors = np.empty((len(texts), self.dim), dtype=np.float32)
        batches = greedy_batches(order, lengths, self.token_budget, self.batch_cap)
        try:
            from tqdm.auto import tqdm
            progress = tqdm(total=len(texts), desc=f"{self.strategy} encode", unit="unit")
        except ImportError:
            progress = None
        started, done, tokens_done = time.monotonic(), 0, 0
        total_tokens = int(lengths.sum())
        for number, indices in enumerate(batches, 1):
            self._encode_batch(texts, indices, vectors, progress)
            done += len(indices)
            tokens_done += int(lengths[indices].sum())
            if number % 25 == 0 or done == len(texts):
                elapsed = max(time.monotonic() - started, 1e-6)
                token_rate = tokens_done / elapsed
                remaining = (total_tokens - tokens_done) / max(token_rate, 1e-6)
                message = (f"{self.strategy}: {done}/{len(texts)} units "
                           f"({format_tokens(tokens_done)}/{format_tokens(total_tokens)} tok), "
                           f"{done / elapsed:.2f} units/s, {int(token_rate)} tok/s, "
                           f"eta {format_duration(remaining)}")
                if torch.cuda.is_available():
                    message += f", peak {torch.cuda.max_memory_allocated() / 2**30:.1f}G"
                (progress.write if progress else self.logger)(message)
        if progress:
            progress.close()
        return vectors

    def _encode_batch(self, texts, indices, vectors, progress):
        import torch

        queue = [np.asarray(indices, dtype=np.int64)]
        while queue:
            current = queue.pop(0)
            try:
                encoded = self.tokenizer([texts[index] for index in current], padding=True,
                                         truncation=True, max_length=MAX_SEQ_LEN,
                                         return_tensors="pt")
                hidden = self._forward(encoded["input_ids"].to(self.input_device),
                                       encoded["attention_mask"].to(self.input_device))
                pooled = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1)
                vectors[current] = pooled.cpu().numpy()
            except Exception as error:
                oom = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
                if not oom or len(current) == 1:
                    raise
                torch.cuda.empty_cache()
                middle = len(current) // 2
                queue = [current[:middle], current[middle:]] + queue
                continue
            if progress:
                progress.update(len(current))


def worker(rank, config):
    import torch

    device = os.environ.get("AIC_DEVICE") or f"cuda:{rank}"
    torch.cuda.set_device(device)
    rows = load_rows(config["captions_path"])
    rows = window_rows(rows, config["canary"], config["shard_index"], config["shard_count"])
    start, end = config["ranges"][rank]
    rows = rows[start:end]
    started = time.monotonic()
    encoder = HFEncoder(config["spec"], rank, config["token_budget"], config["batch_cap"],
                        strategy=config["strategies"][rank])
    writer = PartWriter(Path(config["parts_dir"]) / f"part{rank}", config["spec"]["dim"],
                        resume=config["resume"])
    pending = [row for row in rows if row["index"] not in writer.completed]
    if len(pending) != len(rows):
        log(f"rank {rank}: resuming, {len(rows) - len(pending)} units already written")
    vectors = encoder.encode([row["caption"] for row in pending])
    for row, vector in zip(pending, vectors):
        writer.add(vector, row["index"])
    writer.close()
    elapsed = max(time.monotonic() - started, 1e-6)
    summary = {
        "rank": rank, "strategy": encoder.strategy, "count": len(rows),
        "encoded": len(pending), "elapsed_sec": elapsed,
        "units_per_second": len(pending) / elapsed,
        "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    (Path(config["parts_dir"]) / f"part{rank}" / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    log(f"rank {rank}: {len(pending)} units in {elapsed / 60:.1f} min "
        f"({summary['units_per_second']:.2f} units/s, {encoder.strategy})")
    return encoder if config.get("keep_encoder") else None


def _pump_logs(rank, process, handle):
    for line in process.stdout:
        handle.write(line)
        print(f"[rank{rank}] {line}", end="", flush=True)
    process.stdout.close()


def launch_workers(workers, run, work):
    """Run one fresh subprocess per GPU. Fork would inherit any CUDA context from
    an earlier run in the same kernel, which is exactly what used to hang."""
    assert "__file__" in globals(), "worker launcher needs __file__; run this script as a file"
    script_path = str(Path(__file__).resolve())
    logs_dir = Path(work) / "logs" / run
    shutil.rmtree(logs_dir, ignore_errors=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    log(f"launching {workers} worker subprocesses; full logs in {logs_dir}")
    processes = []
    for rank in range(workers):
        child_env = dict(os.environ, AIC_RANK=str(rank), AIC_RANK_COUNT=str(workers))
        handle = open(logs_dir / f"rank{rank}.log", "w", buffering=1)
        process = subprocess.Popen(
            [sys.executable, script_path], env=child_env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)
        processes.append((rank, process, handle))
    threads = [threading.Thread(target=_pump_logs, args=item, daemon=True)
               for item in processes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for _, process, handle in processes:
        process.wait()
        handle.close()
    failed = {rank: process.returncode for rank, process, _ in processes
              if process.returncode != 0}
    if failed:
        raise RuntimeError(f"worker subprocesses failed: {failed}; inspect {logs_dir}")


def first_match(ordered, case, video_of, fps, starts, ends, tolerance, match_frames):
    for rank, index in enumerate(ordered, 1):
        video_id = video_of[index]
        for target_video, target_frame in case["targets"]:
            if target_video != video_id:
                continue
            if not match_frames:
                return rank
            target_sec = target_frame / fps[target_video]
            if starts[index] - tolerance <= target_sec <= ends[index] + tolerance:
                return rank
    return None


def metrics(ranks):
    found = [rank for rank in ranks if rank is not None]
    total = len(ranks)
    if not total:
        return {"cases": 0}
    return {
        "cases": total,
        "hits": len(found),
        "mrr": sum(1 / rank for rank in found) / total,
        **{f"recall_at_{k}": sum(rank <= k for rank in found) / total for k in (1, 5, 10, 100)},
    }


def run_eval(encoder, output_dir, spec, run):
    import pyarrow.parquet as pq

    queries_path = find_input("official_benchmark_queries_en_v2.json", "AIC_QUERIES_PATH")
    answers_path = find_input("official_benchmark_answers_v2.json", "AIC_ANSWERS_PATH")
    fps_path = find_input("video_fps.json", "AIC_FPS_PATH")
    cases = json.loads(queries_path.read_text())["cases"]
    answers = json.loads(answers_path.read_text())["cases"]
    if [case["id"] for case in cases] != [case["id"] for case in answers]:
        raise RuntimeError("query and answer case ids differ")
    fps = json.loads(fps_path.read_text())

    documents = np.vstack([np.load(path) for path in sorted(Path(output_dir).glob("emb_*.npy"))])
    if documents.shape != (EXPECTED_ROWS, spec["dim"]):
        raise RuntimeError(f"document vectors {documents.shape} do not match "
                           f"({EXPECTED_ROWS}, {spec['dim']})")
    meta = pq.read_table(Path(output_dir) / "meta.parquet").to_pylist()
    video_of = [row["video_id"] for row in meta]
    starts = np.asarray([row["window_start_sec"] for row in meta])
    ends = np.asarray([row["window_end_sec"] for row in meta])

    log(f"evaluating {len(cases)} cases against {documents.shape[0]} captions")
    queries = encoder.encode([case["text"] for case in cases],
                             prompt=f"Instruct: {INSTRUCTION}\nQuery: ")
    ranks, video_ranks = [], []
    for query in queries:
        scores = documents @ query
        k = min(EVAL_K, len(scores) - 1)
        top = np.argpartition(-scores, k)[: k + 1]
        ordered = top[np.argsort(-scores[top])]
        ranks.append(first_match(ordered, cases[len(ranks)], video_of, fps, starts, ends,
                                 TOLERANCE_SECONDS, True))
        video_ranks.append(first_match(ordered, cases[len(video_ranks)], video_of, fps, starts,
                                       ends, TOLERANCE_SECONDS, False))

    by_round = {}
    for name, values in (("caption_window", ranks), ("video_only", video_ranks)):
        rounds = {}
        for case, rank in zip(cases, values):
            rounds.setdefault(case["id"].split("/", 1)[0], []).append(rank)
        by_round[name] = {round_name: metrics(round_values)
                          for round_name, round_values in rounds.items()}
    report = {
        "run": run, "model": spec["model"], "dim": spec["dim"], "strategy": encoder.strategy,
        "query_instruction": INSTRUCTION, "document_instruction": "",
        "tolerance_seconds": TOLERANCE_SECONDS, "k": EVAL_K, "cases": len(cases),
        "metrics": {"caption_window": metrics(ranks), "video_only": metrics(video_ranks)},
        "by_round": by_round,
        "ranks": {case["id"]: rank for case, rank in zip(cases, ranks)},
        "video_ranks": {case["id"]: rank for case, rank in zip(cases, video_ranks)},
    }
    path = Path(output_dir) / f"eval_{run}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    log(f"eval caption-window: {report['metrics']['caption_window']}")
    log(f"eval video-only:     {report['metrics']['video_only']}")


def dry_run(captions_path, spec):
    from transformers import AutoTokenizer

    rows = load_rows(captions_path)
    validate_rows(rows, strict=False)
    tokenizer = AutoTokenizer.from_pretrained(spec["model"])
    lengths = np.asarray([
        len(ids) for ids in tokenizer([row["caption"] for row in rows], truncation=True,
                                      max_length=MAX_SEQ_LEN)["input_ids"]
    ])
    budgets = {name: run["token_budget"] for name, run in RUNS.items()}
    plans = {}
    for name, budget in budgets.items():
        order = np.argsort(lengths, kind="stable")
        batches = greedy_batches(order, lengths, budget, int(os.environ.get("AIC_BATCH_CAP", "64")))
        plans[name] = {"token_budget": budget, "batches": len(batches),
                       "mean_batch_units": round(len(rows) / max(len(batches), 1), 1)}
    report = {
        "captions": str(captions_path), "rows": len(rows),
        "videos": len({row["video_id"] for row in rows}),
        "tokens": {"total": int(lengths.sum()), "mean": round(float(lengths.mean()), 1),
                   **{f"p{p}": int(np.percentile(lengths, p)) for p in (50, 90, 95, 99)},
                   "max": int(lengths.max()), "truncated": int((lengths >= MAX_SEQ_LEN).sum())},
        "plans": plans,
    }
    print(json.dumps(report, indent=2))


def smoke():
    import torch

    model_id = os.environ.get("AIC_SMOKE_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    spec = {"model": model_id, "dim": 1024, "quant": None, "device_mode": "single"}
    texts = [f"probe caption number {index} with some words" for index in range(8)]
    devices = os.environ.get("AIC_SMOKE_DEVICES")
    devices = devices.split(",") if devices else (
        ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cuda:0", "cpu"])
    single = HFEncoder(spec, 0, 2048, 8, strategy="single", devices=devices)
    split = HFEncoder(spec, 0, 2048, 8, strategy="shard", devices=devices)
    reference = single.encode(texts)
    tested = split.encode(texts)
    cosine = (reference * tested).sum(axis=1)
    print(f"single strategy: {single.strategy}; split strategy: {split.strategy}")
    print(f"cosine min/mean: {cosine.min():.6f}/{cosine.mean():.6f}")
    assert cosine.min() > 0.999, cosine
    print("smoke OK")


def self_test():
    import tempfile

    lengths = np.asarray([3, 3, 3, 3, 3])
    batches = greedy_batches(np.arange(5), lengths, token_budget=10, batch_cap=64)
    assert [batch.tolist() for batch in batches] == [[0, 1, 2], [3, 4]], batches
    batches = greedy_batches(np.arange(5), lengths, token_budget=100, batch_cap=2)
    assert [batch.tolist() for batch in batches] == [[0, 1], [2, 3], [4]], batches
    assert partition_by_cost([1, 1, 1, 1], 2) == [(0, 2), (2, 4)]
    assert partition_by_cost([3, 1, 1, 1], 2) == [(0, 1), (1, 4)]
    assert [shard_range(10, index, 3) for index in range(3)] == [(0, 3), (3, 6), (6, 10)]
    assert shard_range(512, 0, 1) == (0, 512)

    assert metrics([1, 10, None])["recall_at_10"] == 2 / 3
    assert format_duration(0) == "0s" and format_duration(90) == "1m30s"
    assert format_duration(3725) == "1h02m" and format_duration(-5) == "0s"
    assert format_tokens(999) == "1.0k" and format_tokens(16_000_000) == "16.0M"
    case = {"targets": [["V1", 250]]}
    assert first_match(np.asarray([0, 1, 2]), case, ["V2", "V1", "V1"], {"V1": 25.0},
                       np.asarray([0, 5, 30]), np.asarray([2, 8, 40]), 10, True) == 2
    assert first_match(np.asarray([0, 1]), case, ["V2", "V2"], {"V1": 25.0},
                       np.asarray([0, 5]), np.asarray([2, 8]), 10, True) is None
    assert first_match(np.asarray([0, 1]), case, ["V2", "V1"], {"V1": 25.0},
                       np.asarray([0, 5]), np.asarray([2, 8]), 10, False) == 2

    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        dim = 4
        rows = [{
            "index": index, "group": "L99", "video_id": f"L99_V{index:03d}",
            "shot_index": 0, "window_index": 0, "window_count": 1,
            "window_start_sec": float(index), "window_end_sec": float(index) + 1.0,
        } for index in range(7)]
        source = np.arange(28, dtype=np.float32).reshape(7, dim)
        writer = PartWriter(root / "parts/part0", dim, shard_size=3)
        writer2 = PartWriter(root / "parts/part1", dim, shard_size=3)
        for index in [0, 2, 4, 6]:
            writer.add(source[index], index)
        for index in [1, 3, 5]:
            writer2.add(source[index], index)
        writer.close()
        writer2.close()
        merge_parts(root / "parts", root / "out", dim, rows, {row["index"]: row for row in rows})
        merged = np.vstack([np.load(path) for path in sorted((root / "out").glob("emb_*.npy"))])
        assert np.array_equal(merged, source), (merged, source)

        # A sharded run hands each process a window with a non-zero start; the
        # merge must accept exactly that window and reject anything else.
        window = rows[4:]
        part_a = PartWriter(root / "window/part0", dim, shard_size=3)
        part_b = PartWriter(root / "window/part1", dim, shard_size=3)
        for index in [4, 6]:
            part_a.add(source[index], index)
        for index in [5]:
            part_b.add(source[index], index)
        part_a.close()
        part_b.close()
        merge_parts(root / "window", root / "out_window", dim, window,
                    {row["index"]: row for row in window})
        merged_window = np.vstack([np.load(path) for path in
                                   sorted((root / "out_window").glob("emb_*.npy"))])
        assert np.array_equal(merged_window, source[4:]), (merged_window, source[4:])

        head = PartWriter(root / "head/part0", dim, shard_size=3)
        head.add(source[0], 0)
        head.close()
        try:
            merge_parts(root / "head", root / "out_head", dim, window,
                        {row["index"]: row for row in window})
        except RuntimeError as error:
            assert "coverage mismatch" in str(error)
        else:
            raise AssertionError("merge accepted vectors from outside the shard window")

        assert window_rows(list(range(10)), True, 1, 5) == list(range(10))
        assert window_rows(list(range(10)), False, 2, 3) == [6, 7, 8, 9]

        resumed = PartWriter(root / "parts/part0", dim, shard_size=3, resume=True)
        assert resumed.completed == {0, 2, 4, 6}, resumed.completed
        resumed.close()

        crashed = PartWriter(root / "parts/part2", dim, shard_size=3)
        for index in range(10, 14):
            crashed.add(source[0], index)  # index 13 stays buffered, as after a kill
        recovered = PartWriter(root / "parts/part2", dim, shard_size=3, resume=True)
        assert recovered.completed == {10, 11, 12}, recovered.completed
        assert recovered.shard == 1, recovered.shard
        recovered.close()

        try:
            merge_parts(root / "parts", root / "out2", dim, rows + [{"index": 7}], {})
        except RuntimeError as error:
            assert "coverage" in str(error)
        else:
            raise AssertionError("merge accepted missing vectors")

        inputs = root / "input"
        nested = inputs / "datasets" / "owner" / "ds"
        nested.mkdir(parents=True)
        (nested / "a.json").write_text("{}")
        os.symlink(nested, inputs / "ds-alias")
        found = find_input("a.json", "AIC_TEST_MISSING", root=inputs)
        assert found.resolve() == (nested / "a.json").resolve(), found
        other = inputs / "other"
        other.mkdir()
        (other / "a.json").write_text("{}")
        try:
            find_input("a.json", "AIC_TEST_MISSING", root=inputs)
        except RuntimeError as error:
            assert "expected one a.json" in str(error)
        else:
            raise AssertionError("find_input accepted two distinct files")
    print("self-test OK")


def main():
    mode = os.environ.get("AIC_MODE") or next(
        (flag[2:] for flag in sys.argv[1:] if flag.startswith("--")), None)
    if mode == "self-test":
        self_test()
        return
    if mode == "smoke":
        smoke()
        return
    run = os.environ.get("AIC_RUN", "qwen3-8b-fp16")
    if run not in RUNS:
        raise SystemExit(f"AIC_RUN must be one of {sorted(RUNS)}")
    spec = dict(RUNS[run])
    if os.environ.get("AIC_MODEL_ID"):
        spec["model"] = os.environ["AIC_MODEL_ID"]
        spec["dim"] = int(os.environ.get("AIC_DIM", spec["dim"]))
    captions_path = find_input("shot_captions_2026.parquet", "AIC_CAPTIONS_PATH")
    if mode == "dry-run":
        dry_run(captions_path, spec)
        return
    ensure_dependencies(spec)
    if not on_kaggle() and os.environ.get("AIC_ALLOW_LOCAL") != "1":
        raise SystemExit("full runs are only supported inside Kaggle (set AIC_ALLOW_LOCAL=1 to override)")

    canary = os.environ.get("AIC_CANARY", "0") == "1"
    evaluate = os.environ.get("AIC_EVAL", "0") == "1"
    resume = os.environ.get("AIC_RESUME", "0") == "1"
    shard_index = int(os.environ.get("AIC_SHARD_INDEX", "0"))
    shard_count = int(os.environ.get("AIC_SHARD_COUNT", "1"))
    if not 0 <= shard_index < shard_count:
        raise SystemExit(f"AIC_SHARD_INDEX {shard_index} outside 0..{shard_count - 1}")
    token_budget = int(os.environ.get("AIC_TOKEN_BUDGET", spec["token_budget"]))
    batch_cap = int(os.environ.get("AIC_BATCH_CAP", "64"))
    workers = int(os.environ.get("AIC_WORKERS", spec["workers"]))
    if workers < 1:
        raise SystemExit(f"AIC_WORKERS must be >= 1, got {workers}")
    work = Path("/kaggle/working") if on_kaggle() else Path(
        os.environ.get("AIC_WORK_DIR", "/tmp/aic_caption_embed"))
    output_dir = work / "embeddings" / run
    parts_dir = work / "parts" / run

    rows = load_rows(captions_path)
    validate_rows(rows, strict=not (canary or os.environ.get("AIC_ALLOW_PARTIAL") == "1"))
    if canary:
        rows = window_rows(rows, True, shard_index, shard_count)
    elif shard_count > 1:
        start, end = shard_range(len(rows), shard_index, shard_count)
        rows = window_rows(rows, False, shard_index, shard_count)
        log(f"shard {shard_index}/{shard_count}: rows {start}..{end} ({len(rows):,})")
    log(f"run={run} canary={canary} rows={len(rows)} workers={workers} "
        f"strategy={spec['device_mode']} token_budget={token_budget} "
        f"shard={shard_index}/{shard_count}")

    rank_env = os.environ.get("AIC_RANK")
    if rank_env is not None:
        # Subprocess worker mode: one process per GPU, so no fork and no inherited
        # CUDA state. The parent sets AIC_RANK/AIC_RANK_COUNT and streams our log.
        count = int(os.environ.get("AIC_RANK_COUNT", str(workers)))
        ranges = partition_by_cost([len(row["caption"]) for row in rows], count)
        worker(int(rank_env), {
            "spec": spec, "ranges": ranges, "captions_path": str(captions_path),
            "parts_dir": str(parts_dir), "token_budget": token_budget, "batch_cap": batch_cap,
            "resume": resume, "strategies": [spec["device_mode"]] * count,
            "canary": canary, "shard_index": shard_index, "shard_count": shard_count,
        })
        return

    if workers > 1:
        ensure_model_cached(spec["model"])

    ranges = partition_by_cost([len(row["caption"]) for row in rows], workers)
    if not resume:
        shutil.rmtree(parts_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)
    config = {
        "spec": spec, "ranges": ranges, "captions_path": str(captions_path),
        "parts_dir": str(parts_dir), "token_budget": token_budget, "batch_cap": batch_cap,
        "resume": resume, "strategies": [spec["device_mode"]] * workers,
        "canary": canary, "shard_index": shard_index, "shard_count": shard_count,
    }
    started = time.monotonic()
    if workers == 1:
        config["keep_encoder"] = True
        encoder = worker(0, config)
    else:
        launch_workers(workers, run, work)
        encoder = None

    rows_by_index = {row["index"]: row for row in rows}
    summaries = merge_parts(parts_dir, output_dir, spec["dim"], rows, rows_by_index)
    shutil.rmtree(parts_dir, ignore_errors=True)
    elapsed = max(time.monotonic() - started, 1e-6)
    count = len(rows)
    info = {
        "run": run, "model": spec["model"], "dim": spec["dim"], "quant": spec["quant"],
        "workers": workers, "strategies": [summary["strategy"] for summary in summaries],
        "max_seq_length": MAX_SEQ_LEN, "document_instruction": "", "query_instruction": INSTRUCTION,
        "count": count, "videos": len({row["video_id"] for row in rows}),
        "output_shard_size": OUTPUT_SHARD_SIZE, "normalized": True, "dtype": "float32",
        "token_budget": token_budget, "batch_cap": batch_cap, "canary": canary,
        "shard_index": shard_index, "shard_count": shard_count,
        "elapsed_sec": elapsed, "units_per_second": count / elapsed,
        "peak_gpu_gib": {str(summary["rank"]): round(summary["peak_gpu_gib"], 2)
                         for summary in summaries},
    }
    manifest_path = Path(captions_path).parent / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        info["captions_sha256"] = manifest.get("sha256")
        info["prompt_sha256"] = manifest.get("prompt_sha256")
    (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    if canary:
        projected = EXPECTED_ROWS / max(count / elapsed, 1e-6) / 3600
        (output_dir / "canary.json").write_text(json.dumps({
            "info": info, "projected_full_run_hours": round(projected, 2),
        }, indent=2) + "\n")
        log(f"canary: {count / elapsed:.2f} units/s -> projected {projected:.1f} h for "
            f"{EXPECTED_ROWS:,} units ({', '.join(info['strategies'])})")
    write_checksums(output_dir)
    if evaluate and not canary and shard_count == 1:
        if encoder is None:
            encoder = HFEncoder(spec, 0, token_budget, batch_cap, strategy=spec["device_mode"])
        run_eval(encoder, output_dir, spec, run)
        write_checksums(output_dir)
    elif evaluate:
        log("eval skipped: canary or sharded run (merge shards, then evaluate once)")
    log(f"DONE: {count:,} caption embeddings -> {output_dir}")


if __name__ == "__main__":
    main()
