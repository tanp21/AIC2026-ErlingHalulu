#!/usr/bin/env python3
"""Embed the PE-Core Batch 1 2026 frame IDs with one OpenCLIP model on T4x2.

The file is intentionally self-contained so it can be sent to Kaggle as a script
notebook. Configuration comes from environment variables prepended by the launcher.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


GROUPS = [f"L{i}" for i in range(21, 31)]
SHARD_SIZE = 50_000


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_manifest():
    matches = sorted(Path("/kaggle/input").rglob("batch1_2026.json"))
    if not matches:
        url = os.environ.get("AIC_MANIFEST_URL")
        if not url:
            raise RuntimeError("manifest is not mounted and AIC_MANIFEST_URL is unset")
        target = Path("/kaggle/working/batch1_2026.json")
        download(url, target)
        matches = [target]
    if len(matches) != 1:
        raise RuntimeError(f"expected one batch1_2026.json, found {matches}")
    return matches[0]


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "curl", "-fL", "--retry", "5", "--retry-delay", "5", "--progress-bar",
        "-o", str(target), url,
    ], check=True)


def ensure_dependencies():
    try:
        import open_clip  # noqa: F401
    except ImportError:
        log("installing open-clip-torch>=3.3.0")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/mlfoundations/open_clip.git"
        ], check=True)
    try:
        import remotezip  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "remotezip"],
                       check=True)


def prepare_video_source(first_entry, canary):
    relative = Path(first_entry["rel"])
    suffix = Path(*relative.parts[1:])
    for direct in (
        Path("/kaggle/input/datasets/tonypeter25/aic-batch1-videos"),
        Path("/kaggle/input/aic-batch1-videos"),
        Path("/kaggle/input/datasets/tommr211/aic-batch1-videos"),
    ):
        if (direct / suffix).exists():
            return direct, {}
    urls = json.loads(os.environ.get("AIC_VIDEO_URLS", "{}"))
    if canary and urls:
        return None, urls
    url = os.environ.get("AIC_VIDEO_ARCHIVE_URL")
    if not url:
        raise RuntimeError("video dataset is not mounted and no signed video URL is set")
    log("using HTTP Range reads; only the current video is kept on disk")
    return None, {}


def video_path(video_id, entry, root, urls, downloads_dir, remote_archive=None):
    if root is None and urls:
        if video_id not in urls:
            raise FileNotFoundError(f"no signed URL for canary video {video_id}")
        target = Path(downloads_dir) / f"{video_id}.mp4"
        if not target.exists():
            log(f"downloading canary video {video_id}")
            download(urls[video_id], target)
        return target
    if root is None and remote_archive is not None:
        target = Path(downloads_dir) / f"{video_id}.mp4"
        target.parent.mkdir(parents=True, exist_ok=True)
        relative = Path(entry["rel"])
        candidates = [entry["rel"], str(Path(*relative.parts[1:]))]
        names = remote_archive.namelist()
        archive_name = next((name for name in candidates if name in names), None)
        if archive_name is None:
            matches = [name for name in names if Path(name).name == relative.name]
            if len(matches) != 1:
                raise FileNotFoundError(f"archive does not contain {entry['rel']}")
            archive_name = matches[0]
        with remote_archive.open(archive_name) as source, open(target, "wb") as sink:
            shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
        return target
    relative = Path(entry["rel"])
    path = Path(root) / Path(*relative.parts[1:])
    if path.exists():
        return path
    path = Path("/kaggle/input/datasets/tommr211") / entry["rel"]
    if path.exists():
        return path
    matches = list(Path("/kaggle/input").rglob(Path(entry["rel"]).name))
    if len(matches) != 1:
        raise FileNotFoundError(f"cannot resolve {entry['rel']}: {matches}")
    return matches[0]


def rgb_digest(path):
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        payload = image.tobytes()
        return {
            "width": image.width,
            "height": image.height,
            "sha256_rgb": hashlib.sha256(payload).hexdigest(),
        }


def extract_exact_frames(video, frame_ids, output, fps, threads=2):
    """Extract exact decode-order frame IDs, matching the PE-Core extraction path."""
    output.mkdir(parents=True, exist_ok=True)
    requested = sorted({int(frame_id) for frame_id in frame_ids})
    for start in range(0, len(requested), 100):
        chunk = requested[start : start + 100]
        relative = [frame_id - chunk[0] for frame_id in chunk]
        select = "+".join(f"eq(n\\,{frame_id})" for frame_id in relative)
        batch = output / ".batch"
        shutil.rmtree(batch, ignore_errors=True)
        batch.mkdir()
        command = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-threads", str(threads),
            "-ss", f"{chunk[0] / float(fps):.6f}",
            "-t", f"{(chunk[-1] - chunk[0] + 2) / float(fps):.6f}",
            "-i", str(video),
            "-vf", f"select='{select}'", "-vsync", "0",
            "-c:v", "png", "-pix_fmt", "rgb24", "-compression_level", "1",
            "-start_number", "0", str(batch / "f%06d.png"),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=1800)
        frames = sorted(batch.glob("f*.png"))
        if len(frames) != len(chunk):
            raise RuntimeError(
                f"{video.name}: ffmpeg returned {len(frames)} frames for {len(chunk)} IDs"
            )
        for frame_id, frame in zip(chunk, frames):
            frame.replace(output / f"{frame_id:06d}.png")
        shutil.rmtree(batch)
    actual = sorted(int(path.stem) for path in output.glob("*.png"))
    if actual != requested:
        raise RuntimeError(f"{video.name}: extracted frame IDs differ from manifest")


class PartWriter:
    def __init__(self, root, rank, dimension, shard_size=SHARD_SIZE):
        self.root = Path(root) / f"part{rank}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dimension = dimension
        self.shard_size = shard_size
        self.buffer = np.empty((shard_size, dimension), dtype=np.float32)
        self.fill = 0
        self.shard = 0
        self.total = 0
        self.meta = open(self.root / "meta.jsonl", "w", buffering=1)

    def add_batch(self, vectors, video_id, frame_ids):
        for vector, frame_id in zip(vectors, frame_ids):
            self.buffer[self.fill] = vector
            self.meta.write(json.dumps({
                "part": int(self.root.name[4:]), "shard": self.shard,
                "row": self.fill, "video_id": video_id, "frame_idx": int(frame_id),
            }) + "\n")
            self.fill += 1
            self.total += 1
            if self.fill == self.shard_size:
                self.flush()

    def flush(self):
        if not self.fill:
            return
        np.save(self.root / f"emb_{self.shard:06d}.npy", self.buffer[: self.fill])
        self.shard += 1
        self.fill = 0

    def close(self):
        self.flush()
        self.meta.close()


def load_model(model_id, naflex, device):
    import open_clip

    kwargs = {"device": device, "precision": "fp16"}
    if naflex:
        kwargs["aug_cfg"] = {"naflex": True, "use_timm": True}
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{model_id}", **kwargs
    )
    model.eval()
    if naflex:
        preprocess = preprocess(max_seq_len=576, patch_size=16)
    return model, preprocess


def collate_images(paths, preprocess, naflex):
    import torch
    from PIL import Image
    from torch.utils.data._utils.collate import default_collate

    samples = []
    for path in paths:
        with Image.open(path) as image:
            samples.append(preprocess(image.convert("RGB")))
    return default_collate(samples) if naflex else torch.stack(samples)


def to_device(batch, device):
    import torch

    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    moved = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    if "patches" in moved:
        moved["seq_len"] = moved["patches"].shape[1]
    return moved


def encode_paths(model, preprocess, paths, naflex, device, batch_size, progress):
    import torch

    offset = 0
    while offset < len(paths):
        size = min(batch_size, len(paths) - offset)
        while True:
            try:
                batch = collate_images(paths[offset : offset + size], preprocess, naflex)
                batch = to_device(batch, device)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    vectors = model.encode_image(batch, normalize=True)
                    vectors = torch.nn.functional.normalize(vectors.float(), dim=-1)
                vectors = vectors.cpu().numpy().astype(np.float32, copy=False)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size == 1:
                    raise
                size = max(1, size // 2)
                batch_size = min(batch_size, size)
                progress.write(f"{device}: OOM, reducing batch size to {batch_size}")
        yield vectors, batch_size
        offset += size


def split_videos(manifest, part_index, part_count):
    """Assign whole videos to balanced, deterministic keyframe partitions."""
    if not 0 <= part_index < part_count:
        raise ValueError(f"part index {part_index} is outside 0..{part_count - 1}")
    bins = [[] for _ in range(part_count)]
    totals = [0] * part_count
    videos = sorted(manifest["videos"].items(), key=lambda item: (-len(item[1]["frames"]), item[0]))
    for video_id, entry in videos:
        target = min(range(part_count), key=lambda index: (totals[index], index))
        bins[target].append(video_id)
        totals[target] += len(entry["frames"])
    return set(bins[part_index]), totals


def worker(rank, config, manifest, selected_video_ids):
    import torch
    from tqdm.auto import tqdm

    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)
    videos = [(video_id, entry) for video_id, entry in sorted(manifest["videos"].items())
              if video_id in selected_video_ids]
    videos = videos[rank::2]
    if config["canary"]:
        videos = videos[:1]
    total = sum(len(entry["frames"]) for _, entry in videos)
    log(f"GPU {rank}: loading {config['model_id']}; videos={len(videos)}; frames={total:,}")
    model, preprocess = load_model(config["model_id"], config["naflex"], device)
    with torch.inference_mode():
        probe = collate_images([], preprocess, config["naflex"]) if False else None
    dimension = 1152 if config["naflex"] else 1536
    writer = PartWriter(config["parts_dir"], rank, dimension)
    batch_size = config["batch_size"]
    digests = {}
    remote_archive = None
    if config["remote_archive_url"]:
        from remotezip import RemoteZip
        remote_archive = RemoteZip(config["remote_archive_url"])
    progress = tqdm(total=total, desc=f"GPU {rank} extract+embed", unit="frame",
                    position=rank, dynamic_ncols=True)
    started = time.monotonic()
    for video_number, (video_id, entry) in enumerate(videos, 1):
        target = Path(config["temp_dir"]) / f"gpu{rank}" / video_id
        error = None
        for attempt in range(2):
            shutil.rmtree(target, ignore_errors=True)
            try:
                extract_exact_frames(video_path(
                    video_id, entry, config["video_root"], config["video_urls"],
                    config["downloads_dir"], remote_archive), entry["frames"], target,
                    entry["fps"])
                error = None
                break
            except Exception as exc:
                error = exc
                progress.write(f"GPU {rank} {video_id}: extraction attempt {attempt + 1} failed: {exc}")
        if error is not None:
            raise error
        if config["video_urls"] or remote_archive is not None:
            (Path(config["downloads_dir"]) / f"{video_id}.mp4").unlink(missing_ok=True)
        paths = [target / f"{int(frame_id):06d}.png" for frame_id in entry["frames"]]
        if config["canary"]:
            digests[video_id] = {
                str(frame_id): rgb_digest(path) for frame_id, path in zip(entry["frames"], paths)
            }
        consumed = 0
        for vectors, batch_size in encode_paths(
                model, preprocess, paths, config["naflex"], device, batch_size, progress):
            frame_ids = entry["frames"][consumed : consumed + len(vectors)]
            writer.add_batch(vectors, video_id, frame_ids)
            consumed += len(vectors)
            progress.update(len(vectors))
            elapsed = max(time.monotonic() - started, 1e-6)
            progress.set_postfix(video=video_id, img_s=f"{progress.n / elapsed:.1f}",
                                 bs=batch_size, ckpt=writer.total,
                                 mem=f"{torch.cuda.memory_allocated(rank) / 2**30:.1f}G")
        if consumed != len(entry["frames"]):
            raise RuntimeError(f"{video_id}: embedded {consumed}, expected {len(entry['frames'])}")
        shutil.rmtree(target)
        if video_number % 10 == 0 or video_number == len(videos):
            progress.write(f"GPU {rank}: {video_number}/{len(videos)} videos, "
                           f"{writer.total:,}/{total:,} frames checkpointed")
    writer.close()
    if remote_archive is not None:
        remote_archive.close()
    progress.close()
    (Path(config["parts_dir"]) / f"part{rank}" / "digests.json").write_text(
        json.dumps(digests, sort_keys=True)
    )
    log(f"GPU {rank}: complete, {writer.total:,} embeddings")


def merge_parts(parts_dir, output_dir, model_id, naflex, expected):
    import pyarrow as pa
    import pyarrow.parquet as pq

    records = []
    for meta_path in sorted(Path(parts_dir).glob("part*/meta.jsonl")):
        with open(meta_path) as handle:
            records.extend(json.loads(line) for line in handle)
    records.sort(key=lambda row: (row["video_id"], row["frame_idx"]))
    if len(records) != expected:
        raise RuntimeError(f"merged {len(records):,} rows, expected {expected:,}")
    keys = [(row["video_id"], row["frame_idx"]) for row in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate video_id/frame_idx pairs in worker output")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dimension = 1152 if naflex else 1536
    buffer = np.empty((SHARD_SIZE, dimension), dtype=np.float32)
    fill = shard_idx = 0
    meta_rows = []
    cache_key = None
    cache = None
    for row in records:
        key = row["part"], row["shard"]
        if key != cache_key:
            cache = np.load(Path(parts_dir) / f"part{key[0]}" / f"emb_{key[1]:06d}.npy",
                            mmap_mode="r")
            cache_key = key
        buffer[fill] = cache[row["row"]]
        meta_rows.append((shard_idx, fill, row["video_id"], row["frame_idx"]))
        fill += 1
        if fill == SHARD_SIZE:
            np.save(output_dir / f"emb_{shard_idx:06d}.npy", buffer)
            shard_idx += 1
            fill = 0
    if fill:
        np.save(output_dir / f"emb_{shard_idx:06d}.npy", buffer[:fill])

    pq.write_table(pa.table({
        "shard_idx": pa.array([r[0] for r in meta_rows], type=pa.int32()),
        "row_idx": pa.array([r[1] for r in meta_rows], type=pa.int32()),
        "video_id": pa.array([r[2] for r in meta_rows], type=pa.string()),
        "frame_idx": pa.array([r[3] for r in meta_rows], type=pa.int32()),
    }), output_dir / "meta.parquet")
    info = {
        "model": model_id, "dim": dimension, "count": len(records),
        "shard_size": SHARD_SIZE, "normalized": True, "dtype": "float32",
        "preprocessing": "naflex-native-576-patches" if naflex else "fixed-384",
    }
    (output_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    digests = {}
    for path in sorted(Path(parts_dir).glob("part*/digests.json")):
        digests.update(json.loads(path.read_text()))
    if digests:
        (output_dir / "pixel_digests.json").write_text(json.dumps(digests, indent=2) + "\n")
    checksums = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "checksums.json":
            digest = hashlib.sha256()
            with open(path, "rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            checksums[path.name] = digest.hexdigest()
    (output_dir / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")


def main():
    if "--self-test" in sys.argv:
        # A compact writer test using production dimensions avoids special-case merge code.
        import tempfile
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for rank in range(2):
                writer = PartWriter(root / "parts", rank, 1536, shard_size=3)
                for i in range(rank, 8, 2):
                    writer.add_batch(np.full((1, 1536), i, np.float32),
                                     f"L21_V{i:03d}", [i])
                writer.close()
                (root / "parts" / f"part{rank}" / "digests.json").write_text("{}")
            merge_parts(root / "parts", root / "out", "test", False, 8)
            table = pq.read_table(root / "out" / "meta.parquet")
            assert table.num_rows == 8
            assert [p.name for p in sorted((root / "out").glob("emb_*.npy"))] == [
                "emb_000000.npy"
            ]
        fake = {"videos": {f"L21_V{i:03d}": {"frames": list(range(i + 1))}
                           for i in range(12)}}
        partitions = [split_videos(fake, index, 4)[0] for index in range(4)]
        assert set.union(*partitions) == set(fake["videos"])
        assert not any(left & right for i, left in enumerate(partitions)
                       for right in partitions[i + 1:])
        print("self-test OK")
        return

    import torch.multiprocessing as mp

    if not __file__.startswith("/kaggle/") and not Path("/kaggle/input").exists():
        raise RuntimeError("full runs are only supported inside Kaggle")
    ensure_dependencies()
    model_id = os.environ["AIC_MODEL_ID"]
    output_name = os.environ["AIC_OUTPUT_NAME"]
    canary = os.environ.get("AIC_CANARY", "0") == "1"
    naflex = os.environ.get("AIC_NAFLEX", "0") == "1"
    work = Path("/kaggle/working")
    output = work / "embeddings" / output_name
    parts = work / ".siglip2_parts" / output_name
    temp = work / ".siglip2_frames" / output_name
    for path in (output, parts, temp):
        shutil.rmtree(path, ignore_errors=True)
    manifest_path = find_manifest()
    manifest = json.loads(manifest_path.read_text())
    first_entry = next(iter(sorted(manifest["videos"].items())))[1]
    video_root, video_urls = prepare_video_source(first_entry, canary)
    part_count = int(os.environ.get("AIC_PART_COUNT", "1"))
    part_indices = [int(value) for value in os.environ.get(
        "AIC_PART_INDICES", os.environ.get("AIC_PART_INDEX", "0")
    ).split(",")]
    if len(part_indices) != len(set(part_indices)):
        raise ValueError("AIC_PART_INDICES contains duplicates")
    selected_video_ids = set()
    partition_totals = None
    for part_index in part_indices:
        video_ids, partition_totals = split_videos(manifest, part_index, part_count)
        selected_video_ids.update(video_ids)
    expected = sum(partition_totals[index] for index in part_indices)
    if canary:
        selected_video_ids = set(sorted(selected_video_ids)[:2])
        expected = sum(len(manifest["videos"][video_id]["frames"])
                       for video_id in selected_video_ids)
    config = {
        "model_id": model_id, "naflex": naflex, "canary": canary,
        "batch_size": int(os.environ.get("AIC_BATCH_SIZE", "16" if naflex else "8")),
        "parts_dir": str(parts), "temp_dir": str(temp),
        "video_root": str(video_root) if video_root else None, "video_urls": video_urls,
        "remote_archive_url": (os.environ.get("AIC_VIDEO_ARCHIVE_URL")
                               if video_root is None and not video_urls else None),
        "downloads_dir": str(work / ".siglip2_downloads" / output_name),
    }
    log(f"manifest={manifest_path}; zero_based_parts={part_indices}/{part_count}; "
        f"selected_videos={len(selected_video_ids)}; selected_frames={expected:,}; "
        f"partition_frames={partition_totals}; canary={canary}")
    mp.spawn(worker, args=(config, manifest, selected_video_ids), nprocs=2, join=True)
    merge_parts(parts, output, model_id, naflex, expected)
    shutil.rmtree(parts)
    shutil.rmtree(temp, ignore_errors=True)
    log(f"DONE: {expected:,} exact-frame embeddings -> {output}")


if __name__ == "__main__":
    main()
