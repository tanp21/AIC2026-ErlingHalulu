"""Embed Batch 2 2026 keyframes with SigLIP2 NaFlex on a single GPU.

    uv run python -m scripts.embedding.embed_batch2_naflex \
        --keyframes-dir data/2026/keyframes --out-dir data/2026/embeddings_batch2/naflex \
        --groups S01,M01,M02,M03,M04,M05,M06,M07,M08,M09,M10

Adapted from the NaFlex path of ``notebooks/kaggle_siglip2_embed.py`` (same model load:
``timm/ViT-SO400M-16-SigLIP2-naflex``, ``aug_cfg={"naflex": True, "use_timm": True}``,
``preprocess(max_seq_len=576, patch_size=16)``, dim 1152), but single-GPU instead of the
Kaggle T4x2 ``mp.spawn`` split, and reading keyframe PNGs that
``scripts.extraction.extract_batch2_keyframes`` already produced locally instead of
re-deriving frame ids from a manifest and re-extracting them from the mounted video zip.

Writes the same ``emb_%06d.npy`` + ``meta.parquet`` + ``info.json`` shard layout as
``scripts/embedding/embed_batch2_pe_core.py`` (same ``ShardWriter``, same resume behaviour),
with ``info.json["preprocessing"] = "naflex-native-576-patches"`` matching the field batch 1's
naflex output already carries.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data._utils.collate import default_collate

import open_clip

MODEL = "timm/ViT-SO400M-16-SigLIP2-naflex"
DIM = 1152
MAX_SEQ_LEN = 576
PATCH_SIZE = 16
DEFAULT_BATCH_CANDIDATES = (4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--keyframes-dir", type=Path, default=Path("data/2026/keyframes"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--groups", default="", help="comma-separated groups, e.g. S01,M01 "
                        "(default: every group directory under --keyframes-dir)")
    p.add_argument("--only", default="", help="comma-separated video ids, for a canary run")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto-tune")
    p.add_argument("--batch-candidates", default=",".join(str(b) for b in DEFAULT_BATCH_CANDIDATES))
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--workers", type=int, default=8,
                        help="threads for PIL decode/preprocess; match vCPUs so the GPU "
                             "isn't left idle between batches")
    return p.parse_args(argv)


def collect_items(keyframes_dir, groups, only):
    items = []
    for path in sorted(keyframes_dir.rglob("*.png")):
        if path.name.endswith(".tmp.png"):
            continue
        video_id = path.parent.name
        group = path.parent.parent.name
        if groups and group not in groups:
            continue
        if only and video_id not in only:
            continue
        items.append((path, video_id, int(path.stem)))
    return items


class ShardWriter:
    """Same resume-safe shard writer as embed_batch2_pe_core.py -- kept as a separate copy
    (not a shared import) so this script stays paste-portable, matching the existing
    notebooks/kaggle_*.py convention in this repo.
    """

    def __init__(self, out_dir, dim, shard_size=50000):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.out_dir / "meta.parquet"
        self.dim = dim
        self.shard_size = shard_size
        self.buf = np.empty((shard_size, dim), dtype=np.float32)
        self.fill = 0
        self.shard_idx = 0
        self.total = 0
        self.done = set()
        self.meta_rows = []
        self._pending = []
        self._resume()

    def _resume(self):
        if not self.meta_path.exists():
            return
        rows = pq.read_table(self.meta_path).to_pylist()
        self.meta_rows = [(r["shard_idx"], r["row_idx"], r["video_id"], r["frame_idx"])
                          for r in rows]
        self.done = {(r[2], r[3]) for r in self.meta_rows}
        self.total = len(self.meta_rows)
        self.shard_idx = self.total // self.shard_size
        self.fill = self.total % self.shard_size
        if self.fill > 0:
            part = self.out_dir / f"emb_{self.shard_idx:06d}.npy"
            assert part.exists(), f"inconsistent state: {part} missing"
            self.buf[: self.fill] = np.load(part)

    def add(self, feat, video_id, frame_idx):
        self.buf[self.fill] = feat
        self._pending.append((self.shard_idx, self.fill, video_id, frame_idx))
        self.fill += 1
        self.total += 1
        if self.fill == self.shard_size:
            self.checkpoint(finalize=True)

    def checkpoint(self, finalize=False):
        if self.fill == 0:
            return
        np.save(self.out_dir / f"emb_{self.shard_idx:06d}.npy", self.buf[: self.fill])
        self.meta_rows.extend(self._pending)
        self._pending.clear()
        self._write_meta()
        if finalize:
            self.shard_idx += 1
            self.fill = 0

    def _write_meta(self):
        table = pa.table({
            "shard_idx": pa.array([r[0] for r in self.meta_rows], type=pa.int32()),
            "row_idx": pa.array([r[1] for r in self.meta_rows], type=pa.int32()),
            "video_id": pa.array([r[2] for r in self.meta_rows], type=pa.string()),
            "frame_idx": pa.array([r[3] for r in self.meta_rows], type=pa.int32()),
        })
        pq.write_table(table, self.meta_path)

    def close(self):
        self.checkpoint(finalize=True)

    def write_info(self, model, dim):
        (self.out_dir / "info.json").write_text(json.dumps({
            "model": model, "dim": dim, "count": self.total, "shard_size": self.shard_size,
            "normalized": True, "dtype": "float32",
            "preprocessing": "naflex-native-576-patches",
        }, indent=2))


def load_model(model_id, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{model_id}", device=device, precision="fp16",
        aug_cfg={"naflex": True, "use_timm": True})
    model.eval()
    preprocess = preprocess(max_seq_len=MAX_SEQ_LEN, patch_size=PATCH_SIZE)
    return model, preprocess


def _load_one(args):
    path, preprocess = args
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def collate_images(paths, preprocess, workers=0):
    """PIL decode + preprocess releases the GIL for most of its work, so threading this
    (rather than leaving it serial) keeps the GPU fed instead of idling between batches --
    measured ~17 img/s (GPU-idle, CPU-decode-bound) with workers=0 on a 12-vCPU box vs. much
    higher with threading; see scripts/embedding/embed_batch2_stream.py's callers.
    """
    if workers > 0:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as pool:
            samples = list(pool.map(_load_one, [(path, preprocess) for path in paths]))
    else:
        samples = [_load_one((path, preprocess)) for path in paths]
    return default_collate(samples)


def to_device(batch, device):
    moved = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
             for key, value in batch.items()}
    if "patches" in moved:
        moved["seq_len"] = moved["patches"].shape[1]
    return moved


def encode_paths(model, preprocess, paths, device, batch_size, workers=0):
    """Yields (vectors, batch_size_used), halving batch_size on OOM like the Kaggle notebook."""
    offset = 0
    while offset < len(paths):
        size = min(batch_size, len(paths) - offset)
        while True:
            try:
                batch = to_device(
                    collate_images(paths[offset:offset + size], preprocess, workers), device)
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
                log(f"OOM, reducing batch size to {batch_size}")
        yield vectors, batch_size
        offset += size


def autotune_batch(model, preprocess, device, sample_path, candidates, workers=0):
    best = candidates[0]
    for b in candidates:
        try:
            batch = to_device(collate_images([sample_path] * b, preprocess, workers), device)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                model.encode_image(batch, normalize=True)
            best = b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return best


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("NaFlex embedding requires a CUDA GPU (fp16 autocast path)")

    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    only = {v.strip() for v in args.only.split(",") if v.strip()}
    items = collect_items(args.keyframes_dir, groups, only)
    log(f"found {len(items)} keyframes under {args.keyframes_dir}"
        + (f" (groups={sorted(groups)})" if groups else "")
        + (f" (only={sorted(only)})" if only else ""))
    if not items:
        log("nothing to do")
        return 0

    log(f"loading {args.model} ...")
    model, preprocess = load_model(args.model, device)

    writer = ShardWriter(args.out_dir, DIM, args.shard_size)
    log(f"resume: {writer.total} committed, {len(writer.done)} keyframes already done")
    items = [it for it in items if (it[1], it[2]) not in writer.done]
    if not items:
        log("all requested keyframes already embedded")
        writer.write_info(args.model, DIM)
        return 0
    log(f"{len(items)} keyframes remaining")

    candidates = sorted(int(b) for b in args.batch_candidates.split(",") if b.strip())
    bs = args.batch_size if args.batch_size > 0 else autotune_batch(
        model, preprocess, device, items[0][0], candidates, workers=args.workers)
    log(f"batch size = {bs}")

    paths = [it[0] for it in items]
    n = 0
    started = time.perf_counter()
    consumed = 0
    for vectors, bs in encode_paths(model, preprocess, paths, device, bs, workers=args.workers):
        batch_items = items[consumed:consumed + len(vectors)]
        for (_, video_id, frame_idx), vector in zip(batch_items, vectors):
            writer.add(vector, video_id, frame_idx)
            n += 1
        consumed += len(vectors)
        if n % 5000 < bs:
            elapsed = time.perf_counter() - started
            log(f"{n}/{len(items)} embedded ({n / elapsed:.1f} img/s)")

    writer.close()
    writer.write_info(args.model, DIM)
    elapsed = time.perf_counter() - started
    log(f"DONE: {n} embedded in {elapsed / 60:.1f} min "
        f"({n / elapsed:.1f} img/s) -> {args.out_dir} (total committed {writer.total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
