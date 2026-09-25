"""Embed Batch 2 2026 keyframes with PE-Core-bigG-14-448 on a single GPU.

    uv run python -m scripts.embedding.embed_batch2_pe_core \
        --keyframes-dir data/2026/keyframes --out-dir data/2026/embeddings_batch2/pe-core \
        --groups S01,M01,M02,M03,M04,M05,M06,M07,M08,M09,M10

Adapted from ``scripts/embedding/embed_all.py`` (same model, ``ShardWriter``/resume logic,
fp16 autocast), but reads keyframes that are already local (no per-group rclone/gdrive
download loop -- a rented GPU box already has ``data/<year>/keyframes/`` populated by
``scripts.extraction.extract_batch2_keyframes``) and is parameterized by ``--groups`` instead
of a hardcoded ``L21``..``L30`` list.

Writes into its own directory (``--out-dir``), never into the live ``data/2026/embeddings/``
-- that directory is production's PE-Core source and this script's shard numbering starts
fresh, so pointing it there would overwrite batch 1's shards. Merge the two afterwards with
``scripts/embedding/assemble_gopt_batch1.py``'s dedup/re-shard pattern.

``--batch-size 0`` (default) autotunes via OOM probing; ``--batch-candidates`` controls the
candidate list probed (default extends past the 32 used for older/smaller GPUs, since a
32GB-class card can usually go higher -- always measure with ``--only`` on 1-2 videos first).
"""
import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import open_clip

MODEL = "hf-hub:timm/PE-Core-bigG-14-448"
IMAGE_SIZE = 448
DEFAULT_BATCH_CANDIDATES = (2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320)


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
    p.add_argument("--batch-candidates", default=",".join(str(b) for b in DEFAULT_BATCH_CANDIDATES),
                        help="comma-separated candidate batch sizes probed largest-first "
                             "when --batch-size is 0")
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers; match vCPUs")
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args(argv)


class KeyframeDataset(Dataset):
    def __init__(self, items, preprocess):
        self.items = items
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, video_id, frame_idx = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
        return self.preprocess(img), video_id, frame_idx


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
    """Accumulates (shard_size, dim) float32 vectors and commits full shards.

    ``meta.parquet`` is rewritten only *after* a shard file is saved, so committed rows
    always have their vectors on disk (resume is consistent). Resuming reads the existing
    ``meta.parquet`` + shard files, so re-running after an interruption never restarts
    shard numbering from 0 (unlike ``scripts/embedding/embed_keyframes.py``).
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
            "model": model, "dim": dim, "count": self.total,
            "shard_size": self.shard_size, "normalized": True, "dtype": "float32",
        }, indent=2))


def autotune_batch(model, device, fp16, candidates):
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    best = candidates[0]
    for b in candidates:
        try:
            x = torch.randn(b, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
            with torch.inference_mode(), ctx:
                model.encode_image(x, normalize=True)
            best = b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return best


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"
    if args.workers > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)
    if device == "cuda":
        # Inherited from embed_all.py: the ViT's one tiny conv (patch embed) doesn't need
        # cuDNN, and disabling it sidesteps a cuDNN/driver sublibrary version mismatch seen
        # on some wheel/driver combos.
        torch.backends.cudnn.enabled = False

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
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, precision="fp16" if fp16 else "fp32", device=device)
    model.eval()
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    with torch.inference_mode(), ctx:
        dim = model.encode_image(
            torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device), normalize=True
        ).shape[1]
    log(f"embed dim = {dim}, fp16 = {fp16}, device = {device}")

    candidates = sorted(int(b) for b in args.batch_candidates.split(",") if b.strip())
    bs = args.batch_size if args.batch_size > 0 else autotune_batch(model, device, fp16, candidates)
    log(f"batch size = {bs}, workers = {args.workers}")

    writer = ShardWriter(args.out_dir, dim, args.shard_size)
    log(f"resume: {writer.total} committed, {len(writer.done)} keyframes already done")
    items = [it for it in items if (it[1], it[2]) not in writer.done]
    if not items:
        log("all requested keyframes already embedded")
        writer.write_info(args.model, dim)
        return 0
    log(f"{len(items)} keyframes remaining")

    ds = KeyframeDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=bs, num_workers=args.workers,
                    pin_memory=device == "cuda", persistent_workers=args.workers > 0,
                    drop_last=False)

    n = 0
    started = time.perf_counter()
    with torch.inference_mode():
        for images, vids, fids in dl:
            images = images.to(device, non_blocking=True)
            with ctx:
                feats = model.encode_image(images, normalize=True)
            feats = feats.float().cpu().numpy()
            fids = fids.tolist() if isinstance(fids, torch.Tensor) else fids
            for i in range(len(vids)):
                writer.add(feats[i], vids[i], int(fids[i]))
                n += 1
            if n % 5000 < bs:
                elapsed = time.perf_counter() - started
                log(f"{n}/{len(items)} embedded ({n / elapsed:.1f} img/s)")

    writer.close()
    writer.write_info(args.model, dim)
    elapsed = time.perf_counter() - started
    log(f"DONE: {n} embedded in {elapsed / 60:.1f} min "
        f"({n / elapsed:.1f} img/s) -> {args.out_dir} (total committed {writer.total})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
