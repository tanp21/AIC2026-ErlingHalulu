#!/usr/bin/env python3
"""Embed all keyframes (L21..L30) with PE-Core-bigG-14-448 on a single GPU.

Pipeline, per group:
  gdown (prefetched in background) -> unzip -> embed -> delete zip + extracted dir

Embeddings are L2-normalized float32, saved as continuous 50000-row shards
`emb_%06d.npy` with `meta.parquet` mapping (shard_idx,row_idx,video_id,frame_idx).
Resume-safe: committed state is reconstructed from meta.parquet + existing shards,
and fully-processed groups (recorded in done_groups.txt) are skipped.

Quality: uses the model's own preprocessing (mean/std/resize from the config),
fp16 autocast, normalize=True. No quality-affecting flags are overridden.
"""
import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pyarrow as pa
import pyarrow.parquet as pq

import open_clip

GROUPS = ["L21", "L22", "L23", "L24", "L25", "L26", "L27", "L28", "L29", "L30"]

# rclone remote pointing at the keyframe zips (authenticated -> no GDrive quota).
RCLONE_SRC = "gdrive:AIC_HCMC/2026/Dataset/zipped/keyframes"

MODEL = "hf-hub:timm/PE-Core-bigG-14-448"
IMAGE_SIZE = 448


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zips-dir", default="/workspace/zips")
    p.add_argument("--extract-dir", default="/workspace/kf_tmp")
    p.add_argument("--out-dir", default="/workspace/embeddings")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto-tune")
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--only", help="comma-separated group names (L21,L26,...)")
    p.add_argument("--self-test", action="store_true", help="test ShardWriter resume logic, then exit")
    return p.parse_args()


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


def collect_items(root: Path):
    items = []
    for p in sorted(root.rglob("*.png")):
        if p.name.endswith(".tmp.png"):
            continue
        items.append((p, p.parent.name, int(p.stem)))
    return items


class ShardWriter:
    """Accumulates (shard_size, dim) float32 vectors and commits full shards.

    meta.parquet is rewritten only *after* a shard file is saved, so committed
    rows always have their vectors on disk (resume is consistent).
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
        self.meta_rows = [
            (r["shard_idx"], r["row_idx"], r["video_id"], r["frame_idx"]) for r in rows
        ]
        self.done = {(r[2], r[3]) for r in self.meta_rows}
        self.total = len(self.meta_rows)
        self.shard_idx = self.total // self.shard_size
        self.fill = self.total % self.shard_size
        if self.fill > 0:
            part = self.out_dir / f"emb_{self.shard_idx:06d}.npy"
            assert part.exists(), f"inconsistent state: {part} missing but meta.parquet has {self.total} rows"
            self.buf[: self.fill] = np.load(part)

    def add(self, feat, video_id, frame_idx):
        self.buf[self.fill] = feat
        self._pending.append((self.shard_idx, self.fill, video_id, frame_idx))
        self.fill += 1
        self.total += 1
        if self.fill == self.shard_size:
            self.checkpoint(finalize=True)

    def checkpoint(self, finalize=False):
        """Durably save the current buffer + meta rows.

        finalize=False snapshots a partial shard without advancing shard_idx/fill,
        so the next group keeps filling the same 50000-row shard. finalize=True
        (shard full, or end of run) advances to the next shard.
        """
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
        if not self.meta_rows:
            return
        table = pa.table(
            {
                "shard_idx": pa.array([r[0] for r in self.meta_rows], type=pa.int32()),
                "row_idx": pa.array([r[1] for r in self.meta_rows], type=pa.int32()),
                "video_id": pa.array([r[2] for r in self.meta_rows], type=pa.string()),
                "frame_idx": pa.array([r[3] for r in self.meta_rows], type=pa.int32()),
            }
        )
        pq.write_table(table, self.meta_path)

    def close(self):
        self.checkpoint(finalize=True)

    def write_info(self, model, dim):
        info = {
            "model": model,
            "dim": dim,
            "count": self.total,
            "shard_size": self.shard_size,
            "normalized": True,
            "dtype": "float32",
        }
        (self.out_dir / "info.json").write_text(json.dumps(info, indent=2))


def autotune_batch(model, device, fp16, max_b=32):
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    best = 1
    for b in [2, 4, 8, 16, 32]:
        if b > max_b:
            break
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


def embed_items(model, preprocess, device, fp16, items, writer, workers, batch_size):
    ds = KeyframeDataset(items, preprocess)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=device == "cuda",
        persistent_workers=workers > 0,
        drop_last=False,
    )
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    n = 0
    with torch.inference_mode():
        for images, vids, fids in tqdm(dl, desc="embed", unit="img"):
            images = images.to(device, non_blocking=True)
            with ctx:
                feats = model.encode_image(images, normalize=True)
            feats = feats.float().cpu().numpy()
            fids = fids.tolist() if isinstance(fids, torch.Tensor) else fids
            for i in range(len(vids)):
                writer.add(feats[i], vids[i], int(fids[i]))
                n += 1
    return n


def download(name, dest: Path, retries=8):
    dest.parent.mkdir(parents=True, exist_ok=True)
    remote = f"{RCLONE_SRC}/{name}.zip"
    for attempt in range(1, retries + 1):
        r = subprocess.run(["rclone", "copyto", "--log-level", "ERROR", remote, str(dest)])
        if r.returncode == 0:
            return
        log(f"rclone attempt {attempt} rc={r.returncode}, retrying in 30s")
        time.sleep(30)
    raise RuntimeError(f"failed to download {name} after {retries} attempts")


def extract(zip_path: Path, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(dest_dir)])
        ok = r.returncode == 0
    except FileNotFoundError:
        ok = False
    if not ok:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_dir)


def _launch_download(name, zips_dir):
    dest = Path(zips_dir) / f"{name}.zip"
    t = threading.Thread(target=download, args=(name, dest), name=name, daemon=True)
    t.start()
    return t


def self_test(tmp=Path("/tmp/opencode/shard_selftest")):
    import shutil as _sh

    _sh.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    w = ShardWriter(tmp, dim=4, shard_size=3)
    for i in range(7):
        w.add(np.full(4, i, dtype=np.float32), f"v{i // 3}", i % 3)
    w.close()
    assert w.total == 7 and w.fill == 0

    # resume
    w2 = ShardWriter(tmp, dim=4, shard_size=3)
    assert w2.total == 7, w2.total
    assert w2.shard_idx == 2 and w2.fill == 1, (w2.shard_idx, w2.fill)
    assert len(w2.done) == 7
    for i in range(7, 9):
        w2.add(np.full(4, i, dtype=np.float32), f"v{i // 3}", i % 3)
    w2.close()
    assert w2.total == 9

    shards = sorted(tmp.glob("emb_*.npy"))
    assert len(shards) == 3, shards
    shapes = [np.load(s).shape[0] for s in shards]
    assert shapes == [3, 3, 3], shapes
    meta = pq.read_table(tmp / "meta.parquet")
    assert meta.num_rows == 9, meta.num_rows
    _sh.rmtree(tmp, ignore_errors=True)
    print("ShardWriter self-test OK: 3 shards x 3 rows, resume continuity verified")
    return True


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"
    if args.workers > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)
    if device == "cuda":
        # ViT has one tiny conv (patch embed); native kernels suffice, avoids
        # cudnn sublibrary version mismatch on some wheel/driver combos.
        torch.backends.cudnn.enabled = False

    log(f"loading {MODEL} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL, precision="fp16" if fp16 else "fp32", device=device
    )
    model.eval()
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    with torch.inference_mode(), ctx:
        dim = model.encode_image(
            torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device), normalize=True
        ).shape[1]
    log(f"embed dim = {dim}, fp16 = {fp16}, device = {device}")

    bs = args.batch_size if args.batch_size > 0 else autotune_batch(model, device, fp16)
    log(f"batch size = {bs}, workers = {args.workers}")

    writer = ShardWriter(args.out_dir, dim, args.shard_size)
    log(f"resume: {writer.total} embeddings committed, {len(writer.done)} keyframes done")

    done_groups_path = Path(args.out_dir) / "done_groups.txt"
    done_groups = set(done_groups_path.read_text().split()) if done_groups_path.exists() else set()

    only = set(args.only.split(",")) if args.only else None
    groups = [g for g in GROUPS if only is None or g in only]
    to_do = [g for g in groups if g not in done_groups]
    log(f"groups: {groups}; skipping done: {sorted(done_groups)}")

    if not to_do:
        log("all groups done, nothing to embed")
    else:
        threads = {0: _launch_download(to_do[0], args.zips_dir)}
        for i, name in enumerate(to_do):
            threads[i].join()
            zip_path = Path(args.zips_dir) / f"{name}.zip"
            if not zip_path.exists():
                raise RuntimeError(f"{zip_path} missing after download (rclone failed for {name})")
            if i + 1 < len(to_do):
                threads[i + 1] = _launch_download(to_do[i + 1], args.zips_dir)

            t0 = time.perf_counter()
            log(f"[{name}] zip ready ({zip_path.stat().st_size / 1e9:.1f} GB), extracting...")
            extract(zip_path, Path(args.extract_dir))
            kf_dir = Path(args.extract_dir) / name
            items = collect_items(kf_dir)
            items = [it for it in items if (it[1], it[2]) not in writer.done]
            log(f"[{name}] {len(items)} keyframes to embed")
            n = embed_items(model, preprocess, device, fp16, items, writer, args.workers, bs)
            writer.checkpoint()  # durable snapshot before deleting the group's files

            zip_path.unlink(missing_ok=True)
            shutil.rmtree(kf_dir, ignore_errors=True)
            with open(done_groups_path, "a") as f:
                f.write(name + "\n")
            done_groups.add(name)
            dt = time.perf_counter() - t0
            log(f"[{name}] embedded {n} in {dt / 60:.1f} min; deleted zip + dir "
                f"(total committed {writer.total})")

    writer.write_info(MODEL, dim)
    writer.close()
    log(f"DONE: {writer.total} embeddings in {writer.out_dir}")


if __name__ == "__main__":
    main()
