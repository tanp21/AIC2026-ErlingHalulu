import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import open_clip


def parse_args():
    p = argparse.ArgumentParser(description="Embed keyframes with PE-Core-bigG-14-448")
    p.add_argument("--keyframes-dir", default="data/2026/keyframes")
    p.add_argument("--out-dir", default="data/2026/embeddings")
    p.add_argument("--model", default="hf-hub:timm/PE-Core-bigG-14-448")
    p.add_argument("--limit", type=int, default=0, help="cap number of keyframes (0 = all)")
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto-tune")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--no-save", action="store_true", help="benchmark only, write nothing")
    p.add_argument("--resume", action="store_true", help="skip keyframes already in meta.csv")
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


def collect_items(keyframes_dir: Path):
    items = []
    for p in sorted(keyframes_dir.rglob("*.png")):
        if p.name.endswith(".tmp.png"):
            continue
        video_id = p.parent.name
        frame_idx = int(p.stem)
        items.append((p, video_id, frame_idx))
    return items


def autotune_batch(model, device, fp16, max_b=32):
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    best = 1
    for b in [1, 2, 4, 8, 16, 32]:
        try:
            x = torch.randn(b, 3, 448, 448, device=device)
            with torch.inference_mode(), ctx:
                model.encode_image(x, normalize=True)
            best = b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return best


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp16 = (not args.no_fp16) and device == "cuda"
    # ViT has one tiny conv (patch embed); cuDNN 9.20 cu13 wheels hit a
    # sublibrary version mismatch, native kernels are negligible here.
    torch.backends.cudnn.enabled = False

    print(f"Loading model {args.model} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, precision="fp16" if fp16 else "fp32", device=device
    )
    model.eval()

    items = collect_items(Path(args.keyframes_dir))
    total = len(items)
    print(f"Found {total} keyframes")

    if args.limit > 0:
        items = items[: args.limit]

    out_dir = Path(args.out_dir)
    meta_csv = out_dir / "meta.csv"
    done = set()
    if args.resume and meta_csv.exists():
        with open(meta_csv) as f:
            next(f, None)
            for line in f:
                vid, fid = line.rstrip("\n").split(",")
                done.add((vid, int(fid)))
        items = [it for it in items if (it[1], it[2]) not in done]
        print(f"Resume: {len(done)} already done, {len(items)} remaining")

    if not items:
        print("Nothing to do.")
        return

    bs = args.batch_size if args.batch_size > 0 else autotune_batch(model, device, fp16)
    print(f"Batch size: {bs}  fp16: {fp16}")

    ds = KeyframeDataset(items, preprocess)
    dl = DataLoader(
        ds,
        batch_size=bs,
        num_workers=args.workers,
        pin_memory=device == "cuda",
        persistent_workers=args.workers > 0,
        drop_last=False,
    )

    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()

    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_f = open(meta_csv, "a")
        if not meta_csv.exists() or meta_csv.stat().st_size == 0:
            meta_f.write("video_id,frame_idx\n")

    shard_buf = []
    shard_idx = 0
    n_processed = 0
    n_batches = -(-len(items) // bs)
    warmup_batches = min(5, n_batches // 3)
    t_start = time.perf_counter()
    timed_start = None

    def flush_shard(final=False):
        nonlocal shard_buf, shard_idx
        if not shard_buf or args.no_save:
            return
        arr = np.ascontiguousarray(np.stack(shard_buf), dtype=np.float32)
        np.save(out_dir / f"emb_{shard_idx:06d}.npy", arr)
        shard_idx += 1
        shard_buf = []

    with torch.inference_mode():
        for bi, (images, vids, fids) in enumerate(dl):
            images = images.to(device, non_blocking=True)
            if bi == warmup_batches:
                torch.cuda.synchronize()
                timed_start = time.perf_counter()

            with ctx:
                feats = model.encode_image(images, normalize=True)

            feats = feats.float().cpu().numpy()
            fids = fids.tolist() if isinstance(fids, torch.Tensor) else fids

            for i in range(len(vids)):
                shard_buf.append(feats[i])
                if not args.no_save:
                    meta_f.write(f"{vids[i]},{fids[i]}\n")
                n_processed += 1
                if len(shard_buf) >= args.shard_size:
                    flush_shard()

    flush_shard()

    if not args.no_save:
        meta_f.close()
        info = {
            "model": args.model,
            "dim": int(feats.shape[1]),
            "count": n_processed,
            "shard_size": args.shard_size,
            "normalized": True,
            "dtype": "float32",
        }
        (out_dir / "info.json").write_text(json.dumps(info, indent=2))

    torch.cuda.synchronize()
    t_end = time.perf_counter()

    if timed_start is None:
        timed_start = t_start
    timed_s = t_end - timed_start
    timed_n = max(n_processed - warmup_batches * bs, 1)
    img_s = timed_n / timed_s if timed_s > 0 else 0.0

    print("=" * 60)
    print(f"Embedded : {n_processed} keyframes in {t_end - t_start:.1f}s")
    print(f"Throughput: {img_s:.2f} img/s (steady-state after warmup)")
    eta_h = (total - n_processed) / img_s / 3600 if img_s > 0 else float("inf")
    print(f"ETA full ({total} keyframes): {eta_h:.1f} h")
    print("=" * 60)


if __name__ == "__main__":
    main()
