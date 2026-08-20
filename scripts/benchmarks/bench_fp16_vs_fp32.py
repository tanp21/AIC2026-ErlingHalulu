import argparse
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import open_clip


def parse_args():
    p = argparse.ArgumentParser(description="fp16 vs fp32 embedding benchmark")
    p.add_argument("--keyframes-dir", default="data/2026/keyframes")
    p.add_argument("--model", default="hf-hub:timm/PE-Core-bigG-14-448")
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--n-images", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--topk", type=int, nargs="+", default=[100, 1000])
    return p.parse_args()


class ImgDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
        return self.preprocess(im)


def embed(model, dl, device, fp16):
    ctx = torch.autocast("cuda", dtype=torch.float16) if fp16 else nullcontext()
    outs = []
    with torch.inference_mode():
        for images in dl:
            images = images.to(device, non_blocking=True)
            with ctx:
                feats = model.encode_image(images, normalize=True)
            outs.append(feats.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def autotune_batch(model, device, max_b):
    best = 1
    for b in [1, 2, 4, 8, 16, 32]:
        if b > max_b:
            break
        try:
            x = torch.randn(b, 3, 448, 448, device=device)
            with torch.inference_mode():
                model.encode_image(x, normalize=True)
            best = b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break
    return best


def topk_overlap(a, b, k):
    idx_a = np.argsort(-(a @ a.T), axis=1)[:, 1 : k + 1]
    idx_b = np.argsort(-(b @ b.T), axis=1)[:, 1 : k + 1]
    ov = np.array([len(np.intersect1d(ta, tb)) for ta, tb in zip(idx_a, idx_b)]) / k
    return ov.mean(), ov.std()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and args.workers > 0:
        torch.multiprocessing.set_start_method("spawn", force=True)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"Loading {args.model} (fp32 weights) ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model, precision="fp32", device=device
    )
    model.eval()

    batch = autotune_batch(model, device, args.batch)
    print(f"Autotuned batch: {batch}", flush=True)

    x = torch.randn(batch, 3, 448, 448, device=device)
    with torch.inference_mode():
        _ = model.encode_image(x, normalize=True)
        with torch.autocast("cuda", dtype=torch.float16):
            _ = model.encode_image(x, normalize=True)
    torch.cuda.synchronize()

    all_paths = [
        p
        for p in sorted(Path(args.keyframes_dir).rglob("*.png"))
        if not p.name.endswith(".tmp.png")
    ]
    print(f"Found {len(all_paths)} keyframes; benchmark {args.n_runs} x {args.n_images}", flush=True)

    rng = random.Random(args.seed)
    cos_all = []
    overlap_runs = {k: [] for k in args.topk}
    t_fp16 = []
    t_fp32 = []

    for run in range(args.n_runs):
        paths = rng.sample(all_paths, args.n_images)
        ds = ImgDataset(paths, preprocess)

        res = {}
        for name, fp16 in [("fp32", False), ("fp16", True)]:
            dl = DataLoader(
                ds, batch_size=batch, num_workers=args.workers,
                pin_memory=device == "cuda",
                persistent_workers=args.workers > 0,
            )
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            emb = embed(model, dl, device, fp16)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            res[name] = emb
            (t_fp32 if name == "fp32" else t_fp16).append(args.n_images / dt)
            print(
                f"  run {run+1}/{args.n_runs} {name}: "
                f"{args.n_images / dt:.2f} img/s ({dt:.1f}s)",
                flush=True,
            )

        e32 = res["fp32"]
        e16 = res["fp16"]
        cos = np.einsum("ij,ij->i", e16, e32)  # both L2-normalized -> cosine
        cos_all.append(cos)

        for k in args.topk:
            m, s = topk_overlap(e16, e32, k)
            overlap_runs[k].append(m)

        print(
            f"run {run+1}/{args.n_runs}: "
            f"fp32={t_fp32[-1]:.2f} img/s  fp16={t_fp16[-1]:.2f} img/s  "
            f"cos_mean={cos.mean():.5f}  cos_min={cos.min():.5f}  "
            + "  ".join(f"top{k}={overlap_runs[k][-1]:.4f}" for k in args.topk),
            flush=True,
        )

    cos_all = np.concatenate(cos_all)
    t_fp16 = np.array(t_fp16)
    t_fp32 = np.array(t_fp32)

    print("\n" + "=" * 60)
    print(f"COSINE fp16-vs-fp32 (same image): mean={cos_all.mean():.6f} "
          f"std={cos_all.std():.6f} min={cos_all.min():.6f}")
    for k in args.topk:
        arr = np.array(overlap_runs[k])
        print(f"TOP-{k} overlap: mean={arr.mean():.4f} std={arr.std():.4f} "
              f"({k}/{k} neighbors identical)")
    print(f"THROUGHPUT fp16: {t_fp16.mean():.2f} +- {t_fp16.std():.2f} img/s")
    print(f"THROUGHPUT fp32: {t_fp32.mean():.2f} +- {t_fp32.std():.2f} img/s")
    print(f"SPEEDUP fp16 vs fp32: {t_fp16.mean() / t_fp32.mean():.2f}x")
    print(f"ETA full 289875 @ fp16: {289875 / t_fp16.mean() / 3600:.1f} h")
    print(f"ETA full 289875 @ fp32: {289875 / t_fp32.mean() / 3600:.1f} h")
    print("=" * 60)


if __name__ == "__main__":
    main()
