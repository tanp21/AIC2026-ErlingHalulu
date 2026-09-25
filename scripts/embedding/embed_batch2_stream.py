"""Stream Batch 2 2026 through both embedding models one group at a time.

    uv run python -m scripts.embedding.embed_batch2_stream \
        --year 2026 --groups S01,M01,M02,M03,M04,M05,M06,M07,M08,M09,M10 \
        --pe-core-out data/2026/embeddings_batch2/pe-core \
        --naflex-out data/2026/embeddings_batch2/naflex \
        --keyframes-tmp-dir data/2026/keyframes_tmp

For each group in order: extract that group's shot-based keyframes to a *temporary* local
directory, embed them with PE-Core, embed them with NaFlex (both models stay loaded in VRAM
the whole run -- no per-group reload), then delete that group's keyframe PNGs before moving
to the next group.

This exists because the team already paid for the alternative once: per
``docs/reports/server-performance-optimization-2026-09.md`` §3, keeping every extracted
keyframe on disk (``data/<year>/keyframes/*.png``) reached **604 GB** for the existing
855,632-keyframe corpus and had to be deleted outright -- nothing in the live-serving path
needs it, and the team's own stated direction ("the new embedding plan") is to pull frames
live rather than stage them all first. ``scripts/extraction/extract_batch2_keyframes.py`` +
``scripts/embedding/embed_batch2_pe_core.py`` + ``scripts/embedding/embed_batch2_naflex.py``
still exist standalone (useful for a canary on 1-2 videos, or if disk is genuinely not a
constraint), but this streaming entry point is the one to use for the full batch: at most one
group's keyframes ever sit on disk, so the container only needs room for the raw videos +
one group's worth of PNGs, not the whole batch's.

Both ``--pe-core-out`` and ``--naflex-out`` are resume-safe (``ShardWriter``, reads existing
``meta.parquet``): rerunning after an interruption skips groups whose keyframes are already
fully committed to both outputs.
"""
import argparse
import json
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from scripts.embedding.embed_batch2_pe_core import (
    IMAGE_SIZE as PE_CORE_IMAGE_SIZE,
    MODEL as PE_CORE_MODEL,
    ShardWriter as PECoreShardWriter,
    autotune_batch as autotune_pe_core,
    collect_items as collect_pe_core_items,
)
from scripts.embedding.embed_batch2_naflex import (
    DIM as NAFLEX_DIM,
    MODEL as NAFLEX_MODEL,
    ShardWriter as NaflexShardWriter,
    autotune_batch as autotune_naflex,
    collect_items as collect_naflex_items,
    encode_paths as naflex_encode_paths,
    load_model as load_naflex_model,
)
from scripts.extraction.extract_batch2_keyframes import compute_keyframes, extract_video_batch

import open_clip

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("AIC_DATA_DIR", ROOT / "data"))

DEFAULT_PE_CORE_BATCH_CANDIDATES = "2,4,8,16,32,48,64,96,128,160,192,224,256,320"
DEFAULT_NAFLEX_BATCH_CANDIDATES = "4,8,16,24,32,48,64,96,128,160,192"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--year", default="2026")
    p.add_argument("--groups", required=True, help="comma-separated groups, in the order "
                        "they will be processed, e.g. S01,M01,M02,...,M10")
    p.add_argument("--data-dir", type=Path, default=DATA)
    p.add_argument("--keyframes-tmp-dir", type=Path, default=None,
                        help="default: <data-dir>/<year>/keyframes_tmp")
    p.add_argument("--pe-core-out", type=Path, required=True)
    p.add_argument("--naflex-out", type=Path, required=True)
    p.add_argument("--pe-core-model", default=PE_CORE_MODEL)
    p.add_argument("--naflex-model", default=NAFLEX_MODEL)
    p.add_argument("--pe-core-batch-candidates", default=DEFAULT_PE_CORE_BATCH_CANDIDATES)
    p.add_argument("--naflex-batch-candidates", default=DEFAULT_NAFLEX_BATCH_CANDIDATES)
    p.add_argument("--pe-core-batch-size", type=int, default=0)
    p.add_argument("--naflex-batch-size", type=int, default=0)
    p.add_argument("--pe-core-workers", type=int, default=8,
                        help="DataLoader workers for PE-Core image decode/preprocess; "
                             "match vCPUs so the GPU is never left waiting on CPU decode")
    p.add_argument("--naflex-workers", type=int, default=8,
                        help="threads for NaFlex image decode/preprocess (same reasoning "
                             "as --pe-core-workers)")
    p.add_argument("--shard-size", type=int, default=50000)
    p.add_argument("--extract-workers", type=int, default=8,
                        help="videos ffmpeg-extracted in parallel per group")
    p.add_argument("--extract-threads", type=int, default=2, help="ffmpeg threads per video")
    p.add_argument("--keep-keyframes", action="store_true",
                        help="don't delete each group's keyframes after embedding "
                             "(defeats the point of streaming -- for debugging only)")
    return p.parse_args(argv)


def extract_group_keyframes(group, shots_dir, videos_dir, out_dir, workers, threads):
    """Same algorithm as extract_batch2_keyframes.py, scoped to one group."""
    jobs = []
    for shot_path in sorted((shots_dir / group).glob("*.json")):
        video_id = shot_path.stem
        video_path = videos_dir / group / f"{video_id}.mp4"
        if not video_path.exists():
            log(f"MISSING video for {shot_path} -- skip")
            continue
        info = json.loads(shot_path.read_text())
        frames = compute_keyframes(info["shots"])
        jobs.append((video_path, frames, out_dir / group / video_id, info["fps"]))

    from concurrent.futures import ThreadPoolExecutor, as_completed

    failures = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(extract_video_batch, video_path, frames, dest, 1, fps, threads): video_path
            for video_path, frames, dest, fps in jobs
        }
        for future in as_completed(futures):
            video_path = futures[future]
            error = future.result()
            if error is not None:
                failures[video_path.name] = error
    if failures:
        raise RuntimeError(f"{group}: keyframe extraction failed for {failures}")
    return sum(len(frames) for _, frames, _, _ in jobs)


def load_pe_core_model(model_id, device):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_id, precision="fp16", device=device)
    model.eval()
    return model, preprocess


def embed_pe_core_group(model, preprocess, device, items, writer, batch_size, workers=0):
    from torch.utils.data import DataLoader

    from scripts.embedding.embed_batch2_pe_core import KeyframeDataset

    ds = KeyframeDataset(items, preprocess)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=workers,
                    persistent_workers=workers > 0, drop_last=False)
    ctx = torch.autocast("cuda", dtype=torch.float16)
    n = 0
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
    return n


def embed_naflex_group(model, preprocess, device, items, writer, batch_size, workers=0):
    paths = [it[0] for it in items]
    consumed = 0
    n = 0
    for vectors, batch_size in naflex_encode_paths(
            model, preprocess, paths, device, batch_size, workers=workers):
        batch_items = items[consumed:consumed + len(vectors)]
        for (_, video_id, frame_idx), vector in zip(batch_items, vectors):
            writer.add(vector, video_id, frame_idx)
            n += 1
        consumed += len(vectors)
    return n


def pe_core_embed_dim(model, device):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        return model.encode_image(
            torch.zeros(1, 3, PE_CORE_IMAGE_SIZE, PE_CORE_IMAGE_SIZE, device=device),
            normalize=True,
        ).shape[1]


def group_keyframe_ids(shots_dir, group):
    """Every (video_id, frame_idx) this group's shots should yield, pure/no side effects."""
    expected = set()
    for shot_path in sorted((shots_dir / group).glob("*.json")):
        video_id = shot_path.stem
        info = json.loads(shot_path.read_text())
        for frame_idx in compute_keyframes(info["shots"]):
            expected.add((video_id, frame_idx))
    return expected


def group_fully_committed(shots_dir, group, pe_core_writer, naflex_writer):
    expected = group_keyframe_ids(shots_dir, group)
    return bool(expected) and expected <= pe_core_writer.done and expected <= naflex_writer.done


def main(argv=None):
    args = parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("streaming embedding requires a CUDA GPU")
    torch.backends.cudnn.enabled = False

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if not groups:
        raise SystemExit("--groups must name at least one group")

    shots_dir = args.data_dir / args.year / "shots"
    videos_dir = args.data_dir / args.year / "videos"
    keyframes_tmp_dir = args.keyframes_tmp_dir or (args.data_dir / args.year / "keyframes_tmp")

    log(f"loading {args.pe_core_model} ...")
    pe_core_model, pe_core_preprocess = load_pe_core_model(args.pe_core_model, device)
    log(f"loading {args.naflex_model} ...")
    naflex_model, naflex_preprocess = load_naflex_model(args.naflex_model, device)

    pe_core_candidates = sorted(int(b) for b in args.pe_core_batch_candidates.split(",") if b.strip())
    naflex_candidates = sorted(int(b) for b in args.naflex_batch_candidates.split(",") if b.strip())
    pe_core_bs = args.pe_core_batch_size or autotune_pe_core(
        pe_core_model, device, True, pe_core_candidates)
    log(f"PE-Core batch size = {pe_core_bs}")

    pe_core_writer = PECoreShardWriter(args.pe_core_out, dim=pe_core_embed_dim(pe_core_model, device),
                                       shard_size=args.shard_size)
    naflex_writer = NaflexShardWriter(args.naflex_out, NAFLEX_DIM, args.shard_size)
    log(f"PE-Core resume: {pe_core_writer.total} committed; "
        f"NaFlex resume: {naflex_writer.total} committed")

    for group in groups:
        if group_fully_committed(shots_dir, group, pe_core_writer, naflex_writer):
            log(f"{group}: already fully embedded in both outputs, skipping")
            continue

        log(f"{group}: extracting keyframes ...")
        started = time.perf_counter()
        n_frames = extract_group_keyframes(
            group, shots_dir, videos_dir, keyframes_tmp_dir,
            args.extract_workers, args.extract_threads)
        log(f"{group}: {n_frames} keyframes extracted in "
            f"{time.perf_counter() - started:.1f}s")

        pe_core_items = collect_pe_core_items(keyframes_tmp_dir, {group}, set())
        pe_core_items = [it for it in pe_core_items if (it[1], it[2]) not in pe_core_writer.done]
        if pe_core_items:
            started = time.perf_counter()
            n = embed_pe_core_group(pe_core_model, pe_core_preprocess, device,
                                    pe_core_items, pe_core_writer, pe_core_bs,
                                    workers=args.pe_core_workers)
            pe_core_writer.checkpoint()
            log(f"{group}: PE-Core embedded {n} in {time.perf_counter() - started:.1f}s")
        else:
            log(f"{group}: PE-Core already has every keyframe")

        naflex_items = collect_naflex_items(keyframes_tmp_dir, {group}, set())
        naflex_items = [it for it in naflex_items if (it[1], it[2]) not in naflex_writer.done]
        if naflex_items:
            naflex_bs = args.naflex_batch_size or autotune_naflex(
                naflex_model, naflex_preprocess, device, naflex_items[0][0], naflex_candidates,
                workers=args.naflex_workers)
            started = time.perf_counter()
            n = embed_naflex_group(naflex_model, naflex_preprocess, device,
                                   naflex_items, naflex_writer, naflex_bs,
                                   workers=args.naflex_workers)
            naflex_writer.checkpoint()
            log(f"{group}: NaFlex embedded {n} in {time.perf_counter() - started:.1f}s "
                f"(batch size {naflex_bs})")
        else:
            log(f"{group}: NaFlex already has every keyframe")

        if not args.keep_keyframes:
            shutil.rmtree(keyframes_tmp_dir / group, ignore_errors=True)
            log(f"{group}: deleted temporary keyframes")

    pe_core_writer.close()
    pe_core_writer.write_info(args.pe_core_model, pe_core_writer.dim)
    naflex_writer.close()
    naflex_writer.write_info(args.naflex_model, NAFLEX_DIM)
    log(f"DONE: PE-Core total {pe_core_writer.total}, NaFlex total {naflex_writer.total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
