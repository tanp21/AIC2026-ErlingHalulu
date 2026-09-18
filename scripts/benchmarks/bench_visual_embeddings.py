#!/usr/bin/env python3
"""Compare PE-Core, SigLIP2 NaFlex, and SigLIP2 GOPT exact-frame retrieval on
all 3 rounds of 2026 competition ground truth.

Extends scripts/benchmarks/bench_naflex_ensemble.py (PE-Core vs NaFlex only,
Round 1-2 - 2026) to a 3-way comparison across Round 1, 2, and 3 - 2026, plus
pairwise and 3-way RRF ensembles. See docs/reports/visual-embedding-model-
benchmark-2026.md for the write-up.
"""
import argparse
import ctypes
import gc
import json
import re
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PE_PROFILE = ROOT / "indexes/profiles/l2026"
NAFLEX_PROFILE = ROOT / "indexes/profiles/naflex-l2026"
GOPT_PROFILE = ROOT / "indexes/profiles/gopt-l2026"
NAFLEX_MODEL = "ViT-SO400M-16-SigLIP2-naflex"
GOPT_MODEL = "ViT-gopt-16-SigLIP2-384"

ROUND_SOURCES = {
    "Test": ROOT / "queries/query-p1-groupA",
    "Round 1 - 2026": ROOT / "queries/SOTUYEN1-bo-de-thi.zip",
    "Round 2 - 2026": ROOT / "queries/SOTUYEN2-bo-de-thi.zip",
    "Round 3 - 2026": None,  # no local query zip; ground truth comes from the DB only
}
REPORTED_ROUNDS = ["Round 1 - 2026", "Round 2 - 2026", "Round 3 - 2026"]

# TRAKE prompts mark events inconsistently across rounds: "E1", "E1:", "Cảnh 1:".
EVENT = re.compile(r"(?:E|Cảnh|Sự kiện)\s*\d+\s*:?", re.IGNORECASE)


def load_cases_from_db(db_path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    cases = []
    for round_name in ROUND_SOURCES:
        rows = db.execute("""select q.filename, q.kind, q.prompt, q.answers_json
                             from queries q join rounds r on r.id=q.round_id where r.name=?""",
                          (round_name,)).fetchall()
        for row in rows:
            answers = json.loads(row["answers_json"])
            if not answers:
                continue
            prompt = row["prompt"]
            events = [prompt]
            if row["kind"] == "trake":
                matches = list(EVENT.finditer(prompt))
                if matches:
                    events = [prompt[m.end(): matches[i + 1].start() if i + 1 < len(matches) else None].strip()
                              for i, m in enumerate(matches)]
            for event_index, text in enumerate(events):
                targets = []
                for answer in answers:
                    frames = answer.get("frame_ids", [answer.get("frame_id")])
                    if (answer["video_id"].startswith("L") and event_index < len(frames)
                            and frames[event_index] is not None):
                        targets.append((answer["video_id"], int(frames[event_index])))
                if targets:
                    cases.append({"id": f"{round_name}/{row['filename']}:{event_index + 1}",
                                  "text": text, "targets": targets})
    if not cases:
        raise ValueError("no answered cases across Test/Round 1-3 - 2026")
    return cases


def load_cases(path):
    document = json.loads(Path(path).read_text())
    cases = [{**case, "targets": [target for target in case["targets"] if target[0].startswith("L")]}
             for case in document["cases"]]
    cases = [case for case in cases if case["targets"]]
    if not cases or document.get("scope") != ["P1", "SOTUYEN1", "SOTUYEN2", "Round 3 - 2026"]:
        raise ValueError("answer artifact is not the approved 3-round scope")
    return cases


def profile_maps(profile):
    return (np.load(profile / "id_video.npy"), np.load(profile / "id_frame.npy"),
            pq.read_table(profile / "video_meta.parquet").to_pylist())


def ensure_snapshot(model_id):
    """Return the local HF snapshot dir for timm/<model_id>, downloading it first if absent.

    Fetches the checkpoint/config via create_model_and_transforms (tolerating a build
    failure from a NaFlex-only vision_cfg field some checkpoints carry -- the config and
    weights are still downloaded before that point) and the tokenizer files separately,
    since a failed model build skips open_clip's own tokenizer download step.
    """
    snapshots = Path.home() / ".cache/huggingface/hub" / f"models--timm--{model_id}" / "snapshots"
    needs_download = not snapshots.exists() or not any(snapshots.iterdir())
    if needs_download:
        import open_clip
        print(f"downloading timm/{model_id} (first use on this machine)...", flush=True)
        try:
            open_clip.create_model_and_transforms(f"hf-hub:timm/{model_id}", device="cpu")
        except TypeError:
            pass  # config/weights already downloaded; only the vision-tower build failed
    snapshot = next(snapshots.iterdir())
    if not (snapshot / "tokenizer.json").exists() and not (snapshot / "tokenizer.model").exists():
        import open_clip
        open_clip.get_tokenizer(f"hf-hub:timm/{model_id}")
    return snapshot


def encode_pe(texts):
    import open_clip
    import torch
    from open_clip.model import TextTransformer
    from safetensors import safe_open

    # The 9.1-GB checkpoint contains a huge visual tower; only text.* is needed here.
    cfg = open_clip.get_model_config("PE-Core-bigG-14-448")["text_cfg"]
    model = TextTransformer(output_dim=1280, **cfg)
    snapshot = next((Path.home() / ".cache/huggingface/hub/models--timm--PE-Core-bigG-14-448/snapshots").iterdir())
    with safe_open(str(snapshot / "open_clip_model.safetensors"), framework="pt") as weights:
        state = {key[5:]: weights.get_tensor(key) for key in weights.keys() if key.startswith("text.")}
    model.load_state_dict(state)
    del state
    model.eval()
    with torch.inference_mode():
        tokenizer = open_clip.get_tokenizer("hf-hub:timm/PE-Core-bigG-14-448")
        vectors = [torch.nn.functional.normalize(model(tokenizer(texts[i:i + 8])).float(), dim=-1).numpy()
                   for i in range(0, len(texts), 8)]
    del model
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return np.vstack(vectors).astype("float32")


def encode_naflex(texts):
    """Text-tower-only load: the installed open_clip (3.3.0) can't build this
    checkpoint's full model (its vision_cfg carries a NaFlex-only `image_seq_len`
    field the installed CLIPVisionCfg rejects), and the bundled local model-name
    registry no longer has a "...-naflex" entry to read text_cfg from either. So
    read text_cfg straight out of the downloaded open_clip_config.json and build
    only the TextTransformer, skipping the vision tower entirely."""
    import open_clip
    import torch
    from functools import partial
    from open_clip.transformer import LayerNorm
    from open_clip.model import TextTransformer
    from open_clip.tokenizer import HFTokenizer

    snapshot = ensure_snapshot(NAFLEX_MODEL)
    model_cfg = json.loads((snapshot / "open_clip_config.json").read_text())["model_cfg"]
    cfg = model_cfg["text_cfg"]
    model = TextTransformer(
        output_dim=model_cfg["embed_dim"], context_length=cfg["context_length"], vocab_size=cfg["vocab_size"],
        width=cfg["width"], heads=cfg["heads"], layers=cfg["layers"],
        mlp_ratio=cfg["mlp_ratio"], no_causal_mask=True, proj_bias=True, pool_type="last",
        norm_layer=partial(LayerNorm, eps=1e-6), act_layer=partial(torch.nn.GELU, approximate="tanh"),
    )
    from safetensors import safe_open
    with safe_open(str(snapshot / "open_clip_model.safetensors"), framework="pt") as weights:
        state = {key[5:]: weights.get_tensor(key) for key in weights.keys() if key.startswith("text.")}
    model.load_state_dict(state)
    del state
    model.eval()
    tokenizer = HFTokenizer(str(snapshot), context_length=cfg["context_length"], clean="canonicalize",
                            local_files_only=True)
    with torch.inference_mode():
        vectors = [torch.nn.functional.normalize(model(tokenizer(texts[i:i + 8])).float(), dim=-1).cpu().numpy()
                   for i in range(0, len(texts), 8)]
    del model
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return np.vstack(vectors).astype("float32")


def encode_gopt(texts):
    """Full model load (small query count makes the lightweight partial-load trick unnecessary)."""
    import open_clip
    import torch

    ensure_snapshot(GOPT_MODEL)
    model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:timm/{GOPT_MODEL}", device="cpu")
    tokenizer = open_clip.get_tokenizer(f"hf-hub:timm/{GOPT_MODEL}")
    model.eval()
    with torch.inference_mode():
        vectors = [torch.nn.functional.normalize(model.encode_text(tokenizer(texts[i:i + 8])).float(), dim=-1).numpy()
                   for i in range(0, len(texts), 8)]
    del model
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    return np.vstack(vectors).astype("float32")


def encode_isolated(model_name, texts, answers):
    """Exit after each small text batch so large CPU model allocations are reclaimed."""
    vectors = []
    with tempfile.TemporaryDirectory(prefix="aic-query-") as temp:
        for start in range(0, len(texts), 8):
            output = Path(temp) / f"{start}.npy"
            subprocess.run([sys.executable, __file__, "--answers", str(answers),
                            "--child-model", model_name, "--child-start", str(start),
                            "--child-end", str(min(start + 8, len(texts))), "--child-output", str(output)],
                           check=True)
            vectors.append(np.load(output))
    return np.vstack(vectors)


def ranks(index, vectors, k):
    return index.search(vectors.astype("float32"), k)[1]


def rrf(left, right, weight, k, constant=60):
    rows = []
    for a, b in zip(left, right):
        scores = {}
        for rank, item in enumerate(a, 1):
            scores[int(item)] = scores.get(int(item), 0.0) + weight / (constant + rank)
        for rank, item in enumerate(b, 1):
            scores[int(item)] = scores.get(int(item), 0.0) + (1 - weight) / (constant + rank)
        rows.append(np.array([item for item, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]],
                             dtype=np.int64))
    return rows


def rrf_equal(ranked_lists, k, constant=60):
    """Equal-weight reciprocal-rank fusion across N ranked lists."""
    rows = []
    for case_rows in zip(*ranked_lists):
        scores = {}
        for row in case_rows:
            for rank, item in enumerate(row, 1):
                scores[int(item)] = scores.get(int(item), 0.0) + 1 / (constant + rank)
        rows.append(np.array([item for item, _ in sorted(scores.items(), key=lambda x: -x[1])[:k]],
                             dtype=np.int64))
    return rows


def first_hit(rows, case, video_ids, frames, video_meta, tolerance):
    lookup = {row["video_id"]: (row["video_idx"], row["fps"]) for row in video_meta}
    for rank, item in enumerate(rows, 1):
        for video_id, target_frame in case["targets"]:
            target_video, fps = lookup[video_id]
            if video_ids[item] == target_video and abs(int(frames[item]) - target_frame) / fps <= tolerance:
                return rank
    return None


def metrics(results):
    ranks_found = [rank for rank in results if rank is not None]
    n = len(results)
    if n == 0:
        return {"mrr": None, "recall_at_1": None, "recall_at_5": None,
                "recall_at_10": None, "recall_at_100": None, "hits": 0, "cases": 0}
    return {"mrr": sum(1 / rank for rank in ranks_found) / n,
            "recall_at_1": sum(rank <= 1 for rank in ranks_found) / n,
            "recall_at_5": sum(rank <= 5 for rank in ranks_found) / n,
            "recall_at_10": sum(rank <= 10 for rank in ranks_found) / n,
            "recall_at_100": sum(rank <= 100 for rank in ranks_found) / n,
            "hits": len(ranks_found), "cases": n}


def round_of(case_id):
    return case_id.split("/", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path,
                        default=ROOT / "artifacts/official_benchmark_answers_v2.json")
    parser.add_argument("--answers-db", type=Path, default=Path("/tmp/aic-collab-server.sqlite3"))
    parser.add_argument("--export-answers", action="store_true")
    parser.add_argument("--child-model", choices=("pe", "naflex", "gopt"))
    parser.add_argument("--child-start", type=int)
    parser.add_argument("--child-end", type=int)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--vectors-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/visual/visual_embedding_benchmark_2026.json")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--tolerance-seconds", type=float, default=10)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert metrics([1, 10, None])["recall_at_10"] == 2 / 3
        assert rrf(np.array([[1, 2]]), np.array([[2, 3]]), .5, 3)[0].tolist() == [2, 1, 3]
        assert rrf_equal([np.array([[1, 2]]), np.array([[2, 3]]), np.array([[3, 1]])], 3)[0].tolist() == [1, 2, 3]
        print("self-test OK")
        return

    if args.export_answers:
        cases = load_cases_from_db(args.answers_db)
        args.answers.parent.mkdir(parents=True, exist_ok=True)
        args.answers.write_text(json.dumps({
            "scope": ["P1", "SOTUYEN1", "SOTUYEN2", "Round 3 - 2026"],
            "source": "Docker collaboration DB snapshot (aic_hcmc_collab-data volume)", "cases": cases,
        }, ensure_ascii=False, indent=2) + "\n")
        print(f"exported {len(cases)} cases -> {args.answers}")
        return

    cases = load_cases(args.answers)
    if args.child_model:
        if args.child_start is None or args.child_end is None or args.child_output is None:
            raise ValueError("child mode requires start, end, and output")
        texts = [case["text"] for case in cases[args.child_start:args.child_end]]
        encoders = {"pe": encode_pe, "naflex": encode_naflex, "gopt": encode_gopt}
        np.save(args.child_output, encoders[args.child_model](texts))
        return

    pe_ids, pe_frames, pe_meta = profile_maps(PE_PROFILE)
    nf_ids, nf_frames, nf_meta = profile_maps(NAFLEX_PROFILE)
    go_ids, go_frames, go_meta = profile_maps(GOPT_PROFILE)
    if not (np.array_equal(pe_ids, nf_ids) and np.array_equal(pe_frames, nf_frames)
            and np.array_equal(pe_ids, go_ids) and np.array_equal(pe_frames, go_frames)):
        raise ValueError("PE-Core, NaFlex, and GOPT profiles have different row mappings")

    texts = [case["text"] for case in cases]
    if args.vectors_dir:
        pe_vectors = np.vstack([np.load(path) for path in sorted(args.vectors_dir.glob("pe_*.npy"))])
        nf_vectors = np.vstack([np.load(path) for path in sorted(args.vectors_dir.glob("naflex_*.npy"))])
        go_vectors = np.vstack([np.load(path) for path in sorted(args.vectors_dir.glob("gopt_*.npy"))])
    else:
        print(f"encoding {len(cases)} events with PE-Core", flush=True)
        pe_vectors = encode_isolated("pe", texts, args.answers)
        print(f"encoding {len(cases)} events with NaFlex", flush=True)
        nf_vectors = encode_isolated("naflex", texts, args.answers)
        print(f"encoding {len(cases)} events with GOPT", flush=True)
        go_vectors = encode_isolated("gopt", texts, args.answers)

    pe_index = faiss.read_index(str(PE_PROFILE / "flat.ip"))
    nf_index = faiss.read_index(str(NAFLEX_PROFILE / "flat.ip"))
    go_index = faiss.read_index(str(GOPT_PROFILE / "flat.ip"))
    pe_ranked = ranks(pe_index, pe_vectors, args.k)
    nf_ranked = ranks(nf_index, nf_vectors, args.k)
    go_ranked = ranks(go_index, go_vectors, args.k)

    methods = {"pe_core": pe_ranked, "naflex": nf_ranked, "gopt": go_ranked}
    methods.update({f"rrf_pe_naflex_{weight:.2f}": rrf(pe_ranked, nf_ranked, weight, args.k)
                    for weight in (0.25, 0.5, 0.75)})
    methods.update({f"rrf_pe_gopt_{weight:.2f}": rrf(pe_ranked, go_ranked, weight, args.k)
                    for weight in (0.25, 0.5, 0.75)})
    methods.update({f"rrf_naflex_gopt_{weight:.2f}": rrf(nf_ranked, go_ranked, weight, args.k)
                    for weight in (0.25, 0.5, 0.75)})
    methods["rrf_all3_equal"] = rrf_equal([pe_ranked, nf_ranked, go_ranked], args.k)

    report = {"source": f"{args.answers}: P1, SOTUYEN1, SOTUYEN2, Round 3 - 2026",
              "metric": "same video and within temporal tolerance", "tolerance_seconds": args.tolerance_seconds,
              "k": args.k, "cases": cases, "methods": {}, "by_round": {}}
    for name, ranked in methods.items():
        found = [first_hit(row, case, pe_ids, pe_frames, pe_meta, args.tolerance_seconds)
                 for row, case in zip(ranked, cases)]
        report["methods"][name] = {**metrics(found), "ranks": found}
        by_round = {}
        for round_name in REPORTED_ROUNDS:
            round_found = [rank for rank, case in zip(found, cases) if round_of(case["id"]) == round_name]
            by_round[round_name] = metrics(round_found)
        aggregate_2026 = [rank for rank, case in zip(found, cases) if round_of(case["id"]) in REPORTED_ROUNDS]
        by_round["Round 1-3 - 2026 (aggregate)"] = metrics(aggregate_2026)
        report["by_round"][name] = by_round

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["by_round"], indent=2))


if __name__ == "__main__":
    main()
