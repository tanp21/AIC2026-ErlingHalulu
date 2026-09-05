#!/usr/bin/env python3
"""Compare PE-Core and SigLIP2 NaFlex exact-frame retrieval on L-2026 ground truth."""
import argparse
import ctypes
import gc
import json
import sqlite3
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import faiss
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PE_PROFILE = ROOT / "indexes/profiles/l2026"
NAFLEX_PROFILE = ROOT / "indexes/profiles/naflex-l2026"
NAFLEX_MODEL = "timm/ViT-SO400M-16-SigLIP2-naflex"


def load_cases_from_db(db_path):
    from backend.collaboration import EVENT

    sources = {
        "Test": ROOT / "queries/query-p1-groupA",
        "Round 1 - 2026": ROOT / "queries/SOTUYEN1-bo-de-thi.zip",
        "Round 2 - 2026": ROOT / "queries/SOTUYEN2-bo-de-thi.zip",
    }
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    cases = []
    for round_name, source in sources.items():
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
        raise ValueError("no answered P1/SOTUYEN cases")
    return cases


def load_cases(path):
    document = json.loads(Path(path).read_text())
    cases = [{**case, "targets": [target for target in case["targets"] if target[0].startswith("L")]}
             for case in document["cases"]]
    cases = [case for case in cases if case["targets"]]
    if not cases or document.get("scope") != ["P1", "SOTUYEN1", "SOTUYEN2"]:
        raise ValueError("answer artifact is not the approved P1/SOTUYEN scope")
    return cases


def profile_maps(profile):
    return (np.load(profile / "id_video.npy"), np.load(profile / "id_frame.npy"),
            pq.read_table(profile / "video_meta.parquet").to_pylist())


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
    import open_clip
    import torch
    from functools import partial
    from open_clip.layers import LayerNorm
    from open_clip.model import TextTransformer
    from open_clip.tokenizer import HFTokenizer
    from safetensors import safe_open

    cfg = open_clip.get_model_config("ViT-SO400M-16-SigLIP2-naflex")["text_cfg"]
    model = TextTransformer(
        output_dim=1152, context_length=cfg["context_length"], vocab_size=cfg["vocab_size"],
        width=cfg["width"], heads=cfg["heads"], layers=cfg["layers"],
        mlp_ratio=cfg["mlp_ratio"], no_causal_mask=True, proj_bias=True, pool_type="last",
        norm_layer=partial(LayerNorm, eps=1e-6), act_layer=partial(torch.nn.GELU, approximate="tanh"),
    )
    snapshot = next((Path.home() / ".cache/huggingface/hub/models--timm--ViT-SO400M-16-SigLIP2-naflex/snapshots").iterdir())
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


def first_hit(rows, case, video_ids, frames, video_meta, tolerance):
    lookup = {row["video_id"]: (row["video_idx"], row["fps"]) for row in video_meta}
    for rank, item in enumerate(rows, 1):
        for video_id, target_frame in case["targets"]:
            target_video, fps = lookup[video_id]
            if video_ids[item] == target_video and abs(int(frames[item]) - target_frame) / fps <= tolerance:
                return rank
    return None


def metrics(results):
    ranks = [rank for rank in results if rank is not None]
    n = len(results)
    return {"mrr": sum(1 / rank for rank in ranks) / n,
            "recall_at_1": sum(rank <= 1 for rank in ranks) / n,
            "recall_at_10": sum(rank <= 10 for rank in ranks) / n,
            "recall_at_100": sum(rank <= 100 for rank in ranks) / n,
            "hits": len(ranks), "cases": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path,
                        default=ROOT / "artifacts/official_benchmark_answers.json")
    parser.add_argument("--answers-db", type=Path, default=Path("/tmp/aic-collab-server.sqlite3"))
    parser.add_argument("--export-answers", action="store_true")
    parser.add_argument("--child-model", choices=("pe", "naflex"))
    parser.add_argument("--child-start", type=int)
    parser.add_argument("--child-end", type=int)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--vectors-dir", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/naflex/naflex_ensemble_official.json")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--tolerance-seconds", type=float, default=10)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        assert metrics([1, 10, None])["recall_at_10"] == 2 / 3
        assert rrf(np.array([[1, 2]]), np.array([[2, 3]]), .5, 3)[0].tolist() == [2, 1, 3]
        print("self-test OK")
        return

    if args.export_answers:
        cases = load_cases_from_db(args.answers_db)
        args.answers.parent.mkdir(parents=True, exist_ok=True)
        args.answers.write_text(json.dumps({
            "scope": ["P1", "SOTUYEN1", "SOTUYEN2"],
            "source": "Docker collaboration DB snapshot", "cases": cases,
        }, ensure_ascii=False, indent=2) + "\n")
        print(f"exported {len(cases)} cases -> {args.answers}")
        return

    cases = load_cases(args.answers)
    if args.child_model:
        if args.child_start is None or args.child_end is None or args.child_output is None:
            raise ValueError("child mode requires start, end, and output")
        texts = [case["text"] for case in cases[args.child_start:args.child_end]]
        np.save(args.child_output, encode_pe(texts) if args.child_model == "pe" else encode_naflex(texts))
        return
    pe_ids, pe_frames, pe_meta = profile_maps(PE_PROFILE)
    nf_ids, nf_frames, nf_meta = profile_maps(NAFLEX_PROFILE)
    if not (np.array_equal(pe_ids, nf_ids) and np.array_equal(pe_frames, nf_frames)):
        raise ValueError("PE-Core and NaFlex profiles have different row mappings")
    texts = [case["text"] for case in cases]
    if args.vectors_dir:
        pe_vectors = np.vstack([np.load(path) for path in sorted(args.vectors_dir.glob("pe_*.npy"))])
        nf_vectors = np.vstack([np.load(path) for path in sorted(args.vectors_dir.glob("naflex_*.npy"))])
        raw_cases = json.loads(args.answers.read_text())["cases"]
        keep = [any(video_id.startswith("L") for video_id, _ in case["targets"])
                for case in raw_cases]
        if len(pe_vectors) == len(keep):
            pe_vectors, nf_vectors = pe_vectors[keep], nf_vectors[keep]
        if len(pe_vectors) != len(cases) or len(nf_vectors) != len(cases):
            raise ValueError("query vector count does not match official cases")
    else:
        print(f"encoding {len(cases)} official P1/SOTUYEN events with PE-Core", flush=True)
        pe_vectors = encode_isolated("pe", texts, args.answers)
        print(f"encoding {len(cases)} official P1/SOTUYEN events with NaFlex", flush=True)
        nf_vectors = encode_isolated("naflex", texts, args.answers)
    pe_index, nf_index = faiss.read_index(str(PE_PROFILE / "flat.ip")), faiss.read_index(str(NAFLEX_PROFILE / "flat.ip"))
    pe_ranked, nf_ranked = ranks(pe_index, pe_vectors, args.k), ranks(nf_index, nf_vectors, args.k)
    methods = {"pe_core": pe_ranked, "naflex": nf_ranked}
    methods.update({f"rrf_pe{weight:.2f}": rrf(pe_ranked, nf_ranked, weight, args.k)
                    for weight in (0.25, 0.5, 0.75)})
    report = {"source": f"{args.answers}: P1, SOTUYEN1, SOTUYEN2",
              "metric": "same video and within temporal tolerance", "tolerance_seconds": args.tolerance_seconds,
              "k": args.k, "cases": cases, "methods": {}}
    for name, ranked in methods.items():
        found = [first_hit(row, case, pe_ids, pe_frames, pe_meta, args.tolerance_seconds)
                 for row, case in zip(ranked, cases)]
        report["methods"][name] = {**metrics(found), "ranks": found}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["methods"], indent=2))


if __name__ == "__main__":
    main()
