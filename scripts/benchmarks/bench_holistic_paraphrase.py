#!/usr/bin/env python3
"""Staged benchmark for holistic-caption paraphrasing (no aspect decomposition;
paraphrases carry the same semantics as the approved caption, only reworded).
See docs/reports/visual-holistic-paraphrase-benchmark-2026.md's plan.

Dev = Test + Round 1 + Round 2 (95 cases). Holdout = Round 3 (42 cases), run exactly
once at the end via --stage holdout --run-holdout, always, regardless of dev outcome.
"""
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.subquery_fusion import Hit, combine_vectors, shot_ranks, weighted_shot_rrf, weighted_shot_score  # noqa: E402
from scripts.benchmarks.bench_query_expansion import (  # noqa: E402
    DEV_ROUNDS, GroupResolver, cases_for_split, encode_and_cache, first_hit_frames,
    load_profile, load_shot_indices, retrieve_cached, round_breakdown, run_caption_baseline,
    run_caption_shotdedup_baseline,
)
from scripts.benchmarks.bench_visual_embeddings import REPORTED_ROUNDS, metrics, round_of  # noqa: E402

HOLDOUT_ROUND = "Round 3 - 2026"
K = 1000  # fixed retrieval depth for every list, pre-registered (matches the baselines)

ART_DIR = ROOT / "artifacts/holistic_paraphrase"
TUNING_LOG = ART_DIR / "tuning_log.jsonl"
COMBO_CACHE = ART_DIR / "combo_retrieval_cache.npz"
HOLDOUT_DONE = ART_DIR / "holdout_done.json"
CONFIG_PATH = ROOT / "benchmarks/visual/holistic_paraphrase_config.json"
BENCHMARK_OUT = ROOT / "benchmarks/visual/holistic_paraphrase_benchmark_2026.json"

CAPTION_FIELD = "caption_en_approved"  # kept distinct from paraphrases.json's "caption" field name


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_paraphrase_cases(path):
    data = json.loads(Path(path).read_text())
    cases = []
    for case in data["cases"]:
        ok = [p for p in case["paraphrases"] if p["status"] == "ok"]
        cases.append({"id": case["id"], "round": case["round"], CAPTION_FIELD: case["caption"],
                      "paraphrases": ok})  # already in fixed strategy order from the generator
    return cases, data


def load_targets(answers_path):
    data = json.loads(Path(answers_path).read_text())
    return {c["id"]: c["targets"] for c in data["cases"]}


def attach_targets(cases, targets_by_id):
    for case in cases:
        case["targets"] = targets_by_id[case["id"]]
    return cases


def collect_texts(cases, n_para=4):
    texts = set()
    for case in cases:
        texts.add(case[CAPTION_FIELD])
        texts.update(p["text"] for p in case["paraphrases"][:n_para])
    return sorted(texts)


def cases_with_enough_paraphrases(cases, n_para):
    return [c for c in cases if len(c["paraphrases"]) >= n_para]


def assert_no_holdout_ids(cases, context):
    leaked = [c["id"] for c in cases if round_of(c["id"]) == HOLDOUT_ROUND]
    if leaked:
        raise SystemExit(f"{context}: holdout ids present in a non-holdout run: {leaked[:5]}")


# ---------------------------------------------------------------------------
# Hit construction, per text and per case.
# ---------------------------------------------------------------------------

def hits_for(text, rows_by_text, scores_by_text, id_time, group_resolver, k):
    rows, scores = rows_by_text[text][:k], scores_by_text[text][:k]
    return [Hit(group_id=group_resolver.group_of(int(r)), payload=int(r), time=float(id_time[int(r)]),
               score=float(s)) for r, s in zip(rows, scores)]


def case_texts(case, *, include_caption, n_para):
    texts = ([case[CAPTION_FIELD]] if include_caption else [])
    texts += [p["text"] for p in case["paraphrases"][:n_para]]
    return texts


def default_weights(n_lists, include_caption, scheme):
    if scheme == "equal":
        return [1.0 / n_lists] * n_lists
    if scheme in ("cap50", "cap67"):
        if not include_caption:
            raise ValueError(f"{scheme} requires include_caption=True")
        cap_weight = 0.5 if scheme == "cap50" else 2 / 3
        n_para = n_lists - 1
        return [cap_weight] + [(1 - cap_weight) / n_para] * n_para
    raise ValueError(scheme)


def first_hit_weighted(results, case, meta_by_idx, id_video, id_frame):
    return first_hit_frames([g.rep_hit.payload for g in results], case, meta_by_idx, id_video, id_frame)


# ---------------------------------------------------------------------------
# Per-config runners.
# ---------------------------------------------------------------------------

def run_rrf_config(cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                   group_resolver, *, include_caption, n_para, weights, k):
    ranks = {}
    for case in cases:
        texts = case_texts(case, include_caption=include_caption, n_para=n_para)
        hit_lists = [hits_for(t, rows_by_text, scores_by_text, id_time, group_resolver, K) for t in texts]
        results = weighted_shot_rrf(hit_lists, weights, k=k, video_of=group_resolver.video_of)
        ranks[case["id"]] = first_hit_weighted(results, case, meta_by_idx, id_video, id_frame)
    return ranks


def run_score_config(cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                     group_resolver, *, include_caption, n_para, weights):
    ranks = {}
    for case in cases:
        texts = case_texts(case, include_caption=include_caption, n_para=n_para)
        hit_lists = [hits_for(t, rows_by_text, scores_by_text, id_time, group_resolver, K) for t in texts]
        results = weighted_shot_score(hit_lists, weights, video_of=group_resolver.video_of)
        ranks[case["id"]] = first_hit_weighted(results, case, meta_by_idx, id_video, id_frame)
    return ranks


def combo_cache_key(texts, weights):
    payload = json.dumps({"texts": texts, "weights": [round(w, 6) for w in weights]}, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def run_embedding_mean_config(cases, vectors_by_text, faiss_index, id_time, id_video, id_frame,
                              meta_by_idx, group_resolver, *, include_caption, n_para, weights):
    cache = {}
    if COMBO_CACHE.exists():
        with np.load(COMBO_CACHE) as cached:
            cache = {k: cached[k] for k in cached.files}
    ranks = {}
    updated = False
    for case in cases:
        texts = case_texts(case, include_caption=include_caption, n_para=n_para)
        key = combo_cache_key(texts, weights)
        if key not in cache:
            vecs = np.stack([vectors_by_text[t] for t in texts])
            combined = combine_vectors(vecs, weights)
            _, indexes = faiss_index.search(combined.reshape(1, -1).astype("float32"), K)
            cache[key] = indexes[0]
            updated = True
        hits = [Hit(group_id=group_resolver.group_of(int(r)), payload=int(r), time=float(id_time[int(r)]))
                for r in cache[key]]
        ordered = sorted(shot_ranks(hits).items(), key=lambda kv: kv[1][0])
        rows = [hit.payload for _, (_, hit) in ordered]
        ranks[case["id"]] = first_hit_frames(rows, case, meta_by_idx, id_video, id_frame)
    if updated:
        ART_DIR.mkdir(parents=True, exist_ok=True)
        tmp = COMBO_CACHE.with_suffix(".tmp.npz")
        with open(tmp, "wb") as handle:
            np.savez(handle, **cache)
        tmp.replace(COMBO_CACHE)
    return ranks


# ---------------------------------------------------------------------------
# Metrics: win/loss/tie, sign test, paired bootstrap.
# ---------------------------------------------------------------------------

def reciprocal(rank):
    return 1.0 / rank if rank else 0.0


def win_loss_tie(ranks_a, ranks_b, case_ids):
    """a vs b, by reciprocal rank per case. Positive = a is better."""
    wins = losses = ties = 0
    for cid in case_ids:
        ra, rb = reciprocal(ranks_a[cid]), reciprocal(ranks_b[cid])
        if ra > rb:
            wins += 1
        elif ra < rb:
            losses += 1
        else:
            ties += 1
    return {"wins": wins, "losses": losses, "ties": ties, "n": len(case_ids)}


def sign_test_p_value(wins, losses):
    """Two-sided exact binomial sign test, p=0.5, no scipy dependency."""
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    from math import comb
    total = sum(comb(n, i) for i in range(k + 1))
    return min(1.0, 2 * total / 2 ** n)


def paired_bootstrap_ci(ranks_a, ranks_b, case_ids, metric_fn, n_resamples=2000, seed=0):
    rng = random.Random(seed)
    ids = list(case_ids)
    deltas = []
    for _ in range(n_resamples):
        sample = [rng.choice(ids) for _ in ids]
        a = metric_fn([ranks_a[i] for i in sample])
        b = metric_fn([ranks_b[i] for i in sample])
        deltas.append(a - b)
    deltas.sort()
    lo = deltas[int(0.025 * n_resamples)]
    hi = deltas[min(int(0.975 * n_resamples), n_resamples - 1)]
    return {"delta": metric_fn([ranks_a[i] for i in ids]) - metric_fn([ranks_b[i] for i in ids]),
           "ci_low": lo, "ci_high": hi}


def _mrr(ranks):
    return sum(reciprocal(r) for r in ranks) / len(ranks) if ranks else 0.0


def _recall_at(ranks, n):
    return sum((r is not None and r <= n) for r in ranks) / len(ranks) if ranks else 0.0


def compare_to_baseline(ranks, baseline_ranks, case_ids):
    wlt = win_loss_tie(ranks, baseline_ranks, case_ids)
    p_value = sign_test_p_value(wlt["wins"], wlt["losses"])
    mrr_ci = paired_bootstrap_ci(ranks, baseline_ranks, case_ids, _mrr)
    r1_ci = paired_bootstrap_ci(ranks, baseline_ranks, case_ids,
                                lambda rs: _recall_at(rs, 1))
    return {"win_loss_tie": wlt, "sign_test_p": p_value, "mrr_delta_ci": mrr_ci, "r1_delta_ci": r1_ci}


def log_tuning(entry):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    with open(TUNING_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def evaluate(name, ranks, cases, baseline_ranks, rounds, *, stage, params=None, log=True):
    ids = [c["id"] for c in cases]
    breakdown = round_breakdown(ranks, cases, rounds)
    comparison = compare_to_baseline(ranks, baseline_ranks, ids) if baseline_ranks is not None else None
    entry = {"stage": stage, "name": name, "params": params or {}, "pooled": breakdown["pooled"],
             "comparison": comparison}
    if log:
        log_tuning(entry)
    pooled = breakdown["pooled"]
    line = (f"{name:42s} MRR={pooled['mrr']:.3f} R@1={pooled['recall_at_1']:.3f} "
            f"R@10={pooled['recall_at_10']:.3f}")
    if comparison:
        wlt = comparison["win_loss_tie"]
        line += (f"  vs baseline W/L/T={wlt['wins']}/{wlt['losses']}/{wlt['ties']} "
                f"p={comparison['sign_test_p']:.3f}")
    print(line)
    return entry, breakdown, ranks


# ---------------------------------------------------------------------------
# Stage 0: sanity.
# ---------------------------------------------------------------------------

def stage0_sanity(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                  group_resolver, vectors_by_text, faiss_index):
    approved = run_caption_baseline(dev_cases, rows_by_text, meta_by_idx, id_video, id_frame,
                                    CAPTION_FIELD, K=K)
    approved_dedup = run_caption_shotdedup_baseline(dev_cases, rows_by_text, id_time, meta_by_idx,
                                                    id_video, id_frame, group_resolver, CAPTION_FIELD, K=K)

    rrf_single = run_rrf_config(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                                meta_by_idx, group_resolver, include_caption=True, n_para=0,
                                weights=[1.0], k=60)
    score_single = run_score_config(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                                    meta_by_idx, group_resolver, include_caption=True, n_para=0,
                                    weights=[1.0])
    emb_single = run_embedding_mean_config(dev_cases, vectors_by_text, faiss_index, id_time, id_video,
                                           id_frame, meta_by_idx, group_resolver, include_caption=True,
                                           n_para=0, weights=[1.0])

    checks = [("weighted_shot_rrf([caption])", rrf_single), ("weighted_shot_score([caption])", score_single),
              ("combine_vectors([caption])", emb_single)]
    failed = False
    for name, ranks in checks:
        mismatches = [cid for cid in approved_dedup if approved_dedup[cid] != ranks[cid]]
        if mismatches:
            failed = True
            print(f"STAGE 0 SANITY FAILED: {name} vs caption_approved_shotdedup, "
                  f"{len(mismatches)}/{len(dev_cases)} mismatches: {mismatches[:5]}")
        else:
            print(f"STAGE 0 SANITY OK: {name} == caption_approved_shotdedup on all {len(dev_cases)} dev cases")
    if failed:
        raise SystemExit(1)

    print_table_line = lambda name, ranks: evaluate(name, ranks, dev_cases, None, DEV_ROUNDS,  # noqa: E731
                                                     stage=0, log=False)
    print_table_line("caption_approved", approved)
    print_table_line("caption_approved_shotdedup", approved_dedup)

    for i, strategy in enumerate(("reorder", "synonym", "restructure", "combined")):
        texts = [c["paraphrases"][i]["text"] for c in dev_cases if len(c["paraphrases"]) > i]
        subset = [c for c in dev_cases if len(c["paraphrases"]) > i]
        strategy_ranks = run_caption_shotdedup_baseline(
            [{**c, CAPTION_FIELD: c["paraphrases"][i]["text"]} for c in subset], rows_by_text, id_time,
            meta_by_idx, id_video, id_frame, group_resolver, CAPTION_FIELD, K=K)
        entry, _, _ = evaluate(f"strategy={strategy} alone (shot-dedup)", strategy_ranks, subset,
                               {c["id"]: approved_dedup[c["id"]] for c in subset}, DEV_ROUNDS, stage=0,
                               log=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paraphrases", type=Path,
                        default=ROOT / "artifacts/holistic_paraphrase/v1/paraphrases.json")
    parser.add_argument("--answers", type=Path, default=ROOT / "artifacts/official_benchmark_answers_v2.json")
    parser.add_argument("--stage", choices=["sanity", "stage1", "stage2", "stage3", "freeze", "holdout"],
                        required=True)
    parser.add_argument("--run-holdout", action="store_true")
    args = parser.parse_args()

    cases, header = load_paraphrase_cases(args.paraphrases)
    targets_by_id = load_targets(args.answers)
    attach_targets(cases, targets_by_id)

    id_video, id_frame, id_time, meta_by_idx, faiss_index = load_profile()
    shot_indices = load_shot_indices(meta_by_idx)
    group_resolver = GroupResolver(id_video, id_frame, id_time, shot_indices)

    dev_cases = cases_for_split(cases, "dev")
    holdout_cases = cases_for_split(cases, "holdout")
    all_2026 = cases_for_split(cases, "2026")
    assert_no_holdout_ids(dev_cases, "dev split")

    texts = collect_texts(cases, n_para=4)
    vectors_by_text = encode_and_cache(texts)
    rows_by_text, scores_by_text = retrieve_cached(texts, vectors_by_text, faiss_index, k=K)

    if args.stage == "sanity":
        assert_no_holdout_ids(dev_cases, "stage sanity")
        stage0_sanity(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                      group_resolver, vectors_by_text, faiss_index)
        return

    if args.stage == "stage1":
        assert_no_holdout_ids(dev_cases, "stage1")
        run_stage1(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                  group_resolver, vectors_by_text, faiss_index)
        return

    if args.stage == "stage2":
        assert_no_holdout_ids(dev_cases, "stage2")
        run_stage2(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                  group_resolver, vectors_by_text, faiss_index)
        return

    if args.stage == "stage3":
        assert_no_holdout_ids(dev_cases, "stage3")
        run_stage3(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
                  group_resolver, vectors_by_text, faiss_index)
        return

    if args.stage == "freeze":
        assert_no_holdout_ids(dev_cases, "freeze")
        # B* (emb-mean, cap+4 paraphrases, cap50) passed the dev acceptance rule in stage 2/3:
        # dMRR=+0.002, dR@1=+2 cases, worse in only 1/3 dev rounds vs caption_approved_shotdedup.
        run_freeze(dev_cases, header, accepted_on_dev=True)
        return

    if args.stage == "holdout":
        if not args.run_holdout:
            raise SystemExit("--stage holdout requires --run-holdout to confirm")
        run_holdout(all_2026, holdout_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                   meta_by_idx, group_resolver, vectors_by_text, faiss_index, header)
        return


def run_stage1(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
              group_resolver, vectors_by_text, faiss_index):
    baseline = run_caption_shotdedup_baseline(dev_cases, rows_by_text, id_time, meta_by_idx, id_video,
                                              id_frame, group_resolver, CAPTION_FIELD, K=K)
    cases2 = cases_with_enough_paraphrases(dev_cases, 2)
    cases3 = cases_with_enough_paraphrases(dev_cases, 3)
    assert len(cases2) == len(dev_cases), f"{len(dev_cases) - len(cases2)} dev cases have < 2 paraphrases"

    configs = [
        ("R1 para3-RRF-equal-k10", cases3, dict(include_caption=False, n_para=3, weights=default_weights(3, False, "equal"), k=10)),
        ("R2 para3-RRF-equal-k60", cases3, dict(include_caption=False, n_para=3, weights=default_weights(3, False, "equal"), k=60)),
        ("R3 cap+2para-RRF-cap50-k10", cases2, dict(include_caption=True, n_para=2, weights=default_weights(3, True, "cap50"), k=10)),
        ("R4 cap+2para-RRF-cap50-k60", cases2, dict(include_caption=True, n_para=2, weights=default_weights(3, True, "cap50"), k=60)),
        ("R5 cap+2para-RRF-equal-k10", cases2, dict(include_caption=True, n_para=2, weights=default_weights(3, True, "equal"), k=10)),
        ("R6 cap+2para-RRF-equal-k60", cases2, dict(include_caption=True, n_para=2, weights=default_weights(3, True, "equal"), k=60)),
    ]
    rrf_results = {}
    for name, cases, params in configs:
        ranks = run_rrf_config(cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                               meta_by_idx, group_resolver, **params)
        entry, _, _ = evaluate(name, ranks, cases, {c["id"]: baseline[c["id"]] for c in cases},
                               DEV_ROUNDS, stage=1, params=params)
        rrf_results[name] = entry

    emb_configs = [
        ("E1 cap+2para-emb-cap50", cases2, dict(include_caption=True, n_para=2, weights=default_weights(3, True, "cap50"))),
        ("E2 cap+2para-emb-equal", cases2, dict(include_caption=True, n_para=2, weights=default_weights(3, True, "equal"))),
        ("E3 para3-emb-equal", cases3, dict(include_caption=False, n_para=3, weights=default_weights(3, False, "equal"))),
    ]
    emb_results = {}
    for name, cases, params in emb_configs:
        ranks = run_embedding_mean_config(cases, vectors_by_text, faiss_index, id_time, id_video, id_frame,
                                          meta_by_idx, group_resolver, **params)
        entry, _, _ = evaluate(name, ranks, cases, {c["id"]: baseline[c["id"]] for c in cases},
                               DEV_ROUNDS, stage=1, params=params)
        emb_results[name] = entry

    best_rrf = max(rrf_results.items(), key=lambda kv: (kv[1]["pooled"]["mrr"], kv[1]["pooled"]["recall_at_1"]))
    best_emb = max(emb_results.items(), key=lambda kv: (kv[1]["pooled"]["mrr"], kv[1]["pooled"]["recall_at_1"]))
    print(f"\nBest RRF (R*): {best_rrf[0]} -> {best_rrf[1]['pooled']}")
    print(f"Best emb-mean (E*): {best_emb[0]} -> {best_emb[1]['pooled']}")


def acceptance_passes(candidate_breakdown, baseline_breakdown):
    """delta MRR >= 0.02 OR >= 2 more R@1 cases, AND not worse (MRR) in > 1 of 3 dev rounds."""
    n = candidate_breakdown["pooled"]["cases"]
    delta_mrr = candidate_breakdown["pooled"]["mrr"] - baseline_breakdown["pooled"]["mrr"]
    delta_r1_cases = round((candidate_breakdown["pooled"]["recall_at_1"]
                            - baseline_breakdown["pooled"]["recall_at_1"]) * n)
    worse_rounds = sum(1 for r in DEV_ROUNDS
                       if candidate_breakdown[r]["mrr"] < baseline_breakdown[r]["mrr"] - 1e-9)
    passes = (delta_mrr >= 0.02 or delta_r1_cases >= 2) and worse_rounds <= 1
    return passes, {"delta_mrr": delta_mrr, "delta_r1_cases": delta_r1_cases, "worse_rounds": worse_rounds}


def print_acceptance(name, breakdown, baseline_breakdown):
    passes, detail = acceptance_passes(breakdown, baseline_breakdown)
    print(f"  ACCEPTANCE[{name}]: {'PASS' if passes else 'fail'} "
         f"(dMRR={detail['delta_mrr']:+.3f}, dR@1={detail['delta_r1_cases']:+d} cases, "
         f"worse_rounds={detail['worse_rounds']})")
    return passes, detail


# ---------------------------------------------------------------------------
# Stage 2: paraphrase count.
# ---------------------------------------------------------------------------

def run_stage2(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
              group_resolver, vectors_by_text, faiss_index):
    """R* = R3 (RRF, cap50, k=10), E* = E1 (emb-mean, cap50) from stage 1. Vary n_para in {3, 4}
    (both include the caption). Re-pick R*/E*, and B* = the better of the two."""
    candidates = {"R3 (stage1, n_para=2)": ("rrf", 2, dict(weights_scheme="cap50", k=10)),
                 "E1 (stage1, n_para=2)": ("emb", 2, dict(weights_scheme="cap50"))}
    for n_para in (3, 4):
        candidates[f"R* n_para={n_para}"] = ("rrf", n_para, dict(weights_scheme="cap50", k=10))
        candidates[f"E* n_para={n_para}"] = ("emb", n_para, dict(weights_scheme="cap50"))

    results = {}
    for name, (kind, n_para, params) in candidates.items():
        cases_n = cases_with_enough_paraphrases(dev_cases, n_para)
        baseline = run_caption_shotdedup_baseline(cases_n, rows_by_text, id_time, meta_by_idx,
                                                  id_video, id_frame, group_resolver, CAPTION_FIELD, K=K)
        weights = default_weights(n_para + 1, True, params["weights_scheme"])
        if kind == "rrf":
            ranks = run_rrf_config(cases_n, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                                   meta_by_idx, group_resolver, include_caption=True, n_para=n_para,
                                   weights=weights, k=params["k"])
        else:
            ranks = run_embedding_mean_config(cases_n, vectors_by_text, faiss_index, id_time, id_video,
                                              id_frame, meta_by_idx, group_resolver, include_caption=True,
                                              n_para=n_para, weights=weights)
        entry, breakdown, _ = evaluate(name, ranks, cases_n,
                                       {c["id"]: baseline[c["id"]] for c in cases_n}, DEV_ROUNDS,
                                       stage=2, params={"kind": kind, "n_para": n_para, **params})
        baseline_breakdown = round_breakdown(baseline, cases_n, DEV_ROUNDS)
        passes, detail = print_acceptance(name, breakdown, baseline_breakdown)
        results[name] = {"kind": kind, "n_para": n_para, "params": params, "breakdown": breakdown,
                         "passes": passes, "detail": detail, "cases": cases_n}

    best_rrf = max((n for n in results if results[n]["kind"] == "rrf"),
                   key=lambda n: (results[n]["breakdown"]["pooled"]["mrr"],
                                 results[n]["breakdown"]["pooled"]["recall_at_1"]))
    best_emb = max((n for n in results if results[n]["kind"] == "emb"),
                   key=lambda n: (results[n]["breakdown"]["pooled"]["mrr"],
                                 results[n]["breakdown"]["pooled"]["recall_at_1"]))
    b_star = max((best_rrf, best_emb),
                key=lambda n: (results[n]["breakdown"]["pooled"]["mrr"],
                              results[n]["breakdown"]["pooled"]["recall_at_1"]))
    print(f"\nR* (stage2) = {best_rrf}: {results[best_rrf]['breakdown']['pooled']}")
    print(f"E* (stage2) = {best_emb}: {results[best_emb]['breakdown']['pooled']}")
    print(f"B* (stage2) = {b_star}: {results[b_star]['breakdown']['pooled']} "
         f"accepted={results[b_star]['passes']}")
    return results, b_star


# ---------------------------------------------------------------------------
# Stage 3: ablations on B*.
# ---------------------------------------------------------------------------

B_STAR = {"kind": "emb", "n_para": 4, "weights_scheme": "cap50"}  # from the real stage1+2 run:
# emb-mean beat RRF throughout and improved monotonically with n_para (2->4); RRF got worse.


def run_stage3(dev_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame, meta_by_idx,
              group_resolver, vectors_by_text, faiss_index, b_star=None):
    b_star = b_star or B_STAR
    n_para = b_star["n_para"]
    cases_n = cases_with_enough_paraphrases(dev_cases, n_para)
    baseline = run_caption_shotdedup_baseline(cases_n, rows_by_text, id_time, meta_by_idx, id_video,
                                              id_frame, group_resolver, CAPTION_FIELD, K=K)
    baseline_breakdown = round_breakdown(baseline, cases_n, DEV_ROUNDS)

    def eval_variant(name, kind, weights_scheme, k=10):
        weights = default_weights(n_para + 1, True, weights_scheme)
        if kind == "rrf":
            ranks = run_rrf_config(cases_n, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                                   meta_by_idx, group_resolver, include_caption=True, n_para=n_para,
                                   weights=weights, k=k)
        else:
            ranks = run_embedding_mean_config(cases_n, vectors_by_text, faiss_index, id_time, id_video,
                                              id_frame, meta_by_idx, group_resolver, include_caption=True,
                                              n_para=n_para, weights=weights)
        entry, breakdown, _ = evaluate(name, ranks, cases_n,
                                       {c["id"]: baseline[c["id"]] for c in cases_n}, DEV_ROUNDS,
                                       stage=3, params={"kind": kind, "n_para": n_para,
                                                        "weights_scheme": weights_scheme, "k": k})
        passes, _ = print_acceptance(name, breakdown, baseline_breakdown)
        return breakdown, passes

    b_star_breakdown, _ = eval_variant("B* baseline", b_star["kind"], b_star["weights_scheme"],
                                       b_star.get("k", 10))
    if b_star["kind"] == "emb":
        alt_breakdown, alt_passes = eval_variant("B* as RRF (same lists/weights)", "rrf",
                                                 b_star["weights_scheme"], b_star.get("k", 10))
    else:
        alt_breakdown, alt_passes = eval_variant("B* as score-fusion (same lists/weights)", "score",
                                                 b_star["weights_scheme"])
    cap67_breakdown, cap67_passes = eval_variant("B* with cap67 weights", b_star["kind"], "cap67",
                                                 b_star.get("k", 10))


# ---------------------------------------------------------------------------
# Freeze and holdout.
# ---------------------------------------------------------------------------

def run_freeze(dev_cases, header, b_star=None, accepted_on_dev=None):
    b_star = b_star or B_STAR
    n_para = b_star["n_para"]
    weights = default_weights(n_para + 1, True, b_star["weights_scheme"])
    config = {
        "kind": b_star["kind"], "include_caption": True, "n_para": n_para, "weights": weights,
        "weights_scheme": b_star["weights_scheme"], "k": b_star.get("k", 10),
        "paraphrase_file_hash": hashlib.sha1(
            Path(header.get("_source_path", "")).read_bytes()).hexdigest()[:16] if header.get("_source_path") else None,
        "prompt_hash": header.get("prompt_hash"), "prompt_version": header.get("prompt_version"),
        "accepted_on_dev": accepted_on_dev,
        "configs_tried": sum(1 for _ in open(TUNING_LOG)) if TUNING_LOG.exists() else 0,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "holdout_roster": [
            {"name": "caption_approved", "kind": "caption", "n_para": 0},
            {"name": "caption_approved_shotdedup", "kind": "shotdedup", "n_para": 0},
            {"name": "3paraphrases_rrf_equal_k10", "kind": "rrf", "include_caption": False,
             "n_para": 3, "weights_scheme": "equal", "k": 10},
            {"name": "caption+2paraphrases_rrf_cap50_k10", "kind": "rrf", "include_caption": True,
             "n_para": 2, "weights_scheme": "cap50", "k": 10},
            {"name": "B*", "kind": b_star["kind"], "include_caption": True, "n_para": n_para,
             "weights_scheme": b_star["weights_scheme"], "k": b_star.get("k", 10)},
        ],
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n")
    print(f"wrote {CONFIG_PATH}")
    print(json.dumps(config, indent=2))
    return config


def _run_roster_entry(entry, cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                      meta_by_idx, group_resolver, vectors_by_text, faiss_index):
    kind = entry["kind"]
    if kind == "caption":
        return run_caption_baseline(cases, rows_by_text, meta_by_idx, id_video, id_frame, CAPTION_FIELD, K=K)
    if kind == "shotdedup":
        return run_caption_shotdedup_baseline(cases, rows_by_text, id_time, meta_by_idx, id_video,
                                              id_frame, group_resolver, CAPTION_FIELD, K=K)
    n_para = entry["n_para"]
    include_caption = entry.get("include_caption", False)
    weights = default_weights(n_para + (1 if include_caption else 0), include_caption,
                              entry["weights_scheme"])
    if kind == "rrf":
        return run_rrf_config(cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                              meta_by_idx, group_resolver, include_caption=include_caption,
                              n_para=n_para, weights=weights, k=entry["k"])
    if kind == "score":
        return run_score_config(cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
                                meta_by_idx, group_resolver, include_caption=include_caption,
                                n_para=n_para, weights=weights)
    if kind == "emb":
        return run_embedding_mean_config(cases, vectors_by_text, faiss_index, id_time, id_video, id_frame,
                                         meta_by_idx, group_resolver, include_caption=include_caption,
                                         n_para=n_para, weights=weights)
    raise ValueError(kind)


def run_holdout(all_2026_cases, holdout_cases, rows_by_text, scores_by_text, id_time, id_video, id_frame,
               meta_by_idx, group_resolver, vectors_by_text, faiss_index, header):
    if not CONFIG_PATH.exists():
        raise SystemExit(f"{CONFIG_PATH} does not exist; run --stage freeze first")
    if HOLDOUT_DONE.exists():
        raise SystemExit(f"{HOLDOUT_DONE} already exists; holdout runs exactly once")
    config = json.loads(CONFIG_PATH.read_text())
    n_para = max(entry.get("n_para", 0) for entry in config["holdout_roster"])
    holdout_ready = cases_with_enough_paraphrases(holdout_cases, n_para)
    all_ready = cases_with_enough_paraphrases(all_2026_cases, n_para)
    if len(holdout_ready) < len(holdout_cases):
        print(f"NOTE: {len(holdout_cases) - len(holdout_ready)} holdout cases have < {n_para} "
             f"paraphrases and are excluded from configs needing them")

    results = {}
    for entry in config["holdout_roster"]:
        cases_n = holdout_ready if entry.get("n_para", 0) >= n_para else holdout_cases
        cases_n_all = all_ready if entry.get("n_para", 0) >= n_para else all_2026_cases
        ranks_holdout = _run_roster_entry(entry, cases_n, rows_by_text, scores_by_text, id_time,
                                          id_video, id_frame, meta_by_idx, group_resolver,
                                          vectors_by_text, faiss_index)
        ranks_all = _run_roster_entry(entry, cases_n_all, rows_by_text, scores_by_text, id_time,
                                      id_video, id_frame, meta_by_idx, group_resolver,
                                      vectors_by_text, faiss_index)
        results[entry["name"]] = {
            "holdout": round_breakdown(ranks_holdout, cases_n, [HOLDOUT_ROUND]),
            "all_2026": round_breakdown(ranks_all, cases_n_all, REPORTED_ROUNDS),
            "ranks_holdout": ranks_holdout, "ranks_all_2026": ranks_all,
        }
        print(f"{entry['name']:40s} HOLDOUT MRR={results[entry['name']]['holdout']['pooled']['mrr']:.3f} "
             f"R@1={results[entry['name']]['holdout']['pooled']['recall_at_1']:.3f}  |  "
             f"ALL-2026 MRR={results[entry['name']]['all_2026']['pooled']['mrr']:.3f} "
             f"R@1={results[entry['name']]['all_2026']['pooled']['recall_at_1']:.3f}")

    baseline_holdout_ranks = results["caption_approved_shotdedup"]["ranks_holdout"]
    for name, data in results.items():
        if name == "caption_approved_shotdedup":
            continue
        common_ids = [c["id"] for c in holdout_ready if c["id"] in data["ranks_holdout"]
                     and c["id"] in baseline_holdout_ranks]
        if common_ids:
            data["vs_baseline_holdout"] = compare_to_baseline(data["ranks_holdout"],
                                                              baseline_holdout_ranks, common_ids)

    BENCHMARK_OUT.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_OUT.write_text(json.dumps({
        "config": config, "results": results,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, default=str))
    HOLDOUT_DONE.write_text(json.dumps({
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster_hash": hashlib.sha1(json.dumps(config["holdout_roster"], sort_keys=True).encode()).hexdigest()[:16],
    }, indent=2) + "\n")
    print(f"\nwrote {BENCHMARK_OUT}")
    print(f"wrote {HOLDOUT_DONE} (holdout is now locked)")


if __name__ == "__main__":
    main()
