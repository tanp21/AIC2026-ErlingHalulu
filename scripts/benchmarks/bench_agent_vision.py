#!/usr/bin/env python3
"""Measure local vision models on the agent's own verification job.

The agent sends every keyframe it wants judged to Gemini as inline JPEG. To decide
whether a local model can take that job over, this compares candidates on **the
same candidate pool**, so the numbers are directly comparable and do not depend on
how well any model can plan or search.

Two design choices carry the measurement:

* **The pool comes from a real search, not from synthetic distractors.** Each of
  the 137 ground-truth cases is run through ``POST /agent/search`` with its English
  caption, and the top hits become the candidates — the distribution the agent
  actually faces. Retrieval is done once and cached, so every model sees an
  identical pool.
* **Scoring uses the Vietnamese query.** Production hands ``verify`` the original
  Vietnamese task text, not the English caption, so that is what is scored here.
  It is also the only fair comparison: Vintern is Vietnamese-tuned, and feeding the
  three models different languages would measure the language, not the model.

The metric that decides the winner is the **separation AUC** — how well a model's
0-10 scores rank gold candidates above non-gold ones within each query. End-to-end
R@1 is deliberately not used: the agent's own docs record the same code scoring
40-56% across runs, which is far too noisy to pick a model with.

Usage:
    uv run python -m scripts.benchmarks.bench_agent_vision pool
    uv run python -m scripts.benchmarks.bench_agent_vision smoke --model 5CD-AI/Vintern-1B-v3_5
    uv run python -m scripts.benchmarks.bench_agent_vision score --models all
    uv run python -m scripts.benchmarks.bench_agent_vision report
    uv run python -m scripts.benchmarks.bench_agent_vision --self-test

``--captions`` adds each candidate's CapRL shot caption to its label, the way
production does, and ``--tag cap`` keeps that arm from overwriting the baseline:

    uv run python -m scripts.benchmarks.bench_agent_vision score \\
        --models Qwen/Qwen3-VL-2B-Instruct --captions --tag cap
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[2]
QUERIES = ROOT / "artifacts/official_benchmark_queries_en_v2.json"
FIXTURE = ROOT / "artifacts/agent/vision-bench"
POOL_JSON = FIXTURE / "pool.json"
IMAGE_CACHE = FIXTURE / "images"
CAPTION_CACHE = FIXTURE / "captions.json"
RESULTS = ROOT / "benchmarks/visual/vlm_verify_2026.json"

#: Matches `python -m agent eval --tolerance 250`, so these numbers sit on the
#: same scale as the ones already in docs/agent.md.
TOLERANCE_FRAMES = 250
#: Candidates per case. Fast mode verifies 15; 20 leaves a little headroom without
#: making the pool easy.
TOP_K = 20
#: The width `Inspector.verify` asks the server for.
IMAGE_WIDTH = 640


# --- pool --------------------------------------------------------------------


def cases() -> list[dict]:
    """The 137 ground-truth cases: English caption, Vietnamese query, gold frames."""
    data = json.loads(QUERIES.read_text())
    return data["cases"] if isinstance(data, dict) and "cases" in data else data


def gold_frames(case: dict) -> dict[str, list[int]]:
    frames: dict[str, list[int]] = {}
    for video_id, frame in case["targets"]:
        frames.setdefault(video_id, []).append(int(frame))
    return frames


def is_gold(video_id: str, frame_idx: int, gold: dict[str, list[int]]) -> bool:
    return any(abs(frame_idx - g) <= TOLERANCE_FRAMES for g in gold.get(video_id, ()))


def build_pool(*, user: str, base_url: str, top_k: int = TOP_K) -> dict:
    """Search once per case and keep the top hits as that case's candidate pool."""
    from agent.search_client import SearchClient

    client = SearchClient(base_url, trusted_user=user, api="batch")
    pools = []
    try:
        for index, case in enumerate(cases(), start=1):
            gold = gold_frames(case)
            reply = client.search([{
                "kind": "visual", "id": "v0", "text": case["text"], "limit": 200,
            }])
            hits = (reply.get("v0") or {}).get("hits") or []
            seen, candidates = set(), []
            for hit in hits:
                key = (hit["video_id"], int(hit["frame_idx"]))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append({
                    "video_id": hit["video_id"],
                    "frame_idx": int(hit["frame_idx"]),
                    "time": float(hit["time"]),
                    "rank": int(hit["rank"]),
                    "score": float(hit["score"]),
                    "gold": is_gold(hit["video_id"], int(hit["frame_idx"]), gold),
                })
                if len(candidates) >= top_k:
                    break
            pools.append({"id": case["id"], "query_vi": case["text_vi"],
                          "candidates": candidates})
            print(f"[{index}/{len(cases())}] {case['id']}: {len(candidates)} candidates, "
                  f"{sum(c['gold'] for c in candidates)} gold", flush=True)
    finally:
        client.close()
    payload = {"tolerance_frames": TOLERANCE_FRAMES, "top_k": top_k, "cases": pools}
    POOL_JSON.parent.mkdir(parents=True, exist_ok=True)
    POOL_JSON.write_text(json.dumps(payload, ensure_ascii=False))
    return payload


def load_pool() -> dict:
    return json.loads(POOL_JSON.read_text())


# --- images ------------------------------------------------------------------


def fetch_images(pool: dict, *, user: str, base_url: str) -> None:
    """Pull every pool frame once and cache it, so models are compared on identical bytes."""
    from agent.search_client import SearchClient

    IMAGE_CACHE.mkdir(parents=True, exist_ok=True)
    wanted = {(c["video_id"], c["frame_idx"])
              for case in pool["cases"] for c in case["candidates"]}
    todo = [key for key in sorted(wanted) if not image_path(*key).exists()]
    if not todo:
        print(f"images: all {len(wanted)} cached")
        return
    print(f"images: fetching {len(todo)} of {len(wanted)}")
    client = SearchClient(base_url, trusted_user=user, api="batch")

    def grab(key):
        video_id, frame_idx = key
        try:
            data = client.image(video_id, frame_idx, IMAGE_WIDTH)
        except Exception:
            return None
        if data:
            image_path(video_id, frame_idx).write_bytes(data)
        return key

    try:
        with ThreadPoolExecutor(6) as pool_exec:
            done = sum(1 for result in pool_exec.map(grab, todo) if result)
    finally:
        client.close()
    print(f"images: {done}/{len(todo)} fetched")


def image_path(video_id: str, frame_idx: int) -> Path:
    return IMAGE_CACHE / f"{video_id}_{frame_idx}_{IMAGE_WIDTH}.jpg"


# --- shot captions -----------------------------------------------------------


def fetch_captions(pool: dict) -> dict[str, str]:
    """Every pool candidate's shot caption, keyed `"<video>:<frame>"`, cached on disk.

    Read through `backend.shots.caption_for_frame` — the exact function the
    `/captions` route calls — rather than over HTTP to a running server. Two
    reasons: the harness already reads ground truth and images off this checkout's
    disk, and going through the route would make the ablation depend on the age of
    whatever container happens to be up. The route's own behaviour is covered by
    `tests/test_shot_captions.py::CaptionRouteTests`; what is measured here is
    whether the text helps, and the text is the same either way.

    Cached rather than re-read per arm on purpose: the two arms must differ in the
    caption and nothing else. 2025 `K*` videos were never captioned and simply do
    not appear — the same missing-caption case production has.
    """
    if CAPTION_CACHE.exists():
        cached = json.loads(CAPTION_CACHE.read_text())
        print(f"captions: {len(cached)} cached")
        return cached

    from backend import shots

    asked = 0
    found: dict[str, str] = {}
    for case in pool["cases"]:
        for candidate in case["candidates"]:
            key = f"{candidate['video_id']}:{candidate['frame_idx']}"
            if key in found or candidate["video_id"][:1] not in shots.CAPTION_YEARS:
                continue
            asked += 1
            try:
                row = shots.caption_for_frame(candidate["video_id"], candidate["time"])
            except (OSError, ValueError):
                continue
            # A dict — caption plus the window bounds. Only the prose is kept; the
            # bounds would tempt a reader to use them, and production does not.
            text = (row or {}).get("caption") or ""
            if text:
                found[key] = text
    CAPTION_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CAPTION_CACHE.write_text(json.dumps(found, ensure_ascii=False))
    print(f"captions: {len(found)}/{asked} 2026 pool frames have one")
    return found


# --- scoring -----------------------------------------------------------------


def shown_gold_rank(rows: list[dict], *, threshold: float | None = None) -> int | None:
    """1-based position of the first gold in the order `CandidatePool.ordered` shows.

    This is the number that matters to the team: not "can the model tell gold from
    non-gold" (that is the AUC) but "does the list it produces put gold nearer the
    top". They can disagree — a model can separate well pairwise and still never move
    the retrieval's own top hit, because `ordered` re-sorts on the verdict.

    Mirrors `agent/pool.py`'s key, minus the scene bonus: the harness scores with an
    empty scene list, so `verify` never sees one. `fused` is the retrieval score the
    pool came in with, and the pool is stored in descending fused order, so index
    order already is the tie-break. `threshold=None` reproduces the no-verifier case,
    which is the baseline this is compared against.
    """
    def key(index: int, row: dict):
        score = row["score"]
        if score is None or threshold is None:
            return (1, 0.0, index)
        if score >= threshold:
            return (0, -score, index)
        if score <= 3.0:
            return (2, -score, index)
        return (1, -score / 10, index)

    order = sorted(range(len(rows)), key=lambda i: key(i, rows[i]))
    for position, index in enumerate(order, start=1):
        if rows[index]["gold"]:
            return position
    return None


def auc(scores: list[float], labels: list[bool]) -> float | None:
    """Probability a gold candidate outranks a non-gold one. Ties count a half."""
    positives = [s for s, label in zip(scores, labels) if label]
    negatives = [s for s, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return wins / (len(positives) * len(negatives))


def verify_prompt(query_vi: str) -> str:
    """The production prompt, minus the plan-derived parts this harness has no plan for.

    `scenes` and `question` come from a DeepSeek planning step that is identical
    for every model being compared, so omitting them costs nothing in fairness and
    keeps the harness from needing a DeepSeek key.
    """
    from agent.prompts import VERIFY_PROMPT

    return VERIFY_PROMPT.format(query=query_vi, scenes="", question="", answer_rule="")


def score_case(model, case: dict, *, usage=None, captions: dict | None = None) -> dict:
    """One verify call over a case's whole pool, exactly as `Inspector.verify` chunks it.

    `captions` turns on the CapRL-context arm: when given, every candidate label
    carries its shot caption exactly as production does. Passing `None` is the
    baseline arm, so the two differ in one argument and nothing else.
    """
    from agent.pool import VERIFIED_MATCH, Candidate, Evidence
    from agent.vision import VERIFY_SCHEMA, candidate_label

    candidates, labelled = [], []
    for number, entry in enumerate(case["candidates"], start=1):
        path = image_path(entry["video_id"], entry["frame_idx"])
        if not path.exists():
            continue
        candidate = Candidate(
            cid=f"C{number}", video_id=entry["video_id"], frame_idx=entry["frame_idx"],
            time=entry["time"],
            evidence={"visual": Evidence(source="visual", kind="visual", rank=entry["rank"],
                                         score=entry["score"])},
            fused=entry["score"],
        )
        caption = None if captions is None else captions.get(
            f"{entry['video_id']}:{entry['frame_idx']}")
        candidates.append((candidate, entry["gold"]))
        labelled.append((candidate_label(candidate, caption=caption), path.read_bytes()))
    if not labelled:
        return {"id": case["id"], "error": "no cached images"}
    started = time.monotonic()
    reply = model.judge(labelled, verify_prompt(case["query_vi"]), VERIFY_SCHEMA,
                        job="verify", query=case["query_vi"])
    seconds = time.monotonic() - started
    by_id = {}
    for item in (reply or {}).get("candidates", []) or []:
        cid = str(item.get("id", "")).strip("[] ")
        if cid:
            by_id[cid] = item
    rows = []
    for candidate, gold in candidates:
        item = by_id.get(candidate.cid)
        rows.append({
            "cid": candidate.cid, "video_id": candidate.video_id,
            "frame_idx": candidate.frame_idx, "gold": gold,
            "score": None if item is None else float(item.get("score", 0) or 0),
            "reason": "" if item is None else str(item.get("reason", ""))[:120],
        })
    scored = [r for r in rows if r["score"] is not None]
    best = max(scored, key=lambda r: r["score"], default=None)
    return {
        "id": case["id"],
        "seconds": seconds,
        "answered": len(scored),
        "asked": len(rows),
        "auc": auc([r["score"] for r in scored], [r["gold"] for r in scored]),
        # The highest-scored candidate, not the first one answered for: `rows` is in
        # pool order, so indexing it would report a property of the pool rather than
        # of the model's ranking.
        "top1_gold": bool(best and best["gold"]),
        # Where gold lands in the list the team actually reads, with and without
        # this model's verdicts. `None` when the pool holds no gold at all — those
        # cases are dropped from the AUC anyway.
        "shown_gold_rank": shown_gold_rank(rows, threshold=VERIFIED_MATCH),
        "pool_gold_rank": shown_gold_rank(rows),
        "rows": rows,
    }


def summarize(case_results: list[dict]) -> dict:
    scored = [r for r in case_results if r.get("auc") is not None]
    seconds = [r["seconds"] for r in case_results if "seconds" in r]
    asked = sum(r.get("asked", 0) for r in case_results)
    answered = sum(r.get("answered", 0) for r in case_results)
    # How many distinct scores a model gives within one pool. `top1_gold_rate` is
    # meaningless for a model that ties: `max` returns the *first* largest, which is
    # the pool's own rank-1, so the metric silently reports the retrieval order's
    # accuracy instead of the model's. Observed exactly that — all three local
    # candidates scored 0.500, and the pool's rank-1 is gold in 0.5055 of cases.
    # A reader seeing 1.5 here knows the top1 column is telling them nothing.
    distinct = [len({row["score"] for row in r["rows"] if row["score"] is not None})
                for r in case_results if r.get("rows")]
    # Only cases where both ranks exist: a case with no gold in the pool has no
    # rank to compare, and one where every answer was dropped is not a data point.
    honoured = [r for r in case_results
                if r.get("shown_gold_rank") is not None
                and r.get("pool_gold_rank") is not None]
    return {
        "cases": len(case_results),
        "cases_with_both_classes": len(scored),
        "separation_auc": round(statistics.mean(r["auc"] for r in scored), 4) if scored else None,
        "top1_gold_rate": round(
            sum(1 for r in scored if r["top1_gold"]) / len(scored), 4) if scored else None,
        "distinct_scores_mean": (
            round(statistics.mean(distinct), 2) if distinct else None),
        # The list the team reads. `mean_gold_rank` is the average position of the
        # first gold keyframe, lower being better; `list_*_rate` is how often this
        # model moved gold up versus down against the retrieval order alone.
        "mean_gold_rank": (
            round(statistics.mean(r["shown_gold_rank"] for r in honoured), 2)
            if honoured else None),
        "pool_mean_gold_rank": (
            round(statistics.mean(r["pool_gold_rank"] for r in honoured), 2)
            if honoured else None),
        "list_improved_rate": (
            round(sum(1 for r in honoured if r["shown_gold_rank"] < r["pool_gold_rank"])
                  / len(honoured), 4) if honoured else None),
        "list_worsened_rate": (
            round(sum(1 for r in honoured if r["shown_gold_rank"] > r["pool_gold_rank"])
                  / len(honoured), 4) if honoured else None),
        "answer_rate": round(answered / asked, 4) if asked else None,
        "seconds_mean": round(statistics.mean(seconds), 2) if seconds else None,
        "seconds_p95": round(sorted(seconds)[int(0.95 * (len(seconds) - 1))], 2) if seconds else None,
    }


def usable_cases(pool: dict) -> list[dict]:
    """Cases whose pool holds both gold and non-gold candidates.

    The rest cannot produce an AUC by definition — 42 of the 137 have no gold in
    the top 20 at all, which is a *retrieval* miss and not something a verifier can
    fix. Scoring them only burns GPU time.
    """
    return [case for case in pool["cases"]
            if 0 < sum(c["gold"] for c in case["candidates"]) < len(case["candidates"])]


#: A pseudo-model name: the baseline every local candidate is really being measured
#: against. Without it an AUC of 0.6 says the model has *some* signal but not
#: whether that is enough, since nothing on the same pool says what good looks like.
#: `gemini:<model-id>` picks a specific one — `gemini-flash-lite-latest` is the
#: cheap tier, worth knowing about if the full model ever gets expensive.
GEMINI = "gemini"
GEMINI_PREFIX = "gemini:"

#: `deepseek:<model-id>` scores the pool through `DeepSeekClient.judge` — the same
#: five vision jobs, on the same images, as Gemini. The API does not document
#: vision, but the OpenAI-compatible endpoint takes `image_url` parts and answers
#: correctly, so it is a real third backend rather than a text-only control.
DEEPSEEK_PREFIX = "deepseek:"


def deepseek_key() -> str:
    from agent.config import ROOT, read_dotenv

    key = os.environ.get("DEEPSEEK_API_KEY") or read_dotenv(ROOT / ".env").get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set, so there is nothing to score")
    return key


def can_see(client, prompt_style: str) -> bool:
    """Ask about a solid-colour image; a model that cannot see will not name it.

    This is not paranoia — `deepseek-v4-pro` answers a vision call with
    schema-shaped JSON whose every score is 0, which reads as a working judge
    scoring badly. Scoring 91 cases before noticing wastes half an hour and
    produces a number that looks like a result.
    """
    import base64
    import io

    from PIL import Image

    from agent.llm import parse_json

    for colour, word in (("red", "red"), ("blue", "blue")):
        buffer = io.BytesIO()
        Image.new("RGB", (320, 240), colour).save(buffer, "JPEG")
        probe = ("What is the dominant colour of this image? "
                 'Reply with JSON: {"colour": "<one word>"}')
        try:
            reply = client.judge(
                [("[probe]", buffer.getvalue())],
                probe if prompt_style == "plain" else probe,
                {"type": "object", "properties": {"colour": {"type": "string"}},
                 "required": ["colour"]},
            )
        except Exception:
            return False
        text = json.dumps(reply).lower()
        if word not in text:
            return False
    return True


def build_model(name: str, *, device: str | None):
    """A `judge()`-compatible object, or raises with the reason it could not be built."""
    from agent.vlm import LocalVLM, spec

    if name.startswith(DEEPSEEK_PREFIX):
        # Judged the same way Gemini judges: same images, same prompt, same schema.
        from agent.llm import DeepSeekClient

        client = DeepSeekClient(deepseek_key(), name[len(DEEPSEEK_PREFIX):] or "deepseek-flash",
                                thinking=False)
        if not can_see(client, "plain"):
            raise RuntimeError(
                f"{client.model} did not identify a solid-colour image, so it cannot be "
                f"scored as a vision judge"
            )
        return client

    if name == GEMINI or name.startswith(GEMINI_PREFIX):
        from agent.llm import GeminiClient

        key = os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set, so there is no baseline to score")
        model = (name[len(GEMINI_PREFIX):] if name.startswith(GEMINI_PREFIX)
                 else os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"))
        # A timeout is not optional here. Without one a stalled request hangs the
        # whole sweep: `gemini-flash-lite-latest` sat at 50/91 for ten minutes with
        # the process asleep and nothing in the log, because a 20-image request had
        # gone quiet and there was nothing to give up on it.
        client = GeminiClient(key, model, timeout=90.0, retries=3, max_backoff=8.0,
                              thinking_level=os.environ.get("GEMINI_THINKING_LEVEL", "low"))
        if name.startswith(GEMINI_PREFIX) and not can_see(client, "plain"):
            raise RuntimeError(f"{model} did not identify a solid-colour image")
        return client

    model = LocalVLM(spec(name), device=device)
    model._model()  # load before timing, so the first case is not the outlier
    return model


def write_results(results: dict, pool: dict) -> dict:
    """Merge `results` into the tracked file and return everything now in it.

    Written after every model rather than at the end of the sweep: a full sweep is
    hours long, and a crash near the end of it must not cost the models that had
    already finished.

    Locked, and re-read inside the lock, because several sweeps can legitimately run
    at once — the local arms need the GPU, the remote ones only the network, so the
    natural way to use the time is both together. Without the lock the read half and
    the write half of two processes can interleave and one model's row disappears.
    """
    import fcntl

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS.with_suffix(".lock"), "w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        payload = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        merged = {**payload.get("models", {}), **results}
        payload.update({
            "source": "scripts/benchmarks/bench_agent_vision.py",
            "metric": "separation AUC of 0-10 verdicts, gold vs non-gold, same pool per model",
            "tolerance_frames": pool["tolerance_frames"],
            "top_k": pool["top_k"],
            "cases": len(pool["cases"]),
            "usable_cases": len(usable_cases(pool)),
            "models": merged,
        })
        # Through a temp file in the same directory so a crash mid-write leaves the
        # previous results intact rather than a truncated JSON nobody can score.
        scratch = RESULTS.with_suffix(".json.tmp")
        scratch.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        scratch.replace(RESULTS)
        fcntl.flock(handle, fcntl.LOCK_UN)
    return payload


def score_models(names: list[str], pool: dict, *, device: str | None = None,
                 limit: int | None = None, tag: str = "",
                 captions: dict | None = None) -> dict:
    from agent import vlm_models
    from agent.vlm import spec

    chosen = usable_cases(pool)[:limit] if limit else usable_cases(pool)
    print(f"scoring {len(chosen)} of {len(pool['cases'])} cases "
          f"(the rest have a single class and no defined AUC)", flush=True)
    out = {}
    for name in names:
        # A remote judge is named as it is; only a local candidate needs resolving
        # through the registry.
        remote = name == GEMINI or name.startswith(GEMINI_PREFIX) or name.startswith(DEEPSEEK_PREFIX)
        label = (name if remote else spec(name).name) + (f" [{tag}]" if tag else "")
        print(f"=== {label} ===", flush=True)
        try:
            model = build_model(name, device=device)
        except Exception as exc:
            out[label] = {"error": f"could not start: {exc}"}
            print(f"  could not start: {exc}", flush=True)
            write_results(out, pool)
            continue
        results = []
        for index, case in enumerate(chosen, start=1):
            try:
                results.append(score_case(model, case, captions=captions))
            except Exception as exc:
                results.append({"id": case["id"], "error": str(exc)})
            if index % 10 == 0:
                print(f"  {index}/{len(chosen)}", flush=True)
        summary = summarize(results)
        # VRAM only means anything for a model that ran here.
        summary["peak_vram_mb"] = None if remote else vlm_models.peak_vram_mb()
        summary["device"] = getattr(model, "device", "remote")
        summary["errors"] = sum(1 for r in results if "error" in r)
        summary["prompt_language"] = getattr(model, "prompt_language", None)
        # How many captions this arm actually had. 0 means the baseline arm, not a
        # failed fetch: `--captions` is the only thing that populates it.
        summary["captions"] = len(captions) if captions else 0
        out[label] = summary
        print(f"  {json.dumps(summary, ensure_ascii=False)}", flush=True)
        write_results({label: summary}, pool)
        del model
        if not remote:
            vlm_models.free()
    return out


def smoke(name: str, pool: dict, *, device: str | None = None) -> None:
    """One real call end to end: raw reply, parsed dict, seconds, VRAM.

    Run this before scoring anything. It is what catches a family that cannot be
    driven the way the interface assumes — an InternVL multi-image call, say —
    without waiting for a full sweep to fail.
    """
    from agent import vlm_models
    from agent.vlm import LocalVLM, format_instruction, spec
    from agent.vision import VERIFY_SCHEMA

    candidate = spec(name)
    model = LocalVLM(candidate, device=device)
    case = next(c for c in pool["cases"] if sum(x["gold"] for x in c["candidates"]) >= 1)
    from agent.pool import Candidate, Evidence
    from agent.vision import candidate_label

    labelled = []
    for number, entry in enumerate(case["candidates"][:3], start=1):
        path = image_path(entry["video_id"], entry["frame_idx"])
        if not path.exists():
            continue
        node = Candidate(cid=f"C{number}", video_id=entry["video_id"],
                         frame_idx=entry["frame_idx"], time=entry["time"],
                         evidence={"visual": Evidence("visual", "visual", entry["rank"],
                                                      entry["score"])})
        labelled.append((candidate_label(node), path.read_bytes()))
    if not labelled:
        raise SystemExit("no cached images; run `pool` then `images` first")
    prompt = verify_prompt(case["query_vi"])
    print(f"model   : {candidate.name} ({candidate.family}, {candidate.dtype})")
    print(f"device  : {model.device}")
    print(f"images  : {len(labelled)}")
    print("--- instruction appended to the prompt ---")
    print(format_instruction(VERIFY_SCHEMA))
    print("--- raw reply ---")
    started = time.monotonic()
    raw = model._generate(
        [(label, vlm_models.resize(vlm_models.to_pil(data), model.image_side))
         for label, data in labelled],
        prompt + "\n\n" + format_instruction(VERIFY_SCHEMA),
        model.max_new_tokens,
    )
    seconds = time.monotonic() - started
    print(raw)
    print(f"--- {seconds:.1f}s, peak VRAM {vlm_models.peak_vram_mb():.0f} MB ---")
    try:
        from agent.llm import parse_json

        print("parsed :", json.dumps(parse_json(raw), ensure_ascii=False)[:400])
    except Exception as exc:
        print(f"parsed : FAILED ({exc})")


# --- self test ---------------------------------------------------------------


def self_test() -> None:
    # labels align with scores: high score + gold is a perfect ranker
    assert auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert auc([1, 2, 3, 4], [True, True, False, False]) == 0.0
    # positives at 1 and 3, negatives at 2 and 4: only (3 vs 2) wins, so 1/4
    assert auc([1, 2, 3, 4], [True, False, True, False]) == 0.25
    # ties count half; one gold against one non-gold at the same score is a coin flip
    assert auc([5, 5], [True, False]) == 0.5
    assert auc([1, 1, 1], [True, True, False]) == 0.5
    assert auc([5], [True]) is None, "a single class cannot separate"
    assert auc([], []) is None

    assert is_gold("L01_V001", 100, {"L01_V001": [100]})
    assert is_gold("L01_V001", 100 + TOLERANCE_FRAMES, {"L01_V001": [100]})
    assert not is_gold("L01_V001", 100 + TOLERANCE_FRAMES + 1, {"L01_V001": [100]})
    assert not is_gold("L01_V002", 100, {"L01_V001": [100]}), "another video is never gold"

    frames = gold_frames({"targets": [["L01_V001", 5], ["L01_V002", 9], ["L01_V001", 7]]})
    assert frames == {"L01_V001": [5, 7], "L01_V002": [9]}, frames

    prompt = verify_prompt("con thuyền đỏ")
    assert "con thuyền đỏ" in prompt
    assert "{" not in prompt.replace("{{", ""), "the template left an unfilled slot"
    print("self-test OK")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?",
                        choices=["pool", "images", "captions", "smoke", "score",
                                 "report", "all"],
                        default="report")
    parser.add_argument("--model", default=None, help="one model, for `smoke`")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int, default=None,
                        help="score only the first N usable cases, for a quick look")
    parser.add_argument("--tag", default="",
                        help="suffix for the result key, so two runs of the same model "
                             "under different settings do not overwrite each other")
    parser.add_argument("--captions", action="store_true",
                        help="put each candidate's CapRL shot caption in its label, as "
                             "production does. Run the same model with and without it to "
                             "measure what the captions are worth.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--user", default=os.environ.get("AIC_AGENT_TRUSTED_USER", "owner@example.com"))
    parser.add_argument("--url", default=os.environ.get("AIC_AGENT_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    from agent.vlm import CANDIDATES

    names = ([c.name for c in CANDIDATES] if args.models == ["all"] else args.models)
    if args.command in ("pool", "all") and not POOL_JSON.exists():
        build_pool(user=args.user, base_url=args.url, top_k=args.top_k)
    if args.command in ("images", "all"):
        fetch_images(load_pool(), user=args.user, base_url=args.url)
    if args.command == "smoke":
        smoke(args.model or names[0], load_pool(), device=args.device)
        return
    if args.command in ("captions", "all"):
        fetch_captions(load_pool())
    if args.command in ("score", "all"):
        pool = load_pool()
        captions = fetch_captions(pool) if args.captions else None
        results = score_models(names, pool, device=args.device, limit=args.limit,
                               tag=args.tag, captions=captions)
        payload = write_results(results, pool)
        print(f"wrote {RESULTS} ({len(payload['models'])} models)")
    report()


def report() -> None:
    if not RESULTS.exists():
        raise SystemExit(f"no results yet at {RESULTS}")
    payload = json.loads(RESULTS.read_text())
    print(f"\n{payload['cases']} cases, tolerance {payload['tolerance_frames']} frames, "
          f"top-{payload['top_k']} pool\n")
    header = f"{'model':<34}{'AUC':>8}{'top1':>8}{'answered':>10}{'s/call':>9}{'VRAM MB':>9}"
    print(header)
    print("-" * len(header))
    for name, row in payload["models"].items():
        # A model can be recorded and still have no AUC: `error` never started, and
        # `excluded` was measured and then thrown out. Both belong in the table —
        # silently dropping them is how a bad backend comes back as a clean run.
        if "error" in row or "excluded" in row:
            print(f"{name:<34}  {row.get('error') or row.get('reason') or 'excluded'}")
            continue
        # Blank rather than 0 for a column an older `summarize` never wrote: a
        # printed 0.000 reads as "measured zero", which is a different claim.
        def cell(key, width, digits=3):
            value = row.get(key)
            return f"{'—':>{width}}" if value is None else f"{value:>{width}.{digits}f}"

        print(f"{name:<34}{cell('separation_auc', 8)}"
              f"{cell('top1_gold_rate', 8)}"
              f"{cell('answer_rate', 10)}"
              f"{cell('seconds_mean', 9, 2)}"
              f"{cell('peak_vram_mb', 9, 0)}")


if __name__ == "__main__":
    main()
