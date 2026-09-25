#!/usr/bin/env python3
"""Score the retrieval agent in **human-review mode** against the team's own round answers.

`docs/reports/agent-benchmark-2026-09.md` measures `judge=model` (fast mode). This measures
the other mode, which the CLI cannot drive: a human-judged run waits for a person, so the
reviewer here is scripted -- it trusts the view the run puts in front of the person and
submits what it offers:

* it waits for the checklist (`summarizing` False) or a per-query deadline, then finishes;
* it makes no accept/reject decisions, so the ranking the run finalises is the agent's own
  evidence-ordered view -- which is what the ordering exists to produce;
* for a Q&A query it types no answer, so the run submits the first chip it read from the
  frames (`suggested from gemini`, marked unconfirmed in the submission).

Ground truth is the **live collaboration database**, not the JSON snapshot: the snapshot
records gold frames only, while a Q&A query's answer string lives in the round's rows
(`/api/collab/rounds`). Tolerance is `agent.evaluate.DEFAULT_TOLERANCE` (250 frames), so
the numbers sit on the same scale as the model-judged baseline.

**Run one query at a time** (`--workers 1`, the default). Two checklists in flight measured
above Gemini's 2M input tokens a minute: the calls come back 429, `vision_summarize` catches
each one by design and leaves that candidate a text-only checklist, so the run finishes
looking whole while having measured something the deployment does not do. Every query records
its `checklist` health (`shown` / `summarised` / `with_vision`) and the report names any query
that ended with no vision checks at all.

Usage (from the repository root):

    uv run python -m scripts.benchmarks.bench_agent_human --rounds 1 2 3 \\
        --exclude query-p2-27-qa query-p2-5-qa query-p2-6-qa query-p2-9-kis \\
        --out artifacts/agent/human-bench

    uv run python -m scripts.benchmarks.bench_agent_human --score artifacts/agent/human-bench

Rows are written per query as they finish, so a stopped run resumes by pointing `--out` at
the same directory: a query with a saved `.json` is not run again.
"""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.config import AgentSettings, ROOT as REPO_ROOT
from agent.evaluate import DEFAULT_TOLERANCE, score_result, summarize
from agent.llm import DeepSeekClient, GeminiClient, OfflineGemini, Usage
from agent.orchestrator import Agent, AgentOptions
from agent.review import HumanReview
from agent.search_client import SearchClient
from agent.tasks import GoldRow, task_from_text
from agent.vlm import build_vision

#: The 2026 rounds as the collaboration server names them, in round order.
ROUND_NAMES = {1: "Round 1 - 2026", 2: "Round 2 - 2026", 3: "Round 3 - 2026"}
#: How long a scripted reviewer waits for the checklist before finishing with what it has.
REVIEW_DEADLINE_SECONDS = 240.0


def fetch_round(settings: AgentSettings, name: str) -> dict:
    """One round with its queries and answers, as the collaboration API returns it."""
    headers = {"X-AIC-User": settings.trusted_user} if settings.trusted_user else {}
    auth = ((settings.server_user, settings.server_password)
            if settings.server_user and settings.server_password else None)
    with httpx.Client(base_url=settings.server_url, headers=headers, auth=auth, timeout=60.0) as http:
        rounds = http.get("/api/collab/rounds").json()
        found = next((row for row in rounds if row["name"] == name), None)
        if found is None:
            raise SystemExit(f"no round named {name!r} on {settings.server_url}")
        return http.get(f"/api/collab/rounds/{found['id']}").json()


def gold_of(query: dict) -> list[GoldRow]:
    """The round's answer rows as gold rows, in the shape `agent.evaluate` reads.

    ONE ROW PER ACCEPTED FRAME, not one per video: `score_result` compares a submitted frame
    against `gold.frames[0]` of every row for that video (the CSV key is one accepted frame a
    row, possibly many rows for one clip), so a video's other accepted frames have to be rows
    of their own or they are not gold at all. Collapsing them into one row's tuple scored a
    submission sitting exactly on the clip's *second* accepted frame as a miss.

    A TRAKE row is the exception and keeps its frames together: `frame_ids` is one frame per
    EVENT, in order, compared as a sequence -- grouping them per video is what the format
    means, and sorting them would destroy the order the answer is.
    """
    rows, answer = [], None
    for row in query.get("answers") or []:
        video_id = str(row.get("video_id") or "").strip()
        if not video_id:
            continue
        answer = answer or (str(row.get("answer") or "").strip() or None)
        events = row.get("frame_ids")
        if events is not None:  # TRAKE: the whole sequence, in event order
            frames = tuple(int(frame) for frame in events if frame is not None)
            if frames:
                rows.append(GoldRow(video_id, frames, None))
        elif row.get("frame_id") is not None:
            rows.append(GoldRow(video_id, (int(row["frame_id"]),), None))
    return [GoldRow(row.video_id, row.frames, answer) for row in rows]


def tasks_and_gold(round_payload: dict, exclude=(), only=()) -> tuple[list, dict, dict]:
    """One round's query files to run and their gold rows, skipping the excluded ones.

    ONE ROUND AT A TIME, because the rounds reuse each other's file names: round 3's
    `query-p2-1-kis` and round 2's are different queries with the same name, so gold keyed
    by filename across rounds would score one round's answers against another's. An
    exclusion is by FILENAME and applies to the round it is given for, which is how the
    team names them ("query-p2-27-qa of round 2", "query-p2-9-kis of round 3").
    """
    excluded, wanted = set(exclude), set(only)
    tasks, golds, kinds = [], {}, {}
    for query in round_payload["queries"]:
        filename = query["filename"]
        if filename in excluded or (wanted and filename not in wanted):
            continue
        gold = gold_of(query)
        if not gold:
            continue  # a query nobody answered has nothing to score against
        tasks.append(task_from_text(filename, query["prompt"]))
        golds[filename] = gold
        kinds[filename] = query["kind"]
    return tasks, golds, kinds


def scripted_reviewer(review: HumanReview, deadline: float):
    """A person who trusts the view: waits for the checklist, then submits the ranking."""

    def watch(snapshot):
        if snapshot["stage"] != "awaiting_review" or review.decided():
            return
        if not snapshot.get("summarizing") or time.monotonic() > deadline:
            review.finish()

    return watch


def checklist_health(snapshot: dict) -> dict:
    """How much of the checklist actually got read, so a degraded run cannot be scored quietly.

    A rate-limited vision backend (Gemini answers 429 above 2M input tokens a minute) fails
    one candidate at a time and `vision_summarize` catches each failure by design, leaving
    that candidate a text-only checklist -- a run that looks finished and is not the one the
    deployment produces. `with_vision` is the count that tells the two apart.
    """
    candidates = snapshot.get("candidates") or []
    return {
        "shown": len(candidates),
        "summarised": sum(1 for candidate in candidates if candidate.get("summary")),
        "with_vision": sum(1 for candidate in candidates if candidate.get("summary") and any(
            check.get("source") == "vision" for check in candidate["summary"]["checks"])),
    }


def run_one(agent: Agent, task, deadline_seconds: float) -> dict:
    review = HumanReview(judge="human")
    deadline = time.monotonic() + deadline_seconds
    last: dict = {}

    def watch(snapshot):
        last.clear()
        last.update(snapshot)
        scripted_reviewer(review, deadline)(snapshot)

    try:
        result = agent.run(task, review=review, progress=watch).to_dict()
    except Exception as exc:  # one query must not stop the round
        result = {"query_id": task.query_id, "kind": task.kind, "query": task.text,
                  "error": f"{type(exc).__name__}: {exc}", "ranking": [], "sequences": [],
                  "answer": None, "elapsed": 0.0}
    result["checklist"] = checklist_health(last)
    result["deadline_hit"] = bool(result.get("elapsed") and result["elapsed"] >= deadline_seconds)
    return result


class PacedGemini:
    """A Gemini client that keeps the run under the account's input-token ceiling.

    The checklist is ~80 calls of ~25 images -- measured at ~37k input tokens each, three in
    flight -- so a single query bursts past Gemini's 2M input tokens a minute. Every call the
    ceiling turns back is retried, and `vision_summarize` catches the ones that stay failed,
    leaving that candidate a text-only checklist: the run finishes looking whole and is not
    the run the deployment produces. Pacing changes *when* a call is asked for, never what is
    asked or what comes back, so the checklists a paced benchmark reads are the same ones a
    live run reads when the quota allows it.
    """

    def __init__(self, client, *, tokens_per_minute: float = 1_200_000.0,
                 call_tokens: float = 40_000.0, clock=time.monotonic, sleep=time.sleep):
        self.client = client
        self.budget = tokens_per_minute
        self.call_tokens = call_tokens
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._spent: collections.deque = collections.deque()  # (time, tokens)

    def _reserve(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                while self._spent and now - self._spent[0][0] > 60.0:
                    self._spent.popleft()
                used = sum(tokens for _, tokens in self._spent)
                if used + self.call_tokens <= self.budget:
                    self._spent.append((now, self.call_tokens))
                    return
                wait = 60.0 - (now - self._spent[0][0]) + 0.05
            self._sleep(max(0.05, min(wait, 5.0)))

    def judge(self, images, prompt, schema, **kwargs):
        self._reserve()
        return self.client.judge(images, prompt, schema, **kwargs)

    def web_search(self, question):
        self._reserve()
        return self.client.web_search(question)

    def __getattr__(self, name):  # usage, model, and anything else the agent reaches for
        return getattr(self.client, name)


def build_agent(settings: AgentSettings, options: AgentOptions):
    usage = Usage()
    deepseek = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_model,
                              settings.deepseek_base_url, thinking=settings.deepseek_thinking,
                              usage=usage, timeout=30.0, retries=2)
    # No `hedge_after`, unlike a live run: hedging duplicates a call that is slow, which at
    # the checklist's size is ~80 calls of ~25 images each and put a SINGLE query over
    # Gemini's 2M input tokens a minute -- the copies come back 429 and cost that candidate
    # its vision checks. A duplicate can only ever lose a race, never change a reply, so the
    # benchmark reads the same checklists without it, just slower than a live run would.
    gemini = (GeminiClient(settings.gemini_api_key, settings.gemini_model,
                           thinking_level=settings.gemini_thinking_level, usage=usage,
                           timeout=40.0, retries=4, max_backoff=20.0)
              if settings.gemini_api_key else OfflineGemini())
    gemini = PacedGemini(gemini)
    vision = build_vision(settings, usage=usage, gemini=gemini)
    search = SearchClient.from_settings(settings)
    return Agent(deepseek, gemini, search, options, vision=vision), usage, search


def read_results(directory: Path) -> list[dict]:
    """The saved per-query results of one round's directory, its own bookkeeping left out."""
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
            if path.name not in {"scores.json", "usage.json", "gold.json"}]


def read_gold(directory: Path) -> dict:
    rows = json.loads((directory / "gold.json").read_text(encoding="utf-8"))
    return {query_id: [GoldRow(row["video_id"], tuple(row["frames"]), row.get("answer"))
                       for row in queries] for query_id, queries in rows.items()}


def score(results: list[dict], golds: dict, tolerance: int) -> dict:
    scores = [score_result(result, golds[result["query_id"]], tolerance)
              for result in results if golds.get(result["query_id"])]
    return {"summary": summarize(scores), "queries": [vars(score) for score in scores]}


def report(summary: dict, title: str, results: list[dict] | None = None) -> None:
    print(f"\n== {title}")
    print(json.dumps(summary.get("rows") or summary, ensure_ascii=False, indent=2))
    degraded = [result for result in results or []
                if not (result.get("checklist") or {}).get("with_vision")]
    if degraded:
        print(f"  DEGRADED (no vision checks at all, so not the deployment's checklist): "
              f"{[result['query_id'] for result in degraded]}")


def print_health(results: list[dict]) -> None:
    for result in results:
        health = result.get("checklist") or {}
        print(f"  {result['query_id']:26} vision {health.get('with_vision', 0):>3}/"
              f"{health.get('shown', 0):<3} summarised {health.get('summarised', 0):>3}"
              f"{' DEADLINE HIT' if result.get('deadline_hit') else ''}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="bench_agent_human", description=__doc__)
    parser.add_argument("--rounds", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--exclude", nargs="*", default=[], help="query filenames to leave out")
    parser.add_argument("--only", nargs="*", help="run just these query filenames (a pilot)")
    parser.add_argument("--out", help="output directory (default artifacts/agent/human-bench/<stamp>)")
    parser.add_argument("--workers", type=int, default=1, help=(
        "queries at once. ONE by default: the checklist is ~80 vision calls with ~25 images "
        "each, and two of those in flight exceeded Gemini's 2M input tokens a minute, which "
        "fails candidates one by one into a text-only checklist -- a run that is not the "
        "deployment's. Increase only if the account's TPM allows it."))
    parser.add_argument("--deadline", type=float, default=REVIEW_DEADLINE_SECONDS)
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--score", help="score an existing output directory instead of running")
    args = parser.parse_args(argv)

    if args.score:
        root = Path(args.score)
        for directory in sorted(path for path in root.iterdir() if path.is_dir()) or [root]:
            results = read_results(directory)
            if not results:
                continue
            scored = score(results, read_gold(directory), args.tolerance)
            (directory / "scores.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
            report(scored["summary"], str(directory), results)
        return

    settings = AgentSettings.from_env()
    out_root = Path(args.out) if args.out else \
        REPO_ROOT / "artifacts" / "agent" / f"human-bench-{datetime.now():%Y%m%d-%H%M%S}"
    options = AgentOptions.for_mode(
        "fast", judge="human", checklist_vision=True,
        checklist_vision_model=settings.checklist_vision_model,
    )
    agent, usage, search = build_agent(settings, options)
    started = time.monotonic()
    try:
        for number in args.rounds:
            payload = fetch_round(settings, ROUND_NAMES.get(number, str(number)))
            tasks, golds, kinds = tasks_and_gold(payload, exclude=args.exclude, only=args.only or ())
            if not tasks:
                print(f"round {number}: no queries selected")
                continue
            directory = out_root / f"round-{number}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "gold.json").write_text(json.dumps(
                {qid: [{"video_id": row.video_id, "frames": list(row.frames), "answer": row.answer}
                       for row in rows] for qid, rows in golds.items()},
                ensure_ascii=False, indent=2), encoding="utf-8")
            pending = [task for task in tasks if not (directory / f"{task.query_id}.json").exists()]
            print(f"\nround {number}: {len(tasks)} queries "
                  f"({len(tasks) - len(pending)} already saved) -> {directory}", flush=True)
            print("kinds:", {kind: sum(1 for task in tasks if kinds[task.query_id] == kind)
                             for kind in sorted(set(kinds.values()))}, flush=True)
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(lambda task: run_one(agent, task, args.deadline), pending):
                    (directory / f"{result['query_id']}.json").write_text(
                        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    health = result.get("checklist") or {}
                    print(f"  {result['query_id']:26} {result['elapsed']:6.1f}s "
                          f"vision {health.get('with_vision', 0)}/{health.get('shown', 0)}"
                          f"{' DEADLINE' if result.get('deadline_hit') else ''}"
                          f"{' ERROR ' + str(result['error'])[:50] if result.get('error') else ''}",
                          flush=True)
    finally:
        search.close()
        (directory / "usage.json").write_text(json.dumps(usage.as_dict(), indent=2), encoding="utf-8")

    (out_root / "usage.json").write_text(json.dumps(usage.as_dict(), indent=2), encoding="utf-8")
    print(f"\nwall clock: {time.monotonic() - started:.0f}s | usage: {usage.as_dict()}")
    for directory in sorted(path for path in out_root.iterdir() if path.is_dir()):
        results = read_results(directory)
        if not results:
            continue
        scored = score(results, read_gold(directory), args.tolerance)
        (directory / "scores.json").write_text(json.dumps(scored, ensure_ascii=False, indent=2),
                                               encoding="utf-8")
        print_health(results)
        report(scored["summary"], str(directory), results)


if __name__ == "__main__":
    main()
