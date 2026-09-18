"""`python -m agent {check,ask,run,eval}`."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

from .config import ROOT, AgentSettings
from .evaluate import DEFAULT_TOLERANCE, score_result, summarize
from .llm import DeepSeekClient, GeminiClient, Usage
from .orchestrator import Agent, AgentOptions
from .plan import analysis_messages, initial_queries, parse_plan
from .report import render
from .search_client import SearchClient
from .submission import result_rows, to_csv, write_submission
from .tasks import load_answers, load_tasks, task_from_text

DEFAULT_OUT = ROOT / "artifacts" / "agent"


def build_agent(settings, options):
    usage = Usage()
    deepseek = DeepSeekClient(
        settings.deepseek_api_key, settings.deepseek_model, settings.deepseek_base_url,
        thinking=settings.deepseek_thinking, usage=usage,
    )
    gemini = GeminiClient(
        settings.gemini_api_key, settings.gemini_model,
        thinking_level=settings.gemini_thinking_level, usage=usage,
    )
    search = SearchClient.from_settings(settings)
    return Agent(deepseek, gemini, search, options), usage


def options_from(args, settings):
    return AgentOptions(max_steps=args.max_steps, time_budget=args.time_budget,
                        plan_thinking=settings.deepseek_plan_thinking)


def save_run(results, directory, server_url, usage=None):
    directory.mkdir(parents=True, exist_ok=True)
    for result in results:
        (directory / f"{result['query_id']}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = write_submission(results, directory)
    (directory / "report.html").write_text(render(results, server_url), encoding="utf-8")
    if usage is not None:
        (directory / "usage.json").write_text(json.dumps(usage.as_dict(), indent=2), encoding="utf-8")
    return archive


def print_result(result):
    print(f"\n== {result['query_id']} ({result['kind']}) {result['elapsed']}s"
          + (f" ERROR {result['error']}" if result["error"] else ""))
    if result.get("answer") is not None:
        print(f"   answer: {result['answer']}  [{result.get('answer_evidence')}]")
    for row, candidate in zip(result_rows(result)[:5], result["ranking"][:5]):
        verdict = candidate.get("verdict")
        print("   " + ",".join(row) + (f"   {verdict['score']:.0f}/10 {verdict['reason']}" if verdict else ""))
    if result["kind"] == "trake":
        print("   " + to_csv(result_rows(result)[:3]).replace("\n", "\n   ").rstrip())


def command_check(args, settings):
    search = SearchClient.from_settings(settings)
    print(f"server {settings.server_url}")
    probe = search.search([{"kind": "visual", "id": "probe", "text": "a boat on a river", "limit": 3}])
    print("  /agent/search:", json.dumps(probe, ensure_ascii=False)[:300])
    deepseek = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_model,
                              settings.deepseek_base_url, thinking=False)
    reply = deepseek.chat([{"role": "user", "content": "Reply with the single word: ok"}], max_tokens=10)
    print(f"deepseek {settings.deepseek_model}: {reply.get('content')!r}")
    gemini = GeminiClient(settings.gemini_api_key, settings.gemini_model,
                          thinking_level=settings.gemini_thinking_level)
    grounded = gemini.web_search("Thủ đô của Việt Nam là gì? Trả lời một từ.")
    print(f"gemini {settings.gemini_model} (google search): {grounded.text[:80]!r}, "
          f"{len(grounded.sources)} sources")


def command_plan(args, settings):
    """Planning only -- no search server needed. For tuning the analysis prompt."""
    deepseek = DeepSeekClient(settings.deepseek_api_key, settings.deepseek_model,
                              settings.deepseek_base_url, thinking=settings.deepseek_plan_thinking)
    tasks = load_tasks(args.path)
    if args.only:
        tasks = [task for task in tasks if task.query_id in set(args.only)]

    def plan_one(task):
        started = time.monotonic()
        try:
            plan = parse_plan(deepseek.chat_json(analysis_messages(task)), task)
        except Exception as exc:
            return task, None, f"{type(exc).__name__}: {exc}", time.monotonic() - started
        return task, plan, None, time.monotonic() - started

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for task, plan, error, seconds in pool.map(plan_one, tasks):
            print(f"\n== {task.query_id} ({seconds:.1f}s)\n{task.text}")
            if error:
                print(f"   ERROR {error}")
                continue
            print(json.dumps(plan.to_dict(), ensure_ascii=False, indent=1))
            for query in initial_queries(plan, task):
                print(f"   -> {query['id']:9} {query['kind']:8} "
                      f"{query.get('text') or query.get('events')}")
    print(f"\nusage: {deepseek.usage.as_dict()}")


def run_tasks(tasks, args, settings):
    agent, usage = build_agent(settings, options_from(args, settings))
    directory = Path(args.out) if args.out else DEFAULT_OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = [result.to_dict() for result in pool.map(agent.run, tasks)]
    for result in results:
        print_result(result)
    archive = save_run(results, directory, settings.server_url, usage)
    print(f"\nsaved {len(results)} results to {directory}\n  submission: {archive}\n"
          f"  report: {directory / 'report.html'}\n  usage: {usage.as_dict()}")
    return results, directory


def command_ask(args, settings):
    task = task_from_text(args.id or f"ask-{args.kind}", args.text, args.kind)
    run_tasks([task], args, settings)


def command_run(args, settings):
    tasks = load_tasks(args.path)
    if args.only:
        tasks = [task for task in tasks if task.query_id in set(args.only)]
    if not tasks:
        sys.exit("no queries found")
    results, directory = run_tasks(tasks, args, settings)
    if args.answers:
        report_scores(results, load_answers(args.answers), args.tolerance, directory)


def report_scores(results, answers, tolerance, directory=None):
    scores = [score_result(result, answers[result["query_id"]], tolerance)
              for result in results if result["query_id"] in answers]
    for score in scores:
        print(f"  {score.query_id:24} hit@{score.hit_rank} video@{score.video_rank}"
              + (f" answer={'OK' if score.answer_ok else 'WRONG'} "
                 f"({score.answer!r} vs {score.gold_answer!r})" if score.kind == "qa" else ""))
    summary = summarize(scores)
    print(json.dumps(summary, indent=2))
    if directory is not None:
        (Path(directory) / "scores.json").write_text(json.dumps(
            {"tolerance": tolerance, "summary": summary, "queries": [vars(s) for s in scores]},
            ensure_ascii=False, indent=2), encoding="utf-8")


def command_eval(args, _settings=None):
    directory = Path(args.run)
    results = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))
               if path.name not in {"usage.json", "scores.json"}]
    report_scores(results, load_answers(args.answers), args.tolerance, directory)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m agent", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def agent_flags(command):
        command.add_argument("--out", help="output directory (default artifacts/agent/<timestamp>)")
        command.add_argument("--max-steps", type=int, default=6, help="agent tool rounds after round 1")
        command.add_argument("--time-budget", type=float, default=300.0, help="seconds per query")
        command.add_argument("--workers", type=int, default=2, help="queries processed concurrently")

    sub.add_parser("check", help="verify server, DeepSeek and Gemini connectivity")
    plan = sub.add_parser("plan", help="show the analysis and round-1 queries only (no server)")
    plan.add_argument("path")
    plan.add_argument("--only", nargs="*")
    plan.add_argument("--workers", type=int, default=4)
    ask = sub.add_parser("ask", help="run one query given on the command line")
    ask.add_argument("text")
    ask.add_argument("--kind", choices=["kis", "qa", "trake"], default="kis")
    ask.add_argument("--id")
    agent_flags(ask)
    run = sub.add_parser("run", help="run a query file, directory or organiser zip")
    run.add_argument("path")
    run.add_argument("--only", nargs="*", help="query ids to run")
    run.add_argument("--answers", help="answer key (csv or zip) to score against afterwards")
    run.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    agent_flags(run)
    evaluate = sub.add_parser("eval", help="score a saved run directory against an answer key")
    evaluate.add_argument("run")
    evaluate.add_argument("--answers", required=True)
    evaluate.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "google_genai", "google_genai.models"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if args.command == "eval":
        return command_eval(args)
    settings = AgentSettings.from_env()
    commands = {"check": command_check, "plan": command_plan, "ask": command_ask, "run": command_run}
    commands[args.command](args, settings)
