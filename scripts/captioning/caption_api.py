#!/usr/bin/env python3
"""Caption every shot window against a local vLLM OpenAI server.

`caption_shots.caption_records` prepares a batch, generates it, then prepares the next,
so video decode and GPU work strictly alternate -- and at the documented `--batch-size 1`
it also gives up continuous batching. Measured throughput was 27 s/shot on Kaggle T4x2
and ~35 s/shot on a 5090 under plain transformers, i.e. hundreds of hours for one year
of the corpus.

This runner splits the two halves instead. A process pool decodes JPEG frames while an
asyncio client keeps `--concurrency` requests in flight against `vllm serve`, so the GPU
never waits on ffmpeg. Everything else is kept from `caption_shots`: whole-video
sharding, atomic per-video parquet parts, and resume by exact key sequence.

    vllm serve internlm/CapRL-Video-4B --served-model-name caprl ...
    python -m scripts.captioning.caption_api --year 2026
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import pyarrow.parquet as pq

from scripts.captioning import frames as frame_io, windows
from scripts.captioning.caption_shots import (
    WINDOW_CAPTION_SCHEMA, ShotRecord, atomic_write_json,
    atomic_write_table, log, records_by_video, select_shard, strip_timestamps,
)

from backend import shots as backend_shots


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "caprl"
# vLLM's own listener. Port 8000 on a Vast.ai instance is the Caddy auth edge, which
# answers 401 without a token; localhost:18000 bypasses it entirely.
DEFAULT_ENDPOINT = "http://127.0.0.1:18000/v1"
RETRY_DELAYS = (2.0, 8.0, 20.0)

# The exact prompt of the "1024-default-8frames" quality baseline. It is not
# caption_shots.PROMPT: the baseline parquet records prompt_sha256 b81f69d3..., and this
# is the string that hashes to it (copied verbatim from
# notebooks/kaggle_caprl_video_benchmark.py:22, where the baseline was produced). Do not
# edit without re-validating against that artifact -- tests assert the hash.
RECIPE_PROMPT = (
    "Describe this video clip in English in great detail and in strictly chronological "
    "order. Cover all visible people and their appearance, objects, actions and "
    "interactions, setting, lighting, camera changes, and readable on-screen text. "
    "Return only one self-contained caption without headings or timestamps."
)
RECIPE_PROMPT_SHA256 = hashlib.sha256(RECIPE_PROMPT.encode()).hexdigest()
BASELINE_PROMPT_SHA256 = "b81f69d3a3ef3a198d2062c8857ed726ddc4d226f414a9c23e91d3f7a7af2ea7"

# The seven shots under manual review plus the hard cases from the three 2026 rounds:
# every total miss (rank > 1000) and the deep-rank near misses from
# docs/reports/visual-search-recall-failure-analysis-2026.md.
PILOT_TARGETS = (
    ("L27_V010", 83), ("L30_V078", 20), ("L30_V078", 21), ("L30_V072", 19),
    ("K17_V003", 102), ("K19_V016", 158), ("L23_V013", 20), ("L23_V013", 21),
    ("L25_V060", 66), ("L25_V012", 81), ("L25_V041", 25), ("L25_V062", 43),
    ("L24_V006", 38), ("L24_V024", 28), ("L24_V024", 29),
    ("L26_V156", 82), ("L26_V156", 61), ("L26_V481", 87), ("L28_V018", 14),
    ("L21_V003", 224), ("L26_V389", 86), ("L26_V482", 28), ("L30_V003", 46),
    ("L29_V021", 43), ("L22_V024", 230),
    # Added on request. Neither is the round's gold shot: R2 p2-30-qa's frame 529
    # sits in L26_V254 shot 7, and R3 p2-24-qa's frame 2819 in L26_V448 shot 44.
    ("L26_V254", 8), ("L26_V448", 38), ("L26_V448", 39),
)


def part_path(output_dir, unit):
    return Path(output_dir) / "parts" / unit.group / f"{unit.video_id}.parquet"


def part_is_complete(path, expected):
    """True only if the part holds exactly the expected units, in order.

    Mirrors `caption_shots.part_is_complete` but keys on the window too, so a part
    written before a shot was split is correctly treated as stale.
    """
    if not Path(path).exists():
        return False
    try:
        rows = pq.read_table(
            path, columns=["video_id", "shot_index", "window_index"]
        ).to_pylist()
    except Exception:
        return False
    return [(r["video_id"], r["shot_index"], r["window_index"]) for r in rows] == [
        (u.video_id, u.shot_index, u.window_index) for u in expected
    ]


def decode_unit(payload):
    """Runs in the process pool: unit -> list of JPEG frames."""
    unit, width, quality, cache_dir = payload
    return frame_io.sample_frames_cached(unit, cache_dir, width, quality)


def build_request(unit, jpegs, args):
    """A pseudo-video of base64 JPEGs, plus the real timeline it was sampled from.

    Prompt and sampling follow the "1024-default-8frames" recipe that was picked as the
    quality baseline (it reads "Mộc Châu" correctly on K17_V003 shot 102; every
    wider-sampling-window config tried did not), with the sampling params tuned on the
    pilot afterwards (see parse_args) and `max_tokens` raised from 1024 to 2048. That
    baseline run called `model.generate(**inputs, max_new_tokens=max_tokens)` with no
    sampling args, which falls back to the checkpoint's own `generation_config.json`
    (do_sample=true, temperature=0.7, top_p=0.8, top_k=20) rather than vLLM's
    SamplingParams default of temperature=1.0/unbounded top_p -- reproduced here
    explicitly since the HTTP API has no checkpoint default to fall back to.

    Sending frames rather than a file keeps frame selection on our side -- exact
    decode-order indices inside the shot window -- while `media_io_kwargs` still gives
    Qwen3-VL the true fps and duration, so its temporal reasoning is not lied to.
    """
    encoded = ",".join(base64.b64encode(jpeg).decode() for jpeg in jpegs)
    body = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "video_url",
                 "video_url": {"url": f"data:video/jpeg;base64,{encoded}"}},
                {"type": "text", "text": RECIPE_PROMPT},
            ],
        }],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # top_k is a vLLM extension field on the request body itself, not an
        # OpenAI-SDK "extra_body" wrapper -- this is a raw JSON POST, so it must be a
        # plain top-level key or vLLM silently ignores it and samples unbounded.
        "top_k": args.top_k,
        "media_io_kwargs": {"video": {
            "fps": len(jpegs) / max(unit.duration_sec, 1e-6),
            "duration": unit.duration_sec,
            "total_num_frames": len(jpegs),
        }},
    }
    # vLLM's default is 1.0, so the key is only sent when a sweep asks for something
    # else -- keeping the production request identical to the validated recipe.
    if args.repetition_penalty != 1.0:
        body["repetition_penalty"] = args.repetition_penalty
    if args.seed is not None:
        body["seed"] = args.seed
    return body


def build_row(unit, jpegs, payload, elapsed, args):
    choice = payload["choices"][0]
    raw = (choice["message"].get("content") or "").strip()
    if not raw:
        raise RuntimeError(f"empty caption for {unit.unit_id}")
    usage = payload.get("usage") or {}
    return {
        "year": unit.year,
        "group": unit.group,
        "video_id": unit.video_id,
        "shot_index": unit.shot_index,
        "window_index": unit.window_index,
        "window_count": unit.window_count,
        "start_frame": unit.start_frame,
        "end_frame": unit.end_frame,
        "start_sec": unit.shot_start_sec,
        "end_sec": unit.shot_end_sec,
        "duration_sec": unit.shot_duration_sec,
        "window_start_sec": unit.window_start_sec,
        "window_end_sec": unit.window_end_sec,
        "window_duration_sec": unit.duration_sec,
        "frames_used": len(jpegs),
        "frame_width": args.frame_width,
        "summary": "",
        "caption": strip_timestamps(raw),
        "onscreen_text": "",
        "raw_output": raw,
        "model_id": args.model_id,
        "prompt_sha256": RECIPE_PROMPT_SHA256,
        "kv_cache_dtype": args.kv_cache_dtype,
        "max_tokens": args.max_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "output_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "finish_reason": choice.get("finish_reason") or "",
        "elapsed_sec": elapsed,
    }


async def caption_unit(client, pool, unit, args, gates):
    loop = asyncio.get_running_loop()
    last_error = None
    for attempt in range(len(RETRY_DELAYS) + 1):
        started = time.perf_counter()
        try:
            async with gates["decode"]:
                jpegs = await loop.run_in_executor(
                    pool, decode_unit,
                    (unit, args.frame_width, args.jpeg_quality, args.frame_cache),
                )
            async with gates["request"]:
                response = await client.post(
                    "/chat/completions", json=build_request(unit, jpegs, args),
                )
            response.raise_for_status()
            return build_row(
                unit, jpegs, response.json(), time.perf_counter() - started, args,
            )
        except Exception as error:  # noqa: BLE001 - retried, then reported per video
            last_error = error
            if attempt < len(RETRY_DELAYS):
                await asyncio.sleep(RETRY_DELAYS[attempt])
    raise RuntimeError(f"{unit.unit_id}: {type(last_error).__name__}: {last_error}")


async def run(units, args):
    import httpx

    by_video = defaultdict(list)
    for unit in units:
        by_video[unit.video_id].append(unit)

    pending, done_units = {}, 0
    for video_id, video_units in by_video.items():
        if part_is_complete(part_path(args.output_dir, video_units[0]), video_units):
            done_units += len(video_units)
            continue
        pending[video_id] = video_units
    log(f"resume: {len(by_video) - len(pending)} videos complete, {len(pending)} to do "
        f"({len(units) - done_units:,} units)")
    if not pending:
        return {"units": 0, "failed_videos": []}

    queue = [unit for video_units in pending.values() for unit in video_units]
    collected = defaultdict(dict)
    failures, started = defaultdict(list), time.perf_counter()
    gates = {
        "decode": asyncio.Semaphore(args.decoders * 2),
        "request": asyncio.Semaphore(args.concurrency),
    }
    cursor = completed = 0

    async def worker(client, pool):
        # Pull from a shared cursor rather than a static stride: a unit that retries
        # must not hold back the slice a fixed partition would have assigned it.
        nonlocal cursor, completed
        while True:
            if cursor >= len(queue):
                return
            unit, cursor = queue[cursor], cursor + 1
            try:
                key = (unit.shot_index, unit.window_index)
                collected[unit.video_id][key] = await caption_unit(
                    client, pool, unit, args, gates,
                )
            except Exception as error:  # noqa: BLE001
                failures[unit.video_id].append(str(error))
            completed += 1
            if completed % 200 == 0:
                rate = completed / (time.perf_counter() - started)
                log(f"{completed:,}/{len(queue):,} units  {rate:.2f} units/s  "
                    f"eta {(len(queue) - completed) / max(rate, 1e-9) / 3600:.1f} h")

    limits = httpx.Limits(max_connections=args.concurrency + 8)
    timeout = httpx.Timeout(args.request_timeout, connect=30.0)
    workers = args.concurrency + args.decoders * 2
    with ProcessPoolExecutor(max_workers=args.decoders) as pool:
        async with httpx.AsyncClient(base_url=args.endpoint, limits=limits,
                                     timeout=timeout) as client:
            await asyncio.gather(*(worker(client, pool) for _ in range(workers)))

    for video_id, video_units in pending.items():
        if failures.get(video_id):
            log(f"FAILED {video_id}: {len(failures[video_id])} units; part not written")
            continue
        rows = [collected[video_id][(unit.shot_index, unit.window_index)]
                for unit in video_units]
        atomic_write_table(rows, WINDOW_CAPTION_SCHEMA,
                           part_path(args.output_dir, video_units[0]))

    elapsed = time.perf_counter() - started
    log(f"{len(queue):,} units in {elapsed / 3600:.2f} h "
        f"({len(queue) / max(elapsed, 1e-9):.2f} units/s)")
    return {
        "units": len(queue),
        "elapsed_sec": elapsed,
        "units_per_second": len(queue) / max(elapsed, 1e-9),
        "failed_videos": sorted(failures),
        "failures": {video: errors[:5] for video, errors in failures.items()},
    }


def merge_parts(units, args):
    """Write shot_captions.parquet only when every part validates."""
    import pyarrow as pa

    tables, missing = [], []
    for video_id, video_units in sorted(records_by_video(units).items()):
        path = part_path(args.output_dir, video_units[0])
        if not part_is_complete(path, video_units):
            missing.append(video_id)
            continue
        tables.append(pq.read_table(path, schema=WINDOW_CAPTION_SCHEMA))
    if missing:
        raise RuntimeError(
            f"{len(missing)} videos incomplete, first {missing[0]}; not merging"
        )
    table = pa.concat_tables(tables)
    if table.num_rows != len(units):
        raise RuntimeError(f"expected {len(units)} rows, got {table.num_rows}")
    target = Path(args.output_dir) / "shot_captions.parquet"
    temporary = target.with_name(f".{target.name}.tmp")
    pq.write_table(table, temporary)
    os.replace(temporary, target)
    log(f"merged {table.num_rows:,} rows -> {target}")
    return table.num_rows


def load_records(data_dir, year, groups):
    """Shot records for a year, without requiring the MP4s to be on disk yet.

    `caption_shots.load_shot_records` raises on a missing video, which is right for its
    all-at-once runner but wrong here: the pilot selects 25 shots out of 282,706, and
    the staging loop deletes each group's videos as soon as it is captioned. Presence is
    checked in `select_units` once the records have been narrowed to what will actually
    be read.
    """
    data_dir, year = Path(data_dir), str(year)
    selected, records = set(groups), []
    for path in sorted((data_dir / year / "shots").glob("*/*.json")):
        group, video_id = path.parent.name, path.stem
        if selected and group not in selected:
            continue
        index = backend_shots.load_path(path)
        if index.fps <= 0:
            raise ValueError(f"invalid fps in {path}")
        video = data_dir / year / "videos" / group / f"{video_id}.mp4"
        records.extend(
            ShotRecord(year, group, video_id, shot_index, int(start), int(end),
                       index.fps, str(video))
            for shot_index, (start, end) in enumerate(index.bounds)
        )
    return records


def select_units(args):
    records = []
    for year in args.years:
        records.extend(load_records(args.data_dir, year, args.groups))
    if not records:
        raise ValueError(f"no shots found for years {args.years} groups {args.groups}")
    if args.targets:
        wanted = set(args.targets)
        records = [r for r in records if (r.video_id, r.shot_index) in wanted]
        found = {(r.video_id, r.shot_index) for r in records}
        for target in sorted(wanted - found):
            log(f"WARNING target not found: {target[0]} shot {target[1]}")
    if not (args.dry_run or args.merge_only):
        missing = sorted({r.video_path for r in records
                          if not Path(r.video_path).exists()})
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} videos are not on disk, first {missing[0]}"
            )
    if args.shard_count > 1:
        records = select_shard(records, args.shard_count, args.shard_index)
        log(f"shard {args.shard_index}/{args.shard_count}: {len(records):,} shots")
    units = windows.build_units(records, args.window_seconds, args.overlap_seconds)
    if args.limit:
        units = units[:args.limit]
    return units


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=backend_shots.DATA)
    parser.add_argument("--years", default="2026", help="comma-separated")
    parser.add_argument("--groups", default="", help="comma-separated")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="--served-model-name on the vLLM server")
    parser.add_argument("--model-id", default="internlm/CapRL-Video-4B",
                        help="recorded in the parquet, not sent to the server")
    parser.add_argument("--kv-cache-dtype", default="auto",
                        help="recorded in the parquet; set to match how vllm serve ran")
    parser.add_argument("--frame-width", type=int, default=0,
                        help="scale frames to this width; 0 keeps the source size "
                             "(the baseline config; matches transformers' do_resize=False)")
    parser.add_argument("--jpeg-quality", type=int, default=frame_io.DEFAULT_QUALITY)
    # 2048, not the baseline's 1024: over a third of the pilot captions hit the 1024 cap
    # and lost their tail mid-sentence. The prompt and sampling are untouched, so this
    # only lets a caption that would have been cut finish itself.
    parser.add_argument("--max-tokens", type=int, default=2048)
    # Generation parameters were tuned on the 137-unit pilot, scored on 25 fidelity
    # checks over the shots with known on-screen text (see docs/caprl-shot-captioning.md).
    # 0.8/0.9/40 read 21/25 on every seed tried (0.7/0.8/20 ranged 17-22), kept the
    # water-gauge numbers, produced no repetition loops, and yielded the most quoted
    # text. 1.0/1.0 with top_k off was the worst (12-20/25, invented an "increments of
    # 8" pattern on the gauge). The checkpoint's own generation_config.json (0.7/0.8/20)
    # is the incumbent that these beat.
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="1.0 is vLLM's default; sent only when different")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--decoders", type=int, default=0,
                        help="frame-decode processes; 0 picks from the CPU count")
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--window-seconds", type=float, default=windows.WINDOW_SECONDS)
    parser.add_argument("--overlap-seconds", type=float, default=windows.OVERLAP_SECONDS)
    parser.add_argument("--frame-cache", type=Path, default=None,
                        help="reuse identical JPEGs across configs (pilot A/B only)")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--pilot", action="store_true",
                        help="only the manual-review and hard-case shots")
    parser.add_argument("--pilot-only", default="",
                        help="with --pilot, narrow to these VIDEO_ID:SHOT pairs "
                             "(comma separated) -- for a box that only has some of "
                             "the pilot videos staged")
    parser.add_argument("--dry-run", action="store_true",
                        help="enumerate units and exit; no GPU, no server")
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args(argv)

    args.years = [year.strip() for year in args.years.split(",") if year.strip()]
    args.groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    args.targets = list(PILOT_TARGETS) if args.pilot else []
    if args.pilot and args.pilot_only:
        wanted = set()
        for pair in args.pilot_only.split(","):
            video_id, _, shot = pair.strip().partition(":")
            if not video_id or not shot:
                parser.error(f"--pilot-only expects VIDEO_ID:SHOT, got {pair!r}")
            wanted.add((video_id, int(shot)))
        unknown = wanted - set(PILOT_TARGETS)
        if unknown:
            parser.error(f"--pilot-only has shots not in PILOT_TARGETS: {sorted(unknown)}")
        args.targets = [t for t in PILOT_TARGETS if t in wanted]
    if args.pilot:
        args.years = args.years or ["2025", "2026"]
        if "2025" not in args.years:
            args.years.append("2025")
    if args.output_dir is None:
        args.output_dir = ROOT / "artifacts" / "captions" / (
            "pilot-v2" if args.pilot else args.years[0]
        )
    if args.decoders < 1:
        args.decoders = max(2, (os.cpu_count() or 4) - 2)
    if args.concurrency < 1 or args.max_tokens < 1 or args.frame_width < 0:
        parser.error("concurrency and max tokens must be positive")
    if args.overlap_seconds >= args.window_seconds:
        parser.error("overlap must be shorter than the window")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("shard index must be in [0, shard count)")
    if args.shard_count > 1:
        args.output_dir /= f"shard-{args.shard_index:02d}-of-{args.shard_count:02d}"
    if not args.endpoint.rstrip("/").endswith("/v1"):
        parser.error("endpoint should be the OpenAI base URL, ending in /v1")
    args.endpoint = args.endpoint.rstrip("/")
    return args


def main(argv=None):
    args = parse_args(argv)
    units = select_units(args)
    summary = windows.summarize(units)
    log(f"{summary['shots']:,} shots -> {summary['units']:,} units "
        f"({summary['split_shots']:,} split, max {summary['max_windows']} windows, "
        f"{summary['total_frames']:,} frames)")
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return 0
    if args.merge_only:
        merge_parts(units, args)
        return 0

    result = asyncio.run(run(units, args))
    atomic_write_json(
        {**summary, **result, "frame_width": args.frame_width,
         "max_tokens": args.max_tokens, "concurrency": args.concurrency,
         "decoders": args.decoders, "kv_cache_dtype": args.kv_cache_dtype,
         "prompt_sha256": RECIPE_PROMPT_SHA256},
        Path(args.output_dir) / "run_summary.json",
    )
    if result["failed_videos"]:
        log(f"{len(result['failed_videos'])} videos failed; "
            f"see {args.output_dir}/run_summary.json. Not merging.")
        return 1
    merge_parts(units, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
