# CapRL shot captioning

Dense per-shot captions from
[`internlm/CapRL-Video-4B`](https://huggingface.co/internlm/CapRL-Video-4B), for the
retrieval failures no CLIP-style embedding can reach: lecture slides, exam papers,
counting ("the *fourth* orange segment"), and two events inside one long shot.

There are two runners. **`scripts.captioning.caption_api` is the one to use**; it talks
to a `vllm serve` process over the OpenAI API and keeps many requests in flight.
`scripts.captioning.caption_shots` is the original in-process runner, kept because it
owns the shared library (shot enumeration, sharding, atomic parts) and the earlier
parquet files under `artifacts/captions/` use its schema.

## What a caption unit is

One unit is a **time window inside one shot**. Shots at or under 20 s are a single unit,
so 98.5 % of the 2026 corpus keeps one row per shot. Longer shots are covered by
equal-length windows overlapping by 2 s (`scripts/captioning/windows.py`), because
TransNetV2 only cuts where the picture changes and a locked-off lecture camera produces
"shots" of 400-660 s. For 2026: **96,624 shots → 103,086 units**, 1,428 shots split, at
most 65 windows on the longest.

Every window gets **8 frames, flat** (`windows.FIXED_FRAME_COUNT`), the frame count of the
quality baseline. They are sampled uniformly from the **inset middle** of the window
(`frames.EDGE_INSET_FRACTION`, 12 % off each end), because a TransNetV2 boundary can
still carry a sliver of the neighbouring scene and the model then describes the cut
instead of the shot. Frames are read straight from the source MP4 with an `-ss` seek and
MJPEG to a pipe (`scripts/captioning/frames.py`); no temporary clips are rendered.

## The caption contract

`caption_api.RECIPE_PROMPT` is the exact 305-character prompt of the **"1024-default-8frames"
quality baseline** (copied from `notebooks/kaggle_caprl_video_benchmark.py`, which
produced that baseline). Its sha256 is pinned to the baseline parquet's `b81f69d3…` by
`tests/captioning/test_caption_api.py`, so the recipe cannot silently drift. A row
carries one self-contained `caption`; `raw_output` keeps the model's text verbatim. The
extended schema's `summary` and `onscreen_text` columns are retained for compatibility
and stay empty under this recipe.

Sampling was tuned on the 137-unit pilot, scored against 25 fidelity checks on the
shots whose on-screen text is known. The checkpoint's own `generation_config.json`
(0.7 / 0.8 / 20) ranged 17–22/25 across seeds; **temperature 0.8, top_p 0.9, top_k 40**
read 21/25 on every seed, kept the water-gauge numbers (7/7), produced no repetition
loops and quoted the most text. `temperature 1.0 / top_p 1.0` with top_k off was the
worst (12–20/25; it invented an "increments of 8" pattern on the gauge). `max_tokens`
is **2048**, not the baseline's 1024: a third of the pilot captions hit 1024 and lost
their tail mid-sentence.

No post-processing is applied to the caption. The baseline prompt transcribes readable
on-screen text, so channel overlays (`HTV`, `VTV`, `THVL`, `Tuổi Trẻ TV`, `60 Giây`) do
appear in the output; if later benchmarking shows they hurt retrieval, a deterministic
filter over `raw_output` can be added then without re-running the model.

## Serving the model

On a Vast.ai vLLM instance the server is the supervisor service `vllm`, listening on
`127.0.0.1:18000`. Point it at CapRL by setting **both** `VLLM_MODEL` and `MODEL_NAME`
in `/etc/environment`, then restarting:

```bash
VLLM_MODEL="internlm/CapRL-Video-4B"
MODEL_NAME="internlm/CapRL-Video-4B"
VLLM_ARGS="--served-model-name caprl --host 127.0.0.1 --port 18000 \
  --download-dir /workspace/models --max-model-len 16384 --max-num-seqs 64 \
  --gpu-memory-utilization 0.92 --async-scheduling --kv-cache-dtype fp8"
```

`--max-model-len 16384` is comfortable now that a unit is at most 8 native-720p frames:
~4.3 k visual+prompt tokens plus a 2048-token answer. (The older 24-frame policy needed
the full 16 k; it was dropped when the baseline recipe was locked.) `fp8` KV roughly
doubles KV capacity and showed no visible cost in the pilot-v2 A/B — the run records
whatever the server actually used, via `--kv-cache-dtype` on the client.

Two traps on this image:

- `supervisorctl stop vllm` can leave the `VLLM::EngineCore` child alive holding ~19 GB
  of VRAM; the next start then fails with *"Free memory on device cuda:0 … is less than
  desired GPU memory utilization"*. Check `nvidia-smi --query-compute-apps=pid,used_memory`
  and kill the orphan.
- `/etc/vllm-args.conf` is split on whitespace with no shell quoting, so a JSON-valued
  flag written there arrives as `{video:1}` and fails to parse.

## Pilot before the full run

```bash
bash scripts/remote/caprl_pilot.sh                    # every pilot shot
CAPRL_PILOT_ONLY="K17_V003:102" bash scripts/remote/caprl_pilot.sh
```

One pass, no config flags: `caption_api.py`'s defaults already are the locked recipe
(baseline prompt, 8 frames, native resolution, checkpoint sampling params), so the script
only waits for the server and runs the client. Output lands in
`artifacts/captions/pilot-v3/baseline-recipe/`, and `pilot_report` renders it beside the
`1024-default-8frames` column for the seven shots both cover.

The smoke test for a fresh server or a code change is `K17_V003:102`, where the baseline
reads "Mộc Châu" correctly. The baseline checkpoint was loaded as fp16 by transformers;
vLLM defaults to bf16. Try bf16 first; if the shot regresses, `CAPRL_MATCH_DTYPE=1`
appends `--dtype float16` to `VLLM_ARGS` and restarts vLLM once.

## The full run

```bash
bash scripts/remote/caprl_stage_groups.sh            # every 2026 group
CAPRL_WIDTH=640 bash scripts/remote/caprl_stage_groups.sh L23 L27
```

The defaults are the production recipe: native width (`CAPRL_WIDTH=0`), 2048 output
tokens, `CAPRL_KV` set to fp8 to match the server, and `CAPRL_DECODERS=32`. The
decoder count matters: the client's process-pool default (nproc−2 = 254 on the 256-vCPU
box) competes with vLLM's own CPU work and measured 0.63 units/s, versus **1.27 units/s
at 32 decoders** — about 23 h for all 103,086 units on one RTX 5090.

The 2026 videos are 88.4 GB across 15 zips (L25_a alone is 12.8 GB), more than fits
alongside the model on a 100 GB instance, so each zip is downloaded, extracted,
captioned and deleted before the next; the next group downloads while the current one
occupies the GPU. Groups run smallest-first so the first captions come back in minutes.

Each video commits atomically to `parts/<group>/<video_id>.parquet`, and a rerun skips
any part whose `(video_id, shot_index, window_index)` sequence already matches — so a
crash costs one video. `--merge-only` writes `shot_captions.parquet` once every part
validates, asserting the row count.

Pull results back often — `workspace_is_volume` is false on these instances, so nothing
survives a recycle:

```bash
bash scripts/remote/pull_captions.sh
```

## Sharding across instances

Whole videos are assigned deterministically and balanced by shot count
(`caption_shots.select_shard`), so two workers never touch the same video:

```bash
CAPRL_SHARD_COUNT=2 CAPRL_SHARD_INDEX=0 bash scripts/remote/caprl_stage_groups.sh
CAPRL_SHARD_COUNT=2 CAPRL_SHARD_INDEX=1 bash scripts/remote/caprl_stage_groups.sh
```

Each writes to its own `shard-NN-of-02` directory and merges its own
`shot_captions.parquet`; concatenate the two shard parquets for the final artifact.

## Checking the plan without a GPU

```bash
python -m scripts.captioning.caption_api --years 2026 --dry-run
# 96,624 shots -> 103,086 units (1,428 split, max 65 windows, 824,688 frames)
```

No caption data is added to the production search index by this offline step.
