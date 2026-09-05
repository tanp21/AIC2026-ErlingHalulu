# CapRL shot captioning

`scripts.captioning.caption_shots` captions the 2026 shot clips with
[`internlm/CapRL-Video-4B`](https://huggingface.co/internlm/CapRL-Video-4B).
It reads each shot directly from its source MP4 with `video_start`/`video_end`;
it does not render temporary clips.

## Setup

Keep vLLM out of the production server environment:

```bash
uv venv .venv-captioning --python 3.12
uv pip install --python .venv-captioning/bin/python -r requirements-captioning.txt
```

## Validate the keyframe mapping

```bash
.venv-captioning/bin/python -m scripts.captioning.caption_shots --map-only
```

This writes the derived artifacts below `artifacts/captions/2026/`:

- `keyframe_shots.parquet`: `(year, group, video_id, frame_idx, shot_index)`;
- `mapping_summary.json`: coverage and keyframe-count validation.

Temporary `*.tmp.png` files are ignored. Every other PNG must fall inside one
shot, and every shot must have at least one keyframe.

## Benchmark before the full run

The default benchmark is stratified across six shot-duration ranges:

```bash
.venv-captioning/bin/python -m scripts.captioning.caption_shots --benchmark-shots 60
```

The runnable defaults use 2 FPS, 4 minimum frames, 32 uniformly sampled maximum
frames, a 24,576-token visual budget, FP16, a 32,768-token model context and a
1,024-token output cap. The bounded video input was validated on Kaggle T4x2;
it keeps long shots inside the context instead of describing only their start.

`benchmark.json` records cold start, peak observed GPU memory, duration-bucket
timings and the weighted ETA for all 96,624 shots. `benchmark_shots.parquet`
contains the sampled captions. Bracketed timestamps are removed because shot
start/end times already live in structured columns.

## Resume a full run

```bash
.venv-captioning/bin/python -m scripts.captioning.caption_shots
```

Each video is committed atomically to `parts/<group>/<video_id>.parquet`.
Rerunning skips complete videos; once every part is present, the command writes
`shot_captions.parquet`. No caption data is added to the production search index
by this offline step.

## Split across three workers

Whole videos are assigned deterministically and balanced by shot count:

```bash
.venv-captioning/bin/python -m scripts.captioning.caption_shots \
  --shard-count 3 --shard-index 0 --tensor-parallel-size 2
```

Run indices 0, 1 and 2 on separate workers. Each writes to a separate
`shard-NN-of-03` directory, so no video or shot is duplicated between workers.
On a single-GPU Vast instance, leave `--tensor-parallel-size` at its default 1.

Video-only remote workers can reuse the mapping built locally instead of copying
the keyframe PNGs:

```bash
.venv-captioning/bin/python -m scripts.captioning.caption_shots --skip-keyframe-mapping
```
