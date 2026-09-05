# CapRL-Video-4B: Kaggle T4x2 benchmark and deployment estimate

Snapshot: **2026-09-01 UTC**

## Result

The runnable Kaggle configuration is:

```text
model                  internlm/CapRL-Video-4B
GPU                    2 x Tesla T4, 14.56 GiB each
precision              FP16 (T4 has no native BF16)
tensor parallel        2
model context          32,768 tokens
video sampling         2 FPS, 4 minimum, 32 maximum frames
visual token budget    24,576
generation cap         1,024 tokens
batch size             1
```

The 32-frame cap samples the entire shot uniformly; it does not take only the
first 32 frames. This is necessary for bounded memory/context on the T4x2 and
also makes very long shots have a finite runtime. The caption remains one
self-contained English caption per shot. The separate keyframe mapping makes
all keyframes inside that shot resolve to the same caption.

The fast benchmark used 12 real Batch 1 shots, two from each of six duration
buckets. It covered 1.56 to 375.68 seconds and completed successfully. Model
load took 256.79 seconds. The weighted inference estimate for all 96,624 shots
is **733.06 hours on one T4x2 worker**. Selecting either of the two observations
in every bucket gives a sensitivity range of **609.12-856.99 hours**; this is
not a statistical confidence interval, because two observations per bucket is
a small sample.

| Shot duration | Population | Samples | Mean seconds/shot | Mean output tokens |
|---|---:|---:|---:|---:|
| <=2 s | 29,940 | 2 | 20.02 | 693.0 |
| 2-5 s | 47,391 | 2 | 21.32 | 641.0 |
| 5-15 s | 16,841 | 2 | 52.48 | 861.0 |
| 15-30 s | 1,653 | 2 | 58.47 | 947.5 |
| 30-120 s | 522 | 2 | 59.90 | 958.0 |
| >120 s | 277 | 2 | 63.57 | 1,024.0 |

The full 60-shot attempt first showed that a 65,536-token KV cache at 90%
memory was too tight: batch size 1 reached 25 shots, then OOM. Reducing the
engine to a 32,768-token context and 80% memory passed that old failure point.
The 12-shot run above additionally bounded the visual input so that all length
buckets fit the 32K context.

## Kaggle completion time

Kaggle documents a 12-hour maximum GPU session and a weekly GPU quota normally
around 30 hours. A T4x2 session consumes one quota hour per wall-clock hour,
not two. See the official [Notebook specifications](https://www.kaggle.com/docs/notebooks),
[GPU quota guide](https://www.kaggle.com/docs/efficient-gpu-usage), and the
[T4x2 announcement](https://www.kaggle.com/product-feedback/361104).

Allowing a new 256.79-second cold start for every 12-hour session:

| Kaggle workers | Inference per worker | Sessions per worker | Total time per worker | Continuous wall time | 30 h/week calendar time |
|---:|---:|---:|---:|---:|---:|
| 1 x T4x2 | 733.06 h | 62 | 737.48 h | 30.7 days | about 25 weeks |
| 3 x T4x2 | 242.26-245.43 h | 21 | up to 246.93 h | about 10.3 days | about 9 weeks |

The three deterministic video shards contain 32,209 / 32,206 / 32,209 shots
and 291 videos each. Their measured-weight estimates are 245.43 / 242.26 /
245.37 inference hours, so the slowest shard determines the 3-worker result.

Kaggle's Terms say one person may not control more than one active account.
The 3-worker scenario is therefore appropriate only for three legitimate
collaborators, not three accounts operated by one person. See the official
[Terms of Use](https://www.kaggle.com/terms).

## Vast.ai snapshot and estimate

The current offers below are verified, rentable, on-demand machines with at
least 0.98 reliability and enough space for a 150 GB allocation. The hourly
column includes 150 GB of storage. Initial staging includes 93 GB of inbound
traffic plus the theoretical transfer-time compute cost. Vast is a live
marketplace, so this is a snapshot, not a quote. Pricing combines compute,
storage and bandwidth and is billed by actual usage; see the official
[Search offers API](https://docs.vast.ai/api-reference/search/search-offers)
and [pricing guide](https://docs.vast.ai/guides/instances/pricing).

| GPU | VRAM | Base $/h | $/h incl. 150 GB | Initial staging | Capacity/risk |
|---|---:|---:|---:|---:|---|
| RTX 4090 | 24 GB | $0.320 | $0.362 | $0.37 | Likely runnable only with the bounded 32-frame config; tight capacity |
| RTX 5090 | 32 GB | $0.400 | $0.469 | $0.63 | Best price/performance candidate |
| A100 SXM4 | 40 GB | $0.667 | $0.694 | $0.15 | More capacity, but worse current value than 5090 |
| RTX PRO 6000 Max-Q | 96 GB | $0.960 | $1.029 | $0.70 | Safest capacity candidate |
| 2 x Tesla T4 | 30 GB total | $0.296 | $0.338 | $0.25 | Measured baseline; not economical for the full run |

No verified rentable RTX A4500 offer matching the filters was available in the
snapshot. Its 20 GB VRAM is below the measured T4x2 aggregate and is too close
to the limit to recommend for a full batch before a canary succeeds.

There is no same-model measurement on the stronger Vast GPUs yet. A throughput
claim based only on FLOPS or Vast DLPerf would be misleading for video decode,
vision prefill, tensor-parallel communication and long text generation. Using
a deliberately broad **2.5x-5x** speed range for a single RTX 5090 gives:

```text
RTX 5090 runtime       146.61-293.22 hours
RTX 5090 run cost      $68.82-$137.64
RTX 5090 total         $69.45-$138.27 including initial staging
```

At a central 3.5x assumption, the RTX 5090 estimate is about **209.45 hours
(8.73 days) and $98.95 total**. Treat this as a planning range. Running the
same 12-shot notebook once on the rented 5090 is the required calibration
before committing the full batch.

The RTX PRO 6000 Max-Q is estimated at 3x-6x because its main advantage is
capacity rather than low hourly price: approximately **122.18-244.35 hours**
and **$126.47-$252.24 total**.

## Vast template to select

Choose the recommended **PyTorch** template with version tag **[Automatic]**,
launch mode **Jupyter + SSH**, and **150 GB disk**. Vast's RTX 5-series guide
requires CUDA 12.8 and PyTorch 2.7 or newer for RTX 5090; `[Automatic]` selects
a compatible image. Do not replace it with an older CUDA image. See Vast's
[template guide](https://docs.vast.ai/guides/instances/choosing/templates) and
[RTX 5-series requirements](https://docs.vast.ai/rtx-5-series).

## Artifacts and commands

- Kaggle benchmark notebook: <https://www.kaggle.com/code/tommr211/caprl-video-4b-fast-benchmark>
- Longest-shot canary: <https://www.kaggle.com/code/tommr211/caprl-video-4b-longest-shot-canary>
- Local Kaggle runner: `notebooks/kaggle_caprl_video_benchmark.py`
- Production runner: `scripts/captioning/caption_shots.py`

Three-way production sharding is available with:

```bash
.venv-captioning/bin/python -m scripts.captioning.caption_shots --shard-count 3 --shard-index 0 --tensor-parallel-size 2
.venv-captioning/bin/python -m scripts.captioning.caption_shots --shard-count 3 --shard-index 1 --tensor-parallel-size 2
.venv-captioning/bin/python -m scripts.captioning.caption_shots --shard-count 3 --shard-index 2 --tensor-parallel-size 2
```

Each shard writes to its own `shard-NN-of-03` output directory and keeps every
video wholly within one shard.
