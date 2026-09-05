# CapRL-Video-4B: local benchmark and Vast.ai estimate

Snapshot: **2026-09-01 00:52 UTC**

## Outcome

The strict default configuration does **not** start on the local RTX 3060
12 GB. vLLM resolved the model context length to 262,144 tokens and requested
92% of GPU memory (11.04 GiB), while only 10.98 GiB was free after the GPU
application had been stopped. Engine initialization therefore failed before
weights were loaded or any caption was generated.

Per the benchmark rule, the run was stopped at that point. Context length,
video frame limits, BF16 precision and quantization were not changed. As a
result there is no measured shots/second value and no defensible full-run ETA
for 96,624 shots on this machine.

The stopped application was restarted and confirmed healthy. Ingress and
Elasticsearch remained healthy.

## What was validated

- Scope: 2026, 873 source MP4 files, 96,624 shots.
- Video input: each shot is read directly from its source MP4 by passing
  `video_start` and `video_end`; no temporary clips are rendered.
- CapRL preprocessing smoke test: a real 2.733-second shot decoded successfully
  as four frames with the model's Qwen video processor defaults.
- Keyframe mapping: 289,875 permanent PNG keyframes mapped to all 96,624 shots;
  8 `*.tmp.png` files were ignored. 96,623 shots have three keyframes and one
  shot has six.
- Output contracts: shot captions and keyframe-to-shot mapping use separate
  Parquet schemas. Every keyframe in one shot therefore resolves to the same
  shot caption.
- Local failed-run artifact: `artifacts/captions/2026/benchmark.json` records
  77.62 seconds elapsed and 1,420 MiB peak observed GPU use. Failure happened
  during engine startup, so the peak is not model residency.

## Benchmark configuration

```text
model                 internlm/CapRL-Video-4B
engine                vLLM 0.28.0
precision             BF16
quantization          none
model context         default (resolved to 262,144)
video sampling        model defaults: 2 FPS, min 4, max 768 frames
generation cap        1,024 tokens
batch size            4
planned sample        60 shots, stratified over 6 duration buckets
caption language      English, very detailed, no timestamps
```

## Vast.ai price snapshot

The table uses currently rentable, verified, on-demand single-GPU offers with
reliability at least 0.98. Vast.ai is a live marketplace, so these rows are a
snapshot rather than a quote. Prices were obtained from the official
[Search offers API](https://docs.vast.ai/api-reference/search/search-offers);
Vast.ai documents that total cost combines compute, storage and bandwidth and
is billed by actual usage in its [pricing guide](https://docs.vast.ai/guides/instances/pricing).

Dataset staging assumes 93 GB downloaded (82.98 GB of videos plus 9.67 GB for
the local model snapshot, rounded up) and a 120 GB disk allocation. `Estimated $/h`
equals base compute plus `120 * storage $/GB/month / 720`. `Initial staging $`
adds inbound bandwidth and the theoretical download-time compute charge; real
staging can be slower because of the source host and setup overhead.

| GPU offer | VRAM | Base $/h | Estimated $/h with 120 GB | Inbound $/GB | Network lower bound | Initial staging $ | Strict-default use |
|---|---:|---:|---:|---:|---:|---:|---|
| RTX 3090 | 24 GB | 0.1493 | 0.1693 | 0.001302 | 0.101 h | 0.138 | Not recommended without a successful strict benchmark |
| RTX 5090 | 32 GB | 0.4000 | 0.4556 | 0.005208 | 0.338 h | 0.638 | Not recommended without a successful strict benchmark |
| RTX A6000 | 46 GB | 0.4000 | 0.4667 | 0.026042 | 0.028 h | 2.435 | Unverified for the 262k default context |
| RTX 6000 Ada | 48 GB | 0.6267 | 0.6600 | 0.004036 | 0.016 h | 0.386 | Unverified for the 262k default context |
| A100 SXM4 | 80 GB | 1.0213 | 1.0613 | 0.013333 | 0.102 h | 1.348 | Recommended first strict benchmark |
| RTX PRO 6000 WS | 96 GB | 1.0000 | 1.0444 | 0.002604 | 0.275 h | 0.529 | Recommended first strict benchmark |

The 80/96 GB recommendations are capacity-based candidates, not claims of a
successful model load. A short benchmark on the rented host is still required
before committing to the full run.

## Full-run cost formula

Because the local benchmark produced no throughput, a single full-run dollar
estimate would invent the missing runtime. For caption runtime `T` hours:

```text
A100 80 GB total       ~= 1.348 + 1.0613 * T  USD
RTX PRO 6000 96 GB     ~= 0.529 + 1.0444 * T  USD
```

These scenario totals show the cost sensitivity without treating them as an
ETA:

| Caption runtime scenario | A100 80 GB | RTX PRO 6000 96 GB |
|---:|---:|---:|
| 24 h | $26.82 | $25.60 |
| 48 h | $52.29 | $50.66 |
| 72 h | $77.76 | $75.73 |

The next safe step is to rent one 80/96 GB offer, run the same 60-shot benchmark
unchanged, then substitute its weighted ETA into the formula above. No Vast.ai
instance was rented during this implementation.
