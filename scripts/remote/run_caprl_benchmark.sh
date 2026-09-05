#!/bin/bash
set -euo pipefail
source "${CAPRL_VENV:-/workspace/aic/.venv-caprl-cu128}/bin/activate"
cd /workspace/aic
mkdir -p artifacts/captions/2026/rtx5090-batch2
python -m scripts.captioning.caption_shots \
  --groups L23 \
  --skip-keyframe-mapping \
  --benchmark-shots 10 \
  --batch-size 2 \
  --max-tokens 384 \
  --gpu-memory-utilization .90 \
  --output-dir artifacts/captions/2026/rtx5090-batch2 \
  2>&1 | tee artifacts/captions/2026/rtx5090-batch2/run.log
