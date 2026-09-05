#!/bin/bash
set -euo pipefail
cd /workspace/aic

curl -fsSL --retry 10 --retry-all-errors -C - \
  -o Videos_L23_a.zip https://aic-data.ledo.io.vn/Videos_L23_a.zip &
download_pid=$!

test -x .venv-caprl/bin/python || uv venv --python /venv/main/bin/python .venv-caprl
UV_TORCH_BACKEND=cu129 uv pip install \
  --python .venv-caprl/bin/python --no-cache \
  --index-strategy unsafe-best-match \
  'https://github.com/vllm-project/vllm/releases/download/v0.28.0/vllm-0.28.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  decord==0.6.0 pyarrow==25.0.1 qwen-vl-utils==0.0.14 transformers==5.15.1

wait "$download_pid"
unzip -tq Videos_L23_a.zip
mkdir -p data/2026/videos/L23
unzip -q -j Videos_L23_a.zip 'video/*.mp4' -d data/2026/videos/L23

.venv-caprl/bin/python -c \
  'import torch, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__)'
