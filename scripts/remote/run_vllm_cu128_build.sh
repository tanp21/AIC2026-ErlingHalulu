#!/bin/bash
set -euo pipefail
cd /workspace/aic

test -d vllm-src/.git || \
  git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm.git vllm-src

test -x .venv-vllm-cu128/bin/python || \
  uv venv --system-site-packages --python /venv/main/bin/python .venv-vllm-cu128

cd vllm-src
../.venv-vllm-cu128/bin/python use_existing_torch.py
uv pip install --python ../.venv-vllm-cu128/bin/python \
  -r requirements/build/cuda.txt
MAX_JOBS=16 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST=12.0 \
  uv pip install --python ../.venv-vllm-cu128/bin/python \
  --no-build-isolation -e .
uv pip install --python ../.venv-vllm-cu128/bin/python \
  decord==0.6.0 pyarrow==25.0.1 qwen-vl-utils==0.0.14 transformers==5.15.1

../.venv-vllm-cu128/bin/python -c \
  'import torch, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__)'
