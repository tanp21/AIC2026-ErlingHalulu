#!/bin/bash
set -euo pipefail
cd /workspace/aic

test -x .venv-caprl-cu128/bin/python || \
  uv venv --python /venv/main/bin/python .venv-caprl-cu128
UV_TORCH_BACKEND=cu128 uv pip install \
  --python .venv-caprl-cu128/bin/python --no-cache \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  vllm==0.11.2 decord==0.6.0 pyarrow==25.0.1 qwen-vl-utils==0.0.14

.venv-caprl-cu128/bin/python -c \
  'import torch, vllm; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), vllm.__version__)'
