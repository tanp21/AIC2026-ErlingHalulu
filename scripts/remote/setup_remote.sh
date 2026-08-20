#!/usr/bin/env bash
# Install dependencies on the vast.ai instance (CUDA 12.8 driver -> cu124 wheels).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

python3 -m pip install --upgrade pip

# PyTorch for CUDA 12.4 (compatible with 12.8 driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install open-clip-torch timm gdown pillow numpy tqdm

# unzip (fast extraction) + rsync (pull shards back) + rclone (GDrive download)
apt-get update -qq && apt-get install -y -qq unzip rsync rclone || true

python3 -c "import torch, open_clip, gdown; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'open_clip ok')"
