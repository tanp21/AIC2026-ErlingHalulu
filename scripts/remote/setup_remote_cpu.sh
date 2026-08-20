#!/usr/bin/env bash
# Install deps on a cheap CPU vast.ai instance: ffmpeg + aria2 + rclone + tqdm.
# Run once, then: bash scripts/remote/run_extract_loop.sh --workers 12
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
apt-get install -y -qq --fix-missing \
  ffmpeg aria2 unzip curl ca-certificates python3 python3-pip || true

# tqdm for progress bars
python3 -m pip install --quiet --upgrade tqdm || true

# rclone 1.75.0 static (force: apt may ship an old DEV build)
curl -sSL https://downloads.rclone.org/v1.75.0/rclone-v1.75.0-linux-amd64.zip -o /tmp/rclone.zip
unzip -o -j /tmp/rclone.zip 'rclone-*-linux-amd64/rclone' -d /usr/local/bin
chmod +x /usr/local/bin/rclone

echo "--- versions ---"
python3 --version
ffmpeg -version | head -1
aria2c --version | head -1
rclone version | head -1
python3 -c "import tqdm; print('tqdm', tqdm.__version__)"
