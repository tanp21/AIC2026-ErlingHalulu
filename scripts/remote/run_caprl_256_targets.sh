#!/bin/bash
set -euo pipefail
cd /workspace/aic
exec > >(tee -a caprl-256-targets.log) 2>&1

uv pip install --python /venv/main/bin/python \
  transformers==4.57.1 decord==0.6.0 pyarrow==25.0.1 qwen-vl-utils==0.0.14

download_video() {
  local archive="$1" year="$2" group="$3" video="$4"
  test -s "data/$year/videos/$group/$video.mp4" && return
  aria2c -c -x16 -s16 -k4M --file-allocation=none \
    -o "$archive" "https://aic-data.ledo.io.vn/$archive"
  mkdir -p "data/$year/videos/$group"
  unzip -q -j "$archive" "*$video.mp4" -d "data/$year/videos/$group"
  rm "$archive"
}

download_video Videos_L27_a.zip 2026 L27 L27_V010
if ! test -s data/2026/videos/L30/L30_V078.mp4 || \
   ! test -s data/2026/videos/L30/L30_V072.mp4; then
  aria2c -c -x16 -s16 -k4M --file-allocation=none \
    -o Videos_L30_a.zip https://aic-data.ledo.io.vn/Videos_L30_a.zip
  mkdir -p data/2026/videos/L30
  unzip -q -j Videos_L30_a.zip '*L30_V078.mp4' '*L30_V072.mp4' \
    -d data/2026/videos/L30
  rm Videos_L30_a.zip
fi
download_video Videos_K17.zip 2025 K17 K17_V003
download_video Videos_K19.zip 2025 K19 K19_V016
download_video Videos_L23_a.zip 2026 L23 L23_V013

# The Hub's Xet client can stall on this shard on some Vast hosts. Resume the
# same blob with parallel HTTP ranges so the completed shard remains cached.
model_root=/workspace/.hf_home/hub/models--internlm--CapRL-Video-4B
model_blob=0dd742aaab6a723395f0cfd5d8f9f6c3efb70b137cfb23f385b6e23cc9996555
if ! test -s "$model_root/blobs/$model_blob"; then
  aria2c -c -x4 -s4 -k4M --file-allocation=none \
    --connect-timeout=30 --timeout=120 --max-tries=20 --retry-wait=5 \
    --allow-overwrite=true --auto-file-renaming=false \
    -d "$model_root/blobs" -o "$model_blob.incomplete" \
    'https://huggingface.co/internlm/CapRL-Video-4B/resolve/main/model-00001-of-00002.safetensors'
  mv "$model_root/blobs/$model_blob.incomplete" "$model_root/blobs/$model_blob"
fi
snapshot_dir=$(find "$model_root/snapshots" -mindepth 1 -maxdepth 1 -type d | head -1)
ln -sfn "../../blobs/$model_blob" "$snapshot_dir/model-00001-of-00002.safetensors"

: "${CAPRL_MAX_FRAMES:=8}"
: "${CAPRL_MAX_NEW_TOKENS:=1024}"
: "${CAPRL_OUTPUT_LABEL:=1024-default-8frames}"
export CAPRL_MAX_FRAMES CAPRL_MAX_NEW_TOKENS CAPRL_OUTPUT_LABEL
HF_HUB_DISABLE_XET=1 PYTHONPATH=/workspace/aic \
  /venv/main/bin/python -u scripts/remote/run_caprl_256_targets.py
