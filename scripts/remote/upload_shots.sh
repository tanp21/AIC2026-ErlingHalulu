#!/usr/bin/env bash
# Package TransNetV2 shots (data/2025/shots) and upload to Drive once.
# Colab then downloads this single small tar.gz (input for keyframe extraction).
set -euo pipefail

RC="$(command -v rclone)" || { echo "rclone is required" >&2; exit 1; }
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="$(readlink -f "$REPO_ROOT/data")"
OUT="/tmp/opencode/shots.tar.gz"
DEST="gdrive:AIC_HCMC/2026/2025_Dataset/"

mkdir -p "$(dirname "$OUT")"
tar -czf "$OUT" -C "$DATA_ROOT/2025" shots
"$RC" copy "$OUT" "$DEST" --progress
echo "Uploaded: $DEST/shots.tar.gz"
