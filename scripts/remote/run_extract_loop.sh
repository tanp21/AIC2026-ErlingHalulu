#!/usr/bin/env bash
# Auto-restart loop: resume từ trạng thái Drive khi crash.
# Usage: bash scripts/remote/run_extract_loop.sh [--workers 12] [--groups K01,K02,...]
set -u
cd "$(dirname "$0")/../.."

while true; do
  python3 -m scripts.extraction.extract_keyframes_remote "$@"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] CLEAN EXIT (all groups done)"
    break
  fi
  echo "[$(date '+%H:%M:%S')] crash rc=$rc, backoff 30s rồi resume"
  sleep 30
done
