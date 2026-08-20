#!/usr/bin/env bash
# Auto-restart loop: resumes from committed state on GPU/process crash.
set -u
cd /workspace
while true; do
  source /venv/main/bin/activate
  python -m scripts.embedding.embed_all
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "[$(date '+%H:%M:%S')] CLEAN EXIT (all groups done)"
    break
  fi
  echo "[$(date '+%H:%M:%S')] crashed rc=$rc, backoff 20s then resume"
  sleep 20
done
