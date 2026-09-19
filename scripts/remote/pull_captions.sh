#!/usr/bin/env bash
# Pull caption artifacts off the rented instance.
#
# `vast-capabilities | jq .instance.workspace_is_volume` is false on this instance, so
# nothing on it survives a recycle or destroy. Run this after every group, not just at
# the end -- the per-video parquet parts are the only copy of finished work.
#
#   bash scripts/remote/pull_captions.sh                       # the 2026 run
#   CAPRL_REMOTE_DIR=.../pilot-v2 bash scripts/remote/pull_captions.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
HOST="${CAPRL_SSH_HOST:-root@137.175.22.196}"
PORT="${CAPRL_SSH_PORT:-28707}"
KEY="${CAPRL_SSH_KEY:-$HOME/.ssh/id_ed25519_vast_caprl}"
REMOTE_DIR="${CAPRL_REMOTE_DIR:-/workspace/aic/artifacts/captions/2026}"
LOCAL_DIR="${CAPRL_LOCAL_DIR:-$REPO_ROOT/artifacts/captions/$(basename "$REMOTE_DIR")}"

mkdir -p "$LOCAL_DIR"
rsync -avP --partial -e "ssh -i $KEY -p $PORT" \
  "$HOST:$REMOTE_DIR/" "$LOCAL_DIR/"

echo "--- $LOCAL_DIR ---"
find "$LOCAL_DIR" -name '*.parquet' | wc -l | xargs echo "parquet files:"
du -sh "$LOCAL_DIR"
