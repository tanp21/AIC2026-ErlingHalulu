#!/usr/bin/env bash
# Launches N run_corpus_remote.py worker processes in the background, each on
# its own disjoint set of video zips (passed as worker0_zips=... env-style
# args), logging to logs/workerN.log and writing to out/workerN.json.
#
# Usage: ./launch_remote_workers.sh <out_dir> <rec_batch_size> <zips_worker0> [<zips_worker1> ...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="$1"; shift
BATCH="$1"; shift

mkdir -p "$OUT_DIR/logs"
i=0
for zips in "$@"; do
  echo "worker$i -> $zips"
  nohup /venv/main/bin/python "$ROOT/scripts/ocr/run_corpus_remote.py" \
    --manifest "$ROOT/data_manifest_2026.json" \
    --zips "$zips" \
    --work "/workspace/scratch$i" \
    --out "$OUT_DIR/worker$i.json" \
    --rec-batch-size "$BATCH" \
    > "$OUT_DIR/logs/worker$i.log" 2>&1 &
  echo $! > "$OUT_DIR/worker$i.pid"
  i=$((i + 1))
done
echo "launched $i workers"
