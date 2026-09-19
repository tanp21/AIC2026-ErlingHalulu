#!/usr/bin/env bash
# Full 2026 run, one zip at a time, on a box with less disk than the corpus.
#
# The 2026 videos are 88.4 GB across 15 zips (L25_a alone is 12.8 GB) and the instance
# has ~92 GB free, most of which the model and the caption parts also want. So each zip
# is downloaded, extracted, captioned, and deleted before the next one starts -- the
# same download/extract/delete discipline scripts/ocr/run_corpus_remote.py uses.
#
# The next zip downloads in the background while the current group is being captioned,
# so the GPU is not waiting on the network between groups.
#
#   bash scripts/remote/caprl_stage_groups.sh                  # every 2026 group
#   CAPRL_WIDTH=640 bash scripts/remote/caprl_stage_groups.sh L23 L27
set -euo pipefail

ROOT="${CAPRL_ROOT:-/workspace/aic}"
PY="${CAPRL_PYTHON:-/venv/main/bin/python}"
ENDPOINT="${CAPRL_ENDPOINT:-http://127.0.0.1:18000/v1}"
BASE_URL="${CAPRL_BASE_URL:-https://aic-data.ledo.io.vn}"
YEAR="${CAPRL_YEAR:-2026}"
# 0 keeps the source resolution: the baseline recipe ("1024-default-8frames") is native.
WIDTH="${CAPRL_WIDTH:-0}"
MAX_TOKENS="${CAPRL_MAX_TOKENS:-2048}"
# Must match the --kv-cache-dtype vllm serve actually runs with; it is recorded per row.
KV="${CAPRL_KV:-fp8}"
CONCURRENCY="${CAPRL_CONCURRENCY:-32}"
# The default (nproc - 2 = 254 on the 256-vCPU box) leaves the CPU pool thrashing
# against vLLM's own scheduling: measured 0.63 units/s at 254 decoders vs 1.36 at 32.
DECODERS="${CAPRL_DECODERS:-32}"
SHARD_COUNT="${CAPRL_SHARD_COUNT:-1}"
SHARD_INDEX="${CAPRL_SHARD_INDEX:-0}"
OUT="${CAPRL_OUT:-$ROOT/artifacts/captions/$YEAR}"
WORK="$ROOT/staging"

# Zip inventory, confirmed by HEAD against the host. A group's videos may be split
# across several zips; all of them must land before the group is captioned.
declare -A ZIPS=(
  [L21]="Videos_L21_a.zip"
  [L22]="Videos_L22_a.zip"
  [L23]="Videos_L23_a.zip"
  [L24]="Videos_L24_a.zip"
  [L25]="Videos_L25_a.zip Videos_L25_b.zip"
  [L26]="Videos_L26_a.zip Videos_L26_b.zip Videos_L26_c.zip Videos_L26_d.zip Videos_L26_e.zip"
  [L27]="Videos_L27_a.zip"
  [L28]="Videos_L28_a.zip"
  [L29]="Videos_L29_a.zip"
  [L30]="Videos_L30_a.zip"
)
# Smallest first: the earliest captions come back soonest, so a bad config is caught
# after minutes rather than after the 12.8 GB group.
DEFAULT_ORDER=(L23 L27 L21 L30 L22 L24 L29 L28 L25 L26)
# `RUN_GROUPS`, not `GROUPS`: GROUPS is a bash special variable (the caller's group
# IDs) with integer semantics, so `GROUPS=(L23 L27)` silently becomes `GROUPS=0`.
RUN_GROUPS=("$@")
if [ "${#RUN_GROUPS[@]}" -eq 0 ] || [ -z "${RUN_GROUPS[0]}" ]; then
  RUN_GROUPS=("${DEFAULT_ORDER[@]}")
fi

command -v aria2c >/dev/null || { apt-get update -qq && apt-get install -y -qq aria2 unzip; }
mkdir -p "$WORK" "$OUT"
cd "$ROOT"

fetch_group() {
  local group="$1"
  for zip in ${ZIPS[$group]}; do
    [ -s "$WORK/$zip" ] && [ ! -e "$WORK/$zip.aria2" ] && continue
    aria2c -d "$WORK" -o "$zip" -x16 -s16 -k4M -c --file-allocation=none \
      --max-tries=10 --retry-wait=10 --console-log-level=warn "$BASE_URL/$zip"
  done
}

log() { echo "[$(date +%H:%M:%S)] $*"; }

next_index=0
for index in "${!RUN_GROUPS[@]}"; do
  group="${RUN_GROUPS[$index]}"
  [ -n "${ZIPS[$group]:-}" ] || { log "no zip mapping for $group, skipping"; continue; }

  if [ "$index" -ge "$next_index" ]; then
    log "downloading $group"
    fetch_group "$group"
  fi
  wait || true          # the prefetch started for this group last iteration

  log "extracting $group"
  mkdir -p "data/$YEAR/videos/$group"
  for zip in ${ZIPS[$group]}; do
    unzip -qq -o -j "$WORK/$zip" '*.mp4' -d "data/$YEAR/videos/$group"
    rm -f "$WORK/$zip"
  done
  log "$group: $(find "data/$YEAR/videos/$group" -name '*.mp4' | wc -l) videos, $(df -h / | awk 'NR==2{print $4}') free"

  # Prefetch the next group while this one occupies the GPU.
  if [ "$((index + 1))" -lt "${#RUN_GROUPS[@]}" ]; then
    next_index=$((index + 1))
    fetch_group "${RUN_GROUPS[$next_index]}" &
  fi

  log "captioning $group"
  PYTHONPATH="$ROOT" "$PY" -u -m scripts.captioning.caption_api \
    --years "$YEAR" --groups "$group" --endpoint "$ENDPOINT" \
    --output-dir "$OUT" --frame-width "$WIDTH" --max-tokens "$MAX_TOKENS" \
    --kv-cache-dtype "$KV" --concurrency "$CONCURRENCY" --decoders "$DECODERS" \
    --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX" \
    2>&1 | tee -a "$OUT/run.log" || log "$group returned non-zero; see $OUT/run_summary.json"

  # Parts are committed per video, so the pixels are no longer needed.
  rm -rf "data/$YEAR/videos/$group"
  log "$group done, $(df -h / | awk 'NR==2{print $4}') free"
done

wait || true
log "all groups processed; merging"
PYTHONPATH="$ROOT" "$PY" -m scripts.captioning.caption_api \
  --years "$YEAR" --output-dir "$OUT" --merge-only \
  --shard-count "$SHARD_COUNT" --shard-index "$SHARD_INDEX"
