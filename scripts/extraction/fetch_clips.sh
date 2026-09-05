#!/usr/bin/env bash
# Bulk-export shot clips from a running server, for video-captioning experiments.
#
#   scripts/extraction/fetch_clips.sh L21_V006                 # every shot of one video
#   scripts/extraction/fetch_clips.sh L21_V006 --min-sec 2 --max-sec 12
#   scripts/extraction/fetch_clips.sh --all --limit 50 --out /data/caption-clips
#
# Credentials come from AIC_CRED ("user:password"), or are read from
# aic-initial-passwords.txt for AIC_USER. Base URL defaults to the local stack.
set -euo pipefail

BASE=${AIC_BASE:-http://127.0.0.1:8000}
OUT=${AIC_CLIP_OUT:-./clips}
MIN_SEC=0
MAX_SEC=1000000
LIMIT=0
VIDEOS=()
ALL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all) ALL=1; shift ;;
    --min-sec) MIN_SEC=$2; shift 2 ;;
    --max-sec) MAX_SEC=$2; shift 2 ;;
    --limit) LIMIT=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --base) BASE=$2; shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) VIDEOS+=("$1"); shift ;;
  esac
done

if [[ -z "${AIC_CRED:-}" ]]; then
  user=${AIC_USER:-$(id -un)}
  password=$(awk -F'\t' -v u="$user" '$1==u{print $2}' aic-initial-passwords.txt 2>/dev/null || true)
  [[ -n "$password" ]] || { echo "set AIC_CRED='user:password' (or AIC_USER)" >&2; exit 2; }
  AIC_CRED="$user:$password"
fi

api() { curl -fsS -u "$AIC_CRED" "$BASE$1"; }

if [[ $ALL -eq 1 ]]; then
  mapfile -t VIDEOS < <(api /videos | python3 -c '
import json,sys
print("\n".join(r["video_id"] for r in json.load(sys.stdin) if r["has_shots"]))')
fi
[[ ${#VIDEOS[@]} -gt 0 ]] || { echo "name at least one video, or pass --all" >&2; exit 2; }

mkdir -p "$OUT"
total=0
for video in "${VIDEOS[@]}"; do
  # index<TAB>seconds for the shots that pass the duration filter
  while IFS=$'\t' read -r index seconds; do
    [[ $LIMIT -gt 0 && $total -ge $LIMIT ]] && break 2
    name=$(curl -fsS -u "$AIC_CRED" -OJ -w '%{filename_effective}' \
      --output-dir "$OUT" "$BASE/clip/$video/$index")
    total=$((total + 1))
    printf '%4d  %s  %.2fs\n' "$total" "$(basename "$name")" "$seconds"
  done < <(api "/shots/$video" | python3 -c "
import json,sys
for s in json.load(sys.stdin)['shots']:
    length = s['end'] - s['start']
    if $MIN_SEC <= length <= $MAX_SEC:
        print(f\"{s['index']}\t{length}\")")
done
echo "$total clips -> $OUT"
