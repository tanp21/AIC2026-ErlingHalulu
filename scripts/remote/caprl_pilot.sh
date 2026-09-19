#!/usr/bin/env bash
# Caption the pilot shots (manual-review targets + hard cases from the three 2026
# rounds) with the locked-in production recipe: caption_api.py's defaults already are
# that recipe (RECIPE_PROMPT, 8 frames, native resolution, max_tokens 2048, and the
# tuned sampling params 0.8 / 0.9 / 40 that beat the checkpoint's own
# generation_config.json across seeds), so this script takes no config flags.
#
#   bash scripts/remote/caprl_pilot.sh                    # every pilot shot
#   CAPRL_PILOT_ONLY="L26_V448:39" bash scripts/remote/caprl_pilot.sh   # just one
set -euo pipefail

ROOT="${CAPRL_ROOT:-/workspace/aic}"
PY="${CAPRL_PYTHON:-/venv/main/bin/python}"
ENDPOINT="${CAPRL_ENDPOINT:-http://127.0.0.1:18000/v1}"
OUT="$ROOT/artifacts/captions/pilot-v3"
CONCURRENCY="${CAPRL_CONCURRENCY:-16}"
DECODERS="${CAPRL_DECODERS:-32}"

cd "$ROOT"
mkdir -p "$OUT"

wait_for_server() {
  for _ in $(seq 1 120); do
    curl -sf --max-time 5 "${ENDPOINT}/models" >/dev/null && return 0
    sleep 10
  done
  echo "vllm did not come up at $ENDPOINT" >&2
  exit 1
}

restart_vllm() {
  # `supervisorctl restart vllm` stops the supervised wrapper but can leave the
  # VLLM::EngineCore child alive still holding ~19 GB, and the next start then dies with
  # "Free memory on device cuda:0 ... is less than desired GPU memory utilization".
  supervisorctl stop vllm || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    echo "killing orphaned GPU process $pid"
    kill -9 "$pid" 2>/dev/null || true
  done
  sleep 8
  supervisorctl start vllm
}

# The baseline run loaded the checkpoint as dtype=torch.float16; vLLM defaults to bf16.
# Set CAPRL_MATCH_DTYPE=1 to force fp16 (one restart) if bf16 turns out not to be enough
# -- try the cheaper option (no restart) first.
if [ "${CAPRL_MATCH_DTYPE:-0}" = "1" ] && ! grep -q -- '--dtype float16' /etc/environment 2>/dev/null; then
  python3 - <<'PY'
from pathlib import Path
path = Path("/etc/environment")
lines = []
for line in path.read_text().splitlines():
    if line.startswith("VLLM_ARGS=") and "--dtype" not in line:
        line = line[:-1] + ' --dtype float16"'
    lines.append(line)
path.write_text("\n".join(lines) + "\n")
PY
  restart_vllm
fi
wait_for_server

PYTHONPATH="$ROOT" "$PY" -u -m scripts.captioning.caption_api \
  --pilot --pilot-only "${CAPRL_PILOT_ONLY:-}" --endpoint "$ENDPOINT" \
  --output-dir "$OUT/baseline-recipe" --concurrency "$CONCURRENCY" \
  --decoders "$DECODERS" \
  2>&1 | tee -a "$OUT/run.log"

PYTHONPATH="$ROOT" "$PY" -m scripts.captioning.pilot_report \
  "$OUT"/baseline-recipe/shot_captions.parquet \
  -o "$OUT/report.md" --split-dir "$OUT/by-shot" --summary "$OUT/summary.md"
