#!/usr/bin/env bash
# Embed the organisers' Batch 2 keyframes for the N (traffic-camera) videos, one zip at a
# time, with PE-Core and SigLIP2 NaFlex.
#
#     ROOT=~/aic PY=python3 ./scripts/extraction/run_btc_keyframes.sh
#
# Run this on a machine with a GPU and internet -- a rented box, not the production host.
# Nothing is uploaded from a developer machine: each 5 GB keyframe archive and the
# map-keyframes zip are pulled straight from the organisers' own mirror, used, and deleted.
#
# Resumable by construction. A range writes zips/<tag>.done only after both models have
# committed it, so a rerun skips finished ranges; within a range the ingest skips images it
# has already converted and each embedder's ShardWriter skips keyframes already committed.
# Interrupt it, rerun it, and it picks up where it stopped.
#
# Disk: one zip (5 GB) plus its converted PNGs (~18 GB) are the peak, because each range's
# keyframes are deleted once embedded. The whole N set is ~48 GB of JPEG; held at once as
# PNG it would be ~180 GB.
set -euo pipefail

ROOT="${ROOT:-$HOME/aic}"
PY="${PY:-python3}"
# 8 matches scripts/embedding/embed_batch2_stream.py's default: each DataLoader worker is a
# forked process holding its own copy of the image, so scaling to nproc on a 16-vCPU box
# costs RAM without feeding the GPU any faster.
WORKERS="${WORKERS:-8}"
BASE="${BASE:-https://aic-data.ledo.io.vn}"
RANGES="${RANGES:-001-010 011-020 021-030 031-040 041-050 051-060 061-070 071-080 081-090 091-100}"

cd "$ROOT"

# The model weights are the slowest thing to fetch, and an unauthenticated download from a
# datacentre IP is rate-limited: measured on a rented 5090, ~3 MB/s without this and ~16 MB/s
# with it (10 GB of PE-Core + NaFlex, so the difference is most of an hour). Current
# huggingface_hub uses Xet, which is why this rather than the older
# HF_HUB_ENABLE_HF_TRANSFER, which it now warns is deprecated.
export HF_XET_HIGH_PERFORMANCE=1

OUT="data/2026/embeddings_n2026"
KEYFRAMES="data/2026/keyframes_tmp"
POINTS="data/2026/keyframes_btc/points.parquet"
MAPS_DIR="data/2026/map-keyframes"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# Download to $2 from $1 unless the local copy already matches the remote size. Plain
# ``curl -C -`` cannot be rerun blindly: against a file that is already complete the server
# answers 416, and under ``set -e`` that kills a run whose only sin was being resumed.
fetch() {
    local url="$1" dest="$2" want have attempt
    want="$(curl -fsIL "$url" | awk 'BEGIN{IGNORECASE=1} /^content-length:/{v=$2} END{gsub(/\r/,"",v); print v}')"
    for attempt in 1 2; do
        have="$(stat -c %s "$dest" 2>/dev/null || echo 0)"
        if [ "$attempt" = 1 ] && [ -n "$want" ] && [ "$want" = "$have" ]; then
            log "have $(basename "$dest") ($have bytes)"
        else
            log "fetching $(basename "$dest") (${have:-0}/$want bytes)"
            # aria2 when it is there: measured on a rented 5090, one curl stream stalled
            # part-way through a 5 GB archive where 16 aria2 connections took a minute.
            if command -v aria2c >/dev/null 2>&1; then
                aria2c -x16 -s16 -c -q --console-log-level=warn \
                       -d "$(dirname "$dest")" -o "$(basename "$dest")" "$url"
            else
                curl -fL -C - --retry 3 --retry-delay 5 -o "$dest" "$url"
            fi
        fi

        # Test the archive either way, not only after a download: a bad copy can arrive at
        # exactly the right byte count, so matching the remote size proves nothing. Seen for
        # real -- a partial fetched by one client and resumed by another left a full-size
        # archive with no readable central directory. Removing it and looping means the run
        # heals itself instead of failing the same way on every retry.
        case "$dest" in
            *.zip)
                unzip -tq "$dest" >/dev/null && return 0
                echo "corrupt archive, refetching from scratch: $dest" >&2
                rm -f "$dest" "$dest.aria2"
                ;;
            *) return 0 ;;
        esac
    done
    echo "still corrupt after a refetch: $dest" >&2
    return 1
}

# data/ is not in the repo (it is a gitignored mount), so on a fresh box the tree under it
# does not exist yet -- and unzip will not create a missing parent for -d.
mkdir -p zips data/2026

# --- Phase A: the organisers' map-keyframes, which maps each image's ordinal to its
# frame_idx and its real time. Small, and needed by every range.
if [ ! -d "$MAPS_DIR" ] || [ -z "$(ls -A "$MAPS_DIR" 2>/dev/null)" ]; then
    fetch "$BASE/aic26-b2-map-keyframes.zip" zips/aic26-b2-map-keyframes.zip
    unzip -oq zips/aic26-b2-map-keyframes.zip -d data/2026
fi
log "map-keyframes: $(find "$MAPS_DIR" -name 'N*.csv' | wc -l) N video(s)"

# --- Phase B: per range, download -> convert -> embed with both models -> delete.
for range in $RANGES; do
    start="${range%%-*}"
    end="${range##*-}"
    tag="N${start}-N${end}"
    zip="zips/Keyframes_${tag}.zip"
    groups="$(seq -f 'N%03g' "$((10#$start))" "$((10#$end))" | paste -sd, -)"

    if [ -f "zips/${tag}.done" ]; then
        log "skip $tag (already embedded)"
        continue
    fi

    fetch "$BASE/Keyframes_${tag}.zip" "$zip"

    log "$tag: converting to <frame_idx>.png"
    "$PY" -m scripts.extraction.ingest_btc_keyframes \
        --zips-dir zips --map-keyframes-dir "$MAPS_DIR" \
        --groups "$groups" --keyframes-dir "$KEYFRAMES" --points "$POINTS" \
        --workers "$WORKERS"

    log "$tag: PE-Core"
    "$PY" -m scripts.embedding.embed_batch2_pe_core \
        --keyframes-dir "$KEYFRAMES" --out-dir "$OUT/pe-core" \
        --groups "$groups" --workers "$WORKERS"

    log "$tag: NaFlex"
    "$PY" -m scripts.embedding.embed_batch2_naflex \
        --keyframes-dir "$KEYFRAMES" --out-dir "$OUT/naflex" \
        --groups "$groups" --workers "$WORKERS"

    # Only now is this range's work safe to throw away.
    rm -rf "$KEYFRAMES" "$zip"
    touch "zips/${tag}.done"
    log "$tag: done"
done

# --- Phase C: the organisers' real keyframe times, joined in once every range is embedded.
# Must be last: ShardWriter rewrites meta.parquet with its own four columns at every
# checkpoint, so a pts_ms column added earlier would be dropped by the next shard rollover.
log "attaching pts_ms"
for model in pe-core naflex; do
    "$PY" -m scripts.embedding.attach_pts_ms \
        --embedding-dir "$OUT/$model" --points "$POINTS"
done

log "DONE -> $OUT"
