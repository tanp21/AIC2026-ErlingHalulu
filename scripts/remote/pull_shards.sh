#!/usr/bin/env bash
# Pull shard .npy files + meta.csv + info.json from vast.ai to local data/2026/embeddings.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DEST="$(readlink -f "$REPO_ROOT/data")/2026/embeddings"
mkdir -p "$DEST"

rsync -avP --info=progress2 "vast:/workspace/embeddings/" "$DEST/"

echo "Shards in $DEST:"
ls -lh "$DEST"/emb_*.npy
