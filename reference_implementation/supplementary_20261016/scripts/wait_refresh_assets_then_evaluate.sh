#!/usr/bin/env bash
# Continue the pre-authorized H experiment only after the 450 assets are frozen.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../results/supplementary_20261016_refresh
while [ ! -f "$OUT/assets_manifest.json" ]; do
  sleep 60
done
bash ./run_refresh_evaluation_20261016.sh >> "$OUT/evaluation_queue.log" 2>&1
