#!/usr/bin/env bash
# Frozen H-stage wrapper-generation queue. Targets are not called in this stage.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../results/supplementary_20261016_refresh
mkdir -p "$OUT"
for generator in gemma3-4b kimi-k3 mimo-v2.6-flash; do
  for seed in 42 43 44; do
    echo "START generator=$generator seed=$seed $(date -u +%FT%TZ)" >> "$OUT/generation_queue.log"
    python3 -u run_refresh_generate.py --generator "$generator" --seed "$seed" \
      > "$OUT/generate_${generator}_seed${seed}.log" 2>&1
    echo "DONE generator=$generator seed=$seed $(date -u +%FT%TZ)" >> "$OUT/generation_queue.log"
  done
done
python3 freeze_refresh_assets.py > "$OUT/freeze_assets.log" 2>&1
echo "REFRESH_ASSETS_FROZEN" >> "$OUT/generation_queue.log"
