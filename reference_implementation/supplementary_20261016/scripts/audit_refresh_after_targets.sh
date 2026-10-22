#!/usr/bin/env bash
# Independent post-generation check; it never changes any target prompt.
set -euo pipefail
cd "$(dirname "$0")"
OUT=../results/supplementary_20261016_refresh
while true; do
  python3 - <<'PY' && break
import json
from pathlib import Path
root=Path("../runs")
targets=("deepseek-v4-flash","llama3.1-8b","qwen3-8b-nothink")
for target in targets:
    p=root/f"refresh_20261016_{target}_responses.jsonl"
    if not p.exists(): raise SystemExit(1)
    latest={}
    for line in p.read_text().splitlines():
        if line.strip():
            row=json.loads(line);latest[row["item_id"]]=row
    if len(latest)!=450 or any(r.get("error") or not r.get("response") for r in latest.values()):
        raise SystemExit(1)
PY
  sleep 120
done
python3 audit_frozen_refresh_assets.py > "$OUT/assets_audit.log" 2>&1
echo "ASSET_PROMPT_AUDIT_COMPLETE $(date -u +%FT%TZ)" >> "$OUT/evaluation_queue.log"
