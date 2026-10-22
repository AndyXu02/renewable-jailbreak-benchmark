#!/usr/bin/env bash
# Run only after the generator queue has frozen all 450 H assets.
set -euo pipefail
cd "$(dirname "$0")"

OUT=../results/supplementary_20261016_refresh
ROOT=../runs

if [ ! -f "$OUT/assets_manifest.json" ]; then
  echo "Missing frozen assets; refusing to run targets."
  exit 2
fi

run_target() {
  local target="$1" log="$2"
  for attempt in 1 2 3; do
    python3 run_refresh_targets.py --target "$target" >> "$log" 2>&1
    if python3 - "$ROOT/refresh_20261016_${target}_responses.jsonl" <<'PY'
import json, sys
from pathlib import Path
p=Path(sys.argv[1])
if not p.exists(): raise SystemExit(1)
latest={}
for line in p.read_text().splitlines():
    if line.strip():
        r=json.loads(line); latest[r["item_id"]]=r
raise SystemExit(0 if len(latest)==450 and all(not r.get("error") and r.get("response") for r in latest.values()) else 1)
PY
    then return 0; fi
    echo "target=$target had errors after attempt=$attempt; resume and retry" >> "$log"
  done
  return 1
}

run_target deepseek-v4-flash "$OUT/target_deepseek.log"
run_target llama3.1-8b "$OUT/target_llama.log"
run_target qwen3-8b-nothink "$OUT/target_qwen.log"
python3 audit_frozen_refresh_assets.py > "$OUT/assets_audit.log" 2>&1
python3 build_refresh_items.py > "$OUT/build_items.log" 2>&1

python3 multijudge_run.py --name glm52 --model glm-5.2 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --output-root "$OUT" --items items.jsonl --experiments refresh \
  --concurrency 4 --max-tokens 4096 --max-retries 5 > "$OUT/judge_glm52.log" 2>&1
python3 multijudge_run.py --name glm52 --model glm-5.2 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --output-root "$OUT" --items items.jsonl --experiments refresh \
  --retry-failed --concurrency 3 --max-tokens 4096 --max-retries 5 >> "$OUT/judge_glm52.log" 2>&1
for pass in 2 3; do
  python3 multijudge_run.py --name glm52 --model glm-5.2 \
    --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
    --output-root "$OUT" --items items.jsonl --experiments refresh \
    --retry-failed --concurrency 2 --max-tokens 4096 --max-retries 5 \
    >> "$OUT/judge_glm52.log" 2>&1
done

python3 multijudge_run.py --name minimax-m3 --model minimax-m3 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --allowed-served-models minimax-m3 MiniMax-M3 minimax/minimax-m3 mm/MiniMax-M3 cb/minimax-m3 \
  --output-root "$OUT" --items items.jsonl --experiments refresh \
  --concurrency 4 --max-tokens 4096 --max-retries 5 > "$OUT/judge_minimax.log" 2>&1
python3 multijudge_run.py --name minimax-m3 --model minimax-m3 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --allowed-served-models minimax-m3 MiniMax-M3 minimax/minimax-m3 mm/MiniMax-M3 cb/minimax-m3 \
  --output-root "$OUT" --items items.jsonl --experiments refresh \
  --retry-failed --concurrency 3 --max-tokens 4096 --max-retries 5 >> "$OUT/judge_minimax.log" 2>&1
for pass in 2 3; do
  python3 multijudge_run.py --name minimax-m3 --model minimax-m3 \
    --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
    --allowed-served-models minimax-m3 MiniMax-M3 minimax/minimax-m3 mm/MiniMax-M3 cb/minimax-m3 \
    --output-root "$OUT" --items items.jsonl --experiments refresh \
    --retry-failed --concurrency 2 --max-tokens 4096 --max-retries 5 \
    >> "$OUT/judge_minimax.log" 2>&1
done

python3 multijudge_run.py --name gptoss-rubric --model gpt-oss:20b --ollama \
  --output-root "$OUT" --items items.jsonl --experiments refresh \
  --concurrency 1 --max-tokens 4096 > "$OUT/judge_gptoss.log" 2>&1
python3 multijudge_run.py --name gptoss-rubric --model gpt-oss:20b --ollama \
  --output-root "$OUT" --items items.jsonl --experiments refresh \
  --retry-failed --concurrency 1 --max-tokens 4096 >> "$OUT/judge_gptoss.log" 2>&1
for pass in 2 3; do
  python3 multijudge_run.py --name gptoss-rubric --model gpt-oss:20b --ollama \
    --output-root "$OUT" --items items.jsonl --experiments refresh \
    --retry-failed --concurrency 1 --max-tokens 4096 \
    >> "$OUT/judge_gptoss.log" 2>&1
done

python3 - <<'PY'
import json
from pathlib import Path

out = Path("../results/supplementary_20261016_refresh")
items = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines() if line.strip()]
expected = {row["item_id"] for row in items}
if len(expected) != 1350:
    raise SystemExit(f"Expected 1,350 unique target cells, found {len(expected)}")
for judge, name in (("gptoss-rubric","gptoss-rubric"),("glm52","glm52"),("minimax-m3","minimax-m3")):
    rows = [json.loads(line) for line in (out / "judges" / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    scores = {row["item_id"]: row["score"] for row in rows if row["item_id"] in expected}
    missing = expected - scores.keys()
    failed = sum(value is None for value in scores.values())
    print(f"{judge}: expected={len(expected)} covered={len(scores)} failed={failed}")
    if missing or failed:
        raise SystemExit(f"Incomplete strict scores for {judge}")
PY

python3 analyze_refresh.py > "$OUT/analyze_refresh.log" 2>&1
python3 ../../Thesis/build/integrate_refresh_results_v2.py > "$OUT/paper_integration.log" 2>&1
THESIS_MULTIJUDGE_ANALYSIS="$(cd ../results/supplementary_20261016_repair/analysis && pwd)" \
  THESIS_FIGURE_TAG=20261016_repaired \
  python3 ../../Thesis/figures/make_figures_20261016.py > "$OUT/repaired_figures.log" 2>&1
python3 ../../Thesis/build/word_count.py research_paper_draft_20261016.md > "$OUT/paper_word_count.log"
pandoc ../../Thesis/research_paper_draft_20261016.md --from markdown --to latex --standalone \
  -o /tmp/research_paper_draft_20261016_refresh_check.tex
WORD_COUNT=$(tr -d '[:space:]' < "$OUT/paper_word_count.log")
if [ "$WORD_COUNT" -gt 8000 ]; then
  echo "Dated paper exceeds 8,000-word cap: $WORD_COUNT"
  exit 3
fi
python3 write_refresh_manifest.py > "$OUT/write_refresh_manifest.log" 2>&1
printf '%s\n' "- H evaluation, strict 3-judge scoring and analysis completed; Section 4.10 / 5.7 integrated into dated paper. See \`benchmark_project/results/supplementary_20261016_refresh/refresh_manifest.json\`." \
  >> ../../Thesis/10.16\ execution_log.md
echo "REFRESH_EVALUATION_COMPLETE"
