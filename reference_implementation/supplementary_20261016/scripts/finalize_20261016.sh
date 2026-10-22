#!/usr/bin/env bash
# Wait for the local follow-up generations to finish, then score and analyze
# every new response. Frozen v2formal files remain read-only.
set -euo pipefail
cd "$(dirname "$0")"

RUNS=../runs
OUT=../results/supplementary_20261016_multijudge

wait_for_lines() {
  local file="$1" expected="$2"
  while true; do
    local n=0
    [ ! -f "$file" ] || n=$(wc -l < "$file" | tr -d ' ')
    echo "$(date '+%F %T') wait $file $n/$expected"
    [ "$n" -ge "$expected" ] && break
    sleep 120
  done
}

# 2 Gemma targets x (3 attacks x 4 scenarios x 20 behaviors).
wait_for_lines "$RUNS/phase2_v3cap1024_gemma4-12b-it_responses.jsonl" 240
wait_for_lines "$RUNS/phase2_v3cap1024_gemma4-12b-obliterated_responses.jsonl" 240

# Four aligned targets x two optimization formats x two scenarios x 20 behaviors.
for fmt in single multi; do
  for model in deepseek-v4-flash llama3.1-8b qwen3-8b-nothink gemma4-12b-it; do
    wait_for_lines "$RUNS/phase2_xover_${fmt}_${model}_responses.jsonl" 40
  done
done

python3 multijudge_build_new_items.py | tee "$OUT/build_new_items.log"
python3 - <<'PY'
import json
from pathlib import Path

checks = {
    **{f"runs/phase2_v3cap1024_{model}_responses.jsonl": 240
       for model in ("gemma4-12b-it", "gemma4-12b-obliterated")},
    **{f"runs/phase2_xover_{fmt}_{model}_responses.jsonl": 40
       for fmt in ("single", "multi")
       for model in ("deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink", "gemma4-12b-it")},
}
for rel, expected in checks.items():
    rows = [json.loads(line) for line in (Path("..") / rel).read_text().splitlines() if line.strip()]
    keys = [(r["sample_id"], r["attack"], r["scenario"]) for r in rows]
    errors = [r for r in rows if r.get("error") or not r.get("response", "").strip()]
    if len(rows) != expected or len(set(keys)) != expected or errors:
        raise SystemExit(f"Invalid follow-up matrix {rel}: n={len(rows)}, unique={len(set(keys))}, errors={len(errors)}")
    if "v3cap1024" in rel and any(int(r.get("generation_max_tokens", 0)) != 1024 for r in rows):
        raise SystemExit(f"Unexpected token cap in {rel}")
print(f"Validated {len(checks)} follow-up matrices")
PY

python3 multijudge_run.py --name glm52 --model glm-5.2 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --items items_new.jsonl --experiments gemma1024,xover --concurrency 6 --max-tokens 2048 --max-retries 6 \
  > "$OUT/glm52_new_run.log" 2>&1
python3 multijudge_run.py --name glm52 --model glm-5.2 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --items items_new.jsonl --experiments gemma1024,xover --retry-failed --concurrency 4 --max-tokens 2048 --max-retries 8 \
  >> "$OUT/glm52_new_run.log" 2>&1
python3 multijudge_run.py --name glm52 --model glm-5.2 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --items items_new.jsonl --experiments gemma1024,xover --retry-failed --concurrency 4 --max-tokens 2048 --max-retries 8 \
  >> "$OUT/glm52_new_run.log" 2>&1

python3 multijudge_run.py --name minimax-m3 --model minimax-m3 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --items items_new.jsonl --experiments gemma1024,xover --concurrency 6 --max-tokens 2048 --max-retries 6 \
  > "$OUT/minimax-m3_new_run.log" 2>&1
python3 multijudge_run.py --name minimax-m3 --model minimax-m3 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --items items_new.jsonl --experiments gemma1024,xover --retry-failed --concurrency 4 --max-tokens 2048 --max-retries 8 \
  >> "$OUT/minimax-m3_new_run.log" 2>&1
python3 multijudge_run.py --name minimax-m3 --model minimax-m3 \
  --api-base-env GLM_API_BASE --api-key-env GLM_API_KEY \
  --items items_new.jsonl --experiments gemma1024,xover --retry-failed --concurrency 4 --max-tokens 2048 --max-retries 8 \
  >> "$OUT/minimax-m3_new_run.log" 2>&1

python3 multijudge_run.py --name gptoss-rubric --model gpt-oss:20b --ollama \
  --items items_new.jsonl --experiments gemma1024,xover --concurrency 1 \
  > "$OUT/gptoss-rubric_new_run.log" 2>&1
python3 multijudge_run.py --name gptoss-rubric --model gpt-oss:20b --ollama \
  --items items_new.jsonl --experiments gemma1024,xover --retry-failed --concurrency 1 \
  >> "$OUT/gptoss-rubric_new_run.log" 2>&1

python3 - <<'PY'
import json
from pathlib import Path

out = Path("../results/supplementary_20261016_multijudge")
items = [json.loads(line) for line in (out / "items_new.jsonl").read_text().splitlines() if line.strip()]
expected = {item["item_id"] for item in items if item["experiment"] in {"gemma1024", "xover"}}
for judge, filename in (("gptoss-rubric", "gptoss-rubric"), ("glm52", "glm52"), ("minimax-m3", "minimax-m3")):
    scores = {}
    for line in (out / "judges" / f"{filename}.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["item_id"] in expected:
            scores[row["item_id"]] = row["score"]
    missing = expected - scores.keys()
    failed = [item_id for item_id in expected if scores.get(item_id) is None]
    print(f"{judge}: expected={len(expected)} covered={len(expected) - len(missing)} failed={len(failed)}")
    if missing or failed:
        raise SystemExit(f"Incomplete judge coverage for {judge}")
PY

MJ_N_PERM=20000 python3 multijudge_analysis.py > "$OUT/analysis_run.log" 2>&1
python3 supplementary_20261016_analysis.py > "$OUT/supp_analysis_run.log" 2>&1
python3 ../../Thesis/build/complete_revision_20261016.py > "$OUT/paper_finalize.log" 2>&1
THESIS_MULTIJUDGE_ANALYSIS="$(cd "$OUT/analysis" && pwd)" \
  python3 ../../Thesis/figures/make_figures_20261016.py > "$OUT/figures_run.log" 2>&1
python3 ../../Thesis/build/word_count.py research_paper_draft_20261016.md > "$OUT/paper_word_count.log"
pandoc ../../Thesis/research_paper_draft_20261016.md --from markdown --to latex --standalone \
  -o /tmp/research_paper_draft_20261016_repair.tex
python3 write_supplementary_manifest.py
echo "FOLLOWUP_ANALYSIS_COMPLETE"
