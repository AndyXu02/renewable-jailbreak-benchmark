#!/usr/bin/env bash
# Supplementary 2026-10-16 job queue (local M1 Max + DeepSeek API).
# Ollama jobs wait for the gpt-oss re-scoring to finish (Ollama blocks model
# swaps under load); the MLX Gemma-IT crossover waits for the 1,024-token Gemma reruns.
set -u
cd "$(dirname "$0")"
LOG=../results/supplementary_20261016_multijudge
MLX_PY=../.venv_mlx312/bin/python
XOVER="--attacks A1_pair --scenarios S1_direct,S2_multiturn --max-tokens 1024 --generate"

ollama_lane() {
  while pgrep -f "name gptoss-rubric" >/dev/null; do sleep 60; done
  python3 pair_format_crossover.py --format single
  python3 pair_format_crossover.py --format multi
  for fmt in single multi; do
    for m in llama3.1-8b qwen3-8b-nothink deepseek-v4-flash; do
      python3 run_phase2_twostage.py --model "$m" $XOVER \
        --pair-file "../runs/pair_xover_${fmt}_20261016.jsonl" --dataset-tag "xover_${fmt}"
    done
  done
  echo "OLLAMA_LANE_DONE"
}

mlx_lane() {
  while pgrep -f "run_phase2_twostage.py --model gemma4" >/dev/null; do sleep 60; done
  while [ ! -f ../runs/pair_xover_multi_20261016.jsonl ] || [ "$(wc -l < ../runs/pair_xover_multi_20261016.jsonl)" -lt 20 ]; do sleep 60; done
  for fmt in single multi; do
    "$MLX_PY" run_phase2_twostage.py --model gemma4-12b-it $XOVER --backend-override mlx \
      --pair-file "../runs/pair_xover_${fmt}_20261016.jsonl" --dataset-tag "xover_${fmt}"
  done
  echo "MLX_LANE_DONE"
}

LANE="${1:-all}"
[ "$LANE" = "mlx" ] || ollama_lane >> "$LOG/queue_ollama.log" 2>&1 &
[ "$LANE" = "ollama" ] || mlx_lane >> "$LOG/queue_mlx.log" 2>&1 &
wait
