#!/usr/bin/env python3
"""Evaluate frozen refresh assets; one Ollama target per process."""

import argparse
import datetime as dt
import json
from pathlib import Path

from config import RUNS_DIR
from multijudge_run import ROOT, load_env
from run_inference import get_backend

GENERATORS = ("gemma3-4b", "kimi-k3", "mimo-v2.6-flash")
SEEDS = (42, 43, 44)
TARGETS = ("deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", required=True, choices=TARGETS)
    args = p.parse_args()
    load_env(ROOT / ".env")
    frozen = ROOT / "results/supplementary_20261016_refresh/assets_frozen.jsonl"
    if not frozen.exists():
        raise SystemExit("Freeze and hash all proposals before target evaluation")
    assets = [json.loads(line) for line in frozen.read_text().splitlines() if line.strip()]
    if len(assets) != 450:
        raise SystemExit(f"Expected 450 asset rows, got {len(assets)}")
    quarantined = [r["sample_id"] for r in assets if r["strategy"] == "quarantined_model_version"]
    empty = [r["sample_id"] for r in assets if not r["prompt"].strip()]
    if quarantined or empty:
        raise SystemExit(f"Refuse incomplete target run: quarantined={len(quarantined)}, empty={len(empty)}")
    keys = [(r["generator"], r["seed_requested"], r["sample_id"]) for r in assets]
    if len(set(keys)) != 450:
        raise SystemExit("Duplicate generator/seed/behavior assets")

    output = RUNS_DIR / f"refresh_20261016_{args.target}_responses.jsonl"
    previous = [json.loads(line) for line in output.read_text().splitlines() if line.strip()] if output.exists() else []
    done = {r["item_id"] for r in previous if not r.get("error") and r.get("response")}
    backend = get_backend(args.target)
    pending = [r for r in assets if f"{args.target}|{r['generator']}|{r['seed_requested']}|{r['sample_id']}" not in done]
    print(f"target={args.target} assets={len(assets)} pending={len(pending)}", flush=True)
    for n, asset in enumerate(pending, 1):
        result = {}
        for attempt in range(1, 4):
            result = backend.generate(
                [{"role": "user", "content": asset["prompt"]}],
                temperature=0.0, max_tokens=1024, seed=42,
            )
            if result.get("response") and not result.get("error"):
                break
        row = {
            "item_id": f"{args.target}|{asset['generator']}|{asset['seed_requested']}|{asset['sample_id']}",
            "experiment": "refresh", "kind": "harmful", "model": args.target,
            "behavior_id": asset["sample_id"], "category": asset["category"],
            "behavior": asset["behavior"], "generator": asset["generator"],
            "generator_model_requested": asset["generator_model_requested"],
            "generator_served_model": asset["served_model"],
            "generator_seed": asset["seed_requested"],
            "generator_template_valid": asset["normalized_valid"],
            "generator_raw_first_pass_valid": asset["first_pass_valid"],
            "generator_format_normalization": asset["normalization"],
            "generator_fallback": asset["fallback_reason"],
            "scenario": "S1_direct", "defense": "D0_none", "attack": "A7_refreshed",
            "attack_prompt": asset["prompt"], "response": result.get("response") or "",
            "response_length": len(result.get("response") or ""),
            "served_model": result.get("model"),
            "finish_reason": result.get("finish_reason"),
            "usage": result.get("usage"),
            "finish_metadata_available": result.get("finish_reason") is not None,
            "error": result.get("error"), "generation_max_tokens": 1024,
            "generation_attempts": attempt, "temperature": 0.0, "seed": 42,
            "timestamp": dt.datetime.now().isoformat(),
        }
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if n % 10 == 0 or n == len(pending) or row["error"]:
            print(f"  [{n}/{len(pending)}] error={row['error']} chars={row['response_length']}", flush=True)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
