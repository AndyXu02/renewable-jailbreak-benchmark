#!/usr/bin/env python3
"""Collect the 2026-10-16 supplementary generations into items_new.jsonl.

  gemma1024 : both Gemma 4 variants re-generated at a 1,024-token cap (full Experiment 1 grid)
  xover     : PAIR format-crossover prompts (single- vs multi-turn optimized) evaluated on the
              four aligned targets in the Single-turn and Multi-turn scenarios
Judge with multijudge_run.py --items items_new.jsonl --experiments gemma1024,xover.
"""

import json
import os

from bootstrap_ci import RUNS
from multijudge_build_items import OUT, item

SOURCES = (
    [("gemma1024", "none", f"phase2_v3cap1024_{m}_responses.jsonl") for m in ("gemma4-12b-it", "gemma4-12b-obliterated")]
    + [("xover", fmt, f"phase2_xover_{fmt}_{m}_responses.jsonl")
       for fmt in ("single", "multi") for m in ("deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink", "gemma4-12b-it")]
)


def main():
    items, missing = [], []
    for exp, fmt, fname in SOURCES:
        path = os.path.join(RUNS, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            attack = r["attack"] if exp == "gemma1024" else f"A1_pair_opt_{fmt}"
            it = item(exp, "harmful", r, r["model"], attack, r["sample_id"], None, optimization_format=fmt)
            it["item_id"] = f"{exp}|{fmt}|" + it["item_id"].split("|", 1)[1]
            items.append(it)
    ids = [i["item_id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    with open(os.path.join(OUT, "items_new.jsonl"), "w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    counts = {}
    for i in items:
        k = f"{i['experiment']}/{i['model']}/{i['optimization_format']}"
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps({"total": len(items), "counts": counts, "missing": missing}, indent=1))


if __name__ == "__main__":
    main()
