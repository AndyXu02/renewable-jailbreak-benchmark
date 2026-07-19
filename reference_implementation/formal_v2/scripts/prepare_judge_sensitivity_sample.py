#!/usr/bin/env python3
"""Create a deterministic, balanced Phase-2 sample for a second LLM judge.

For each model, the default design selects one record from every
category × scenario cell (5 × 4 = 20 records). Attack families are rotated
across cells and models, yielding an approximately balanced attack mix without
using any human labels or looking at the primary judge score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


# Fixed offsets make selections stable as additional model files become
# available. They also preserve the already-judged DeepSeek (0) and Llama (1)
# strata when the sample grows from 20/40 rows to the final 100.
MODEL_ATTACK_OFFSETS = {
    "deepseek-v4-flash": 0,
    "llama3.1-8b": 1,
    "qwen3-8b-nothink": 2,
    "gemma4-12b-it": 0,
    "gemma4-12b-obliterated": 1,
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def stable_rank(record: dict, seed: int) -> str:
    key = "|".join(
        str(record.get(k, ""))
        for k in ("sample_id", "model", "category", "attack", "scenario")
    )
    return hashlib.sha256(f"{seed}|{key}".encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = [row for path in args.inputs for row in load_jsonl(path)]
    models = sorted({str(row.get("model")) for row in rows})
    categories = sorted({str(row.get("category")) for row in rows})
    scenarios = sorted({str(row.get("scenario")) for row in rows})
    attacks = sorted({str(row.get("attack")) for row in rows})
    if (len(categories), len(scenarios), len(attacks)) != (5, 4, 3):
        raise RuntimeError(
            "Expected the formal 5-category × 4-scenario × 3-attack design; "
            f"found {len(categories)} × {len(scenarios)} × {len(attacks)}"
        )

    cells: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        cells[(str(row.get("model")), str(row.get("category")),
               str(row.get("scenario")), str(row.get("attack")))].append(row)

    selected = []
    for model in models:
        model_offset = MODEL_ATTACK_OFFSETS.get(
            model, int(hashlib.sha256(model.encode()).hexdigest(), 16) % len(attacks)
        )
        for category_i, category in enumerate(categories):
            for scenario_i, scenario in enumerate(scenarios):
                attack = attacks[(model_offset + category_i + scenario_i) % len(attacks)]
                candidates = cells[(model, category, scenario, attack)]
                if not candidates:
                    raise RuntimeError(f"Empty design cell: {(model, category, scenario, attack)}")
                chosen = min(candidates, key=lambda row: stable_rank(row, args.seed))
                chosen = dict(chosen)
                chosen["sensitivity_sampling"] = {
                    "design": "one_per_model_category_scenario_attack_rotated",
                    "seed": args.seed,
                    "selection_uses_primary_score": False,
                }
                selected.append(chosen)

    keys = [
        (r.get("sample_id"), r.get("model"), r.get("attack"), r.get("scenario"))
        for r in selected
    ]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate records in sensitivity sample")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(selected)} records to {args.output}")
    for field in ("model", "category", "scenario", "attack"):
        print(field, dict(sorted(Counter(str(r.get(field)) for r in selected).items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
