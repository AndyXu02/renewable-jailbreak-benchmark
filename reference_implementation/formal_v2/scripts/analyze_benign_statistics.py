#!/usr/bin/env python3
"""Paired utility and over-refusal tests for benign defense conditions."""

import argparse
import csv
import json
from pathlib import Path

from scipy.stats import binomtest, wilcoxon


DEFENSES = ["D1_system_prompt", "D2_perturbation", "D3_rule_input", "D4_rule_output"]


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, idx in enumerate(order):
        value = min(1.0, p_values[idx] * (m - rank))
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    scores = {}
    for record in records:
        score = record.get("judge_llm", {}).get("score")
        if score is None:
            continue
        key = (record.get("model_id"), record.get("sample_id"), record.get("defense"))
        scores[key] = int(score)

    rows = []
    for defense in DEFENSES:
        pairs = []
        for model, sample_id, condition in scores:
            if condition != "D0_none":
                continue
            baseline = scores[(model, sample_id, "D0_none")]
            defended = scores.get((model, sample_id, defense))
            if defended is not None:
                pairs.append((baseline, defended))

        baseline_scores = [a for a, _ in pairs]
        defended_scores = [b for _, b in pairs]
        utility_test = wilcoxon(defended_scores, baseline_scores, alternative="two-sided")
        b = sum(a > 2 and d <= 2 for a, d in pairs)
        c = sum(a <= 2 and d > 2 for a, d in pairs)
        mcnemar_p = binomtest(min(b, c), n=b + c, p=0.5).pvalue if b + c else 1.0
        rows.append(
            {
                "defense": defense,
                "paired_n": len(pairs),
                "baseline_mean_utility": sum(baseline_scores) / len(pairs),
                "defended_mean_utility": sum(defended_scores) / len(pairs),
                "mean_utility_change": sum(d - a for a, d in pairs) / len(pairs),
                "wilcoxon_statistic": float(utility_test.statistic),
                "wilcoxon_p": float(utility_test.pvalue),
                "new_overrefusals": b,
                "resolved_overrefusals": c,
                "mcnemar_exact_p": float(mcnemar_p),
            }
        )

    utility_adjusted = holm_adjust([row["wilcoxon_p"] for row in rows])
    orr_adjusted = holm_adjust([row["mcnemar_exact_p"] for row in rows])
    for row, utility_p, orr_p in zip(rows, utility_adjusted, orr_adjusted):
        row["wilcoxon_p_holm"] = utility_p
        row["mcnemar_p_holm"] = orr_p

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(rows)} paired comparisons -> {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
