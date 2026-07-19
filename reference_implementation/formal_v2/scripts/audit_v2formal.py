#!/usr/bin/env python3
"""Audit completeness and integrity of the harmful-v2 formal route."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


PHASE2_MODELS = [
    "deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink",
    "gemma4-12b-it", "gemma4-12b-obliterated",
]
REPRESENTATIVE_MODELS = ["deepseek-v4-flash", "qwen3-8b-nothink"]
A6_MODELS = ["deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def error_value(row: dict):
    return row.get("error") or (row.get("metadata") or {}).get("error")


def audit_file(path: Path, expected: int, key_fields: tuple[str, ...], score_field: str | None,
               categories: dict[str, int] | None = None, d2_variants: bool = False) -> dict:
    result = {"path": str(path), "expected_records": expected, "exists": path.exists()}
    if not path.exists():
        result["status"] = "missing"
        return result
    rows = load(path)
    keys = {tuple(row.get(field) for field in key_fields) for row in rows}
    result.update({
        "records": len(rows),
        "unique_keys": len(keys),
        "empty_responses": sum(not str(row.get("response") or "").strip() for row in rows),
        "errors": sum(bool(error_value(row)) for row in rows),
        "sha256": sha256(path),
    })
    if score_field:
        result["unscored"] = sum(not isinstance(row.get(score_field), int) for row in rows)
    if categories:
        observed = Counter(
            row.get("category") or (row.get("metadata") or {}).get("category") for row in rows
        )
        result["categories"] = dict(observed)
        result["category_balance_ok"] = dict(observed) == categories
    if d2_variants:
        d2 = [row for row in rows if row.get("defense") == "D2_perturbation"]
        result["d2_records"] = len(d2)
        result["d2_three_nonempty_variants"] = sum(
            len(row.get("response_variants") or (row.get("metadata") or {}).get("defense_result", {}).get("response_variants", [])) == 3
            and all(row.get("response_variants") or (row.get("metadata") or {}).get("defense_result", {}).get("response_variants", []))
            for row in d2
        )
    gates = [
        len(rows) == expected,
        len(keys) == expected,
        result["empty_responses"] == 0,
        result["errors"] == 0,
        not score_field or result["unscored"] == 0,
        not categories or result["category_balance_ok"],
        not d2_variants or result["d2_three_nonempty_variants"] == result["d2_records"],
    ]
    result["status"] = "complete" if all(gates) else "incomplete"
    return result


def audit_derived(path: Path, expected_data_rows: int | None = None,
                  required_json_paths: tuple[tuple[str, ...], ...] = ()) -> dict:
    """Audit a derived CSV/JSON/text artifact rather than a response JSONL."""
    result = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        result["status"] = "missing"
        return result
    result.update({"size_bytes": path.stat().st_size, "sha256": sha256(path)})
    gates = [path.stat().st_size > 0]
    if path.suffix == ".csv":
        rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        data_rows = max(0, len(rows) - 1)
        result["data_rows"] = data_rows
        if expected_data_rows is not None:
            result["expected_data_rows"] = expected_data_rows
            gates.append(data_rows == expected_data_rows)
    elif path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            result["valid_json"] = True
        except Exception as exc:
            payload = {}
            result.update({"valid_json": False, "parse_error": str(exc)})
            gates.append(False)
        for key_path in required_json_paths:
            value = payload
            for key in key_path:
                value = value.get(key) if isinstance(value, dict) else None
            ok = value is not None
            result["required_" + "_".join(key_path)] = ok
            gates.append(ok)
    result["status"] = "complete" if all(gates) else "incomplete"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    runs = args.project / "runs"
    audits = {}
    phase2_categories = {name: 48 for name in (
        "economic harm", "fraud/deception", "harassment/discrimination",
        "malware/hacking", "physical harm",
    )}
    phase3_categories = {name: 80 for name in phase2_categories}
    for model in PHASE2_MODELS:
        name = f"phase2_{model}"
        audits[name] = audit_file(
            runs / f"phase2_v2formal_{model}.jsonl", 240,
            ("sample_id", "attack", "scenario"), "score_5scale", phase2_categories,
        )
    for model in REPRESENTATIVE_MODELS:
        audits[f"phase3_{model}"] = audit_file(
            runs / f"phase3_v2formal_{model}.jsonl", 400,
            ("sample_id", "attack", "scenario", "defense"), "score_5scale",
            phase3_categories, d2_variants=True,
        )
        audits[f"benign_{model}"] = audit_file(
            runs / f"benign_v2formal_{model}_responses.gptoss_utility.jsonl", 150,
            ("sample_id", "model_id", "attack", "scenario", "defense"), "utility_score",
            {"benign_sensitive": 150}, d2_variants=True,
        )
    for model in A6_MODELS:
        audits[f"a6_{model}"] = audit_file(
            runs / f"phase4_a6_v2formal_{model}_responses.gptoss.jsonl", 50,
            ("sample_id", "model_id", "attack", "scenario", "defense"), "score_5scale",
            {name: 10 for name in phase2_categories},
        )
        audits[f"a6_baselines_{model}"] = audit_file(
            runs / f"matched_v2_{model}_a0_a2_responses.gptoss.jsonl", 100,
            ("sample_id", "model_id", "attack", "scenario", "defense"), "score_5scale",
            {name: 20 for name in phase2_categories},
        )
        audits[f"a6_pair_{model}"] = audit_file(
            runs / f"matched_v2_{model}_pair_responses.gptoss.jsonl", 50,
            ("sample_id", "model_id", "attack", "scenario", "defense"), "score_5scale",
            {name: 10 for name in phase2_categories},
        )
    audits["judge_sensitivity"] = audit_file(
        runs / "judge_sensitivity_v2formal_glm52.jsonl", 100,
        ("sample_id", "model", "attack", "scenario"), "score_glm52",
        {name: 20 for name in phase2_categories},
    )

    # Derived evidence is part of the completion claim, not an optional
    # afterthought.  Row-count gates catch stale partial summaries.
    results = args.project / "results" / "v2formal_20260715"
    derived_csv = {
        "phase2_summary_model": ("phase2_by_model.csv", 5),
        "phase2_summary_attack": ("phase2_by_model_attack.csv", 15),
        "phase2_summary_scenario": ("phase2_by_model_scenario.csv", 20),
        "phase3_summary": ("phase3_by_model_defense.csv", 10),
        "phase3_paired_statistics": ("phase3_clustered_paired.csv", 24),
        "benign_summary": ("benign_by_model_defense.csv", 10),
        "benign_deepseek_statistics": ("benign_deepseek_paired.csv", 4),
        "benign_qwen_statistics": ("benign_qwen_paired.csv", 4),
        "a6_summary": ("summary_by_model_attack.csv", 12),
        "a6_paired_statistics": ("paired_vs_a6.csv", 9),
        "a6_strategy_summary": ("a6_by_generator_strategy.csv", 6),
        "judge_sensitivity_primary_summary": ("judge_sensitivity_primary_by_model.csv", 5),
        "judge_sensitivity_glm_summary": ("judge_sensitivity_glm_by_model.csv", 5),
    }
    for name, (filename, expected_rows) in derived_csv.items():
        audits[name] = audit_derived(results / filename, expected_rows)
    audits["phase2_clustered_interaction"] = audit_derived(
        results / "phase2_clustered_bootstrap_interaction.json",
        required_json_paths=(("phase2",), ("interaction_test", "p_value")),
    )
    audits["phase3_paired_statistics_json"] = audit_derived(
        results / "phase3_clustered_paired.json"
    )
    audits["a6_statistics_json"] = audit_derived(results / "results.json")
    audits["results_discussion_draft"] = audit_derived(
        args.project / "docs" / "results_chapter_draft_zh.md"
    )

    summary = Counter(item["status"] for item in audits.values())
    report = {"summary": dict(summary), "all_complete": summary == {"complete": len(audits)}, "artifacts": audits}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    for name, item in audits.items():
        if item["status"] != "complete":
            print(f"{item['status']:10s} {name}: {item.get('records', 0)}/{item['expected_records']}")
    return 0 if report["all_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
