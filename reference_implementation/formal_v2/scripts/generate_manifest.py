#!/usr/bin/env python3
"""Freeze the completed harmful-v2 formal route in a reproducibility manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path, root: Path) -> dict:
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "records": sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if path.suffix in {".jsonl", ".csv"} else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project.resolve()
    audit_path = args.audit or root / "results" / "v2formal_20260715" / "completion_audit.json"
    output = args.output or root / "results" / "v2formal_20260715" / "formal_results_manifest.json"
    if not audit_path.exists():
        raise SystemExit(f"Completion audit not found: {audit_path}")
    audit = json.loads(audit_path.read_text())
    if not audit.get("all_complete"):
        incomplete = [name for name, item in audit.get("artifacts", {}).items()
                      if item.get("status") != "complete"]
        raise SystemExit(
            "Refusing to freeze an incomplete formal route. Incomplete artifacts: "
            + ", ".join(incomplete)
        )

    result_files = {}
    for name, item in audit["artifacts"].items():
        path = Path(item["path"])
        if not path.is_absolute():
            path = root / path
        result_files[name] = describe(path, root)

    input_paths = {
        "harmful_v2": root / "data/processed/harmful_behaviors_v2.jsonl",
        "harmful_v2_provenance": root / "data/processed/harmful_behaviors_v2.provenance.json",
        "pair_prompts": root / "runs/pair_attacks_harmful_v2.jsonl",
        "pair_provenance": root / "runs/pair_attacks_harmful_v2.provenance.json",
        "a6_wrappers": root / "data/processed/phase4_a6_harmful_v2_gemma3-4b_50seed.jsonl",
        "a6_provenance": root / "data/processed/phase4_a6_harmful_v2_gemma3-4b_50seed.provenance.json",
        "benign_controls": root / "data/processed/benign_prompts_30.jsonl",
        "matched_pair_inputs": root / "data/processed/matched_v2_pair_50_inputs.jsonl",
    }
    code_paths = {
        name: root / "scripts" / filename for name, filename in {
            "phase2_runner": "run_phase2_twostage.py",
            "phase3_runner": "run_phase3.py",
            "inference_backend": "run_inference.py",
            "defenses": "defenses.py",
            "secondary_judge": "judge_external_glm52.py",
            "cluster_statistics": "bootstrap_ci.py",
            "phase3_statistics": "analyze_phase3_statistics.py",
            "benign_statistics": "analyze_benign_statistics.py",
            "a6_statistics": "analyze_a6_v2formal.py",
            "descriptive_summaries": "summarize_v2formal.py",
            "primary_rejudge": "rejudge_gptoss.py",
            "benign_utility_judge": "rejudge_benign_gptoss.py",
            "completion_audit": "audit_v2formal.py",
            "manifest_generator": "generate_manifest.py",
        }.items()
    }
    missing = [str(path) for path in [*input_paths.values(), *code_paths.values()] if not path.exists()]
    if missing:
        raise SystemExit("Cannot freeze manifest; missing inputs/code: " + ", ".join(missing))

    manifest = {
        "manifest_version": 2,
        "generated_at": datetime.now().astimezone().isoformat(),
        "project": "Toward Renewable and Scenario-Aware Jailbreak Benchmarking for Open-Weight LLMs",
        "formal_route": "harmful_v2 / v2formal",
        "completion_audit": describe(audit_path, root),
        "dataset": {
            "harmful_records": 50,
            "categories": 5,
            "records_per_category": 10,
            "subcategory_labels_unique": 50,
            "labeling": "model-authored, definition-driven, automatically validated; no human annotation",
            "claim_boundary": "protocol-based reconstruction, not human-validated ground truth",
        },
        "target_models": {
            "deepseek-v4-flash": {
                "backend": "OpenAI-compatible API",
                "final_max_tokens_by_experiment": {"phase2": 1024, "phase3": 1024, "benign": 1024, "a6_matched": 1024},
            },
            "llama3.1-8b": {
                "model_id": "llama3.1:8b",
                "backend": "Ollama",
                "final_max_tokens_by_experiment": {"phase2": 512, "a6_matched": 1024},
            },
            "qwen3-8b-nothink": {
                "model_id": "qwen3:8b",
                "backend": "Ollama",
                "think": False,
                "final_max_tokens_by_experiment": {"phase2": 512, "phase3": 1024, "benign": 1024, "a6_matched": 1024},
            },
            "gemma4-12b-it": {
                "model_id": "google/gemma-4-12B-it",
                "backend": "MLX text path",
                "final_max_tokens_by_experiment": {"phase2": 256},
            },
            "gemma4-12b-obliterated": {
                "model_id": "OBLITERATUS/Gemma-4-12B-OBLITERATED",
                "backend": "MLX text path",
                "final_max_tokens_by_experiment": {"phase2": 256},
            },
        },
        "shared_decoding": {"temperature": 0.0, "seed": 42, "multiturn_intermediate_max_tokens": 128},
        "phase2": {
            "design": "20 stratified behaviors × 3 attacks × 4 scenarios",
            "records_per_model": 240,
            "attacks": ["A0_direct", "A1_pair", "A2_persona"],
            "scenarios": ["S1_direct", "S2_multiturn", "S3_longcontext", "S4_agent"],
            "pair_note": "Qwen3-8B no-think is the PAIR optimization target; its A1 result is in-sample.",
        },
        "phase3": {
            "representative_models": ["deepseek-v4-flash", "qwen3-8b-nothink"],
            "design": "20 behaviors × 2 attacks × 2 scenarios × 5 defenses",
            "records_per_model": 400,
            "defenses": {
                "D0_none": "no defense",
                "D1_system_prompt": "safety system prompt",
                "D2_perturbation": "three character-perturbed variants plus refusal/non-refusal majority aggregation",
                "D3_rule_input": "fixed keyword input filter",
                "D4_rule_output": "fixed keyword output filter",
            },
            "naming_boundary": "D3/D4 are rule baselines, not learned guard models",
        },
        "benign_utility": {
            "design": "30 benign-sensitive prompts × matched five defenses",
            "utility_scale": "1-5; over-refusal if utility <= 2",
        },
        "a6": {
            "generator": "gemma3:4b",
            "design": "50 literal-seed wrappers transferred to three targets under S1/D0",
            "literal_seed_protocol": "generator template contains <<BEHAVIOR>> exactly once; program injects seed verbatim",
            "valid_model_templates": 40,
            "deterministic_fallback_templates": 10,
            "matched_comparators": ["A0_direct", "A2_persona", "A1_pair"],
        },
        "judges": {
            "primary": {"model_id": "gpt-oss:20b", "backend": "Ollama", "scale": "1-5"},
            "sensitivity": {"model_id": "glm-5.2", "backend": "external OpenAI-compatible API", "sample_n": 100,
                            "sampling": "20 per model, score-blind, category/scenario balanced with fixed attack rotation"},
        },
        "statistics": {
            "bootstrap_unit": "harmful behavior",
            "interaction_permutation": "complete scenario profiles permuted within behavior × attack clusters",
            "phase3_paired_test": "behavior-level sign-flip with behavior bootstrap confidence intervals",
            "multiple_comparisons": "Holm correction",
            "a6": "behavior-paired bootstrap and exact McNemar with Holm correction",
        },
        "known_limitations": [
            "Harmful-v2 labels were not independently human annotated.",
            "Target-specific generation caps/backends limit unconditional cross-model causal ranking.",
            "Gemma 256-vs-512 overlap sensitivity is reported, but does not remove every decoding-budget concern.",
            "S4 is agent-lite prompt injection, not real tool execution.",
            "PAIR was optimized on Qwen, so Qwen A1 is not an out-of-sample transfer estimate.",
            "D3/D4 are deliberately simple keyword baselines and incur over-refusal.",
            "Automated judges remain imperfect; primary and secondary absolute rates differ.",
        ],
        "inputs": {name: describe(path, root) for name, path in input_paths.items()},
        "code": {name: describe(path, root) for name, path in code_paths.items()},
        "formal_result_files": result_files,
        "excluded": [
            "legacy seed_prompts results",
            "diagnostic partials and smoke runs",
            "pre-literal-fix A6 wrappers",
            "pre-channel-fix Gemma outputs",
            "old phi4-mini main-result scores",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen formal manifest -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
