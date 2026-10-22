#!/usr/bin/env python3
"""Build the response-redacted public result package from the private experiment tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


FORMAL_FILES = {
    "phase2_deepseek-v4-flash": "phase2_v2formal_deepseek-v4-flash.jsonl",
    "phase2_llama3.1-8b": "phase2_v2formal_llama3.1-8b.jsonl",
    "phase2_qwen3-8b-nothink": "phase2_v2formal_qwen3-8b-nothink.jsonl",
    "phase2_gemma4-12b-it": "phase2_v2formal_gemma4-12b-it.jsonl",
    "phase2_gemma4-12b-obliterated": "phase2_v2formal_gemma4-12b-obliterated.jsonl",
    "phase3_deepseek-v4-flash": "phase3_v2formal_deepseek-v4-flash.jsonl",
    "phase3_qwen3-8b-nothink": "phase3_v2formal_qwen3-8b-nothink.jsonl",
    "benign_deepseek-v4-flash": "benign_v2formal_deepseek-v4-flash_responses.gptoss_utility.jsonl",
    "benign_qwen3-8b-nothink": "benign_v2formal_qwen3-8b-nothink_responses.gptoss_utility.jsonl",
    "a6_deepseek-v4-flash": "phase4_a6_v2formal_deepseek-v4-flash_responses.gptoss.jsonl",
    "a6_llama3.1-8b": "phase4_a6_v2formal_llama3.1-8b_responses.gptoss.jsonl",
    "a6_qwen3-8b-nothink": "phase4_a6_v2formal_qwen3-8b-nothink_responses.gptoss.jsonl",
    "a6_baselines_deepseek-v4-flash": "matched_v2_deepseek-v4-flash_a0_a2_responses.gptoss.jsonl",
    "a6_baselines_llama3.1-8b": "matched_v2_llama3.1-8b_a0_a2_responses.gptoss.jsonl",
    "a6_baselines_qwen3-8b-nothink": "matched_v2_qwen3-8b-nothink_a0_a2_responses.gptoss.jsonl",
    "a6_pair_deepseek-v4-flash": "matched_v2_deepseek-v4-flash_pair_responses.gptoss.jsonl",
    "a6_pair_llama3.1-8b": "matched_v2_llama3.1-8b_pair_responses.gptoss.jsonl",
    "a6_pair_qwen3-8b-nothink": "matched_v2_qwen3-8b-nothink_pair_responses.gptoss.jsonl",
    "judge_sensitivity": "judge_sensitivity_v2formal_glm52.jsonl",
}

SUPPLEMENTARY_RESULTS = {
    "multijudge_20261016": (
        "supplementary_20261016_repair",
        ["items.jsonl", "items_new.jsonl", "protocol.json", "analysis/*", "judges/*.jsonl", "repair_attempts/*.jsonl"],
    ),
    "multijudge_20261016/judges_prerepair": ("supplementary_20261016_multijudge/judges", ["*.jsonl"]),
    "multijudge_20261016/prerepair_manifest": ("supplementary_20261016_multijudge", ["supplementary_manifest.json"]),
    "refresh_20261016": (
        "supplementary_20261016_refresh",
        [
            "assets_frozen.jsonl",
            "assets_audit_v2.json",
            "assets_audit_v2.jsonl",
            "assets_manifest.json",
            "items.jsonl",
            "refresh_manifest.json",
            "analysis/*",
            "judges/*.jsonl",
        ],
    ),
    "generator_effects_20261019": ("supplementary_20261019_generator_effects", ["*"]),
    "supplementary_20260926": ("supplementary_20260926", ["*"]),
}
SUPPLEMENTARY_RUNS = {
    "refresh_20261016/generation": "refresh_20261016_*_seed*.jsonl",
    "refresh_20261016/targets": "refresh_20261016_*_responses.jsonl",
    "followups_20261016/pair_format_crossover": "pair_xover_*_20261016.jsonl",
    "followups_20261016/pair_format_crossover_responses": "phase2_xover_*_responses.jsonl",
    "followups_20261016/gemma_cap1024_responses": "phase2_v3cap1024_*_responses.jsonl",
}
SUPPLEMENTARY_SCRIPTS = [
    "multijudge_build_items.py",
    "multijudge_build_new_items.py",
    "multijudge_run.py",
    "multijudge_analysis.py",
    "repair_judge_scores.py",
    "cluster_sensitivity.py",
    "supplementary_analysis.py",
    "supplementary_20261016_analysis.py",
    "pair_format_crossover.py",
    "run_inference.py",
    "run_refresh_generate.py",
    "freeze_refresh_assets.py",
    "audit_frozen_refresh_assets.py",
    "build_refresh_items.py",
    "run_refresh_targets.py",
    "analyze_refresh.py",
    "refresh_generator_effects.py",
    "write_supplementary_manifest.py",
    "write_repair_manifest.py",
    "write_refresh_manifest.py",
    "test_strict_judge.py",
    "test_refresh_parsing.py",
    "run_queue_20261016.sh",
    "run_refresh_generation_20261016.sh",
    "run_refresh_evaluation_20261016.sh",
    "wait_refresh_assets_then_evaluate.sh",
    "audit_refresh_after_targets.sh",
    "finalize_20261016.sh",
]
# Human-reference items and their judge records are calibration-only and never published.
PRIVATE_ITEM_PREFIX = "human120|"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256_bytes(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_record(row: dict) -> dict:
    metadata = row.get("metadata") or {}
    behavior = str(row.get("behavior") or metadata.get("judge_behavior") or metadata.get("behavior") or "")
    variants = row.get("response_variants") or (metadata.get("defense_result") or {}).get("response_variants") or []
    return {
        "sample_id": row.get("sample_id"),
        "model_id": row.get("model_id") or row.get("model"),
        "category": row.get("category") or metadata.get("category"),
        "attack": row.get("attack"),
        "scenario": row.get("scenario"),
        "defense": row.get("defense"),
        "behavior_sha256": sha256_bytes(behavior) if behavior else None,
        "score_5scale": row.get("score_5scale"),
        "utility_score": row.get("utility_score"),
        "score_glm52": row.get("score_glm52"),
        "variant_count": len(variants),
        "blocked": str(row.get("response") or "").startswith("[BLOCKED"),
        "had_error": bool(row.get("error") or metadata.get("error")),
        "generation_max_tokens": row.get("generation_max_tokens"),
        "generator_strategy": metadata.get("generator_strategy"),
    }


def describe(path: Path, root: Path) -> dict:
    records = None
    if path.suffix == ".jsonl":
        records = sum(1 for line in path.open(encoding="utf-8") if line.strip())
    return {
        "path": str(path.relative_to(root)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": records,
    }


def copy_public(source_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.suffix != ".jsonl":
        shutil.copy2(source_path, target_path)
        return
    lines = [
        line for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not str(json.loads(line).get("item_id", "")).startswith(PRIVATE_ITEM_PREFIX)
    ]
    target_path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def build_supplementary(source: Path, release: Path) -> Path:
    root = release / "artifacts" / "supplementary"
    if root.exists():
        shutil.rmtree(root)
    for label, (folder, patterns) in SUPPLEMENTARY_RESULTS.items():
        base = source / "results" / folder
        for pattern in patterns:
            matches = sorted(path for path in base.glob(pattern) if path.is_file() and not path.name.startswith("human120"))
            if not matches:
                raise SystemExit(f"Missing supplementary input: {base / pattern}")
            for path in matches:
                copy_public(path, root / label / path.relative_to(base))
    for label, pattern in SUPPLEMENTARY_RUNS.items():
        matches = sorted((source / "runs").glob(pattern))
        if not matches:
            raise SystemExit(f"Missing supplementary run: {pattern}")
        for path in matches:
            copy_public(path, root / label / path.name)
    scripts = release / "reference_implementation" / "supplementary_20261016" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name in SUPPLEMENTARY_SCRIPTS:
        shutil.copy2(source / "scripts" / name, scripts / name)
    shutil.copy2(source / "docs" / "refresh_protocol_20261016.md", release / "docs" / "refresh_protocol_20261016.md")
    return root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    source = args.source_project.resolve()
    release = args.release_root.resolve()
    runs = source / "runs"
    scored = release / "artifacts" / "formal" / "scored"
    raw = release / "artifacts" / "formal" / "raw"

    for label, filename in FORMAL_FILES.items():
        source_path = runs / filename
        if not source_path.exists():
            raise SystemExit(f"Missing formal input: {source_path}")
        write_jsonl(scored / f"{label}.jsonl", [public_record(row) for row in load_jsonl(source_path)])
        raw.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, raw / f"{label}.jsonl")

    pair_rows = []
    for row in load_jsonl(runs / "pair_attacks_harmful_v2.jsonl"):
        pair_rows.append({
            "sample_id": row.get("sample_id"),
            "category": row.get("category"),
            "best_prompt": row.get("best_prompt"),
            "generator_model": row.get("attacker_model"),
            "rounds": row.get("total_rounds"),
            "success_on_optimization_target": row.get("success"),
        })
    write_jsonl(release / "data" / "research" / "pair_prompts_harmful_v2.jsonl", pair_rows)
    raw_data = release / "data" / "research" / "raw"
    raw_data.mkdir(parents=True, exist_ok=True)
    shutil.copy2(runs / "pair_attacks_harmful_v2.jsonl", raw_data / "pair_attacks_harmful_v2.full.jsonl")
    shutil.copy2(
        runs / "pair_attacks_harmful_v2.provenance.json",
        raw_data / "pair_attacks_harmful_v2.provenance.json",
    )

    a6_source = source / "data" / "processed" / "phase4_a6_harmful_v2_gemma3-4b_50seed.jsonl"
    a6_rows = []
    for row in load_jsonl(a6_source):
        a6_rows.append({
            "id": row.get("id"),
            "seed_sample_id": row.get("seed_sample_id"),
            "category": row.get("category"),
            "source": row.get("source"),
            "type": row.get("type"),
            "behavior": row.get("behavior"),
            "judge_behavior": row.get("judge_behavior"),
            "generator_model": row.get("generator_model"),
            "generator_backend": row.get("generator_backend"),
            "generator_strategy": row.get("generator_strategy"),
            "generator_template": row.get("generator_template"),
            "literal_behavior_preserved": row.get("literal_behavior_preserved"),
        })
    write_jsonl(release / "data" / "research" / "a6_wrappers_harmful_v2.jsonl", a6_rows)
    shutil.copy2(a6_source, raw_data / "a6_wrappers_harmful_v2.full.jsonl")
    processed = source / "data" / "processed"
    shutil.copy2(
        processed / "harmful_behaviors_v2.provenance.json",
        raw_data / "harmful_behaviors_v2.provenance.json",
    )
    shutil.copy2(
        processed / "phase4_a6_harmful_v2_gemma3-4b_50seed.provenance.json",
        raw_data / "a6_wrappers_harmful_v2.provenance.json",
    )
    shutil.copy2(
        processed / "matched_v2_pair_50_inputs.jsonl",
        raw_data / "matched_v2_pair_50_inputs.jsonl",
    )

    tracked = []
    folders = (
        release / "data" / "research",
        raw_data,
        scored,
        raw,
        release / "artifacts" / "formal" / "tables",
    )
    for folder in folders:
        for path in sorted(folder.iterdir()):
            if path.is_file() and path.name != "release_manifest.json":
                tracked.append(describe(path, release))
    supplementary = build_supplementary(source, release)
    for folder in (supplementary, release / "reference_implementation" / "supplementary_20261016"):
        tracked.extend(describe(path, release) for path in sorted(folder.rglob("*")) if path.is_file())
    manifest = {
        "release_version": "0.2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_note": (
            "Full raw formal matrices, response-redacted score matrices, and the October 2026 supplementary "
            "layer (three-judge panel with strict repairs, follow-ups, and wrapper refresh). Human-reference "
            "items and labels are excluded."
        ),
        "formal_scored_records": sum(item.get("records") or 0 for item in tracked if "/scored/" in item["path"]),
        "formal_raw_records": sum(item.get("records") or 0 for item in tracked if "/formal/raw/" in item["path"]),
        "supplementary_records": sum(item.get("records") or 0 for item in tracked if "/supplementary/" in item["path"]),
        "files": tracked,
    }
    manifest_path = release / "artifacts" / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(FORMAL_FILES)} response-redacted matrices and {len(tracked)} tracked release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
