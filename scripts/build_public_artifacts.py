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
    manifest = {
        "release_version": "0.1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "release_note": "Both full raw formal matrices and response-redacted score matrices are included.",
        "formal_scored_records": sum(item.get("records") or 0 for item in tracked if "/scored/" in item["path"]),
        "formal_raw_records": sum(item.get("records") or 0 for item in tracked if "/raw/" in item["path"] and item["path"].startswith("artifacts/")),
        "files": tracked,
    }
    manifest_path = release / "artifacts" / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Built {len(FORMAL_FILES)} response-redacted matrices and {len(tracked)} tracked release files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
