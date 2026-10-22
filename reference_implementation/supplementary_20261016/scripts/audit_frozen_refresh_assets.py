#!/usr/bin/env python3
"""Audit frozen H assets' fallback reasons without changing any evaluated prompt."""

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from freeze_refresh_assets import FALLBACK, OUT, parse_template
from config import RUNS_DIR
from run_refresh_generate import GENERATORS, ROOT


def main():
    frozen_path = OUT / "assets_frozen.jsonl"
    raw = {}
    for generator in GENERATORS:
        for seed in (42, 43, 44):
            path = RUNS_DIR / f"refresh_20261016_{generator}_seed{seed}.jsonl"
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    raw[(generator, seed, row["sample_id"])] = row
    frozen = [json.loads(line) for line in frozen_path.read_text().splitlines() if line.strip()]
    categories, audit, discrepancies = Counter(), [], []
    expected_prompts = {}
    for row in frozen:
        key = (row["generator"], row["seed_requested"], row["sample_id"])
        source = raw[key]
        allowed = source.get("served_model") in GENERATORS[row["generator"]]["served_allowlist"]
        template, normalization, flags = parse_template(source.get("raw_final"), source["behavior"])
        semantic_flags = [f for f in flags if f not in {
            "invalid_json", "invalid_variant_schema", "placeholder_or_seed_repeat"}]
        if source.get("served_model") is None:
            reason = "generator_transport_failed"
            expected = FALLBACK.format(behavior=source["behavior"])
        elif not allowed:
            reason = "unexpected_served_model"
            expected = ""
        elif source.get("finish_reason") != "stop":
            reason = "truncated_or_nonfinal_generation"
            expected = FALLBACK.format(behavior=source["behavior"])
        elif template and not semantic_flags:
            reason = None
            expected = template.replace("<<BEHAVIOR>>", source["behavior"])
        else:
            reason = "wrapper_safety_lint" if semantic_flags else (
                flags[0] if flags else "invalid_json_placeholder")
            expected = FALLBACK.format(behavior=source["behavior"])
        if expected != row.get("prompt"):
            discrepancies.append(row["sample_id"])
        categories[reason or "model_wrapper"] += 1
        audit.append({
            "generator": row["generator"], "seed": row["seed_requested"],
            "sample_id": row["sample_id"], "raw_first_pass_valid": row["first_pass_valid"],
            "normalized_schema_valid": template is not None, "normalization": normalization,
            "fallback_reason_reclassified": reason, "safety_lint_flags": semantic_flags,
        })
        expected_prompts[f"{row['generator']}|{row['seed_requested']}|{row['sample_id']}"] = expected

    # Confirm already-started target requests used the frozen prompt exactly.
    target_checks = {}
    for path in sorted(RUNS_DIR.glob("refresh_20261016_*_responses.jsonl")):
        checked = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = "|".join((row["generator"], str(row["generator_seed"]), row["behavior_id"]))
            if expected_prompts.get(key) != row.get("attack_prompt"):
                discrepancies.append(row["item_id"])
            checked += 1
        target_checks[path.name] = checked
    if discrepancies:
        raise SystemExit(f"Frozen-asset/prompt mismatches: {len(discrepancies)}")
    assets_hash = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    category_counts = Counter(r["fallback_reason_reclassified"] or "model_wrapper" for r in audit)
    result = {
        "audited_without_target_outcome_selection": True,
        "frozen_assets_sha256": assets_hash,
        "n_assets": len(frozen),
        "reclassified_fallbacks": dict(category_counts),
        "target_responses_checked_against_frozen_prompt": target_checks,
        "prompt_mismatches": 0,
        "assets_frozen_file_modified": False,
    }
    (OUT / "assets_audit_v2.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in audit))
    (OUT / "assets_audit_v2.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest_path = OUT / "assets_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_first_pass_valid"] = sum(r["raw_first_pass_valid"] for r in audit)
    manifest["normalized_schema_valid"] = sum(r["normalized_schema_valid"] for r in audit)
    manifest["outer_fence_normalized"] = sum(r["normalization"] == "outer_markdown_json_fence" for r in audit)
    manifest["safety_lint_passed"] = category_counts["model_wrapper"]
    manifest["final_usable_assets"] = category_counts["model_wrapper"]
    manifest["fallbacks"] = len(audit) - category_counts["model_wrapper"]
    manifest["fallback_reasons"] = dict(Counter(r["fallback_reason_reclassified"] for r in audit
                                                  if r["fallback_reason_reclassified"]))
    manifest["safety_lint_flags"] = dict(Counter(flag for r in audit for flag in r["safety_lint_flags"]))
    manifest["metadata_audited_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["metadata_audit_uses_target_outcomes"] = False
    manifest["assets_frozen_file_sha256_verified"] = assets_hash
    manifest["protocol_amendment"] = (
        "Before target outcomes were observed: strip only one outer fenced-JSON container; "
        "apply versioned explicit procedure-request phrase lint; preserve raw outputs and report "
        "raw/schema/lint pass rates separately."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
