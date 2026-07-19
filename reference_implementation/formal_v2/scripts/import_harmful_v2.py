#!/usr/bin/env python3
"""Validate and import the model-authored harmful-behavior v2 dataset."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from config import DATA_PROCESSED


EXPECTED_CATEGORIES = {
    "malware/hacking": 10,
    "fraud/deception": 10,
    "physical harm": 10,
    "harassment/discrimination": 10,
    "economic harm": 10,
}
REQUIRED_FIELDS = {
    "id",
    "behavior",
    "category",
    "subcategory",
    "type",
    "source",
    "validity",
    "malicious_intent",
    "harm_rationale",
    "expected_safe_behavior",
    "contains_seeded_operational_details",
    "tags",
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
    return rows


def validate(rows: list[dict]) -> dict:
    errors = []
    expected_ids = [f"harmful_v2_{i:04d}" for i in range(50)]
    actual_ids = [r.get("id") for r in rows]

    if len(rows) != 50:
        errors.append(f"Expected 50 records, found {len(rows)}")
    if actual_ids != expected_ids:
        errors.append("IDs must be consecutive from harmful_v2_0000 to harmful_v2_0049")
    if len(set(actual_ids)) != len(actual_ids):
        errors.append("Duplicate IDs detected")

    behaviors = [str(r.get("behavior", "")).strip() for r in rows]
    if len({b.casefold() for b in behaviors}) != len(behaviors):
        errors.append("Duplicate behavior strings detected")

    category_counts = collections.Counter(r.get("category") for r in rows)
    if dict(category_counts) != EXPECTED_CATEGORIES:
        errors.append(f"Category balance mismatch: {dict(category_counts)}")

    subcategories = [r.get("subcategory") for r in rows]
    if len(set(subcategories)) != 50:
        errors.append("Each record must have a unique subcategory")

    for index, row in enumerate(rows):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"Record {index} missing fields: {sorted(missing)}")
        if row.get("type") != "harmful":
            errors.append(f"{row.get('id')}: type must be harmful")
        if row.get("validity") != "clearly_harmful":
            errors.append(f"{row.get('id')}: validity must be clearly_harmful")
        if row.get("malicious_intent") is not True:
            errors.append(f"{row.get('id')}: malicious_intent must be true")
        if row.get("contains_seeded_operational_details") is not False:
            errors.append(f"{row.get('id')}: seeded operational details are not allowed")
        required_tags = {"clearly_harmful", "explicit_malicious_intent", row.get("subcategory")}
        if not required_tags.issubset(set(row.get("tags", []))):
            errors.append(f"{row.get('id')}: required tags missing")

    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors))

    return {
        "record_count": len(rows),
        "category_counts": dict(category_counts),
        "unique_behaviors": len(set(behaviors)),
        "unique_subcategories": len(set(subcategories)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Source JSONL file")
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_PROCESSED / "harmful_behaviors_v2.jsonl",
    )
    args = parser.parse_args()

    source_bytes = args.input.read_bytes()
    rows = load_jsonl(args.input)
    summary = validate(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    provenance = {
        **summary,
        "dataset_version": "harmful_behaviors_v2",
        "authoring_method": "local_model_generated_then_automatically_validated",
        "human_annotation_used": False,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "output_sha256": digest,
        "output": str(args.output),
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
