#!/usr/bin/env python3
"""Build H-experiment scored items from the three complete target matrices."""

import json
from pathlib import Path

from config import RUNS_DIR

OUT = Path(__file__).resolve().parent.parent / "results/supplementary_20261016_refresh"
TARGETS = ("deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink")


def main():
    rows = []
    audit_path = Path(__file__).resolve().parent.parent / "results/supplementary_20261016_refresh/assets_audit_v2.jsonl"
    audit = {}
    for line in audit_path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            audit[(r["generator"], int(r["seed"]), r["sample_id"])] = r
    for target in TARGETS:
        path = RUNS_DIR / f"refresh_20261016_{target}_responses.jsonl"
        latest = {}
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                latest[row["item_id"]] = row
        data = list(latest.values())
        if len(data) != 450:
            raise SystemExit(f"{target}: expected 450 responses, found {len(data)}")
        keys = [r["item_id"] for r in data]
        if len(set(keys)) != 450:
            raise SystemExit(f"{target}: duplicate item keys")
        errors = [r for r in data if r.get("error") or not r.get("response", "").strip()]
        if errors:
            raise SystemExit(f"{target}: {len(errors)} errors/empty responses; do not judge")
        rows.extend({
            **r, "primary_score": None, "blocked": False,
            "generator_seed": int(r["generator_seed"]),
            "generator_fallback": audit[(r["generator"], int(r["generator_seed"]), r["behavior_id"])]["fallback_reason_reclassified"],
            "target_finish_metadata_available": bool(r.get("finish_metadata_available")),
            "optimization_format": None,
        } for r in data)
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / "items.jsonl"
    destination.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"validated {len(rows)} items -> {destination}")


if __name__ == "__main__":
    main()
