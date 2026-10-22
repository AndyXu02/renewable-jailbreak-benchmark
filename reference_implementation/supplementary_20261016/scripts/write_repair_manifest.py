#!/usr/bin/env python3
"""Hash the strict-scoring repair overlay and the paper artifacts derived from it."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary_20261016_repair"

files = [
    Path("scripts/multijudge_run.py"),
    Path("scripts/repair_judge_scores.py"),
    Path("scripts/cluster_sensitivity.py"),
    Path("scripts/test_strict_judge.py"),
    Path("scripts/supplementary_20261016_analysis.py"),
    Path("scripts/write_repair_manifest.py"),
    Path("results/supplementary_20261016_repair/items.jsonl"),
    Path("results/supplementary_20261016_repair/items_new.jsonl"),
    Path("results/supplementary_20261016_repair/selection.jsonl"),
    Path("results/supplementary_20261016_repair/score_changes.jsonl"),
    Path("results/supplementary_20261016_repair/protocol.json"),
    Path("../Thesis/research_paper_draft_20261016.md"),
    Path("../Thesis/research_paper_draft_20261019.md"),
    Path("../Thesis/10.16 final_update.md"),
    Path("../Thesis/10.16 execution_log.md"),
    Path("../AGENTS.md"),
    Path("../Thesis/figures/make_figures_20261016.py"),
]
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "judges").glob("*.jsonl")))
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "repair_attempts").glob("*.jsonl")))
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "analysis").glob("*") if p.is_file()))
files.extend(sorted(
    Path("../Thesis/figures") / p.name
    for pattern in ("*_20261016_repaired.png", "*_20261019.png")
    for p in (ROOT.parent / "Thesis/figures").glob(pattern)
))
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "strict-final-only score repair, reanalysis and revised dated paper",
    "frozen_v2formal_modified": False,
    "human_reference_labels_in_manifest": False,
    "sha256": {},
}
for rel in files:
    path = ROOT / rel
    if path.is_file():
        manifest["sha256"][str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / "repair_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Repair manifest covers {len(manifest['sha256'])} files")
