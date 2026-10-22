#!/usr/bin/env python3
"""Hash the frozen H-generation assets, target outputs, judge scores and analyses."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary_20261016_refresh"
files = [
    Path("docs/refresh_protocol_20261016.md"),
    Path("scripts/run_refresh_generate.py"),
    Path("scripts/freeze_refresh_assets.py"),
    Path("scripts/audit_frozen_refresh_assets.py"),
    Path("scripts/run_refresh_targets.py"),
    Path("scripts/build_refresh_items.py"),
    Path("scripts/analyze_refresh.py"),
    Path("scripts/run_refresh_evaluation_20261016.sh"),
    Path("scripts/run_refresh_generation_20261016.sh"),
    Path("scripts/audit_refresh_after_targets.sh"),
    Path("scripts/test_refresh_parsing.py"),
    Path("scripts/write_refresh_manifest.py"),
    Path("scripts/refresh_generator_effects.py"),
    Path("../Thesis/build/integrate_refresh_results_20261016.py"),
    Path("../Thesis/10.16 execution_log.md"),
    Path("../Thesis/AGENTS.md"),
    Path("../Thesis/research_paper_draft_20261016.md"),
    Path("../Thesis/research_paper_draft_20261019.md"),
    Path("../Thesis/build/integrate_refresh_results_v2.py"),
    Path("../Thesis/figures/make_figure_refresh_20261019.py"),
    Path("../Thesis/10.16 final_update.md"),
    Path("../AGENTS.md"),
]
files.extend(sorted(p.relative_to(ROOT) for p in ROOT.glob("runs/refresh_20261016_*_seed*.jsonl")))
files.extend(sorted(p.relative_to(ROOT) for p in ROOT.glob("runs/refresh_20261016_*_responses.jsonl")))
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "judges").glob("*.jsonl")))
files.extend(sorted(p.relative_to(ROOT) for p in OUT.glob("*")
                    if p.is_file() and p.name != "refresh_manifest.json"))
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "analysis").glob("*") if p.is_file()))
files.extend(sorted(p.relative_to(ROOT) for p in (ROOT / "results/supplementary_20261019_generator_effects").glob("*")
                    if p.is_file()))
files.extend(sorted(
    Path("../Thesis/figures") / p.name
    for pattern in ("*_20261016_repaired.png", "*_20261019.png")
    for p in (ROOT.parent / "Thesis/figures").glob(pattern)
))
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "2026-10-16 generated-wrapper refresh experiment",
    "frozen_assets_before_target_generation": True,
    "target_results_used_for_asset_selection": False,
    "human_labels_included": False,
    "sha256": {},
}
for path in files:
    absolute = ROOT / path
    if absolute.is_file():
        manifest["sha256"][str(path)] = hashlib.sha256(absolute.read_bytes()).hexdigest()
(OUT / "refresh_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Refresh manifest covers {len(manifest['sha256'])} files")
