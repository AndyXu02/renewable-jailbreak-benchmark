#!/usr/bin/env python3
"""Refresh the dated supplementary SHA-256 manifest without touching frozen outputs."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/supplementary_20261016_multijudge"

files = [
    Path("scripts/multijudge_build_items.py"),
    Path("scripts/multijudge_run.py"),
    Path("scripts/multijudge_analysis.py"),
    Path("scripts/multijudge_build_new_items.py"),
    Path("scripts/supplementary_20261016_analysis.py"),
    Path("scripts/pair_format_crossover.py"),
    Path("scripts/run_queue_20261016.sh"),
    Path("scripts/finalize_20261016.sh"),
    Path("scripts/write_supplementary_manifest.py"),
    Path("runs/pair_xover_single_20261016.jsonl"),
    Path("runs/pair_xover_multi_20261016.jsonl"),
    Path("runs/phase2_v3cap1024_gemma4-12b-it_responses.jsonl"),
    Path("runs/phase2_v3cap1024_gemma4-12b-obliterated_responses.jsonl"),
    Path("results/supplementary_20261016_multijudge/items.jsonl"),
    Path("results/supplementary_20261016_multijudge/items_new.jsonl"),
    Path("../Thesis/research_paper_draft_20261016.md"),
    Path("../Thesis/build/complete_revision_20261016.py"),
    Path("../Thesis/figures/make_figures_20261016.py"),
    Path("../Thesis/10.16 final_update.md"),
    Path("../AGENTS.md"),
]
files.extend(sorted(p.relative_to(ROOT) for p in ROOT.glob("runs/phase2_xover_*_responses.jsonl")))
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "analysis").glob("*") if p.is_file()))
files.extend(sorted(p.relative_to(ROOT) for p in (OUT / "judges").glob("*.jsonl")))
files.extend(
    sorted(Path("../Thesis/figures") / p.name for p in (ROOT.parent / "Thesis/figures").glob("*_20261016.png"))
)

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "scope": "2026-10-16 supplementary multi-judge re-scoring and follow-up experiments",
    "frozen_v2formal_modified": False,
    "human_reference_labels_in_manifest": False,
    "sha256": {},
}
for rel in files:
    path = ROOT / rel
    if path.is_file():
        manifest["sha256"][str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / "supplementary_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Manifest covers {len(manifest['sha256'])} files")
