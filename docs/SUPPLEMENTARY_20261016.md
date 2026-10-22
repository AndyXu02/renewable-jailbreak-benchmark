# Supplementary Layer: Multi-Judge Panel and Wrapper Refresh

This layer provides the final scoring system and wrapper renewability validation for the thesis results.

## Contents

### Multi-Judge Panel (`artifacts/supplementary/multijudge_20261016/`)

| File | Description |
|------|-------------|
| `items.jsonl` | 2,900 formal experiment responses (Experiments 1-3) |
| `items_new.jsonl` | 800 follow-up responses: PAIR format crossover (320) + Gemma cap sensitivity (480) |
| `judges/*.jsonl` | Final scores from three judges with strict validation |
| `judges_prerepair/*.jsonl` | Pre-repair scores for audit trail |
| `repair_attempts/`, `protocol.json` | Repair protocol and re-scoring attempts for 193 flagged records |
| `analysis/*.csv` | Result tables, defense effects, benign utility, judge agreement, human-anchor metrics |

**Judges:**
- gpt-oss 20B (validated rubric)
- GLM-5.2
- MiniMax-M3

All panel scores use strict final-answer validation. Figures in the thesis use panel-median scores.

### Wrapper Refresh (`artifacts/supplementary/refresh_20261016/`)

| File | Description |
|------|-------------|
| `generation/` | Raw outputs from three generators × three seeds |
| `assets_frozen.jsonl` | 450 hash-frozen wrapper assets |
| `assets_audit_v2.*`, `assets_manifest.json` | Schema, placeholder, and lint audit |
| `targets/*.jsonl` | 1,350 target responses (DeepSeek V4 Flash, Llama 3.1 8B, Qwen3 8B) |
| `items.jsonl` | Structured refresh items |
| `judges/*.jsonl` | Three-judge panel scores |
| `analysis/*.csv` | Refresh effect tables and generator statistics |

**Generators:**
- Gemma 3 4B
- Kimi K3
- MiMo V2.6 Flash

**Protocol:** Each of 50 behaviors × 3 generators × 3 seeds produces one validated wrapper template. Three target models evaluate all 450 wrappers.

Frozen protocol: [`refresh_protocol_20261016.md`](refresh_protocol_20261016.md)

### Generator Effects (`artifacts/supplementary/generator_effects_20261019/`)

Post hoc permutation tests for generator main effects and generator-by-target interactions.

### Earlier Supplementary (`artifacts/supplementary/supplementary_20260926/`)

Read-only attack-by-scenario re-analysis from earlier iterations.

## Code

Scripts are in `reference_implementation/supplementary_20261016/scripts/`. They share core modules with `formal_v2/`:
- `config.py`, `utils.py`, `judge.py`
- `attacks.py`, `scenarios.py`, `defenses.py`
- `iterative_pair.py`, `generate_model_attacks.py`
- `bootstrap_ci.py`, `analyze_phase3_statistics.py`

Updated `run_inference.py` handles follow-ups and refresh targets.

## Excluded Materials

The 120-item human calibration set, individual human ratings, and their judge records are not published. Only aggregate agreement metrics (`analysis/human_anchor.csv`) are included.

## Reproducing Analyses

From `reference_implementation/supplementary_20261016/scripts/`:

```bash
# Multi-judge panel analysis
python3 analyze_multijudge.py

# Wrapper refresh analysis
python3 analyze_refresh.py

# Generator effect tests
python3 refresh_generator_effects.py

# Strict judge validation tests
python3 -m unittest test_strict_judge.py
python3 -m unittest test_refresh_parsing.py
```

Pre-computed outputs are in corresponding `analysis/` directories.
