# October 2026 supplementary layer

This layer backs the final thesis results (three-judge panel, follow-up checks, and
wrapper refresh). It reads the frozen formal responses without modifying them; every
file is pinned by SHA-256 in `artifacts/release_manifest.json`.

## Contents of `artifacts/supplementary/`

| Path | What it contains |
|---|---|
| `multijudge_20261016/items.jsonl` | The 2,900 formal experiment responses as judged items (Experiments 1-3). |
| `multijudge_20261016/items_new.jsonl` | 800 follow-up items: 320 PAIR format-crossover and 480 Gemma 1,024-token cap responses. |
| `multijudge_20261016/judges/` | Final per-item scores from gpt-oss 20B (validated rubric), GLM-5.2, and MiniMax-M3 after strict final-answer repairs. All paper figures use these scores. |
| `multijudge_20261016/judges_prerepair/` | The same judges before repair, for audit. |
| `multijudge_20261016/repair_attempts/`, `protocol.json` | Strict-parser repair protocol and the re-scoring attempts for the 193 flagged relay records. |
| `multijudge_20261016/analysis/` | Panel-median result tables (Exp. 1-3, defenses, benign utility, crossover, cap sensitivity, judge agreement, clustered sensitivity, aggregate human-anchor metrics). |
| `refresh_20261016/generation/` | Raw outputs of three generators (Gemma 3 4B, Kimi K3, MiMo V2.6 Flash) x three seeds (42, 43, 44). |
| `refresh_20261016/assets_frozen.jsonl`, `assets_audit_v2.*`, `assets_manifest.json` | The 450 hash-frozen wrapper assets and their schema/placeholder/lint audit. |
| `refresh_20261016/targets/`, `items.jsonl`, `judges/`, `analysis/` | 1,350 target responses (DeepSeek V4 Flash, Llama 3.1 8B, Qwen3 8B), three-judge scores, and refresh tables. |
| `generator_effects_20261019/` | Post hoc generator main and generator-by-target permutation tests. |
| `followups_20261016/` | Raw PAIR format-crossover prompt sets and target responses, and Gemma cap-sensitivity responses. |
| `supplementary_20260926/` | Earlier read-only attack-by-scenario re-analysis. |

The frozen refresh protocol is in [`refresh_protocol_20261016.md`](refresh_protocol_20261016.md).

## Excluded by design

The 120-item human reference set, its individual and consensus human labels, and the
judge records for those items are calibration-only and are not published. Only
aggregate agreement metrics (`analysis/human_anchor.csv`) are included, so the
human-anchor analysis cannot be recomputed from this release. Run logs and provider
credentials are also excluded.

## Code

`reference_implementation/supplementary_20261016/scripts/` holds the scripts as used.
They import shared modules (`config.py`, `utils.py`, `judge.py`, `attacks.py`,
`scenarios.py`, `defenses.py`, `iterative_pair.py`, `generate_model_attacks.py`,
`bootstrap_ci.py`, `analyze_phase3_statistics.py`) that are byte-identical to the copies
in `reference_implementation/formal_v2/scripts/`; `run_inference.py` is the updated
version used for the follow-ups and refresh. The scripts expect the original private
project layout (`runs/`, `results/`) and read relay credentials from a local `.env`.
