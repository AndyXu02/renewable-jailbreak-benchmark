# Renewable & Scenario-Aware Jailbreak Benchmark

[中文版 README](README_CN.md)

A reproducible benchmark for testing language model safety across different attack methods, deployment scenarios, and defense mechanisms.

## Overview

This benchmark evaluates:
- **50 harmful behavior objectives** across five risk categories
- **5 target models** with 4 prompt-delivery scenarios
- **Attack methods**: Direct prompts, persona framing, PAIR optimization, and renewable generated wrappers
- **Defense mechanisms**: System prompts, character perturbation, input/output filtering
- **Benign baselines** to measure utility and over-refusal
- **Multi-judge evaluation** with strict validation

The repository includes:
- **3,700 experimental responses** with validated judge scores (Experiments 1-3 plus follow-ups)
- **1,350 wrapper-refresh responses** from renewable asset protocol
- **Portable runner** supporting Ollama, OpenAI-compatible APIs, and mock backends
- **Complete analysis scripts** for reproducing all reported results

## Quick Start

### Installation

Requires Python 3.10+. No mandatory dependencies for core verification.

```bash
git clone https://github.com/AndyXu02/renewable-jailbreak-benchmark.git
cd renewable-jailbreak-benchmark
python3 -m pip install -e .
```

### Verify Data Integrity

Check checksums and record counts (no model required):

```bash
rjbench-verify
python3 -m unittest discover -s tests -v
```

Expected output: `Release verification passed: ... files; 3000 raw + 3000 compact formal records`

### Run With Mock Backend

Test the full pipeline without a model:

```bash
rjbench-run \
  --backend mock \
  --model mock-safe-model \
  --data data/smoke/smoke_prompts.jsonl \
  --attacks A0_direct,A2_persona \
  --scenarios S1_direct,S2_multiturn \
  --defenses D0_none,D1_system_prompt \
  --output runs/mock_responses.jsonl

rjbench-judge \
  --backend mock \
  --model mock-judge \
  --input runs/mock_responses.jsonl \
  --output runs/mock_judged.jsonl

rjbench-summarize runs/mock_judged.jsonl \
  --group-by model_id,attack,scenario,defense \
  --output runs/mock_summary.csv
```

### Run With Ollama

Install [Ollama](https://ollama.com/) and pull a model:

```bash
ollama pull llama3.2:3b

rjbench-run \
  --backend ollama \
  --model llama3.2:3b \
  --data data/research/harmful_behaviors_v2.jsonl \
  --max-samples 5 \
  --attacks A0_direct,A2_persona \
  --scenarios S1_direct,S2_multiturn \
  --defenses D0_none,D1_system_prompt \
  --max-tokens 512 \
  --output runs/llama3.2-3b.jsonl

rjbench-judge \
  --backend ollama \
  --model llama3.2:3b \
  --input runs/llama3.2-3b.jsonl \
  --output runs/llama3.2-3b_judged.jsonl

rjbench-summarize runs/llama3.2-3b_judged.jsonl \
  --group-by attack,scenario,defense \
  --output runs/llama3.2-3b_summary.csv
```

## Repository Structure

```
├── src/rjbench/              # Portable CLI runner
├── reference_implementation/ # Original experimental scripts
│   ├── formal_v2/           # Experiments 1-3 implementation
│   └── supplementary_20261016/ # Multi-judge panel and refresh
├── artifacts/
│   ├── formal/raw/          # 2,900 experimental responses
│   └── supplementary/       # Panel scores, follow-ups, refresh
├── data/
│   ├── research/            # Study behavior sets and PAIR prompts
│   └── smoke/               # Minimal test data
├── docs/                    # Detailed guides and protocols
└── tests/                   # Unit and integration tests
```

## Experimental Records

| Dataset | Records | Description |
|---------|---------|-------------|
| Formal (Exp 1-3) | 2,900 | 5 targets × 4 attacks × 4 scenarios × 4 defenses + benign baselines |
| Follow-ups | 800 | PAIR format crossover (320) + Gemma cap sensitivity (480) |
| Wrapper refresh | 1,350 | 50 behaviors × 3 generators × 3 seeds × 3 targets |

All responses include three-judge panel scores (gpt-oss 20B, GLM-5.2, MiniMax-M3) with strict validation. See [`docs/SUPPLEMENTARY_20261016.md`](docs/SUPPLEMENTARY_20261016.md) for detailed mapping.

## Analysis

Reproduce statistical analyses:

```bash
# Install analysis dependencies
python3 -m pip install -e ".[analysis]"

# Run analysis scripts
cd reference_implementation/supplementary_20261016/scripts
python3 analyze_multijudge.py
python3 analyze_refresh.py
python3 refresh_generator_effects.py
```

Analysis outputs are pre-computed in `artifacts/supplementary/*/analysis/`.

## Citation

```bibtex
@mastersthesis{xu2026renewable,
  author = {Xu, ChengSheng Andy},
  title = {Renewable and Scenario-Aware Jailbreak Benchmarking for Language Models},
  school = {Monash University},
  year = {2026},
  month = {October}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Security

This repository contains research materials on adversarial prompts. See [SECURITY.md](SECURITY.md) for responsible use guidelines.
