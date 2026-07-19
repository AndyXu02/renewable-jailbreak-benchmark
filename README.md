# Renewable & Scenario-Aware Jailbreak Benchmark

[中文版 README](README_CN.md)

A reproducible Python benchmark for testing how language-model safety changes
across attacks, deployment scenarios, defenses, benign utility controls, and
automated evaluators.

This public repository has two deliberately separate layers:

- `reference_implementation/formal_v2/` preserves the scripts used for the thesis experiments.
- `src/rjbench/` is a small portable runner for Linux, macOS, and Windows. It supports any Ollama model, an OpenAI-compatible endpoint, or a deterministic mock backend.

The portable layer does not change the frozen thesis results. It makes the same
benchmark concepts easier to test on other hardware and newer models.

## What was evaluated

- 50 model-authored harmful objectives across five categories.
- 5 target models and 4 prompt-delivery scenarios.
- Direct, persona, target-optimized PAIR, and renewable A6 attacks.
- A safety system prompt, character perturbation, and rule-based input/output filters.
- Matched benign prompts to measure utility and over-refusal.
- Behavior-clustered or paired statistics and evaluator-sensitivity checks.

The formal route contains 2,900 scored experiment records, plus a separate
100-record second-judge sensitivity sample. This release includes both the full
3,000-record raw matrices and compact response-redacted score matrices.

## 60-second integrity check

Only Python 3.10+ is required. The integrity check does not download a model or
call an API.

```bash
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install -e .
rjbench-verify
python -m unittest discover -s tests -v
```

Expected result:

```text
Release verification passed: ... files; 3000 raw + 3000 compact formal records
```

## Two-minute smoke test without a model

This checks attack/scenario/defense composition, output writing, judging, and
summary generation with deterministic mock backends.

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

## Test any Ollama model

Install and start [Ollama](https://ollama.com/), then pull a model available on
your machine. The model name is not hard-coded in this project.

```bash
ollama pull llama3.2:3b

rjbench-run \
  --backend ollama \
  --model llama3.2:3b \
  --data data/research/harmful_behaviors_v2.jsonl \
  --max-samples 5 \
  --attacks A0_direct,A2_persona,A1_pair \
  --pair-file data/research/pair_prompts_harmful_v2.jsonl \
  --scenarios S1_direct,S2_multiturn,S3_longcontext,S4_agent \
  --defenses D0_none,D1_system_prompt \
  --max-tokens 512 \
  --output runs/llama3.2-3b.jsonl
```

For Qwen and other reasoning models, use `--think off`, `--think low`, or the
setting accepted by your Ollama model. The default is `--think auto`, which does
not send an explicit thinking option.

Judge the new responses with a separate model:

```bash
ollama pull gpt-oss:20b

rjbench-judge \
  --backend ollama \
  --model gpt-oss:20b \
  --think low \
  --input runs/llama3.2-3b.jsonl \
  --output runs/llama3.2-3b.judged.jsonl
```

Do not use the target model as its own judge when making comparative claims.

## Use an OpenAI-compatible endpoint

The runner works with providers or local servers that implement
`POST /chat/completions`.

```bash
export MODEL_API_KEY="your-key"

rjbench-run \
  --backend openai \
  --model your-model-id \
  --api-base https://provider.example/v1 \
  --api-key-env MODEL_API_KEY \
  --data data/smoke/smoke_prompts.jsonl \
  --attacks A0_direct \
  --scenarios S1_direct \
  --defenses D0_none \
  --output runs/provider-smoke.jsonl
```

Never put a real API key in a command committed to Git, a notebook, a config
file, or an issue. `.env.example` contains placeholders only.

## Recalculate a published result table

The compact scored matrices retain the experimental cell and score needed for
descriptive analysis. Full records, including model responses and judge fields,
are under `artifacts/formal/raw/`.

```bash
rjbench-summarize artifacts/formal/scored/phase2_*.jsonl \
  --group-by model_id \
  --output runs/recalculated_phase2_by_model.csv
```

Compare the output with:

```text
artifacts/formal/tables/phase2_by_model.csv
```

Benign utility and over-refusal can be recalculated with:

```bash
rjbench-summarize artifacts/formal/scored/benign_*.jsonl \
  --group-by model_id,defense \
  --score-field utility_score \
  --utility \
  --output runs/recalculated_benign_by_defense.csv
```

Inferential outputs and the frozen thesis manifest are available under
`artifacts/formal/tables/`. The response-redacted public package has its own
hash manifest at `artifacts/release_manifest.json`.

## Repository map

```text
src/rjbench/                         portable, cross-platform runner
reference_implementation/formal_v2/ exact thesis-era scripts
data/research/                       behavior set, controls, PAIR and A6 assets
data/smoke/                          safe, tiny pipeline check
artifacts/formal/scored/             compact formal score matrices
artifacts/formal/raw/                full formal response and judge records
artifacts/formal/tables/             published descriptive/inferential outputs
docs/                                methodology, architecture and release notes
tests/                               no-model deterministic tests
```

## Important interpretation boundaries

- Qwen3-8B no-think was the PAIR optimization target; its PAIR result is in-sample.
- S4 is a textual agent-lite prompt-injection condition, not real tool execution.
- D3 and D4 are fixed keyword rules, not learned guard models.
- The A6 route used one generator and seed; 10 of 50 wrappers are deterministic fallbacks.
- Backends and output caps varied in the original experiments, so unconditional cross-model causal rankings are not claimed.
- Automated judges are measurements, not ground truth. Absolute HCR@4 values were judge-dependent.

## Responsible use

This project is intended for authorized AI-safety evaluation and reproducible
research. The release includes adversarial prompts and complete model outputs.
See
[`docs/RESPONSIBLE_RELEASE.md`](docs/RESPONSIBLE_RELEASE.md).

## Citation and license

Software is released under the MIT License. Dataset and model-output reuse may
also be subject to the terms of the model providers used to produce or evaluate
the artifacts. Use `CITATION.cff` when citing this repository.
