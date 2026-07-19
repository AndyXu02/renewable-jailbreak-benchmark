# Reproduction levels

## Level 1: verify the release

Run `rjbench-verify`. This verifies SHA-256 hashes and record counts for the
public data, score matrices, and result tables. No model is required.

## Level 2: reproduce descriptive tables

Run `rjbench-summarize` on response-redacted matrices. This reproduces means,
Risk@3, and HCR@4 without exposing the original model completions.

The original inferential analyses used NumPy and SciPy. Install them with:

```bash
python -m pip install -e ".[analysis]"
```

The exact analysis scripts are in `reference_implementation/formal_v2/scripts/`.
Their original path expectations may need to be recreated in a temporary
workspace; do not edit them and then describe the result as an exact code-hash
replication.

## Level 3: verify the executable pipeline

Use the mock commands in the main README. These cover the complete software
path but do not make scientific claims about model safety.

## Level 4: evaluate a new model

Use `rjbench-run`, then score the responses with a separate judge through
`rjbench-judge`. Record the exact model identifier, model digest or revision,
backend version, prompt data hash, temperature, seed, output cap, and date.

For a fair comparison with the thesis, keep all factors fixed except the new
target model. If the model requires different output caps or chat formatting,
report the deviation rather than silently treating the result as directly
comparable.

## Level 5: full historical regeneration

Full regeneration requires the original model versions, provider access,
Ollama, Apple-Silicon MLX for the two Gemma artifacts, and substantial compute.
Because hosted models can change, exact response-level regeneration is not
guaranteed. The frozen response-redacted matrices and manifests support result
verification even when a historical endpoint is no longer available.

