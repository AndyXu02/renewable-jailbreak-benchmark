# Data layout

- `smoke/` contains two small records for testing the software. They are not
  research results.
- `research/harmful_behaviors_v2.jsonl` contains 50 model-authored harmful
  objectives without seeded operational instructions.
- `research/benign_prompts_30.jsonl` contains benign but safety-sensitive controls.
- `research/pair_prompts_harmful_v2.jsonl` contains saved PAIR prompts. Qwen3-8B
  no-think was the optimization target, so its PAIR result is in-sample.
- `research/a6_wrappers_harmful_v2.jsonl` contains renewable wrappers generated
  under the literal-placeholder protocol. Ten of fifty are deterministic fallbacks.

Compact score matrices omit response text for fast analysis. Full formal model
responses, judge fields, PAIR rounds, and A6 generator records are also included
under `artifacts/formal/raw/` and `research/raw/`.
