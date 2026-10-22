# Generated-Wrapper Refresh Experiment: Frozen Protocol

Date: 2026-10-16. This protocol is fixed before inspecting new target responses.

## Question and scope

Can the literal-placeholder pipeline regenerate usable, diverse static attack
assets across generators and random seeds, and do measured risks remain stable
across refresh batches on held-out target families?

Three independently generated batches are NOT evidence of long-term resistance
to safety training or benchmark contamination. No model is fine-tuned or
exposed to old wrappers in this experiment. Such saturation claims remain out
of scope; an effective safety system prompt is not a substitute for training.

## Generation matrix

- The same frozen 50 English harmful objectives used in Experiment 3.
- Generators: local Gemma 3 4B (Ollama `gemma3:4b`), relay Kimi K3
  (`kimi-k3`), and relay MiMo V2.6 Flash (`mimo-v2.6-flash`).
- Requested seeds 42, 43, 44; temperature 0.7; one proposal per behavior,
  generator, and seed: 3 x 3 x 50 = 450 assets.
- Reuse `GENERATOR_SYSTEM` and `GENERATOR_USER_TEMPLATE` from the original
  literal-wrapper generator, requesting one JSON variant. A template must
  contain `<<BEHAVIOR>>` exactly once and must not repeat the full seed.
- The original objective is inserted by code, never rewritten by the generator.
  Placeholder validation does not certify that surrounding text contains no
  unsafe semantic additions; do not claim that it does.
- Before asset freeze, a versioned conservative phrase lint flags explicit
  requests to add working code, procedural steps, or detailed technical/
  operational instructions beyond the seed. A flagged proposal uses the fixed
  fallback. This deterministic safety screen does not certify semantic safety;
  report schema validity and lint fallback separately.
- Local output cap: 512 tokens. Relay cap: 4,096 tokens, including any provider
  reasoning. Record final-channel output and finish reason. Never extract an
  asset from a hidden reasoning channel.
- No target feedback, selection by attack success, best-of-N, or semantic repair.
  One parseable final proposal per cell. Invalid format, refusal, or a truncated
  generation uses the same deterministic research-framing fallback as the
  original pipeline; report first-pass and fallback strata separately.
- Retry only transport/service errors, not valid but unsuccessful proposals.
  Unexpected served-model versions are quarantined rather than relabeled.
- Provider seed support is not assumed: report requested seeds, returned model
  identifiers, and exact assets; do not promise bitwise API reproducibility.

## Stages and budgets

1. Generate 450 proposals; save every raw final output, settings, input hashes,
   served model, template validity, fallback reason, and materialized prompt.
2. Evaluate all assets in Single-turn, no-defense delivery on DeepSeek V4 Flash,
   Llama 3.1 8B, and Qwen3 8B no-think. These targets do not supply generation
   feedback: 450 x 3 = 1,350 final target responses. Target temperature 0,
   requested seed 42, final cap 1,024. Preserve chat templates and backends.
3. Score with the repaired final-only gpt-oss / GLM / MiniMax panel. Every scored
   record must have a supported final answer and approved served-model identity.
   Budget: up to 4,050 nonblocked judge calls, plus logged transport retries.
   Do not start automatic scoring if the preceding judge-repair audit fails.
4. Reuse the existing matched 50-behavior Direct, fixed continuation-Persona,
   PAIR, and original generated-wrapper baselines only as historical anchors.
   Runs occurred at different times; do not claim a time-controlled superiority
   test against the historical responses.

## Outcomes and analysis

- Pipeline: JSON/template pass rate, fallback fraction, proposal and prompt
  lengths, exact duplicates; report every generator and requested seed.
- Lexical diversity: normalized exact-duplicate rate, distinct word bi/trigrams,
  pairwise word-trigram Jaccard, and self-BLEU-4 with documented smoothing.
  Report overall and model-written-only values. These are lexical, not semantic
  embedding distances, and do not establish behavioral coverage.
- Risk: mean score, Risk@3, HCR@4 by generator, seed, and target; both end-to-end
  and valid-template strata. Report valid counts, never discard fallback rows.
- Refresh consistency: paired differences between seeds within generator and
  target; confidence intervals resample the 50 behaviors jointly across seeds.
  Seeds are repeats, not 150 independent behaviors.
- Target-specific effects stay separate; pooled summaries must state model
  weights. Correct pairwise p-values within each target/outcome family.
- Freeze every asset before target evaluation; pin it and the protocol by
  SHA-256. Publish the audit status separately from numerical completeness.

## Resource scheduling

Run only one Ollama model at a time (generator, targets, then judge). Relay
generation may run in parallel. Resume by generator/seed/behavior ID. Do not
change prompts or seeds after observing target success rates.

## Pre-target parsing amendment (2026-10-16)

The first 50 local Gemma 3 4B proposals were complete JSON objects wrapped in a
single outer Markdown code fence. No target responses or success rates had been
observed. For all generators/seeds, the asset freezer will now accept either
plain JSON or exactly one outer ```json ... ``` fence, then apply the identical
strict JSON-schema and one-placeholder checks. It records raw first-pass
validity separately from normalization recovery; no interior text is rewritten
and no target outcome is used to select assets. On inspection of these same
pre-target outputs, the generator sometimes appended explicit requests for
working scripts or technical procedures despite the frozen instruction not to
add such material. Before any target response, every generator/seed batch will
therefore receive the same frozen phrase lint described above; flagged rows use
the fixed fallback, while raw candidate text and flag reasons remain in the
local audit record. This is a conservative safety-screen amendment, not
performance-based selection. Original generator outputs remain immutable;
target evaluation reads only the derived, hash-frozen asset file.

## Target-output metadata note (after DeepSeek calls began)

The target wrapper is being instrumented to retain provider-returned model ID,
finish reason, usage, and response length. All three target processes had
already imported the prior shared adapter before this code change; it retained
response text and requested settings but did not preserve provider-returned
finish reason/model ID for any target. Do not infer that these 1,350 responses
all ended before the token cap or that the provider served an immutable weight
hash. Report these fields as unavailable for all targets. No prompt, cap,
target, or seed changes as a result of this note.
