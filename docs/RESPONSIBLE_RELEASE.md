# Responsible release

This repository supports defensive red teaming and includes jailbreak prompt
assets. The release contains complete formal target-model responses and judge
fields for response-level reproduction, alongside compact score-only matrices.
The supplementary layer adds three-judge scores, follow-up responses, and the
wrapper-refresh assets and responses. The human reference items and individual
human labels are calibration-only and are not released.

The harmful behavior set describes malicious objectives but was designed not to
seed executable code, material quantities, exploit parameters, or procedural
instructions. It was model-authored and automatically validated; it is not a
human-validated taxonomy of harm.

When redistributing or extending the raw responses:

1. Review university, supervisor, ethics, model-license, and provider terms.
2. Never add credentials or private provider tokens.
3. Preserve model, backend, decoding, and judge provenance.
4. Document any filtering or transformation of response text.
5. Publish checksums and a versioned data card.

Use this project only on systems and models you are authorized to evaluate.
