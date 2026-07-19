# Formal provenance

The thesis route is identified as `harmful_v2 / v2formal`.

- Phase 2: 1,200 records.
- Phase 3 harmful: 800 records.
- Matched benign utility: 300 records.
- Matched A0/A2/PAIR/A6 comparison: 600 records.
- Second-judge sensitivity: 100 additional records.

The original project completion audit passed 36 checks and the frozen manifest
pinned 59 source, input, run, and result artifacts. The original manifest is
included in `artifacts/formal/tables/formal_results_manifest.json` for historical
reference. The public release has a separate manifest covering both the full raw
matrices and the compact score-only representations.

The portable runner is a post-thesis usability layer. Results produced through
it must not be presented as byte-identical regenerations of the formal route.
