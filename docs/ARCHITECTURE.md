# Architecture

## Portable path

```text
JSONL behavior set
  -> attack transform
  -> scenario message builder
  -> defense wrapper
  -> mock / Ollama / OpenAI-compatible backend
  -> JSONL response records
  -> independent judge backend
  -> grouped summary
```

The portable path uses only the Python standard library. Ollama and remote API
calls use HTTP, so Linux, Windows, Intel macOS, and Apple Silicon can run the
same code. It does not load PyTorch or MLX models directly.

## Formal reference path

`reference_implementation/formal_v2/scripts/` contains copies of the scripts
used by the thesis route. They are retained for provenance and code inspection.
Some of those scripts assume the original directory layout, Ollama models, or
Apple-Silicon MLX. They are not the recommended entry point for a new machine.

## Why both paths exist

Rewriting the formal scripts would make it difficult to tell whether published
results came from the released code. Keeping exact reference copies preserves
that evidence. The portable layer separately improves usability for new models
and platforms without changing historical claims.

