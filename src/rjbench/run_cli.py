from __future__ import annotations

import argparse
from pathlib import Path

from .backends import create_backend
from .io import load_jsonl, split_csv
from .pipeline import run_matrix, load_pair_map


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a portable jailbreak benchmark matrix")
    parser.add_argument("--backend", choices=["mock", "ollama", "openai"], default="mock")
    parser.add_argument("--model", default="mock-safe-model", help="Any Ollama or API model identifier")
    parser.add_argument("--api-base")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--think", choices=["auto", "off", "on", "low", "medium", "high"], default="auto")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--pair-file", type=Path)
    parser.add_argument("--attacks", default="A0_direct")
    parser.add_argument("--scenarios", default="S1_direct")
    parser.add_argument("--defenses", default="D0_none")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.data)
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    backend = create_backend(
        args.backend,
        args.model,
        api_base=args.api_base,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        think=args.think,
    )
    results = run_matrix(
        backend=backend,
        model_id=args.model,
        rows=rows,
        attacks=split_csv(args.attacks),
        scenarios=split_csv(args.scenarios),
        defenses=split_csv(args.defenses),
        output=args.output,
        decoding={"temperature": args.temperature, "max_tokens": args.max_tokens, "seed": args.seed},
        pair_map=load_pair_map(args.pair_file),
        resume=not args.no_resume,
    )
    print(f"Completed {len(results)} unique records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

