#!/usr/bin/env python3
"""Generate one literal-placeholder wrapper for each behavior and seed.

Protocol: docs/refresh_protocol_20261016.md. No target feedback, retries for
format failures, or selection by attack score. Transport errors are retried;
invalid, refused, truncated, or wrong-version proposals use the frozen fallback
and remain separately marked.
"""

import argparse
import datetime as dt
import json
import os
import re
import time
from pathlib import Path

import requests

from config import HARMFUL_BEHAVIORS_V2, RUNS_DIR
from generate_model_attacks import FALLBACK_STRATEGIES, GENERATOR_SYSTEM, GENERATOR_USER_TEMPLATE
from multijudge_run import ROOT, load_env
from utils import load_jsonl

OUT = ROOT / "results/supplementary_20261016_refresh"
GENERATORS = {
    "gemma3-4b": {
        "backend": "ollama", "request_model": "gemma3:4b",
        "served_allowlist": {"gemma3:4b"},
    },
    "kimi-k3": {
        "backend": "relay", "request_model": "kimi-k3",
        "served_allowlist": {"kimi-k3", "Kimi-K3", "moonshotai/kimi-k3", "moonshotai/Kimi-K3"},
    },
    "mimo-v2.6-flash": {
        "backend": "relay", "request_model": "mimo-v2.6-flash",
        "served_allowlist": {"mimo-v2.6-flash", "MiMo-V2.6-Flash", "mimo/mimo-v2.6-flash"},
    },
}
SEEDS = (42, 43, 44)


def messages(behavior, category):
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": GENERATOR_USER_TEMPLATE.format(
            n_variants=1, behavior=behavior, category=category or "unknown")},
    ]


def parse_template(raw, behavior):
    text = re.sub(r"<think>.*?</think>\s*", "", raw or "", flags=re.DOTALL).strip()
    if "<think>" in text:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    variants = data.get("variants") if isinstance(data, dict) else None
    if not isinstance(variants, list) or len(variants) != 1 or not isinstance(variants[0], dict):
        return None
    template = str(variants[0].get("prompt_template") or "").strip()
    if template.count("<<BEHAVIOR>>") != 1 or behavior in template:
        return None
    return template


def call_local(generator_model, msgs, seed, max_tokens):
    payload = {
        "model": generator_model, "messages": msgs, "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens, "seed": seed},
    }
    response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    data = response.json()
    msg = data.get("message") or {}
    return msg.get("content") or "", data.get("model"), data.get("done_reason")


def call_relay(generator_model, msgs, seed, max_tokens, api_base, api_key, retries=3):
    url = api_base.rstrip("/")
    url = url if url.endswith("/chat/completions") else url + "/chat/completions"
    payload = {"model": generator_model, "messages": msgs, "temperature": 0.7,
               "max_tokens": max_tokens, "seed": seed}
    history = []
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers={"Authorization": f"Bearer {api_key}"},
                                     json=payload, timeout=240)
            response.raise_for_status()
            data = response.json()
            choice = (data.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            raw = msg.get("content") or ""
            result = (raw, data.get("model"), choice.get("finish_reason"))
            history.append({"attempt": attempt + 1, "http": response.status_code, "error": None})
            return (*result, history)
        except (requests.RequestException, ValueError, KeyError) as exc:
            history.append({"attempt": attempt + 1, "error": type(exc).__name__})
            if attempt < retries:
                time.sleep(min(2 ** attempt, 15))
    return "", None, None, history


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generator", required=True, choices=GENERATORS)
    p.add_argument("--seed", required=True, type=int, choices=SEEDS)
    p.add_argument("--max-tokens", type=int, default=None)
    args = p.parse_args()

    load_env(ROOT / ".env")
    spec = GENERATORS[args.generator]
    max_tokens = args.max_tokens or (512 if spec["backend"] == "ollama" else 4096)
    api_base, api_key = os.environ.get("GLM_API_BASE"), os.environ.get("GLM_API_KEY")
    if spec["backend"] == "relay" and (not api_base or not api_key):
        raise SystemExit("Missing GLM_API_BASE / GLM_API_KEY in benchmark_project/.env")

    OUT.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"refresh_20261016_{args.generator}_seed{args.seed}.jsonl"
    samples = [r for r in load_jsonl(HARMFUL_BEHAVIORS_V2) if r.get("type") == "harmful"]
    if len(samples) != 50:
        raise SystemExit(f"Expected 50 frozen harmful behaviors, found {len(samples)}")
    existing = {r["sample_id"] for r in load_jsonl(path)} if path.exists() else set()
    print(f"generator={args.generator} seed={args.seed} rows={len(samples)} done={len(existing)}", flush=True)

    for index, sample in enumerate(samples, 1):
        sample_id, behavior = sample["id"], sample["behavior"]
        if sample_id in existing:
            continue
        msgs = messages(behavior, sample.get("category"))
        attempts = []
        try:
            if spec["backend"] == "ollama":
                raw, served, finish = call_local(spec["request_model"], msgs, args.seed, max_tokens)
            else:
                raw, served, finish, attempts = call_relay(
                    spec["request_model"], msgs, args.seed, max_tokens, api_base, api_key)
        except (requests.RequestException, ValueError) as exc:
            raw, served, finish = "", None, None
            attempts.append({"error": type(exc).__name__})

        correct_version = served in spec["served_allowlist"]
        valid_finish = finish == "stop"
        template = parse_template(raw, behavior) if correct_version and valid_finish else None
        if template:
            strategy, fallback_reason = "model_generated_literal_wrapper", None
            prompt = template.replace("<<BEHAVIOR>>", behavior)
        elif not correct_version:
            strategy, fallback_reason, prompt = "quarantined_model_version", "unexpected_served_model", ""
        else:
            strategy = "research_framing"
            fallback_reason = "invalid_json_placeholder_or_finish"
            prompt = FALLBACK_STRATEGIES[0][1].format(behavior=behavior)
            template = FALLBACK_STRATEGIES[0][1].replace("{behavior}", "<<BEHAVIOR>>")

        record = {
            "sample_id": sample_id, "behavior_id": sample_id, "generator": args.generator,
            "generator_model_requested": spec["request_model"], "served_model": served,
            "seed_requested": args.seed, "seed_guaranteed_by_provider": spec["backend"] == "ollama",
            "temperature": 0.7, "max_tokens": max_tokens, "finish_reason": finish,
            "category": sample.get("category"), "behavior": behavior, "kind": "harmful",
            "first_pass_valid": template is not None and fallback_reason is None,
            "fallback_reason": fallback_reason, "strategy": strategy,
            "prompt_template": template, "prompt": prompt,
            "raw_final": raw, "transport_attempts": attempts,
            "prompt_sha256": __import__("hashlib").sha256(
                json.dumps(msgs, ensure_ascii=False).encode()).hexdigest(),
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  [{index}/50] {sample_id} served={served} finish={finish} "
              f"valid={record['first_pass_valid']} fallback={fallback_reason}", flush=True)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
