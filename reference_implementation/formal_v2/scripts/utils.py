"""
Shared utilities for the benchmark pipeline.
"""

import json
import hashlib
import datetime
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: list[dict], path: Path) -> None:
    """Save a list of dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def stratified_sample(items: list[dict], n: int, category_key: str = "category") -> list[dict]:
    """Select records deterministically in category round-robin order."""
    if n >= len(items):
        return list(items)
    by_category = defaultdict(list)
    for item in items:
        by_category[item.get(category_key, "unknown")].append(item)
    selected = []
    categories = sorted(by_category)
    index = 0
    while len(selected) < n and any(index < len(by_category[c]) for c in categories):
        for category in categories:
            if index < len(by_category[category]):
                selected.append(by_category[category][index])
                if len(selected) == n:
                    break
        index += 1
    return selected


def generate_run_id(model: str, phase: str, timestamp: str = None) -> str:
    """Generate a unique run identifier."""
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{phase}_{model}_{timestamp}"


def hash_prompt(prompt: str) -> str:
    """Create a short hash of a prompt for deduplication."""
    return hashlib.md5(prompt.encode()).hexdigest()[:12]


def format_messages(system: str, messages: list[dict]) -> list[dict]:
    """Format into standard chat message format.
    
    Args:
        system: System prompt string
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts
    
    Returns:
        Full message list with system prompt prepended.
    """
    formatted = []
    if system:
        formatted.append({"role": "system", "content": system})
    formatted.extend(messages)
    return formatted


def create_experiment_record(
    sample_id: str,
    model_id: str,
    attack: str,
    scenario: str,
    defense: str,
    prompt: Any,
    response: str,
    metadata: dict = None,
) -> dict:
    """Create a standardized experiment output record."""
    record = {
        "sample_id": sample_id,
        "model_id": model_id,
        "attack": attack,
        "scenario": scenario,
        "defense": defense,
        "prompt": prompt,
        "response": response,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    if metadata:
        record["metadata"] = metadata
    return record
