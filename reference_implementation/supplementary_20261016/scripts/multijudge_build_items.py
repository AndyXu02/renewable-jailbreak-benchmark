#!/usr/bin/env python3
"""Collect every frozen scored response into one item file for multi-judge re-scoring.

Read-only with respect to frozen artifacts. Writes
results/supplementary_20261016_multijudge/items.jsonl with one record per
scored response (Experiments 1-3, matched benign control) plus the 120-item
human-consensus set used to anchor every judge.
"""

import csv
import json
import os

from bootstrap_ci import PHASE2_MODELS, RESULTS, RUNS, load_jsonl

OUT = os.path.join(RESULTS, "supplementary_20261016_multijudge")
EXP2_MODELS = ["deepseek-v4-flash", "qwen3-8b-nothink"]
EXP3_MODELS = ["deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink"]
CONSENSUS = os.path.join(os.path.dirname(RESULTS), "human_audit", "consensus", "human_consensus_120.csv")


def behavior_text(r):
    meta = r.get("metadata") or {}
    return str(meta.get("judge_behavior") or r.get("behavior") or meta.get("behavior") or "")


def primary_score(r, field="score_5scale"):
    value = r.get(field)
    if value is None and isinstance(r.get("judge_llm"), dict):
        value = r["judge_llm"].get("score")
    return int(value)


def item(experiment, kind, r, model, attack, behavior_id, primary, **extra):
    response = str(r.get("response") or "")
    return {
        "item_id": f"{experiment}|{model}|{attack}|{r.get('scenario')}|{r.get('defense') or 'D0_none'}|{behavior_id}",
        "experiment": experiment,
        "kind": kind,
        "model": model,
        "attack": attack,
        "scenario": r.get("scenario"),
        "defense": r.get("defense") or "D0_none",
        "behavior_id": behavior_id,
        "category": r.get("category"),
        "behavior": behavior_text(r),
        "response": response,
        "blocked": response.startswith("[BLOCKED"),
        "primary_score": primary,
        **extra,
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    items = []
    for model in PHASE2_MODELS:
        for r in load_jsonl(os.path.join(RUNS, f"phase2_v2formal_{model}.jsonl")):
            items.append(item("exp1", "harmful", r, model, r["attack"], r["sample_id"], primary_score(r)))
    for model in EXP2_MODELS:
        for r in load_jsonl(os.path.join(RUNS, f"phase3_v2formal_{model}.jsonl")):
            items.append(item("exp2", "harmful", r, model, r["attack"], r["sample_id"], primary_score(r)))
        for r in load_jsonl(os.path.join(RUNS, f"benign_v2formal_{model}_responses.gptoss_utility.jsonl")):
            items.append(item("benign", "benign", r, model, "none", r["sample_id"], int(r["utility_score"])))
    for model in EXP3_MODELS:
        sources = [
            (f"matched_v2_{model}_a0_a2_responses.gptoss.jsonl", None),
            (f"matched_v2_{model}_pair_responses.gptoss.jsonl", "A1_pair"),
            (f"phase4_a6_v2formal_{model}_responses.gptoss.jsonl", "A6_model_generated"),
        ]
        for fname, forced_attack in sources:
            for r in load_jsonl(os.path.join(RUNS, fname)):
                attack = forced_attack or r["attack"]
                bid = "harmful_v2_" + r["sample_id"].split("harmful_v2_")[1][:4]
                items.append(item("exp3", "harmful", r, model, attack, bid, primary_score(r)))
    with open(CONSENSUS, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            r = {"response": row["response"], "behavior": row["behavior"], "scenario": row["scenario"],
                 "defense": row["defense"], "category": row["category"]}
            items.append(item("human120", "harmful", r, row["model"], row["attack"], row["audit_id"], None,
                              human_score=int(round(float(row["consensus_score"]))),
                              human_hcr3=int(row["consensus_hcr3"]), human_hcr4=int(row["consensus_hcr4"])))

    ids = [i["item_id"] for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids"
    experiment_items = [i for i in items if i["experiment"] != "human120"]
    human_items = [i for i in items if i["experiment"] == "human120"]
    with open(os.path.join(OUT, "items.jsonl"), "w", encoding="utf-8") as f:
        for i in experiment_items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    # Keep human labels in a separate ignored analysis artifact, never in the
    # general experiment item matrix that might be included in a data release.
    with open(os.path.join(OUT, "human120_items.jsonl"), "w", encoding="utf-8") as f:
        for i in human_items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    counts = {}
    for i in experiment_items:
        counts[i["experiment"]] = counts.get(i["experiment"], 0) + 1
    print(json.dumps({"total_experiment_items": len(experiment_items), "human_reference_items": len(human_items),
                      "by_experiment": counts,
                      "blocked": sum(i["blocked"] for i in items),
                      "judge_calls_needed": sum(not i["blocked"] and i["response"].strip() != "" for i in items)}, indent=2))


if __name__ == "__main__":
    main()
