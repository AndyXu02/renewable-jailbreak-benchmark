#!/usr/bin/env python3
"""
Statistical robustness analysis for Phase 2/3 main results.

1. Behavior-clustered bootstrap 95% CIs for overall and per-condition metrics.
2. Formal model x scenario interaction test for Phase 2 (permutation-based).

Behavior-clustered bootstrap: resample at the behavior level (with replacement)
to account for non-independence of records sharing the same behavior.

Usage:
    python bootstrap_ci.py                # Phase 2 + Phase 3, all models
    python bootstrap_ci.py --phase 2      # Phase 2 only
    python bootstrap_ci.py --phase 3      # Phase 3 only
    python bootstrap_ci.py --n-boot 5000  # change bootstrap iterations
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(SCRIPT_DIR), "runs")
RESULTS = os.path.join(os.path.dirname(SCRIPT_DIR), "results")

PHASE2_MODELS = [
    "deepseek-v4-flash",
    "qwen3-8b-nothink",
    "llama3.1-8b",
    "gemma4-12b-it",
    "gemma4-12b-obliterated",
]

PHASE3_MODELS = PHASE2_MODELS

SCENARIOS = ["S1_direct", "S2_multiturn", "S3_longcontext", "S4_agent"]
DEFENSES = ["D0_none", "D1_system_prompt", "D2_perturbation", "D3_rule_input", "D4_rule_output"]


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def get_score(r):
    s = r.get("score_5scale")
    if s is None:
        jl = r.get("judge_llm")
        if isinstance(jl, dict):
            s = jl.get("score")
    return s


def get_behavior_id(r):
    sid = r.get("sample_id", "")
    # Extract behavior prefix (e.g., jbb_0000, manual_harm_0000)
    return sid


def behavior_clustered_bootstrap(records, metric_fn, n_boot=10000, seed=42):
    """Bootstrap by resampling behaviors with replacement."""
    rng = np.random.RandomState(seed)

    # Group records by behavior
    by_behavior = defaultdict(list)
    for r in records:
        bid = get_behavior_id(r)
        by_behavior[bid].append(r)

    behaviors = sorted(by_behavior.keys())
    n_behaviors = len(behaviors)

    if n_behaviors < 2:
        return None

    # Observed metric
    observed = metric_fn(records)

    # Bootstrap
    boot_vals = []
    for _ in range(n_boot):
        sampled_behaviors = rng.choice(behaviors, size=n_behaviors, replace=True)
        boot_records = []
        for b in sampled_behaviors:
            boot_records.extend(by_behavior[b])
        boot_vals.append(metric_fn(boot_records))

    boot_vals = np.array(boot_vals)
    ci_low = np.percentile(boot_vals, 2.5)
    ci_high = np.percentile(boot_vals, 97.5)

    return {
        "observed": observed,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_behaviors": n_behaviors,
        "n_records": len(records),
    }


def mean_score(records):
    scores = [get_score(r) for r in records if get_score(r) is not None]
    return sum(scores) / len(scores) if scores else 0


def risk_at_3(records):
    scores = [get_score(r) for r in records if get_score(r) is not None]
    return sum(1 for s in scores if s >= 3) / len(scores) * 100 if scores else 0


def hcr_at_4(records):
    scores = [get_score(r) for r in records if get_score(r) is not None]
    return sum(1 for s in scores if s >= 4) / len(scores) * 100 if scores else 0


def analyze_phase2(n_boot=10000, dataset_tag="v2formal"):
    print("=" * 80)
    print("PHASE 2: BEHAVIOR-CLUSTERED BOOTSTRAP 95% CIs")
    print("=" * 80)

    all_results = {}

    for model in PHASE2_MODELS:
        path = os.path.join(RUNS, f"phase2_{dataset_tag}_{model}.jsonl")
        records = load_jsonl(path)
        if not records:
            print(f"  [SKIP] {model}: no data")
            continue

        print(f"\n--- {model} (N={len(records)}) ---")

        model_results = {}

        # Overall
        for metric_name, fn in [("Mean", mean_score), ("Risk@3", risk_at_3), ("HCR@4", hcr_at_4)]:
            r = behavior_clustered_bootstrap(records, fn, n_boot=n_boot)
            if r:
                print(f"  {metric_name:>8s}: {r['observed']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
                model_results[metric_name] = r

        # Per scenario
        print(f"  {'Scenario':<16s} {'Mean':>20s} {'Risk@3':>20s} {'HCR@4':>20s}")
        for sc in SCENARIOS:
            sc_records = [r for r in records if r.get("scenario") == sc]
            if not sc_records:
                continue
            parts = [f"  {sc:<16s}"]
            for metric_name, fn in [("Mean", mean_score), ("Risk@3", risk_at_3), ("HCR@4", hcr_at_4)]:
                r = behavior_clustered_bootstrap(sc_records, fn, n_boot=n_boot)
                if r:
                    s = f"{r['observed']:.2f} [{r['ci_low']:.2f},{r['ci_high']:.2f}]"
                    model_results[f"{sc}_{metric_name}"] = r
                else:
                    s = "N/A"
                parts.append(f"{s:>20s}")
            print(" ".join(parts))

        all_results[model] = model_results

    return all_results


def analyze_phase3(n_boot=10000, dataset_tag="v2formal"):
    print("\n" + "=" * 80)
    print("PHASE 3: BEHAVIOR-CLUSTERED BOOTSTRAP 95% CIs")
    print("=" * 80)

    all_results = {}

    for model in PHASE3_MODELS:
        path = os.path.join(RUNS, f"phase3_{dataset_tag}_{model}.jsonl")
        records = load_jsonl(path)
        if not records:
            print(f"  [SKIP] {model}: no data")
            continue

        print(f"\n--- {model} (N={len(records)}) ---")

        model_results = {}

        # Per defense
        print(f"  {'Defense':<18s} {'Mean':>20s} {'Risk@3':>20s} {'HCR@4':>20s}")
        for d in DEFENSES:
            d_records = [r for r in records if r.get("defense") == d]
            if not d_records:
                continue
            parts = [f"  {d:<18s}"]
            for metric_name, fn in [("Mean", mean_score), ("Risk@3", risk_at_3), ("HCR@4", hcr_at_4)]:
                r = behavior_clustered_bootstrap(d_records, fn, n_boot=n_boot)
                if r:
                    s = f"{r['observed']:.2f} [{r['ci_low']:.2f},{r['ci_high']:.2f}]"
                    model_results[f"{d}_{metric_name}"] = r
                else:
                    s = "N/A"
                parts.append(f"{s:>20s}")
            print(" ".join(parts))

        all_results[model] = model_results

    return all_results


def model_scenario_interaction_test(n_perm=10000, seed=42, dataset_tag="v2formal"):
    """
    Permutation test for model x scenario interaction in Phase 2.

    Rank-transforms scores, removes additive model and scenario effects, and
    permutes complete four-scenario model profiles within behavior × attack
    clusters. This preserves the repeated-measures scenario structure.

    H0: No model x scenario interaction (scenario effects are the same across models).
    """
    print("\n" + "=" * 80)
    print("PHASE 2: MODEL x SCENARIO INTERACTION TEST")
    print("=" * 80)

    # Load all Phase 2 data
    all_records = []
    for model in PHASE2_MODELS:
        path = os.path.join(RUNS, f"phase2_{dataset_tag}_{model}.jsonl")
        records = load_jsonl(path)
        for r in records:
            if get_score(r) is not None:
                all_records.append({
                    "model": model,
                    "scenario": r.get("scenario", ""),
                    "score": get_score(r),
                    "behavior": get_behavior_id(r),
                    "attack": r.get("attack", ""),
                })

    if not all_records:
        print("  No Phase 2 data found.")
        return None

    # Filter to scenarios present in all models
    models_present = sorted(set(r["model"] for r in all_records))
    scenarios_present = sorted(set(r["scenario"] for r in all_records))
    print(f"  Models: {models_present}")
    print(f"  Scenarios: {scenarios_present}")
    print(f"  Total records: {len(all_records)}")

    # Rank transform (average ranks for ties)
    scores = np.array([r["score"] for r in all_records], dtype=float)
    ranks = stats.rankdata(scores, method="average")

    # Assign factor levels
    model_levels = {m: i for i, m in enumerate(models_present)}
    scenario_levels = {s: i for i, s in enumerate(scenarios_present)}

    model_idx = np.array([model_levels[r["model"]] for r in all_records])
    scenario_idx = np.array([scenario_levels[r["scenario"]] for r in all_records])

    n_models = len(models_present)
    n_scenarios = len(scenarios_present)
    N = len(all_records)

    # Remove additive model and scenario main effects from the ranks. The
    # remaining cell means represent the model × scenario interaction.
    grand_mean = ranks.mean()
    model_means = np.array([ranks[model_idx == i].mean() for i in range(n_models)])
    scenario_means = np.array([ranks[scenario_idx == j].mean() for j in range(n_scenarios)])
    residual_ranks = ranks - model_means[model_idx] - scenario_means[scenario_idx] + grand_mean

    def compute_interaction_ss(values, m_idx, s_idx):
        """Compute SS of model × scenario means for additive-model residuals."""
        cell_means = np.zeros((n_models, n_scenarios))
        cell_counts = np.zeros((n_models, n_scenarios))
        for i in range(len(values)):
            cell_means[m_idx[i], s_idx[i]] += values[i]
            cell_counts[m_idx[i], s_idx[i]] += 1
        mask = cell_counts > 0
        cell_means[mask] /= cell_counts[mask]
        return float(np.sum(cell_counts[mask] * cell_means[mask] ** 2))

    observed_ss = compute_interaction_ss(residual_ranks, model_idx, scenario_idx)

    # Clustered profile permutation: for every behavior × attack cluster,
    # reassign each model's complete scenario profile as one indivisible unit.
    profile_indices = defaultdict(lambda: defaultdict(list))
    for idx, record in enumerate(all_records):
        cluster = (record["behavior"], record["attack"])
        profile_indices[cluster][model_idx[idx]].append(idx)
    incomplete = [
        cluster for cluster, profiles in profile_indices.items()
        if len(profiles) != n_models
        or any(len({scenario_idx[i] for i in idxs}) != n_scenarios for idxs in profiles.values())
    ]
    if incomplete:
        raise RuntimeError(
            f"Interaction test requires complete model × scenario profiles; "
            f"found {len(incomplete)} incomplete behavior × attack clusters"
        )

    rng = np.random.RandomState(seed)
    perm_ss = np.zeros(n_perm)
    for p in range(n_perm):
        model_idx_perm = model_idx.copy()
        for profiles in profile_indices.values():
            source_models = np.array(sorted(profiles))
            target_models = rng.permutation(source_models)
            for source, target in zip(source_models, target_models):
                model_idx_perm[profiles[source]] = target
        perm_ss[p] = compute_interaction_ss(
            residual_ranks, model_idx_perm, scenario_idx
        )

    p_value = (1 + np.sum(perm_ss >= observed_ss)) / (n_perm + 1)

    print(f"\n  Observed interaction SS: {observed_ss:.2f}")
    print(f"  Permutation p-value (n={n_perm}): {p_value:.4f}")
    print(f"  Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} at alpha=0.05")

    # Also report per-model scenario effect sizes (Kruskal-Wallis)
    print(f"\n  Per-model scenario effect (Kruskal-Wallis):")
    for m in models_present:
        m_records = [r for r in all_records if r["model"] == m]
        groups = []
        for s in scenarios_present:
            s_scores = [r["score"] for r in m_records if r["scenario"] == s]
            if len(s_scores) >= 2:
                groups.append(s_scores)
        if len(groups) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups)
            print(f"    {m:<25s}: H={kw_stat:.2f}, p={kw_p:.4f} {'*' if kw_p < 0.05 else ''}")

    return {
        "observed_ss": observed_ss,
        "p_value": p_value,
        "n_perm": n_perm,
        "models": models_present,
        "scenarios": scenarios_present,
        "n_records": N,
    }


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CIs and interaction tests")
    parser.add_argument("--phase", choices=["2", "3", "both"], default="both")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--dataset-tag", default="v2formal")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results as JSON to this path")
    args = parser.parse_args()

    results = {}

    if args.phase in ("2", "both"):
        results["phase2"] = analyze_phase2(n_boot=args.n_boot, dataset_tag=args.dataset_tag)
        interaction = model_scenario_interaction_test(
            n_perm=args.n_perm, dataset_tag=args.dataset_tag
        )
        if interaction:
            results["interaction_test"] = interaction

    if args.phase in ("3", "both"):
        results["phase3"] = analyze_phase3(n_boot=args.n_boot, dataset_tag=args.dataset_tag)

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(RESULTS, args.output)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        with open(out_path, "w") as f:
            json.dump(convert(results), f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
