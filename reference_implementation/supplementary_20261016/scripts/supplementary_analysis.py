#!/usr/bin/env python3
"""Read-only supplementary analyses on the frozen v2formal scores.

No inference or re-judging is performed and no frozen artifact is modified.
Outputs are written to results/supplementary_20260926/.

1. Model x scenario interaction test variants (same statistic and clustered
   profile permutation as bootstrap_ci.model_scenario_interaction_test):
   all five models (reproduction), aligned models only, PAIR only, and
   static attacks only.
2. Attack x scenario HCR@4 / Risk@3 with behavior-clustered bootstrap CIs.
3. Within-model scenario effect under PAIR (Cochran's Q on paired HCR@4).
4. Heuristic truncation audit for the 256-token Gemma runs.
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

from bootstrap_ci import PHASE2_MODELS, RUNS, RESULTS, SCENARIOS, load_jsonl

OUT = os.path.join(RESULTS, "supplementary_20260926")
ALIGNED = ["deepseek-v4-flash", "qwen3-8b-nothink", "llama3.1-8b", "gemma4-12b-it"]
ATTACKS = ["A0_direct", "A2_persona", "A1_pair"]
N_PERM = 100000
N_BOOT = 10000


def load_phase2():
    records = []
    for model in PHASE2_MODELS:
        for r in load_jsonl(os.path.join(RUNS, f"phase2_v2formal_{model}.jsonl")):
            records.append({
                "model": model,
                "scenario": r["scenario"],
                "attack": r["attack"],
                "behavior": r["sample_id"],
                "category": r["category"],
                "score": int(r["score_5scale"]),
                "response": r.get("response") or "",
                "response_length": int(r.get("response_length") or len(r.get("response") or "")),
                "max_tokens": r.get("generation_max_tokens"),
            })
    return records


def interaction_test(records, models, attacks, n_perm=N_PERM, seed=42):
    recs = [r for r in records if r["model"] in models and r["attack"] in attacks]
    models_present = sorted({r["model"] for r in recs})
    scenarios_present = sorted({r["scenario"] for r in recs})
    ranks = stats.rankdata(np.array([r["score"] for r in recs], dtype=float), method="average")
    m_idx = np.array([models_present.index(r["model"]) for r in recs])
    s_idx = np.array([scenarios_present.index(r["scenario"]) for r in recs])
    n_m, n_s = len(models_present), len(scenarios_present)
    model_means = np.array([ranks[m_idx == i].mean() for i in range(n_m)])
    scen_means = np.array([ranks[s_idx == j].mean() for j in range(n_s)])
    resid = ranks - model_means[m_idx] - scen_means[s_idx] + ranks.mean()

    def ss(mi):
        sums = np.zeros((n_m, n_s))
        counts = np.zeros((n_m, n_s))
        np.add.at(sums, (mi, s_idx), resid)
        np.add.at(counts, (mi, s_idx), 1)
        mask = counts > 0
        means = np.zeros_like(sums)
        means[mask] = sums[mask] / counts[mask]
        return float(np.sum(counts[mask] * means[mask] ** 2))

    observed = ss(m_idx)
    profiles = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(recs):
        profiles[(r["behavior"], r["attack"])][m_idx[i]].append(i)
    profile_list = [(np.array(sorted(p)), {k: np.array(v) for k, v in p.items()})
                    for p in profiles.values()]
    rng = np.random.RandomState(seed)
    perm = np.zeros(n_perm)
    for p in range(n_perm):
        mi = m_idx.copy()
        for source, idxs in profile_list:
            target = rng.permutation(source)
            for s, t in zip(source, target):
                mi[idxs[s]] = t
        perm[p] = ss(mi)
    return {
        "models": models_present,
        "attacks": sorted(set(attacks)),
        "n_records": len(recs),
        "n_clusters": len(profile_list),
        "observed_ss": observed,
        "p_value": float((1 + np.sum(perm >= observed)) / (n_perm + 1)),
        "n_perm": n_perm,
    }


def clustered_ci(recs, fn, n_boot=N_BOOT, seed=42):
    by_b = defaultdict(list)
    for r in recs:
        by_b[r["behavior"]].append(r)
    behaviors = sorted(by_b)
    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(len(behaviors), size=len(behaviors), replace=True)
        boots.append(fn([x for i in sample for x in by_b[behaviors[i]]]))
    return fn(recs), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def rate(threshold):
    return lambda recs: 100.0 * sum(r["score"] >= threshold for r in recs) / len(recs)


def cochran_q(matrix):
    """Cochran's Q for a subjects x conditions binary matrix."""
    x = np.asarray(matrix, dtype=float)
    k = x.shape[1]
    col = x.sum(axis=0)
    row = x.sum(axis=1)
    total = x.sum()
    denom = k * total - np.sum(row ** 2)
    if denom == 0:
        return 0.0, 1.0
    q = (k - 1) * (k * np.sum(col ** 2) - total ** 2) / denom
    return float(q), float(stats.chi2.sf(q, k - 1))


def main():
    os.makedirs(OUT, exist_ok=True)
    records = load_phase2()
    summary = {"source": "runs/phase2_v2formal_*.jsonl (frozen, read-only)"}

    variants = {
        "all_models_all_attacks": (PHASE2_MODELS, ATTACKS),
        "aligned_models_all_attacks": (ALIGNED, ATTACKS),
        "all_models_pair_only": (PHASE2_MODELS, ["A1_pair"]),
        "aligned_models_pair_only": (ALIGNED, ["A1_pair"]),
        "aligned_models_static_only": (ALIGNED, ["A0_direct", "A2_persona"]),
    }
    summary["interaction_tests"] = {}
    for name, (models, attacks) in variants.items():
        res = interaction_test(records, models, attacks)
        summary["interaction_tests"][name] = res
        print(f"{name:<32s} n={res['n_records']:5d} p={res['p_value']:.5f}")

    rows = []
    for model in PHASE2_MODELS:
        for attack in ATTACKS:
            for scen in SCENARIOS:
                cell = [r for r in records if r["model"] == model and r["attack"] == attack and r["scenario"] == scen]
                h, hl, hh = clustered_ci(cell, rate(4))
                k, kl, kh = clustered_ci(cell, rate(3))
                rows.append({"model": model, "attack": attack, "scenario": scen, "n": len(cell),
                             "hcr_at_4": round(h, 2), "hcr_ci_low": round(hl, 2), "hcr_ci_high": round(hh, 2),
                             "risk_at_3": round(k, 2), "risk_ci_low": round(kl, 2), "risk_ci_high": round(kh, 2)})
    with open(os.path.join(OUT, "attack_by_scenario_ci.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary["pair_within_model_scenario_cochran_q"] = {}
    for model in PHASE2_MODELS:
        by_b = defaultdict(dict)
        for r in records:
            if r["model"] == model and r["attack"] == "A1_pair":
                by_b[r["behavior"]][r["scenario"]] = int(r["score"] >= 4)
        matrix = [[by_b[b][s] for s in SCENARIOS] for b in sorted(by_b)]
        q, p = cochran_q(matrix)
        summary["pair_within_model_scenario_cochran_q"][model] = {
            "n_behaviors": len(matrix), "Q": q, "df": 3, "p_value": p,
            "hcr_at_4_by_scenario": dict(zip(SCENARIOS, np.mean(matrix, axis=0).round(4).tolist())),
        }

    terminal = tuple(".!?)]}\"'`*:")
    summary["gemma_256_token_truncation_audit"] = {}
    for model in ["gemma4-12b-it", "gemma4-12b-obliterated"]:
        recs = [r for r in records if r["model"] == model]
        lengths = np.array([r["response_length"] for r in recs])
        cutoff = 0.9 * np.percentile(lengths, 99)
        flagged = [r for r in recs if r["response_length"] >= cutoff and not r["response"].rstrip().endswith(terminal)]
        dist = defaultdict(int)
        for r in flagged:
            dist[r["score"]] += 1
        summary["gemma_256_token_truncation_audit"][model] = {
            "n": len(recs),
            "max_tokens": sorted({str(r["max_tokens"]) for r in recs}),
            "median_chars": float(np.median(lengths)),
            "p99_chars": float(np.percentile(lengths, 99)),
            "near_cap_char_cutoff": float(cutoff),
            "likely_truncated_n": len(flagged),
            "likely_truncated_share": len(flagged) / len(recs),
            "likely_truncated_score_distribution": dict(sorted(dist.items())),
            "likely_truncated_hcr_at_4": (sum(r["score"] >= 4 for r in flagged) / len(flagged)) if flagged else None,
            "method": "response >= 90% of the model's 99th-percentile character length and not ending in terminal punctuation",
        }

    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "interaction_tests"}, indent=2))


if __name__ == "__main__":
    main()
