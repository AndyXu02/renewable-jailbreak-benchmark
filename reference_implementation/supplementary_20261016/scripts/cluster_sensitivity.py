#!/usr/bin/env python3
"""Compare attack-specific and whole-behavior model-profile permutations.

Both tests use the paper's rank-residual interaction statistic. The conservative
dependency sensitivity applies one model permutation to all attacks within a
behavior, retaining cross-attack as well as cross-scenario dependence.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from multijudge_analysis import ALIGNED, ANALYSIS, PANEL, load, scored


def profile_test(records, whole_behavior, n_perm=20000, seed=42):
    behaviors = sorted({r["behavior_id"] for r in records})
    attacks = sorted({r["attack"] for r in records})
    models = sorted({r["model"] for r in records})
    scenarios = sorted({r["scenario"] for r in records})
    shape = (len(behaviors), len(attacks), len(models), len(scenarios))
    tensor = np.full(shape, np.nan)
    ranks = rankdata([r["score"] for r in records])
    lookups = [{v: i for i, v in enumerate(values)} for values in (behaviors, attacks, models, scenarios)]
    for row, rank in zip(records, ranks):
        idx = tuple(d[row[k]] for d, k in zip(lookups, ("behavior_id", "attack", "model", "scenario")))
        if not np.isnan(tensor[idx]):
            raise ValueError("Duplicate experimental cell")
        tensor[idx] = rank
    if np.isnan(tensor).any():
        raise ValueError("Permutation requires a complete crossed matrix")
    residual = (tensor - tensor.mean(axis=(0, 1, 3), keepdims=True)
                - tensor.mean(axis=(0, 1, 2), keepdims=True) + tensor.mean())
    weight = shape[0] * shape[1]
    observed = weight * np.square(residual.mean(axis=(0, 1))).sum()
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_perm, 500):
        batch = min(500, n_perm - start)
        order = rng.random((batch, shape[0], 1 if whole_behavior else shape[1], shape[2])).argsort(axis=-1)
        order = np.broadcast_to(order, (batch, shape[0], shape[1], shape[2]))
        permuted = np.take_along_axis(residual[None, ...], order[..., None], axis=3)
        statistics = weight * np.square(permuted.mean(axis=(1, 2))).sum(axis=(1, 2))
        exceed += int(np.sum(statistics >= observed - 1e-9))
    p = (exceed + 1) / (n_perm + 1)
    return {"p_value": p, "observed_ss": float(observed), "n_perm": n_perm,
            "n_behaviors": shape[0], "n_attacks": shape[1],
            "permutation_units": shape[0] if whole_behavior else shape[0] * shape[1],
            "mc_standard_error": float(np.sqrt(p * (1 - p) / (n_perm + 1)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=20000)
    args = parser.parse_args()
    items, _ = load()
    variants = {
        "all_models_all_attacks": (None, {"A0_direct", "A2_persona", "A1_pair"}),
        "aligned_all_attacks": (ALIGNED, {"A0_direct", "A2_persona", "A1_pair"}),
        "aligned_pair_only": (ALIGNED, {"A1_pair"}),
        "aligned_static_only": (ALIGNED, {"A0_direct", "A2_persona"}),
    }
    output = defaultdict(dict)
    for judge in PANEL + ["panel"]:
        recs = scored(items, judge, experiment="exp1")
        if len(recs) != 1200:
            raise ValueError(f"Incomplete experiment for {judge}")
        for name, (models, attacks) in variants.items():
            cell = [r for r in recs if r["attack"] in attacks and (models is None or r["model"] in models)]
            output[judge][name] = {
                label: profile_test(cell, whole, args.n_perm)
                for label, whole in (("behavior_attack", False), ("whole_behavior", True))
            }
            print(judge, name, {k: round(v["p_value"], 5) for k, v in output[judge][name].items()}, flush=True)
    out = Path(ANALYSIS)
    out.mkdir(exist_ok=True)
    (out / "cluster_sensitivity.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
