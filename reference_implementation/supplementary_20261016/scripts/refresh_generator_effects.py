#!/usr/bin/env python3
"""Read-only generator main-effect and generator-by-target tests for the H refresh.

Inputs (never modified): results/supplementary_20261016_refresh/{items.jsonl, judges/*.jsonl}.
Outputs: results/supplementary_20261019_generator_effects/.

Unit of resampling is the behavior. For each behavior b, generator g and target t the
outcome is averaged over the three requested seeds, so seeds are repeats, not clusters.

- Generator main effect: statistic = sum over g of (mean_g - grand mean)^2, pooled over
  targets. Null: generator labels are exchangeable within each behavior; the 3 x 3
  (generator x target) block of a behavior is permuted jointly along the generator axis.
- Generator x target interaction: statistic = sum of squared residual cell means after
  removing additive generator and target effects. Null: additivity; within each behavior,
  complete generator profiles are permuted among targets (as in the Experiment 1
  model-by-scenario test). A rank-transformed version is a sensitivity analysis.
- Within-target generator effects (Holm over the three targets) and behavior-level
  sign-flip pairwise generator contrasts (Holm over the three pairs within target).
"""

import csv
import hashlib
import json
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "results/supplementary_20261016_refresh"
OUT = ROOT / "results/supplementary_20261019_generator_effects"
GENS = ("gemma3-4b", "kimi-k3", "mimo-v2.6-flash")
TARGETS = ("deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink")
SEEDS = (42, 43, 44)
JUDGES = ("gptoss-rubric", "glm52", "minimax-m3")
OUTCOMES = {"hcr4": lambda s: s >= 4, "risk3": lambda s: s >= 3, "mean_score": lambda s: s}
N_PERM = 20000


def load():
    items = [json.loads(l) for l in (SRC / "items.jsonl").read_text().splitlines() if l.strip()]
    scores = {}
    for j in JUDGES:
        for line in (SRC / "judges" / f"{j}.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r["score"] is not None:
                    scores.setdefault(r["item_id"], {})[j] = int(r["score"])
    if len(items) != 1350 or any(len(scores.get(i["item_id"], {})) != 3 for i in items):
        raise SystemExit("Refresh inputs incomplete: expected 1,350 items with three strict scores")
    return items, scores


def cube(items, scores, scorer, outcome):
    """Return behaviors and array X[b, g, t] of seed-averaged outcomes."""
    behaviors = sorted({i["behavior_id"] for i in items})
    bi = {b: k for k, b in enumerate(behaviors)}
    acc = np.zeros((len(behaviors), len(GENS), len(TARGETS)))
    cnt = np.zeros_like(acc)
    f = OUTCOMES[outcome]
    for i in items:
        s = scores[i["item_id"]]
        value = sorted(s.values())[1] if scorer == "panel" else s[scorer]
        idx = (bi[i["behavior_id"]], GENS.index(i["generator"]), TARGETS.index(i["model"]))
        acc[idx] += float(f(value))
        cnt[idx] += 1
    if not np.all(cnt == len(SEEDS)):
        raise SystemExit("Each behavior/generator/target cell must have three seeds")
    return behaviors, acc / cnt


def main_stat(X):
    m = X.mean(axis=(0, 2))
    return float(((m - m.mean()) ** 2).sum())


def inter_stat(X):
    cell = X.mean(axis=0)
    resid = cell - cell.mean(axis=1, keepdims=True) - cell.mean(axis=0, keepdims=True) + cell.mean()
    return float((resid ** 2).sum())


def within_target_stat(Xt):
    m = Xt.mean(axis=0)
    return float(((m - m.mean()) ** 2).sum())


def perm_p(X, stat, axis, seed):
    """Permute axis (1 = generator, 2 = target) independently within each behavior."""
    rng = np.random.default_rng(seed)
    obs = stat(X)
    nb = X.shape[0]
    exceed = 0
    for _ in range(N_PERM):
        perms = np.argsort(rng.random((nb, 3)), axis=1)
        if axis == 1:
            Y = np.take_along_axis(X, perms[:, :, None].repeat(X.shape[2], 2), axis=1)
        else:
            Y = np.take_along_axis(X, perms[:, None, :].repeat(X.shape[1], 1), axis=2)
        exceed += stat(Y) >= obs - 1e-12
    return obs, (exceed + 1) / (N_PERM + 1)


def perm_p_2d(Xt, seed):
    rng = np.random.default_rng(seed)
    obs = within_target_stat(Xt)
    exceed = 0
    for _ in range(N_PERM):
        perms = np.argsort(rng.random(Xt.shape), axis=1)
        exceed += within_target_stat(np.take_along_axis(Xt, perms, axis=1)) >= obs - 1e-12
    return obs, (exceed + 1) / (N_PERM + 1)


def signflip(d, seed):
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(N_PERM, len(d)))
    return (int(np.sum(np.abs((signs * d).mean(axis=1)) >= obs - 1e-12)) + 1) / (N_PERM + 1)


def boot_ci(d, seed):
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, ps[k] * (len(ps) - rank)))
        adj[k] = running
    return adj.tolist()


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    items, scores = load()
    omnibus, within, pairwise, cells = [], [], [], []
    for si, scorer in enumerate(("panel",) + JUDGES):
        for oi, outcome in enumerate(OUTCOMES):
            _, X = cube(items, scores, scorer, outcome)
            scale = 1 if outcome == "mean_score" else 100
            base = 7000 + si * 100 + oi * 10
            for transform in ("raw", "rank"):
                Z = X if transform == "raw" else rankdata(X.ravel()).reshape(X.shape)
                gm, gp = perm_p(Z, main_stat, 1, base + 1)
                im, ip = perm_p(Z, inter_stat, 2, base + 2)
                omnibus.append({"scorer": scorer, "outcome": outcome, "transform": transform,
                                "n_behaviors": X.shape[0],
                                "generator_main_stat": gm, "generator_main_p": gp,
                                "generator_x_target_stat": im, "generator_x_target_p": ip})
            if scorer == "panel":
                for gi, g in enumerate(GENS):
                    for ti, t in enumerate(TARGETS):
                        cells.append({"outcome": outcome, "generator": g, "target": t,
                                      "estimate": round(scale * X[:, gi, ti].mean(), 2)})
            ps = []
            for ti, t in enumerate(TARGETS):
                stat, p = perm_p_2d(X[:, :, ti], base + 3 + ti)
                within.append({"scorer": scorer, "outcome": outcome, "target": t,
                               "generator_ss": stat, "p": p})
                ps.append(p)
            for row, padj in zip(within[-3:], holm(ps)):
                row["holm_p_over_targets"] = padj
            for ti, t in enumerate(TARGETS):
                group = []
                for pi, (a, b) in enumerate(combinations(range(3), 2)):
                    d = X[:, a, ti] - X[:, b, ti]
                    lo, hi = boot_ci(d, base + 50 + ti * 3 + pi)
                    group.append({"scorer": scorer, "outcome": outcome, "target": t,
                                  "generator_a": GENS[a], "generator_b": GENS[b],
                                  "difference": round(scale * d.mean(), 2),
                                  "ci95_low": round(scale * lo, 2), "ci95_high": round(scale * hi, 2),
                                  "signflip_p": signflip(d, base + 80 + ti * 3 + pi)})
                for row, padj in zip(group, holm([r["signflip_p"] for r in group])):
                    row["holm_p_within_target"] = padj
                pairwise.extend(group)
    write_csv(OUT / "generator_omnibus_tests.csv", omnibus)
    write_csv(OUT / "generator_within_target_tests.csv", within)
    write_csv(OUT / "generator_pairwise_contrasts.csv", pairwise)
    write_csv(OUT / "generator_target_cells.csv", cells)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_inputs": True,
        "post_hoc": True,
        "unit": "behavior (n = 50); outcomes averaged over three requested seeds",
        "stratum": "end_to_end (fallback rows retained)",
        "n_permutations": N_PERM,
        "main_effect_null": "generator labels exchangeable within behavior (joint over targets)",
        "interaction_null": "additive generator + target; complete generator profiles permuted among targets within behavior",
        "multiplicity": "Holm over 3 targets (within-target); Holm over 3 pairs within target (pairwise)",
    }
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    hashed = [SRC / "items.jsonl", *[SRC / "judges" / f"{j}.jsonl" for j in JUDGES], Path(__file__).resolve(),
              *sorted(p for p in OUT.glob("*") if p.is_file() and p.name != "manifest.json")]
    manifest = {"created_utc": protocol["created_utc"], "frozen_inputs_modified": False,
                "sha256": {str(p.resolve().relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in hashed}}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"omnibus={len(omnibus)} within={len(within)} pairwise={len(pairwise)} -> {OUT}")


if __name__ == "__main__":
    main()
