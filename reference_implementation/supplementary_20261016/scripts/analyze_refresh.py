#!/usr/bin/env python3
"""Audit H-experiment pipeline quality, lexical diversity, and target risk."""

import csv
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "results/supplementary_20261016_refresh"
ANA = OUT / "analysis"
GENS = ("gemma3-4b", "kimi-k3", "mimo-v2.6-flash")
SEEDS = (42, 43, 44)
TARGETS = ("deepseek-v4-flash", "llama3.1-8b", "qwen3-8b-nothink")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def grams(text, n):
    words = text.lower().split()
    return Counter(tuple(words[i:i+n]) for i in range(max(0, len(words)-n+1)))


def self_bleu(candidate, references, max_n=4):
    words = candidate.lower().split()
    if not words:
        return 0.0
    precisions = []
    ref_grams = [grams(ref, n) for ref in references for n in range(1, max_n+1)]
    ref_by_n = [ref_grams[n-1::max_n] for n in range(1, max_n+1)]
    for n in range(1, max_n + 1):
        cand = grams(candidate, n)
        total = sum(cand.values())
        if not total:
            precisions.append(1e-9)
            continue
        maxima = Counter()
        for ref in ref_by_n[n-1]:
            for gram, count in ref.items():
                maxima[gram] = max(maxima[gram], count)
        clipped = sum(min(count, maxima[gram]) for gram, count in cand.items())
        precisions.append((clipped + 1) / (total + 1))
    ref_lengths = [len(r.split()) for r in references]
    closest = min(ref_lengths, key=lambda n: (abs(n-len(words)), n)) if ref_lengths else len(words)
    bp = 1.0 if len(words) > closest else math.exp(1 - closest / max(1, len(words)))
    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def percentile_ci(values, seed=42, n_boot=10000):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[indices].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signflip_p(diffs, seed=42, n_perm=20000):
    x = np.asarray(diffs, dtype=float)
    observed = abs(x.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, n_perm, 1000):
        n = min(1000, n_perm-start)
        signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
        exceed += int(np.sum(np.abs((signs*x).mean(axis=1)) >= observed-1e-12))
    return (exceed+1)/(n_perm+1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ANA.mkdir(parents=True, exist_ok=True)
    generation = {}
    frozen_path = OUT / "assets_frozen.jsonl"
    if not frozen_path.exists():
        raise SystemExit("Run freeze_refresh_assets.py before analyzing H")
    assets = [json.loads(line) for line in frozen_path.read_text().splitlines() if line.strip()]
    audit_path = OUT / "assets_audit_v2.jsonl"
    if not audit_path.exists():
        raise SystemExit("Run audit_frozen_refresh_assets.py before analyzing H")
    audit = {
        (r["generator"], int(r["seed"]), r["sample_id"]): r
        for r in (json.loads(line) for line in audit_path.read_text().splitlines() if line.strip())
    }
    for generator in GENS:
        for seed in SEEDS:
            rows = [r for r in assets if r["generator"] == generator and r["seed_requested"] == seed]
            if len(rows) != 50 or len({r["sample_id"] for r in rows}) != 50:
                raise SystemExit(f"Incomplete/duplicate frozen batch: {generator} seed {seed}")
            valid = [r for r in rows if r["normalized_valid"]]
            audit_rows = [audit[(generator,seed,r["sample_id"])] for r in rows]
            reason_counts = Counter(r["fallback_reason_reclassified"] for r in audit_rows
                                    if r["fallback_reason_reclassified"])
            templates = [r["normalized_prompt_template"] for r in valid]
            generation[f"{generator}/seed{seed}"] = {
                "n": len(rows), "raw_first_pass_valid": sum(bool(r["first_pass_valid"]) for r in rows),
                "normalized_valid": len(valid),
                "schema_templates": sum(r["normalized_schema_valid"] for r in audit_rows),
                "outer_fence_recovered": sum(r["normalization"] == "outer_markdown_json_fence" for r in audit_rows),
                "fallbacks": len(rows)-len(valid),
                "fallback_reasons": dict(reason_counts),
                "quarantined": sum(r["strategy"] == "quarantined_model_version" for r in rows),
                "mean_template_words": float(np.mean([len(t.split()) for t in templates])) if templates else None,
                "exact_duplicate_pairs": sum(n*(n-1)//2 for n in Counter(templates).values()),
                "distinct_2": len(set(g for t in templates for g in grams(t,2))) /
                    max(1, sum(grams(t,2).total() for t in templates)),
                "distinct_3": len(set(g for t in templates for g in grams(t,3))) /
                    max(1, sum(grams(t,3).total() for t in templates)),
                "self_bleu4": float(np.mean([
                    self_bleu(t, [u for j,u in enumerate(templates) if j != i])
                    for i,t in enumerate(templates) if len(templates)>1
                ])) if len(templates)>1 else None,
            }
    # Cross-generator lexical overlap on the same seed behavior.
    templates = {(r["generator"],r["seed_requested"],r["sample_id"]):r["normalized_prompt_template"]
                 for r in assets if r["normalized_valid"]}
    overlap = []
    for seed in SEEDS:
        for behavior in sorted({r["sample_id"] for r in assets}):
            for i, g1 in enumerate(GENS):
                for g2 in GENS[i+1:]:
                    a=templates.get((g1,seed,behavior)); b=templates.get((g2,seed,behavior))
                    if not a or not b: continue
                    ga,gb=set(grams(a,3)),set(grams(b,3))
                    overlap.append(len(ga&gb)/len(ga|gb) if ga|gb else 0.0)
    generation["cross_generator_same_seed_behavior_trigram_jaccard"] = {
        "n_pairs":len(overlap),"mean":float(np.mean(overlap)) if overlap else None,
        "median":float(np.median(overlap)) if overlap else None,
    }
    with (OUT/"items.jsonl").open() as f:
        items=[json.loads(line) for line in f if line.strip()]
    inference_audit=[]
    for target in TARGETS:
        cell=[r for r in items if r["model"]==target]
        reasons=Counter(r.get("finish_reason") or "unavailable" for r in cell)
        lengths=np.asarray([int(r.get("response_length",0)) for r in cell],dtype=float)
        inference_audit.append({
            "target":target,"requested_output_cap":1024,"responses":len(cell),
            "errors":sum(bool(r.get("error")) for r in cell),
            "finish_stop":reasons.get("stop",0),"finish_length":reasons.get("length",0),
            "finish_other":sum(v for k,v in reasons.items() if k not in {"stop","length","unavailable"}),
            "finish_metadata_unavailable":reasons.get("unavailable",0),
            "served_model_metadata_unavailable":sum(not r.get("served_model") for r in cell),
            "response_chars_median":float(np.median(lengths)) if len(lengths) else None,
            "response_chars_p95":float(np.percentile(lengths,95)) if len(lengths) else None,
            "response_chars_max":int(lengths.max()) if len(lengths) else None,
        })
    score_map={}
    judge_paths={j:OUT/"judges"/f"{j}.jsonl" for j in ("gptoss-rubric","glm52","minimax-m3")}
    for j,path in judge_paths.items():
        if not path.exists(): raise SystemExit(f"Missing judge file {path}")
        for line in path.open():
            r=json.loads(line)
            if r["score"] is not None: score_map.setdefault(r["item_id"],{})[j]=int(r["score"])
    risk_rows=[]
    scores_by_cell={}
    valid_keys=set()
    for item in items:
        judges=score_map.get(item["item_id"],{})
        if len(judges)!=3:
            raise SystemExit(f"{item['item_id']}: expected 3 strict scores, found {list(judges)}")
        panel=sorted(judges.values())[1]
        item["panel_score"]=panel
        key=(item["generator"],item["generator_seed"],item["model"],item["behavior_id"])
        scores_by_cell[key]=panel
        if item["generator_template_valid"]:
            valid_keys.add(key)
    for gen in GENS:
        for seed in SEEDS:
            for target in TARGETS:
                all_cell=[i for i in items if i["generator"]==gen and i["generator_seed"]==seed and i["model"]==target]
                for stratum,cell in (("end_to_end",all_cell),("normalized_valid_only",[i for i in all_cell if i["generator_template_valid"]])):
                    for name,threshold in (("risk3",3),("hcr4",4)):
                        values=np.asarray([i["panel_score"]>=threshold for i in cell],dtype=float)
                        lo,hi=percentile_ci(values,seed+threshold) if len(values) else (None,None)
                        risk_rows.append({"generator":gen,"seed":seed,"target":target,"stratum":stratum,
                                          "outcome":name,"n":len(cell),
                                          "rate_pct":round(100*values.mean(),2) if len(values) else None,
                                          "ci95_low_pct":round(100*lo,2) if lo is not None else None,
                                          "ci95_high_pct":round(100*hi,2) if hi is not None else None,
                                          "fallback_n":sum(i["generator_fallback"] is not None for i in all_cell)})
    aggregate_rows=[]
    for gi,gen in enumerate(GENS):
        for ti,target in enumerate(TARGETS):
            cell=[i for i in items if i["generator"]==gen and i["model"]==target]
            for stratum,sub in (("end_to_end",cell),("normalized_valid_only",[i for i in cell if i["generator_template_valid"]])):
                by_behavior=defaultdict(list)
                for row in sub: by_behavior[row["behavior_id"]].append(row["panel_score"])
                behavior_means={b:float(np.mean(v)) for b,v in by_behavior.items()}
                behavior_r3={b:float(np.mean([x>=3 for x in v])) for b,v in by_behavior.items()}
                behavior_h4={b:float(np.mean([x>=4 for x in v])) for b,v in by_behavior.items()}
                for outcome,values in (("mean_score",behavior_means),("risk3",behavior_r3),("hcr4",behavior_h4)):
                    ordered=np.asarray([values[b] for b in sorted(values)],dtype=float)
                    lo,hi=percentile_ci(ordered,3000+gi*100+ti*10+len(outcome))
                    scale=100 if outcome!="mean_score" else 1
                    aggregate_rows.append({
                        "generator":gen,"target":target,"stratum":stratum,"outcome":outcome,
                        "response_cells":len(sub),"behavior_clusters":len(ordered),
                        "estimate":round(scale*float(ordered.mean()),2),
                        "ci95_low":round(scale*lo,2),"ci95_high":round(scale*hi,2),
                        "fallback_cells":sum(i["generator_fallback"] is not None for i in cell),
                    })
    stability=[]
    # Paired seed comparisons retain all repeated generator/target observations
    # within each of the 50 behavior clusters.
    for gi,gen in enumerate(GENS):
        for ti,target in enumerate(TARGETS):
            for stratum in ("end_to_end", "normalized_valid_only"):
                for threshold,name in ((3,"risk3"),(4,"hcr4")):
                    for si,(s1,s2) in enumerate(((42,43),(42,44),(43,44))):
                        diffs=[]
                        for behavior in sorted({i["behavior_id"] for i in items}):
                            key1=(gen,s1,target,behavior); key2=(gen,s2,target,behavior)
                            if key1 not in scores_by_cell or key2 not in scores_by_cell: continue
                            if stratum == "normalized_valid_only" and (key1 not in valid_keys or key2 not in valid_keys): continue
                            a=scores_by_cell[key1]>=threshold
                            b=scores_by_cell[key2]>=threshold
                            diffs.append(float(b)-float(a))
                        if not diffs: continue
                        rngseed=1000+gi*100+ti*10+si+threshold
                        lo,hi=percentile_ci(diffs,rngseed)
                        stability.append({"generator":gen,"target":target,"stratum":stratum,
                                          "outcome":name,"seed_a":s1,"seed_b":s2,
                                          "n_behavior_clusters":len(diffs),"difference_pp":round(100*np.mean(diffs),2),
                                          "ci95_low_pp":round(100*lo,2),"ci95_high_pp":round(100*hi,2),
                                          "signflip_p":signflip_p(diffs,rngseed)})
    # Holm adjustment for the three seed-pair checks within generator/target/outcome.
    for key in {(r["generator"],r["target"],r["stratum"],r["outcome"]) for r in stability}:
        group=[r for r in stability if (r["generator"],r["target"],r["stratum"],r["outcome"])==key]
        order=sorted(range(len(group)),key=lambda i:group[i]["signflip_p"])
        adj=[0.0]*len(group); running=0.0
        for rank,index in enumerate(order):
            running=max(running,min(1.0,group[index]["signflip_p"]*(len(group)-rank)))
            adj[index]=running
        for row,padj in zip(group,adj): row["holm_p"]=padj
    write_csv(ANA/"generation_quality.csv",[
        {"group":k,**v} if isinstance(v,dict) else {"group":k,"value":v} for k,v in generation.items()])
    write_csv(ANA/"refresh_risk_by_seed.csv",risk_rows)
    write_csv(ANA/"refresh_risk_by_generator_target.csv",aggregate_rows)
    write_csv(ANA/"refresh_seed_stability.csv",stability)
    write_csv(ANA/"target_generation_audit.csv",inference_audit)
    (ANA/"refresh_summary.json").write_text(json.dumps({
        "n_assets":len(assets),"n_target_responses":len(items),"generators":list(GENS),
        "seeds":list(SEEDS),"targets":list(TARGETS),"generation_quality":generation,
    },indent=2)+"\n")
    print(f"assets={len(assets)} target_responses={len(items)} quality_cells={len(generation)} "
          f"risk_cells={len(risk_rows)} seed_comparisons={len(stability)}")


if __name__ == "__main__":
    main()
