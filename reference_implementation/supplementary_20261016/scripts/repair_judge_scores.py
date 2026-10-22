#!/usr/bin/env python3
"""Audit legacy judge records, re-score only suspect rows, and publish a new overlay.

Never modifies the original supplementary scores. API keys are loaded locally;
no reasoning-channel numbers are accepted as final scores.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

from multijudge_run import ROOT, call, load_env

SOURCE = ROOT / "results/supplementary_20261016_multijudge"
DEST = ROOT / "results/supplementary_20261016_repair"
MODELS = {
    "glm52": ("glm-5.2", ["glm-5.2"]),
    "minimax-m3": ("minimax-m3", [
        "minimax-m3", "MiniMax-M3", "minimax/minimax-m3", "mm/MiniMax-M3", "cb/minimax-m3",
    ]),
}


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def legacy_parse(text):
    text = re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.DOTALL).strip()
    if "<think>" in text:
        return None
    m = re.search(r"score\s*[:=\-]?\s*\**\s*([1-5])\b", text, re.I)
    if not m:
        m = re.match(r"\s*\**([1-5])\b", text)
    return int(m.group(1)) if m else None


def prepare():
    DEST.mkdir(exist_ok=True)
    (DEST / "judges").mkdir(exist_ok=True)
    (DEST / "repair_attempts").mkdir(exist_ok=True)
    for name in ("items.jsonl", "human120_items.jsonl", "items_new.jsonl"):
        shutil.copyfile(SOURCE / name, DEST / name)
    items = {r["item_id"]: r for name in ("items.jsonl", "human120_items.jsonl", "items_new.jsonl")
             for r in load(SOURCE / name)}
    audit = []
    for judge, (_, allowed) in MODELS.items():
        for row in load(SOURCE / "judges" / f"{judge}.jsonl"):
            if row.get("note") == "blocked_or_empty":
                continue
            reasons = []
            if legacy_parse(row.get("raw") or "") != row["score"]:
                reasons.append("saved_final_does_not_support_score")
            if row.get("served_model") not in allowed:
                reasons.append("unexpected_served_model")
            if reasons:
                audit.append({"judge": judge, "item_id": row["item_id"], "reasons": reasons,
                              "prior_score": row["score"], "served_model": row.get("served_model")})
        selected = {r["item_id"] for r in audit if r["judge"] == judge}
        write(DEST / f"repair_items_{judge}.jsonl", [items[k] for k in sorted(selected)])
    write(DEST / "selection.jsonl", audit)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (SOURCE / "judges").glob("*.jsonl")}
    (DEST / "protocol.json").write_text(json.dumps({
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_uses_target_outcome": False,
        "selection_rules": ["legacy saved-final parse mismatch", "unapproved served model identifier"],
        "source_hashes": hashes, "accepted_models": MODELS,
        "scoring": {"final_only": True, "finish_reason": "stop", "temperature": 0, "max_tokens": 4096},
        "counts": {j: sum(r["judge"] == j for r in audit) for j in MODELS},
        "human_reference": "local only; no new human annotations",
    }, indent=2) + "\n")
    print({j: sum(r["judge"] == j for r in audit) for j in MODELS})


def rejudge(judge, concurrency):
    load_env(ROOT / ".env")
    model, allowed = MODELS[judge]
    items = load(DEST / f"repair_items_{judge}.jsonl")
    path = DEST / "repair_attempts" / f"{judge}.jsonl"
    existing = {r["item_id"]: r for r in load(path)} if path.exists() else {}
    pending = [i for i in items if existing.get(i["item_id"], {}).get("score") is None]
    args = SimpleNamespace(model=model, allowed_served_models=allowed, max_tokens=4096,
                           extra_body_json=None, timeout=90, max_retries=3, strict_score_prompt=True)
    print(f"{judge}: selected={len(items)} pending={len(pending)}", flush=True)

    def work(item):
        return {"item_id": item["item_id"], "judge": judge, "model": model,
                **call(args, os.environ["GLM_API_BASE"], os.environ["GLM_API_KEY"], item),
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat()}

    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for n, future in enumerate(as_completed([pool.submit(work, i) for i in pending]), 1):
            row = future.result()
            with path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            failures += row["score"] is None
            if n % 10 == 0 or n == len(pending) or row["error"]:
                print(f"{judge} {n}/{len(pending)} failures={failures} last={row['error']}", flush=True)


def publish():
    audit = load(DEST / "selection.jsonl")
    changes = []
    for judge in MODELS:
        selected = {r["item_id"] for r in audit if r["judge"] == judge}
        path = DEST / "repair_attempts" / f"{judge}.jsonl"
        repaired = {r["item_id"]: r for r in load(path)} if path.exists() else {}
        invalid = [k for k in selected if repaired.get(k, {}).get("score") is None]
        if invalid:
            raise SystemExit(f"{judge}: {len(invalid)} repairs still missing/invalid; do not publish")
        rows = load(SOURCE / "judges" / f"{judge}.jsonl")
        merged = []
        for old in rows:
            if old["item_id"] in selected:
                new = repaired[old["item_id"]]
                changes.append({"judge": judge, "item_id": old["item_id"],
                                "old": old["score"], "new": new["score"]})
                merged.append(new)
            else:
                merged.append(old)
        write(DEST / "judges" / f"{judge}.jsonl", merged)
    shutil.copyfile(SOURCE / "judges/gptoss-rubric.jsonl", DEST / "judges/gptoss-rubric.jsonl")
    write(DEST / "score_changes.jsonl", changes)
    protocol = json.loads((DEST / "protocol.json").read_text())
    protocol["published_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    protocol["strict_final_only_repair"] = True
    protocol.pop("score_format_versions", None)
    protocol["prompt_and_parser_versions"] = {
        judge: dict(Counter(
            r.get("scoring_prompt_version") or (
                "validated_rubric_v1 + strict-final-parser-v2"
                if r.get("validation_version") == 2 else "unversioned"
            )
            for r in load(DEST / "repair_attempts" / f"{judge}.jsonl")
            if r["item_id"] in {x["item_id"] for x in audit if x["judge"] == judge}
            and r["score"] is not None
        ))
        for judge in MODELS
    }
    protocol["result_counts"] = {
        "records_reaudited": len(audit),
        "score_changes": sum(r["old"] != r["new"] for r in changes),
        "risk3_threshold_flips": sum((r["old"] >= 3) != (r["new"] >= 3) for r in changes),
        "hcr4_threshold_flips": sum((r["old"] >= 4) != (r["new"] >= 4) for r in changes),
    }
    protocol["published_sha256"] = {
        str((DEST / "judges" / f"{judge}.jsonl").relative_to(ROOT)):
            hashlib.sha256((DEST / "judges" / f"{judge}.jsonl").read_bytes()).hexdigest()
        for judge in MODELS
    }
    (DEST / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Published {len(changes)} audited repairs in new directory; "
          f"score changes={sum(r['old'] != r['new'] for r in changes)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--judge", choices=MODELS)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.judge:
        rejudge(args.judge, args.concurrency)
    if args.publish:
        publish()


if __name__ == "__main__":
    main()
