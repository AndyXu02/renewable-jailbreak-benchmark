from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the public release manifest")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest_path = args.manifest or root / "artifacts" / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    total_records = 0
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.exists():
            failures.append(f"missing: {item['path']}")
            continue
        actual_hash = sha256(path)
        if actual_hash != item["sha256"]:
            failures.append(f"hash mismatch: {item['path']}")
        if item.get("records") is not None:
            actual_records = sum(1 for line in path.open(encoding="utf-8") if line.strip())
            total_records += actual_records
            if actual_records != item["records"]:
                failures.append(f"record mismatch: {item['path']}")
    if failures:
        print("Release verification FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1
    formal_manifest_path = root / "artifacts" / "formal" / "tables" / "formal_results_manifest.json"
    formal_manifest = json.loads(formal_manifest_path.read_text(encoding="utf-8"))
    formal_code_count = 0
    for item in formal_manifest.get("code", {}).values():
        filename = Path(item["path"]).name
        reference = root / "reference_implementation" / "formal_v2" / "scripts" / filename
        if not reference.exists() or sha256(reference) != item["sha256"]:
            print(f"Formal reference code mismatch: {filename}")
            return 1
        formal_code_count += 1
    print(
        "Release verification passed: "
        f"{len(manifest['files'])} files; "
        f"{manifest.get('formal_raw_records', 0)} raw + "
        f"{manifest.get('formal_scored_records', 0)} compact formal records "
        f"({total_records} tracked JSONL records in total); "
        f"{formal_code_count} formal code hashes matched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
