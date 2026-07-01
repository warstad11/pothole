"""Correct sensor dataset labels so that label=1 means POTHOLE, not "any road anomaly".

Background
----------
"Data 3 - Just Sensor" ships with an embedded ``label`` column where every
anomaly category (FatigueCrack, Spall, Aligator cracking, Faulting, Bump,
Shoving, Construction joint, Slippage cracks) is labeled 1. For a pothole
detection task this is label noise: only ``Pothole*`` files are true
positives. The filename prefix preserves the true category for every file,
so the correction is fully recoverable and reversible.

This script rewrites the ``label`` column:
  - files whose stem starts with ``pothole`` -> all rows label=1
  - every other Data 3 file               -> all rows label=0

It writes a JSON manifest next to the dataset recording, for every file,
the category, the original label values, and the new label, so the change
is auditable and reversible.

Usage:
    python tools/relabel_sensor_data.py [--dry-run]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA3 = ROOT / "data" / "sensor" / "Data 3 - Just Sensor"

# Known Data 3 filename prefixes -> human-readable category
CATEGORIES = [
    "Pothole",
    "FatigueCrack",
    "Undamaged",
    "Construction joint",
    "Spall",
    "Aligator cracking",
    "Faulting",
    "Bump",
    "Shoving",
    "Slippage cracks",
]


def categorize(stem: str) -> str:
    for cat in CATEGORIES:
        if stem.lower().startswith(cat.lower()):
            return cat
    return "UNKNOWN"


def relabel_data3(dry_run: bool, data_dir: Path = None) -> dict:
    data_dir = data_dir or DATA3
    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found", file=sys.stderr)
        sys.exit(1)

    manifest = {
        "dataset": str(data_dir),
        "relabeled_at": datetime.now(timezone.utc).isoformat(),
        "rule": "label=1 iff filename stem starts with 'pothole'; all other categories are NOT potholes",
        "dry_run": dry_run,
        "files": [],
    }
    counts = {}
    changed = 0
    empty = []

    for f in sorted(data_dir.glob("*.csv")):
        if f.name == "relabel_manifest.json":
            continue
        cat = categorize(f.stem)
        new_label = 1 if cat == "Pothole" else 0
        try:
            df = pd.read_csv(f)
        except pd.errors.EmptyDataError:
            empty.append(f.name)
            manifest["files"].append(
                {"file": f.name, "category": cat, "rows": 0, "note": "ZERO-BYTE FILE"})
            continue

        if df.empty:
            empty.append(f.name)
            manifest["files"].append(
                {"file": f.name, "category": cat, "rows": 0, "note": "EMPTY FILE"}
            )
            continue

        if "label" not in df.columns:
            manifest["files"].append(
                {"file": f.name, "category": cat, "rows": len(df), "note": "no label column"}
            )
            continue

        orig = sorted(df["label"].dropna().unique().tolist())
        needs_change = orig != [new_label]
        manifest["files"].append(
            {
                "file": f.name,
                "category": cat,
                "rows": len(df),
                "original_labels": orig,
                "new_label": new_label,
                "changed": bool(needs_change),
            }
        )
        counts[cat] = counts.get(cat, 0) + 1
        if needs_change:
            changed += 1
            if not dry_run:
                df["label"] = new_label
                df.to_csv(f, index=False)

    unknown = [e["file"] for e in manifest["files"] if e["category"] == "UNKNOWN"]
    manifest["summary"] = {
        "total_files": sum(counts.values()),
        "files_relabeled": changed,
        "category_counts": counts,
        "empty_files": empty,
        "unknown_category_files": unknown,
    }
    return manifest


def audit_data1() -> dict:
    """Data 1 has per-row, time-localized labels which we keep. Just audit health."""
    data1 = ROOT / "data" / "sensor" / "Data 1"
    report = {"dataset": str(data1), "empty_files": [], "files_missing_label": [], "total": 0}
    if not data1.exists():
        return report
    for f in sorted(data1.glob("*.csv")):
        report["total"] += 1
        df = pd.read_csv(f)
        if df.empty:
            report["empty_files"].append(f.name)
        elif "label" not in df.columns:
            report["files_missing_label"].append(f.name)
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report changes without writing")
    ap.add_argument("--path", type=Path, default=None,
                    help="dataset dir of Data 3-style files to relabel "
                         f"(default: {DATA3})")
    args = ap.parse_args()
    data_dir = args.path or DATA3

    manifest = relabel_data3(args.dry_run, data_dir)
    s = manifest["summary"]
    print(f"Data 3: {s['total_files']} files, {s['files_relabeled']} relabeled"
          f"{' (DRY RUN — nothing written)' if args.dry_run else ''}")
    for cat, n in sorted(s["category_counts"].items(), key=lambda kv: -kv[1]):
        lbl = 1 if cat == "Pothole" else 0
        print(f"  {cat:25s} {n:4d} files -> label {lbl}")
    if s["empty_files"]:
        print(f"  WARNING: {len(s['empty_files'])} empty files: {s['empty_files'][:5]}...")
    if s["unknown_category_files"]:
        print(f"  WARNING: unknown categories: {s['unknown_category_files']}")

    out = data_dir / "relabel_manifest.json"
    if not args.dry_run:
        out.write_text(json.dumps(manifest, indent=2))
        print(f"Manifest written to {out}")

    d1 = audit_data1()
    print(f"Data 1 audit: {d1['total']} files, "
          f"{len(d1['empty_files'])} empty, {len(d1['files_missing_label'])} missing label column")
    if d1["empty_files"]:
        print(f"  Empty (consider removing): {d1['empty_files']}")


if __name__ == "__main__":
    main()
