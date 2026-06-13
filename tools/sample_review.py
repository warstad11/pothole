"""Seeded per-drive uniform sampling of the transfer review queue.

The full event inventory exceeds practical labeling capacity. This script
marks a uniform random subset per (drive, kind) for review, recording the
sampling fraction in the manifest. Uniform random sampling of trigger
events leaves precision estimates unbiased (it subsamples the numerator
and denominator together); probe subsampling likewise leaves the
background-rate estimate unbiased. Run BEFORE any labeling (pre-registered
addendum, docs/TRANSFER_PROTOCOL.md §4a).

Usage: .venv/bin/python tools/sample_review.py [--events 200] [--probes 60]
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "results" / "transfer" / "review_items.json"
SEED = 1337  # distinct from the runner seed; fixed before labeling


def main():
    target_events = int(sys.argv[sys.argv.index("--events") + 1]) \
        if "--events" in sys.argv else 200
    target_probes = int(sys.argv[sys.argv.index("--probes") + 1]) \
        if "--probes" in sys.argv else 60

    data = json.loads(F.read_text())
    items = data["items"]
    labels_f = ROOT / "results" / "transfer" / "labels.jsonl"
    if labels_f.exists() and labels_f.read_text().strip():
        raise SystemExit("labels.jsonl already has entries — sampling after "
                         "labeling began would break pre-registration. Abort.")

    by = defaultdict(list)
    for it in items:
        by[(it["drive"], it["kind"])].append(it)

    totals = {"event": sum(len(v) for (d, k), v in by.items() if k == "event"),
              "probe": sum(len(v) for (d, k), v in by.items() if k == "probe")}
    frac = {"event": min(1.0, target_events / max(totals["event"], 1)),
            "probe": min(1.0, target_probes / max(totals["probe"], 1))}

    rng = random.Random(SEED)
    sample_meta = {}
    for (drive, kind), group in sorted(by.items()):
        n = len(group)
        k = max(1, round(frac[kind] * n)) if n else 0
        chosen = set(it["id"] for it in rng.sample(group, min(k, n)))
        for it in group:
            it["in_review_sample"] = it["id"] in chosen
        sample_meta.setdefault(drive, {})[kind] = {
            "total": n, "sampled": len(chosen),
            "fraction": round(len(chosen) / n, 3) if n else None}

    data["manifest"]["review_sampling"] = {
        "seed": SEED, "target_events": target_events,
        "target_probes": target_probes,
        "global_fraction": {k: round(v, 3) for k, v in frac.items()},
        "per_drive": sample_meta,
        "note": "uniform per-drive subsample; precision and probe-rate "
                "estimates are unbiased under uniform sampling",
    }
    F.write_text(json.dumps(data, indent=1))
    n_e = sum(1 for it in items if it["kind"] == "event" and it["in_review_sample"])
    n_p = sum(1 for it in items if it["kind"] == "probe" and it["in_review_sample"])
    print(f"sampled {n_e}/{totals['event']} events + {n_p}/{totals['probe']} probes "
          f"for review (seed {SEED})")


if __name__ == "__main__":
    main()
