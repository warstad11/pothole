"""Compute transfer-gap metrics from blinded review labels.

Inputs : results/transfer/review_items.json  (runner output, provenance)
         results/transfer/labels.jsonl       (one JSON per submitted label)
Outputs: results/transfer/metrics.json / metrics.md

Per docs/TRANSFER_PROTOCOL.md:
- precision per pipeline = TP/(TP+FP) over labeled trigger events, with
  Wilson 95% CI; 'unsure' labels excluded from both counts (reported).
- probes (random non-triggered moments) estimate the background pothole
  rate in non-triggered time -> estimated misses and a recall PROXY.
- paired pipeline comparison: exact McNemar (binomial) on discordant
  pothole-labeled items.
- inter-rater agreement (Cohen's kappa) when >= 2 reviewers overlap.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TR = ROOT / "results" / "transfer"
CLIP_SECONDS = 4.0
PIPES = ("image", "sensor", "hybrid")


def wilson(k, n, z=1.96):
    if n == 0:
        return None, None
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def binom_two_sided(k, n):
    """Exact two-sided binomial test p-value for p0=0.5 (no scipy needed)."""
    if n == 0:
        return None
    def pmf(i):
        return math.comb(n, i) * 0.5 ** n
    pk = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= pk + 1e-12))


def cohen_kappa(pairs):
    """pairs: list of (label_a, label_b) over the same items."""
    if not pairs:
        return None
    cats = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    n = len(pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    pe = sum((sum(1 for a, _ in pairs if a == c) / n) *
             (sum(1 for _, b in pairs if b == c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def load(primary_reviewer=None):
    data = json.loads((TR / "review_items.json").read_text())
    items = {it["id"]: it for it in data["items"]}
    labels_path = TR / "labels.jsonl"
    raw = []
    if labels_path.exists():
        for line in labels_path.read_text().splitlines():
            if line.strip():
                raw.append(json.loads(line))
    # latest label wins per (item, reviewer)
    by_rev = defaultdict(dict)
    for r in raw:
        by_rev[r.get("reviewer", "user")][r["item_id"]] = r["label"]
    if not by_rev:
        return data["manifest"], items, {}, {}, None
    if primary_reviewer is None:
        primary_reviewer = max(by_rev, key=lambda k: len(by_rev[k]))
    return data["manifest"], items, by_rev, by_rev.get(primary_reviewer, {}), primary_reviewer


def compute(primary_reviewer=None):
    manifest, items, by_rev, labels, primary = load(primary_reviewer)
    platforms = sorted({it["platform"] for it in items.values()})
    out = {"primary_reviewer": primary,
           "reviewers": {r: len(l) for r, l in by_rev.items()},
           "labeled": len(labels), "total_items": len(items),
           "per_platform": {}, "paired_tests": {}, "agreement": None,
           "caveat": "recall figures are ESTIMATES from probe sampling, "
                     "not exhaustive ground truth"}

    for plat in platforms + ["all"]:
        sel = [it for it in items.values()
               if plat == "all" or it["platform"] == plat]
        drive_secs = sum(d["duration_s"] for d in manifest["drives"]
                         if plat == "all" or d["platform"] == plat)
        block = {"pipelines": {}, "probes": {}}

        # probes -> background pothole rate in non-triggered time
        probes = [it for it in sel if it["kind"] == "probe" and it["id"] in labels]
        pk = sum(1 for it in probes if labels[it["id"]] == "pothole")
        pn = sum(1 for it in probes if labels[it["id"]] in ("pothole", "not_pothole"))
        rate = pk / pn if pn else None
        lo, hi = wilson(pk, pn) if pn else (None, None)
        block["probes"] = {"labeled": pn, "pothole": pk,
                           "background_rate": rate, "rate_ci95": [lo, hi],
                           "unsure": sum(1 for it in probes
                                         if labels[it["id"]] == "unsure")}

        for pipe in PIPES:
            trig = [it for it in sel if it["kind"] == "event"
                    and it["provenance"][pipe]["triggered"]]
            lab = [it for it in trig if it["id"] in labels]
            tp = sum(1 for it in lab if labels[it["id"]] == "pothole")
            fp = sum(1 for it in lab if labels[it["id"]] == "not_pothole")
            uns = sum(1 for it in lab if labels[it["id"]] == "unsure")
            n = tp + fp
            prec = tp / n if n else None
            plo, phi = wilson(tp, n) if n else (None, None)
            # estimated misses: probe background rate extrapolated over the
            # drive time NOT covered by this pipeline's events
            est_miss = est_recall = None
            if rate is not None and drive_secs:
                covered = sum(
                    (it["provenance"][pipe].get("t_end", it["time"]) -
                     it["provenance"][pipe].get("t_start", it["time"])) + CLIP_SECONDS
                    for it in trig)
                uncovered = max(drive_secs - covered, 0.0)
                est_miss = rate * (uncovered / CLIP_SECONDS)
                est_recall = tp / (tp + est_miss) if (tp + est_miss) > 0 else None
            block["pipelines"][pipe] = {
                "events": len(trig), "labeled": len(lab),
                "tp": tp, "fp": fp, "unsure": uns,
                "precision": prec, "precision_ci95": [plo, phi],
                "events_per_min": round(len(trig) / (drive_secs / 60), 2)
                                  if drive_secs else None,
                "est_missed_potholes": round(est_miss, 1) if est_miss is not None else None,
                "est_recall_proxy": round(est_recall, 3) if est_recall is not None else None,
            }
        out["per_platform"][plat] = block

    # paired exact McNemar on pothole-labeled event items
    pothole_items = [it for it in items.values() if it["kind"] == "event"
                     and labels.get(it["id"]) == "pothole"]
    for a, b in (("hybrid", "image"), ("hybrid", "sensor"), ("sensor", "image")):
        only_a = sum(1 for it in pothole_items
                     if it["provenance"][a]["triggered"]
                     and not it["provenance"][b]["triggered"])
        only_b = sum(1 for it in pothole_items
                     if it["provenance"][b]["triggered"]
                     and not it["provenance"][a]["triggered"])
        out["paired_tests"][f"{a}_vs_{b}"] = {
            f"only_{a}": only_a, f"only_{b}": only_b,
            "p_value_exact_mcnemar": binom_two_sided(only_a, only_a + only_b)}

    # inter-rater agreement on overlapping items (all reviewer pairs pooled)
    revs = list(by_rev)
    pairs = []
    for i in range(len(revs)):
        for j in range(i + 1, len(revs)):
            common = set(by_rev[revs[i]]) & set(by_rev[revs[j]])
            pairs += [(by_rev[revs[i]][c], by_rev[revs[j]][c]) for c in common]
    out["agreement"] = {"overlap_n": len(pairs), "cohen_kappa": cohen_kappa(pairs)}
    return out


def to_markdown(m):
    L = ["# Transfer-gap metrics (manual review)", "",
         f"Primary reviewer: `{m['primary_reviewer']}` — "
         f"{m['labeled']}/{m['total_items']} items labeled. "
         f"Reviewers: {m['reviewers']}", ""]
    if m["agreement"] and m["agreement"]["overlap_n"]:
        L += [f"Inter-rater agreement: Cohen's kappa = "
              f"{m['agreement']['cohen_kappa']:.3f} "
              f"(n={m['agreement']['overlap_n']} overlapping labels)", ""]
    for plat, b in m["per_platform"].items():
        L += [f"## {plat}", "",
              "| pipeline | events | labeled | TP | FP | unsure | precision "
              "| 95% CI | events/min | est. missed | est. recall* |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for pipe, r in b["pipelines"].items():
            ci = (f"[{r['precision_ci95'][0]:.2f}, {r['precision_ci95'][1]:.2f}]"
                  if r["precision_ci95"][0] is not None else "—")
            prec = f"{r['precision']:.3f}" if r["precision"] is not None else "—"
            L.append(f"| {pipe} | {r['events']} | {r['labeled']} | {r['tp']} "
                     f"| {r['fp']} | {r['unsure']} | {prec} | {ci} "
                     f"| {r['events_per_min']} | {r['est_missed_potholes']} "
                     f"| {r['est_recall_proxy']} |")
        p = b["probes"]
        rate = f"{p['background_rate']:.3f}" if p["background_rate"] is not None else "—"
        L += ["", f"Probes: {p['labeled']} labeled, {p['pothole']} contained a "
              f"pothole (background rate {rate}, "
              f"unsure {p['unsure']}).", ""]
    L += ["## Paired comparisons (exact McNemar on pothole-labeled items)", ""]
    for k, t in m["paired_tests"].items():
        pv = t["p_value_exact_mcnemar"]
        L.append(f"- **{k.replace('_', ' ')}**: discordant "
                 f"{ {kk: vv for kk, vv in t.items() if kk.startswith('only')} }, "
                 f"p = {pv:.4f}" if pv is not None else
                 f"- **{k.replace('_', ' ')}**: no discordant pairs")
    L += ["", "\\* recall is a probe-sampling ESTIMATE "
          "(see docs/TRANSFER_PROTOCOL.md), not exhaustive ground truth."]
    return "\n".join(L)


def main():
    primary = sys.argv[sys.argv.index("--reviewer") + 1] \
        if "--reviewer" in sys.argv else None
    m = compute(primary)
    (TR / "metrics.json").write_text(json.dumps(m, indent=1))
    (TR / "metrics.md").write_text(to_markdown(m))
    print(f"wrote {TR/'metrics.json'} and metrics.md "
          f"({m['labeled']}/{m['total_items']} labeled)")


if __name__ == "__main__":
    main()
