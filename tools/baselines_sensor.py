"""Trivial sensor baselines: majority-class and Z-threshold peak detection.

Why: under heavy class imbalance, learned-model metrics mean little without
floor references. The majority-class baseline shows what "predict nothing"
scores; the Z-threshold heuristic (Mednis et al.-style amplitude trigger) is
the classic non-learned pothole detector. Any learned model in the paper
must clearly beat both.

Protocol matches the learned models exactly: same windowing (100/20), same
labels (app/services/labels.py), same frozen split manifest. The Z-threshold
is selected on the VAL partition by F1 and reported on TEST.

Usage:
    .venv/bin/python tools/baselines_sensor.py "data/sensor/Data 3 - Just Sensor"
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)

from app.services.models.sensor.classical import RandomForestWrapper, WINDOW_SIZE, STRIDE
from app.services.splits import get_or_create_split, partition_files


def windows_for(files, wrapper):
    """Raw windows + labels using the exact pipeline the learned models use."""
    X, y = [], []
    for f in files:
        try:
            df = wrapper._load_file(f)
        except Exception:
            continue
        if df is None or len(df) < WINDOW_SIZE:
            continue
        cols = [c for c in df.select_dtypes(include=[np.number]).columns
                if any(k in str(c).lower() for k in ('acc', 'ax', 'ay', 'az'))][:3]
        if not cols:
            continue
        data = np.nan_to_num(df[cols].values.astype(np.float64))
        labels = np.nan_to_num(df['label'].astype(float).values)
        for i in range(0, len(df) - WINDOW_SIZE + 1, STRIDE):
            X.append(data[i:i + WINDOW_SIZE])
            y.append(1 if labels[i:i + WINDOW_SIZE].max() >= 0.5 else 0)
    return X, np.array(y)


def z_scores(windows):
    """Z-threshold score per window: max absolute deviation of the
    acceleration magnitude from its window median (orientation-robust
    variant of the classic vertical-axis amplitude trigger)."""
    out = []
    for w in windows:
        mag = np.sqrt((w ** 2).sum(axis=1))
        out.append(float(np.abs(mag - np.median(mag)).max()))
    return np.array(out)


def report(name, y_true, y_pred, scores=None):
    m = {
        "baseline": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "samples": int(len(y_true)),
        "eval_split": "test",
    }
    if scores is not None and len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, scores)
        m["roc_auc"] = float(auc(fpr, tpr))
    print(f"  {name:22s} acc={m['accuracy']:.3f} P={m['precision']:.3f} "
          f"R={m['recall']:.3f} F1={m['f1']:.3f}"
          + (f" AUC={m['roc_auc']:.3f}" if 'roc_auc' in m else ""))
    return m


def main():
    ds = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sensor/Data 3 - Just Sensor")
    print(f"Trivial baselines on {ds} (shared manifest, test partition)")

    wrapper = RandomForestWrapper("baselines", {"output_dir": "runs_verify/baselines"})
    files = wrapper._scan_files(ds)
    manifest = get_or_create_split(ds, files, seed=42, label_fn=wrapper._file_level_label)
    parts = partition_files(files, manifest)

    Xtr, ytr = windows_for(parts['train'], wrapper)
    Xva, yva = windows_for(parts['val'], wrapper)
    Xte, yte = windows_for(parts['test'], wrapper)
    print(f"  windows: {len(ytr)} train / {len(yva)} val / {len(yte)} test "
          f"(test positives: {int(yte.sum())})")

    results = []

    # 1. Majority class (from TRAIN)
    maj = int(np.bincount(ytr).argmax())
    results.append(report(f"majority-class ({maj})", yte, np.full_like(yte, maj)))

    # 2. Z-threshold: threshold selected on VAL by F1, reported on TEST
    sva, ste = z_scores(Xva), z_scores(Xte)
    grid = np.quantile(np.concatenate([sva]), np.linspace(0.5, 0.999, 60))
    best_thr, best_f1 = grid[0], -1.0
    for thr in grid:
        f1 = f1_score(yva, (sva >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    print(f"  z-threshold selected on val: thr={best_thr:.3f} (val F1={best_f1:.3f})")
    results.append(report("z-threshold (Mednis)", yte, (ste >= best_thr).astype(int), ste))

    # hybrid datasets end in a generic 'sensor' dir — use the parent's name
    ds_tag = ds.parent.name if ds.name.lower() == "sensor" else ds.name
    out = Path("runs_verify") / f"baselines_{ds_tag.replace(' ', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"dataset": str(ds), "manifest_counts": manifest.get("counts"),
                               "z_threshold": float(best_thr), "results": results}, indent=2))
    print(f"  written: {out}")


if __name__ == "__main__":
    main()
