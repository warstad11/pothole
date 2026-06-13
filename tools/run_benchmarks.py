"""Overnight benchmark campaign: image vs sensor vs hybrid fusion.

Protocol (see docs/METHODOLOGY.md):
- Image datasets are re-pooled into stratified 70/15/15 benchmark copies
  (seed 42, originals untouched); Pothole600_Segmentation keeps its
  published split. Sensor datasets use the shared split manifests.
- train fits, val selects, evaluate() reports TEST only.
- Every row records timing (train s, eval s, ms/sample), device, seed,
  epochs, and an automatic sanity DIAGNOSIS for accuracy <0.5 or >0.95.
- Results: results/benchmarks/results.jsonl (one row per test, append-only,
  resumable), regenerated results.csv + results.md after every row.

Run:
    .venv/bin/python tools/run_benchmarks.py            # full campaign
    .venv/bin/python tools/run_benchmarks.py --smoke    # 20-min plumbing test
"""

import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

SMOKE = "--smoke" in sys.argv
SEED = 42

OUT = ROOT / "results" / "benchmarks"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = OUT / ("results_smoke.jsonl" if SMOKE else "results.jsonl")
# Namespaced by mode: smoke-epoch extractor features must never be
# reused by the full run (cache tags don't encode the weights).
CACHE = OUT / ("cache_smoke" if SMOKE else "cache")
CACHE.mkdir(exist_ok=True)
RUNS = ROOT / ("runs_bench_smoke" if SMOKE else "runs_bench")
RUNS.mkdir(exist_ok=True)
BENCH_DATA = ROOT / "data" / "bench" / "image"

EPOCHS = {
    "yolov8": 2 if SMOKE else 30,
    "faster_rcnn": 1 if SMOKE else 4,
    "unet": 2 if SMOKE else 25,
    "resnet18": 2 if SMOKE else 10,
    "cnn": 3 if SMOKE else 40,
    "lstm": 3 if SMOKE else 40,
    "fusion": 5 if SMOKE else 15,
}

IMAGE_DATASETS = {
    # name -> pool source dirs. NOTE: Pothole-600 itself contains ONLY
    # pothole images — image-level classification metrics are degenerate
    # without negatives, so the NormalRoads negative set (600 no-pothole
    # images, collected for exactly this purpose) is pooled in. Recorded in
    # bench_composition.json.
    "Pothole600": [ROOT / "data/image/RealPothole600_Converted",
                   ROOT / "data/image/NormalRoads"],
    "PotholeV1": [ROOT / "data/image/raw/Pothole.v1-raw.yolov8"],
    "Data3Images": [ROOT / "data/image/raw/Data 3 - Images/train",
                    ROOT / "data/image/raw/Data 3 - Images/val"],
}
SEG_DATASET = ROOT / "data/image/Pothole600_Segmentation"  # published split, real masks
SENSOR_DATASETS = {
    "Data1": ROOT / "data/sensor/Data 1",
    "Data3": ROOT / "data/sensor/Data 3 - Just Sensor",
    "Data1HybridSensor": ROOT / "data/hybrid/Data 1 - Both/sensor",
}
HYBRID = ROOT / "data/hybrid/Data 1 - Both"

if SMOKE:
    IMAGE_DATASETS = {"Pothole600": IMAGE_DATASETS["Pothole600"]}
    SENSOR_DATASETS = {"Data3": SENSOR_DATASETS["Data3"],
                       "Data1HybridSensor": SENSOR_DATASETS["Data1HybridSensor"]}


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def done_keys():
    # Error rows do NOT count as done: a transient failure must retry on
    # resume. The retry appends a fresh row; tables keep the last per key.
    keys = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                row = json.loads(line)
                if not row.get("error"):
                    keys.add(row["key"])
            except Exception:
                pass
    return keys


def diagnose(metrics):
    """Automatic double-check for out-of-range accuracy, from the confusion
    matrix (no re-run needed). Returns (flags, diagnosis)."""
    flags, notes = [], []
    acc = metrics.get("accuracy")
    cm = metrics.get("confusion_matrix")
    if acc is None or not cm:
        return flags, None
    cm = np.array(cm, dtype=float)
    total = cm.sum()
    if total == 0:
        return flags, None
    true_dist = cm.sum(axis=1) / total      # rows = ground truth
    pred_dist = cm.sum(axis=0) / total      # cols = predictions
    majority = float(true_dist.max())

    if acc > 0.95 or acc < 0.5:
        flags.append("ACC_OUT_OF_RANGE")
        if min(true_dist) == 0:
            notes.append("eval set is SINGLE-CLASS — accuracy/precision are "
                         "not meaningful (dataset composition, not a code bug)")
        if min(pred_dist) == 0:
            notes.append(f"model predicts ONE class only "
                         f"(pred dist {pred_dist.round(3).tolist()})")
        if acc > 0.95 and acc <= majority + 0.02:
            notes.append(f"accuracy ~= majority-class floor ({majority:.3f}) — "
                         f"class imbalance; judge by F1/AUC instead")
        elif acc > 0.95:
            notes.append(f"accuracy {acc:.3f} exceeds majority floor "
                         f"{majority:.3f} — genuinely strong OR leakage; "
                         f"verify eval_split=test and split provenance")
        if acc < 0.5:
            notes.append(f"below chance (majority floor {majority:.3f}) — "
                         f"check label semantics for this combination")
    return flags, ("; ".join(notes) if notes else None)


def write_row(row):
    row["timestamp"] = datetime.now(timezone.utc).isoformat()
    row["seed"] = SEED
    with open(RESULTS, "a") as f:
        f.write(json.dumps(row) + "\n")
    regenerate_tables()


def metric_row(metrics, train_s, eval_s):
    samples = metrics.get("samples") or 0
    return {
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "accuracy": metrics.get("accuracy"),
        "f1": metrics.get("f1"),
        "roc_auc": metrics.get("roc_auc"),
        "map50": metrics.get("map50"),
        "mean_iou": metrics.get("mean_iou"),
        "mask_source": metrics.get("mask_source"),
        "eval_split": metrics.get("eval_split"),
        "samples": samples,
        "device": metrics.get("device"),
        "train_seconds": round(train_s, 1),
        "eval_seconds": round(eval_s, 1),
        "ms_per_sample": round(eval_s / samples * 1000, 2) if samples else None,
        "confusion_matrix": metrics.get("confusion_matrix"),
    }


def regenerate_tables():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    if not rows:
        return
    # Keep the LAST row per key (a retried test supersedes its error row).
    by_key = {}
    for r in rows:
        by_key[r["key"]] = r
    df = pd.DataFrame(list(by_key.values()))
    df.to_csv(OUT / ("results_smoke.csv" if SMOKE else "results.csv"), index=False)

    md = ["# Benchmark results", "",
          f"Generated {datetime.now().isoformat()} — seed {SEED}, "
          f"budgets {EPOCHS}", ""]
    cols = ["algorithm", "dataset", "precision", "recall", "accuracy", "f1",
            "roc_auc", "train_seconds", "eval_seconds", "ms_per_sample",
            "eval_split", "device", "flags", "diagnosis"]

    def fmt(df_part, extra_cols=()):
        use = [c for c in list(cols[:2]) + list(extra_cols) + cols[2:] if c in df_part.columns]
        d = df_part[use].copy()
        for c in d.columns:
            if d[c].dtype == float:
                d[c] = d[c].round(3)
        return d.to_markdown(index=False)

    for phase, title, extra in [("image", "Image models", ("map50", "mean_iou", "mask_source")),
                                ("sensor", "Sensor models", ()),
                                ("hybrid", "Hybrid fusion (Data 1 - Both)",
                                 ("fusion_type", "image_algo", "image_src", "sensor_algo"))]:
        part = df[df.get("phase") == phase]
        if len(part):
            md += [f"## {title}", "", fmt(part, extra), ""]
            # Averages across datasets per algorithm
            if phase in ("image", "sensor"):
                num = part.groupby("algorithm")[
                    [c for c in ("precision", "recall", "accuracy", "f1", "roc_auc")
                     if c in part.columns]].mean(numeric_only=True).round(3)
                md += [f"### {title} — average across datasets", "",
                       num.to_markdown(), ""]
    errs = df[df.get("error").notna()] if "error" in df.columns else []
    if len(errs):
        md += ["## Errors", "", errs[["key", "error"]].to_markdown(index=False), ""]
    md += ["## Caveats", "",
           "- **Pothole600 (image-level rows)**: positives are Pothole-600 "
           "(Fan et al.) frames; negatives are pooled from the separate "
           "NormalRoads collection because Pothole-600 ships no negative "
           "images. Classification-style metrics on this pool can be "
           "inflated by source separability (camera/scene statistics), so "
           "near-perfect accuracy here is NOT evidence of pothole "
           "discrimination. Cite the U-Net true-IoU row "
           "(`image|unet|Pothole600Seg`, real pixel masks, published split) "
           "for literature-comparable Pothole-600 performance.",
           "- **PotholeV1**: label is perfectly confounded with image "
           "source — every positive is a Roboflow pothole photo, every "
           "negative an asphalt/unpaved image from a different collection "
           "(shipped that way in the raw export). Verified: 0 augmentation "
           "leaks, 2/1265 exact cross-split duplicates. Near-perfect scores "
           "(ResNet18 1.0) measure source recognition as much as pothole "
           "detection. **Data3Images is the only image pool whose positives "
           "and negatives share one source (Clemson)** — treat it as the "
           "honest image-level benchmark (2/1600 exact cross-split dups; "
           "near-duplicate consecutive frames remain a residual threat, see "
           "METHODOLOGY §8).",
           "- Image P/R/F1 use a uniform image-level decision threshold of "
           "0.5 on per-image max box confidence (YOLO/FRCNN) or mask/clf "
           "probability (U-Net/ResNet18); AUC is threshold-free.",
           "- Sensor accuracy on imbalanced datasets (Data3 ~3.3% positive) "
           "tracks the majority floor; judge by F1/AUC against the recorded "
           "majority and Z-threshold baseline rows.", ""]
    (OUT / ("results_smoke.md" if SMOKE else "results.md")).write_text("\n".join(md))


# ===================================================================== #
# Phase 0 — benchmark image datasets (stratified clean splits)          #
# ===================================================================== #

def build_bench_image_dataset(name, sources):
    from app.services.labels import infer_label_from_yolo_txt, infer_label_from_path
    dst = BENCH_DATA / name
    if (dst / "data.yaml").exists():
        log(f"P0 {name}: bench copy exists, reusing")
        return dst
    log(f"P0 {name}: pooling from {len(sources)} source(s)")
    pool = {}
    for src in sources:
        for p in sorted(list(src.rglob("*.jpg")) + list(src.rglob("*.png"))):
            if {"label", "labels", "masks", "mask"} & set(p.parts):
                continue
            lbl = infer_label_from_yolo_txt(p)
            if lbl is None:
                # negative-only pools (e.g. NormalRoads) carry no label txts
                lbl = infer_label_from_path(p)
            if lbl is None:
                continue
            pool.setdefault(p.stem, (p, lbl))   # dedupe by stem
    stems = sorted(pool)
    rng = np.random.RandomState(SEED)
    by_label = {}
    for s in stems:
        by_label.setdefault(pool[s][1], []).append(s)
    assign = {}
    for lbl, group in sorted(by_label.items()):
        group = sorted(group)
        rng.shuffle(group)
        n = len(group)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        for s in group[:n_tr]:
            assign[s] = "train"
        for s in group[n_tr:n_tr + n_va]:
            assign[s] = "valid"
        for s in group[n_tr + n_va:]:
            assign[s] = "test"

    from app.services.labels import infer_label_from_yolo_txt as _resolve
    import re

    def label_txt_for(img):
        # reuse the centralized candidate logic by probing the same paths
        cands = [img.parent.parent / "labels" / img.with_suffix(".txt").name,
                 img.parent / "labels" / img.with_suffix(".txt").name,
                 img.with_suffix(".txt")]
        if "images" in img.parts:
            parts = list(img.parts)
            parts[parts.index("images")] = "labels"
            cands.append(Path(*parts).with_suffix(".txt"))
        for c in cands:
            if c.exists():
                return c
        return None

    counts = {"train": [0, 0], "valid": [0, 0], "test": [0, 0]}
    for s, split in assign.items():
        img, lbl = pool[s]
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dst / split / "images" / img.name)
        txt = label_txt_for(img)
        target_txt = dst / split / "labels" / (img.stem + ".txt")
        if txt is not None:
            shutil.copy(txt, target_txt)
        else:
            target_txt.write_text("")   # explicit negative
        counts[split][lbl] += 1
    yaml_text = (f"train: {dst / 'train' / 'images'}\n"
                 f"val: {dst / 'valid' / 'images'}\n"
                 f"test: {dst / 'test' / 'images'}\n"
                 f"nc: 1\nnames: ['pothole']\n")
    (dst / "data.yaml").write_text(yaml_text)
    log(f"P0 {name}: train neg/pos={counts['train']}, valid={counts['valid']}, "
        f"test={counts['test']}")
    (dst / "bench_composition.json").write_text(json.dumps(
        {"seed": SEED, "counts": counts, "sources": [str(s) for s in sources]}, indent=2))
    return dst


# ===================================================================== #
# Phase 1 — image models                                                #
# ===================================================================== #

def make_image_wrapper(algo, out_dir):
    if algo == "yolov8":
        from app.services.models.image import YOLOWrapper
        return YOLOWrapper(out_dir.name, {"output_dir": str(out_dir), "model_name": "yolov8n.pt"})
    if algo == "faster_rcnn":
        from app.services.models.image import FasterRCNNWrapper
        return FasterRCNNWrapper(out_dir.name, {"output_dir": str(out_dir)})
    if algo == "unet":
        from app.services.models.image import UNetWrapper
        return UNetWrapper(out_dir.name, {"output_dir": str(out_dir)})
    if algo == "resnet18":
        from app.services.models.image import ResNet18ClassifierWrapper
        return ResNet18ClassifierWrapper(out_dir.name, {"output_dir": str(out_dir)})
    raise ValueError(algo)


WEIGHT_FILES = {"yolov8": "full_model.pt", "faster_rcnn": "faster_rcnn.pth",
                "unet": "unet.pth", "resnet18": "resnet18.pth"}


def run_image_phase(done):
    log("PHASE image: start")
    bench = {name: build_bench_image_dataset(name, srcs)
             for name, srcs in IMAGE_DATASETS.items()}
    algos = ["yolov8", "unet", "resnet18", "faster_rcnn"]  # FRCNN last (slowest)
    # U-Net runs on the bench copies like everyone else (comparable
    # image-level metrics) PLUS one extra row on the published Pothole-600
    # segmentation split for TRUE pixel-mask IoU (literature-comparable).
    jobs = [(a, n, bench[n]) for a in algos for n in bench]
    jobs.append(("unet", "Pothole600Seg", SEG_DATASET))
    for algo, name, ds_path in jobs:
            key = f"image|{algo}|{name}"
            if key in done:
                log(f"skip {key} (done)")
                continue
            out_dir = RUNS / f"img_{algo}_{name}"
            row = {"key": key, "phase": "image", "modality": "image",
                   "algorithm": algo, "dataset": name,
                   "dataset_path": str(ds_path), "epochs": EPOCHS[algo]}
            try:
                w = make_image_wrapper(algo, out_dir)
                t0 = time.time()
                if algo == "yolov8":
                    w.train(ds_path, epochs=EPOCHS[algo], single_cls=True)
                    w.save()
                else:
                    w.train(ds_path, epochs=EPOCHS[algo])
                    w.save()
                train_s = time.time() - t0
                t0 = time.time()
                metrics = w.evaluate(ds_path)
                eval_s = time.time() - t0
                row.update(metric_row(metrics, train_s, eval_s))
                row["flags"], row["diagnosis"] = diagnose(metrics)
                log(f"DONE {key}: acc={row['accuracy']} f1={row['f1']} "
                    f"auc={row['roc_auc']} train={train_s:.0f}s "
                    f"{'FLAG: ' + str(row['diagnosis']) if row['flags'] else ''}")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                log(f"ERROR {key}: {e}\n{traceback.format_exc()}")
            write_row(row)


# ===================================================================== #
# Phase 2 — sensor models + floor baselines                             #
# ===================================================================== #

def make_sensor_wrapper(algo, out_dir):
    if algo == "rf":
        from app.services.models.sensor import RandomForestWrapper
        return RandomForestWrapper(out_dir.name, {"output_dir": str(out_dir)})
    from app.services.models.sensor import CNN1DWrapper
    return CNN1DWrapper(out_dir.name, {"output_dir": str(out_dir),
                                       "arch": "lstm" if algo == "lstm" else "cnn"})


def run_sensor_phase(done):
    log("PHASE sensor: start")
    for algo in ["rf", "cnn", "lstm"]:
        for name, ds in SENSOR_DATASETS.items():
            key = f"sensor|{algo}|{name}"
            if key in done:
                log(f"skip {key} (done)")
                continue
            out_dir = RUNS / f"snr_{algo}_{name}"
            row = {"key": key, "phase": "sensor", "modality": "sensor",
                   "algorithm": algo, "dataset": name, "dataset_path": str(ds),
                   "epochs": EPOCHS.get(algo, 0) if algo != "rf" else None}
            try:
                w = make_sensor_wrapper(algo, out_dir)
                t0 = time.time()
                if algo == "rf":
                    w.train(ds)
                else:
                    w.train(ds, epochs=EPOCHS[algo])
                w.save()
                train_s = time.time() - t0
                t0 = time.time()
                metrics = w.evaluate(ds)
                eval_s = time.time() - t0
                row.update(metric_row(metrics, train_s, eval_s))
                row["flags"], row["diagnosis"] = diagnose(metrics)
                log(f"DONE {key}: acc={row['accuracy']} f1={row['f1']} "
                    f"auc={row['roc_auc']} train={train_s:.0f}s "
                    f"{'FLAG: ' + str(row['diagnosis']) if row['flags'] else ''}")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                log(f"ERROR {key}: {e}\n{traceback.format_exc()}")
            write_row(row)

    # Floor baselines (majority + z-threshold) via the shared tool logic
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_curve, auc)
    sys.path.insert(0, str(ROOT / "tools"))
    from baselines_sensor import windows_for, z_scores
    from app.services.models.sensor.classical import RandomForestWrapper
    from app.services.splits import get_or_create_split, partition_files
    for name, ds in SENSOR_DATASETS.items():
        for base in ["majority", "zthresh"]:
            key = f"sensor|baseline_{base}|{name}"
            if key in done:
                continue
            row = {"key": key, "phase": "sensor", "modality": "sensor",
                   "algorithm": f"baseline_{base}", "dataset": name}
            try:
                wrapper = RandomForestWrapper("baseline", {"output_dir": str(RUNS / "baseline_tmp")})
                files = wrapper._scan_files(ds)
                manifest = get_or_create_split(ds, files, seed=SEED,
                                               label_fn=wrapper._file_level_label)
                parts = partition_files(files, manifest)
                Xtr, ytr = windows_for(parts['train'], wrapper)
                Xva, yva = windows_for(parts['val'], wrapper)
                Xte, yte = windows_for(parts['test'], wrapper)
                t0 = time.time()
                if base == "majority":
                    pred = np.full_like(yte, int(np.bincount(ytr).argmax()))
                    scores = None
                else:
                    sva, ste = z_scores(Xva), z_scores(Xte)
                    grid = np.quantile(sva, np.linspace(0.5, 0.999, 60))
                    thr = max(grid, key=lambda t: f1_score(yva, (sva >= t).astype(int), zero_division=0))
                    pred = (ste >= thr).astype(int)
                    scores = ste
                    row["z_threshold"] = float(thr)
                eval_s = time.time() - t0
                m = {"accuracy": float(accuracy_score(yte, pred)),
                     "precision": float(precision_score(yte, pred, zero_division=0)),
                     "recall": float(recall_score(yte, pred, zero_division=0)),
                     "f1": float(f1_score(yte, pred, zero_division=0)),
                     "samples": int(len(yte)), "eval_split": "test",
                     "confusion_matrix": [[int(((yte == 0) & (pred == 0)).sum()), int(((yte == 0) & (pred == 1)).sum())],
                                          [int(((yte == 1) & (pred == 0)).sum()), int(((yte == 1) & (pred == 1)).sum())]]}
                if scores is not None and len(set(yte)) > 1:
                    fpr, tpr, _ = roc_curve(yte, scores)
                    m["roc_auc"] = float(auc(fpr, tpr))
                row.update(metric_row(m, 0.0, eval_s))
                row["flags"], row["diagnosis"] = diagnose(m)
                log(f"DONE {key}: acc={row['accuracy']} f1={row['f1']}")
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"
                log(f"ERROR {key}: {e}\n{traceback.format_exc()}")
            write_row(row)


# ===================================================================== #
# Phase 3 — hybrid fusion on Data 1 - Both                              #
# ===================================================================== #

def cached_extract(tag, fn):
    path = CACHE / f"{tag}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return d["feats"], d["labels"], list(d["stems"])
    feats, labels, stems = fn()
    if feats is None or len(feats) == 0:
        raise RuntimeError(f"extraction produced nothing for {tag}")
    np.savez(path, feats=feats, labels=labels, stems=np.array(stems))
    return feats, labels, list(stems)


def inline_metrics(y_true, scores, thr=0.5):
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_curve, auc)
    pred = (np.asarray(scores) >= thr).astype(int)
    y = np.asarray(y_true).astype(int)
    m = {"accuracy": float(accuracy_score(y, pred)),
         "precision": float(precision_score(y, pred, zero_division=0)),
         "recall": float(recall_score(y, pred, zero_division=0)),
         "f1": float(f1_score(y, pred, zero_division=0)),
         "samples": int(len(y)), "eval_split": "test",
         "confusion_matrix": [[int(((y == 0) & (pred == 0)).sum()), int(((y == 0) & (pred == 1)).sum())],
                              [int(((y == 1) & (pred == 0)).sum()), int(((y == 1) & (pred == 1)).sum())]]}
    if len(set(y)) > 1:
        fpr, tpr, _ = roc_curve(y, scores)
        m["roc_auc"] = float(auc(fpr, tpr))
    return m


def run_hybrid_phase(done):
    log("PHASE hybrid: start (Data 1 - Both)")
    from app.services.splits import get_or_create_split, stem_partition_map
    from app.services.models.hybrid.fusion import FeatureFusionWrapper, LateFusionWrapper

    # Authoritative labels + frozen split (sensor-dir manifest, shared with
    # the sensor components trained in phase 2 -> component_split_shared=true)
    index = json.loads((HYBRID / "dataset_index.json").read_text())
    stem_label = {Path(k).stem: int(v["label"]) for k, v in index.items()}
    sensor_dir = HYBRID / "sensor"
    files = sorted(sensor_dir.glob("*.csv"))
    manifest = get_or_create_split(sensor_dir, files, seed=SEED,
                                   label_fn=lambda p: stem_label.get(p.stem))
    part_map = stem_partition_map(manifest)

    image_algos = ["yolov8", "unet", "faster_rcnn"]
    image_srcs = list(IMAGE_DATASETS.keys())
    sensor_algos = ["rf", "cnn", "lstm"]
    if SMOKE:
        image_algos, image_srcs, sensor_algos = ["yolov8"], image_srcs[:1], ["rf"]

    # ---- extract embeddings + scores for every extractor ----
    img_data = {}   # (algo, src) -> {"feats": (F, l, s), "scores": (S, l, s)}
    for algo in image_algos:
        for src in image_srcs:
            wdir = RUNS / f"img_{algo}_{src}"
            wpath = wdir / WEIGHT_FILES[algo]
            if not wpath.exists():
                log(f"hybrid: missing weights {wpath}; skipping extractor {algo}/{src}")
                continue
            try:
                w = make_image_wrapper(algo, wdir)
                w.load(wpath)
                tag = f"hyb_img_{algo}_{src}"
                feats = cached_extract(tag + "_feats", lambda: w.extract_features(HYBRID / "images"))
                scores = cached_extract(tag + "_scores", lambda: w.extract_scores(HYBRID / "images"))
                img_data[(algo, src)] = {"feats": feats, "scores": scores}
                log(f"hybrid: extractor {algo}/{src} ready "
                    f"({feats[0].shape[0]} feats x {feats[0].shape[1] if feats[0].ndim > 1 else 1})")
            except Exception as e:
                log(f"ERROR extractor {algo}/{src}: {e}\n{traceback.format_exc()}")

    snr_data = {}
    for algo in sensor_algos:
        wdir = RUNS / f"snr_{algo}_Data1HybridSensor"
        wfile = wdir / ("model.joblib" if algo == "rf" else "model.pth")
        if not wfile.exists():
            log(f"hybrid: missing sensor weights {wfile}; skipping {algo}")
            continue
        try:
            w = make_sensor_wrapper(algo, wdir)
            w.load(wfile)
            tag = f"hyb_snr_{algo}"
            feats = cached_extract(tag + "_feats", lambda: w.extract_features(sensor_dir))
            scores = cached_extract(tag + "_scores", lambda: w.extract_scores(sensor_dir))
            snr_data[algo] = {"feats": feats, "scores": scores}
            log(f"hybrid: sensor extractor {algo} ready")
        except Exception as e:
            log(f"ERROR sensor extractor {algo}: {e}\n{traceback.format_exc()}")

    def aligned(img_tuple, snr_tuple):
        """Stem-aligned arrays restricted to manifest stems w/ index labels."""
        i_arr, _, i_stems = img_tuple
        s_arr, _, s_stems = snr_tuple
        i_map = {s: k for k, s in enumerate(i_stems)}
        s_map = {s: k for k, s in enumerate(s_stems)}
        common = [s for s in i_stems
                  if s in s_map and s in stem_label and part_map.get(s)]
        I = np.array([np.atleast_1d(i_arr[i_map[s]]) for s in common])
        S = np.array([np.atleast_1d(s_arr[s_map[s]]) for s in common])
        y = np.array([stem_label[s] for s in common])
        split = [part_map[s] for s in common]
        return common, I, S, y, split

    # ---- unimodal-on-hybrid baselines (identical test rows) ----
    for (algo, src), d in img_data.items():
        key = f"hybrid|image_only|{algo}|{src}"
        if key in done:
            continue
        row = {"key": key, "phase": "hybrid", "modality": "image-only-on-hybrid",
               "algorithm": f"{algo} (image only)", "fusion_type": "none",
               "image_algo": algo, "image_src": src, "dataset": "Data1-Both"}
        try:
            stems, I, S, y, split = aligned(d["scores"], list(snr_data.values())[0]["scores"] if snr_data else d["scores"])
            te = [i for i, p in enumerate(split) if p == "test"]
            m = inline_metrics(y[te], I[te].reshape(-1))
            row.update(metric_row(m, 0.0, 0.0))
            row["flags"], row["diagnosis"] = diagnose(m)
            log(f"DONE {key}: acc={row['accuracy']} f1={row['f1']} auc={row['roc_auc']}")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            log(f"ERROR {key}: {e}")
        write_row(row)

    for algo, d in snr_data.items():
        key = f"hybrid|sensor_only|{algo}"
        if key in done:
            continue
        row = {"key": key, "phase": "hybrid", "modality": "sensor-only-on-hybrid",
               "algorithm": f"{algo} (sensor only)", "fusion_type": "none",
               "sensor_algo": algo, "dataset": "Data1-Both"}
        try:
            scores, _, stems = d["scores"]
            sel = [(k, s) for k, s in enumerate(stems)
                   if s in stem_label and part_map.get(s) == "test"]
            y = np.array([stem_label[s] for _, s in sel])
            sc = np.array([np.atleast_1d(scores[k])[0] for k, _ in sel])
            m = inline_metrics(y, sc)
            row.update(metric_row(m, 0.0, 0.0))
            row["flags"], row["diagnosis"] = diagnose(m)
            log(f"DONE {key}: acc={row['accuracy']} f1={row['f1']} auc={row['roc_auc']}")
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
            log(f"ERROR {key}: {e}")
        write_row(row)

    # ---- fusion combinatorics ----
    for (ia, isrc), idata in img_data.items():
        for sa, sdata in snr_data.items():
            for ftype in ["feature", "late"]:
                key = f"hybrid|{ftype}|{ia}|{isrc}|{sa}"
                if key in done:
                    log(f"skip {key} (done)")
                    continue
                row = {"key": key, "phase": "hybrid", "modality": "hybrid",
                       "algorithm": f"{ia}+{sa} ({ftype})", "fusion_type": ftype,
                       "image_algo": ia, "image_src": isrc, "sensor_algo": sa,
                       "dataset": "Data1-Both", "epochs": EPOCHS["fusion"],
                       "component_split_shared": True}
                try:
                    pdir = RUNS / f"fus_{ftype}_{ia}_{isrc}_{sa}"
                    pdir.mkdir(exist_ok=True)
                    kind = "scores" if ftype == "late" else "feats"
                    stems, I, S, y, split = aligned(idata[kind], sdata[kind])
                    data = {}
                    if ftype == "late":
                        data["img_score"] = I.reshape(-1)
                        data["snr_score"] = S.reshape(-1)
                        csv_name = "scores.csv"
                    else:
                        for i in range(I.shape[1]):
                            data[f"img_{i}"] = I[:, i]
                        for i in range(S.shape[1]):
                            data[f"snr_{i}"] = S[:, i]
                        csv_name = "features.csv"
                    data["label"] = y
                    data["stem"] = stems
                    data["split"] = split
                    pd.DataFrame(data).to_csv(pdir / csv_name, index=False)

                    if ftype == "late":
                        fw = LateFusionWrapper(pdir.name, {"output_dir": str(pdir)})
                    else:
                        fw = FeatureFusionWrapper(pdir.name, {
                            "output_dir": str(pdir),
                            "input_dim_image": int(I.shape[1]),
                            "input_dim_sensor": int(S.shape[1])})
                    t0 = time.time()
                    tr = fw.train(pdir, epochs=EPOCHS["fusion"]) or {}
                    if tr.get("status") == "failed":
                        raise RuntimeError(f"fusion train failed: {tr.get('reason')}")
                    train_s = time.time() - t0
                    t0 = time.time()
                    metrics = fw.evaluate(pdir)
                    eval_s = time.time() - t0
                    row.update(metric_row(metrics, train_s, eval_s))
                    row["selection"] = {k: v for k, v in tr.items()
                                        if k in ("val_acc", "val_f1", "alpha", "threshold")}
                    row["flags"], row["diagnosis"] = diagnose(metrics)
                    log(f"DONE {key}: acc={row['accuracy']} f1={row['f1']} "
                        f"auc={row['roc_auc']} "
                        f"{'FLAG: ' + str(row['diagnosis']) if row['flags'] else ''}")
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                    log(f"ERROR {key}: {e}\n{traceback.format_exc()}")
                write_row(row)


def main():
    t_start = time.time()
    log(f"BENCHMARK CAMPAIGN start (smoke={SMOKE}) budgets={EPOCHS}")
    (OUT / "config.json").write_text(json.dumps({
        "seed": SEED, "epochs": EPOCHS, "smoke": SMOKE,
        "image_datasets": {k: [str(s) for s in v] for k, v in IMAGE_DATASETS.items()},
        "sensor_datasets": {k: str(v) for k, v in SENSOR_DATASETS.items()},
        "hybrid_dataset": str(HYBRID),
        "started": datetime.now(timezone.utc).isoformat()}, indent=2))
    done = done_keys()
    log(f"{len(done)} rows already complete (resume mode)" if done else "fresh run")

    run_sensor_phase(done_keys())     # cheap, gives early results + hybrid components
    run_image_phase(done_keys())      # YOLO/UNet/classifier on MPS, FRCNN last on CPU
    run_hybrid_phase(done_keys())     # needs phase 1 + 2 weights

    log(f"ALL PHASES COMPLETE in {(time.time()-t_start)/3600:.1f}h — "
        f"results: {RESULTS}")


if __name__ == "__main__":
    main()
