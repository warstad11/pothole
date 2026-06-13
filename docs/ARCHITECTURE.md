# Architecture

System design and data flow. Companion documents:
[METHODOLOGY.md](METHODOLOGY.md) (experimental protocol),
[DATASETS.md](DATASETS.md) (data provenance and label semantics).

## Overview

A local-first research platform with three layers:

```
FastAPI web app (main.py)  ──►  jobs table (SQLite, SQLModel)
                                      │ polled by
Background worker (worker.py) ────────┘
   ├── train_image    → YOLOv8 | Faster R-CNN | U-Net | ResNet18 classifier
   ├── train_sensor   → Random Forest | 1D-CNN | BiLSTM
   ├── train_hybrid   → Feature fusion | Late fusion (6 image×sensor pairings)
   ├── run_inference  → InferenceEngine over a real drive (video + IMU)
   └── validate_datasets / run_tests
```

The web UI submits jobs; the worker executes them and writes results back
into the job record plus `runs/<job_id>/` artifacts (checkpoints,
`metrics.json`, split/normalization/config sidecars).

## Module map

| Path | Responsibility |
|---|---|
| `app/services/labels.py` | **Single source of truth for labels** (annotation files → embedded columns → path keywords → skip) |
| `app/services/splits.py` | Frozen, stratified, file-level train/val/test manifests shared by all stages |
| `app/services/models/base.py` | `BaseModel` contract: `train / evaluate / predict / save / load` |
| `app/services/models/image/yolo.py` | Ultralytics YOLOv8 wrapper + SPPF embeddings + per-image scores |
| `app/services/models/image/torchvision.py` | Faster R-CNN wrapper + FPN embeddings + internal mAP |
| `app/services/models/image/unet.py` | U-Net (smp, resnet34 encoder); true pixel masks (Pothole-600) or box-derived pseudo-masks, recorded as `mask_source` |
| `app/services/models/image/classifier.py` | ResNet18 image-level classification baseline ("why detectors?") |
| `app/services/models/sensor/classical.py` | RandomForest on 68 per-window features (time-domain stats, jerk, kurtosis/skew, ZCR + FFT energy/dominant-frequency/spectral-entropy per axis incl. magnitude) via `features.py` |
| `app/services/models/sensor/dl.py` | 1D-CNN and BiLSTM (config `arch`) on raw windows; train-split global normalization persisted as `norm_stats.json` |
| `app/services/models/sensor/dataset.py` | Windowing (100 samples, stride 20) + normalization |
| `app/services/models/hybrid/fusion.py` | `FeatureFusionWrapper` (MLP over concatenated embeddings), `LateFusionWrapper` (α-weighted score average) |
| `app/services/metrics/detection.py` | IoU, greedy matching, VOC-2010+ all-point AP / mAP |
| `app/services/inference/engine.py` | Drive processing: video frames + sensor windows → fused score timeline |
| `app/services/inference/events.py` | Segment-merge event extraction + NMS backstop |
| `app/services/ingestion/` | SensorLogger/iPhone/nuScenes parsers, resampling to `SENSOR_RESAMPLE_HZ` (100 Hz), video/sensor alignment |
| `app/core/reproducibility.py` | Global seeding helpers |
| `tools/relabel_sensor_data.py` | Pothole-strict relabeling of Data 3-style datasets (auditable manifest) |
| `tools/baselines_sensor.py` | Majority-class + Z-threshold floor baselines on the shared manifest |
| `tools/benchmark_device.py` | CPU vs MPS parity + speed gate |

## Design decisions (and why)

**One label module.** Label inference used to be duplicated in ~6 places
with diverging keyword lists; two models could train on contradictory
ground truth for the same file. All wrappers now call `labels.py`, and the
priority order (annotation file → embedded column → path keywords → skip)
is documented there. "Skip" is a deliberate policy: guessing 0 manufactures
label noise.

**Frozen split manifests.** Splits are data artifacts, not runtime
randomness. `split_manifest.json` lives *with the dataset*, so every model
trained on that dataset shares one partition of reality, results are
reproducible across processes/machines, and adding files cannot silently
re-roll history. The cost — new files are excluded until you explicitly
re-split — is the point.

**Wrappers own their protocol.** Each wrapper's `train()` fits on the train
partition and selects checkpoints on val; `evaluate()` reports the test
partition. The worker passes metrics through verbatim (`None` stays `None`,
with `degraded`/`roc_undefined` flags) — the reporting layer is not allowed
to invent values. This keeps the honesty guarantees in one place per model
rather than scattered through the job runner.

**Per-file fusion granularity.** Hybrid samples are paired *by file stem*
(image `42_Pothole_7.jpg` ↔ sensor `42_Pothole_7.csv`), with
`dataset_index.json` as the authoritative pairing/label map when present.
Positional pairing (row i of one directory listing with row i of another)
is refused — it silently scrambles modality pairs when either side skips a
file.

**Embeddings at native dimensionality.** Zero-padding 256-d embeddings to
512 added constant columns that carried no information but inflated the
claimed embedding size. Every extractor now returns native dims and the
fusion head's `config.json` records what it was trained against; the
inference engine validates dims and disables fusion loudly on mismatch
rather than silently substituting a heuristic per frame.

**Train-split normalization for the sensor CNN.** Per-window z-scoring made
every window unit-variance, erasing amplitude — the most discriminative
feature separating potholes from milder anomalies. Normalization statistics
are now computed on the train partition, saved beside the checkpoint, and
applied identically at training, evaluation, and deployment. (Legacy
checkpoints without stats fall back to per-window scaling, flagged at load.)

**Segment-merge events.** A pothole that keeps scores above threshold for
\>1 s used to produce multiple "events" (point-wise emission + non-transitive
NMS). Contiguous supra-threshold runs now merge into one event with peak
time/score and extent; NMS remains as a backstop.

**Honest failure over silent fallback.** Recurring pattern across the
codebase: when something cannot be computed correctly (no persisted split,
unlabelable file, missing features, dimension mismatch, failed training),
the code raises or skips *visibly* instead of returning zeros, diagonals,
or heuristic substitutes. For a research platform, a wrong number is much
more expensive than a missing one.

## Data flow: hybrid training (`train_hybrid`)

```
dataset_index.json ──┐
images/ ──► best image model (per manifest-filtered job history)
                └─► extract_features() or extract_scores()  [cached, versioned _v2]
sensor/ ──► best sensor model
                └─► extract_features() or extract_scores()
        align by stem ──► features.csv / scores.csv
                          (img_*, snr_* | img_score, snr_score, label, stem, split)
        split column from the dataset's split_manifest.json
        component_split_shared verified & recorded
        ──► FeatureFusionWrapper (scaler+MLP, val-selected checkpoint)
            or LateFusionWrapper (α, τ selected on val by F1)
        ──► evaluate() on test → metrics.json
        best pairing selected by val metric; winner deployed to models/latest/hybrid/
```

## Inference pipeline (`run_inference`)

1. Video → frames at ~5 Hz (effective rate printed); per-frame image score
   + embedding.
2. Sensor CSV → 1 s windows at the trained model's normalization; per-window
   score + embedding.
3. Timeline alignment (constant offset model; see METHODOLOGY §8.7).
4. Per-timestamp fusion: feature-fusion head when dims validate, else
   `max(v, s)` fallback — the `method` field records which produced each
   score.
5. Segment-merge event extraction at the configured threshold; optional
   `strict_filter` gate (off by default, recorded when on).
6. Events persisted with v/s/fused scores and trigger source; ffmpeg clips
   extracted for human review.

## Adding a model

Subclass `BaseModel`; implement `train/evaluate/predict/save/load` honoring
the manifest protocol (use `splits.get_or_create_split` +
`partition_files`); resolve labels via `labels.py`; report metrics with
`eval_split`, `seed`, and framing keys; for hybrid participation add
`extract_features` (embeddings) and `extract_scores` (probabilities), both
returning `(array, labels, stems)` with atomic appends and a length check.
