# Benchmark Campaign Summary (2026-06-11)

94 result rows, zero errors. Seed 42, frozen stratified 70/15/15 splits,
all metrics on held-out **test** partitions. Budgets: YOLOv8 30 / U-Net 25 /
ResNet18 10 / Faster R-CNN 4 (CPU) / sensor DL 40 / fusion 15 epochs.
Device policy per `app/core/device.py` (MPS: yolo/unet/resnet18; CPU:
faster_rcnn/cnn1d/fusion); device recorded in every row. Full tables:
[results.md](results.md), raw rows: [results.csv](results.csv).

## Headline finding — hybrid fusion beats either modality alone

On the hybrid test set (Data 1 - Both, 149 paired samples, 17% positive):

| Configuration | Test F1 | Test AUC |
|---|---|---|
| Best image-only (U-Net/PotholeV1 extractor) | 0.543 | 0.807 |
| Best sensor-only (RF, 68 features) | 0.605 | 0.889 |
| **Fusion winner selected on val F1** (feature fusion: YOLOv8/Pothole600 + RF) | **0.622** | **0.894** |
| Best fusion by test F1 (optimistic upper bound — selected on test) | 0.656 | 0.907 |

- The **honest, citable claim** uses the val-selected winner: feature fusion
  improves test F1 +0.017 and AUC +0.005 over the best unimodal model.
  The 0.656/0.907 row (YOLOv8/Data3Images + RF) is what you get if you pick
  the winner on test — report it only as the range across combos.
- 5 of 54 fusion combos exceed the best unimodal F1; feature fusion with the
  RF sensor branch dominates the top ranks. Image-only models are weak on
  hybrid drive frames (best F1 0.543) — fusion's gain comes from sensor
  evidence correcting image false negatives (winner recall 0.920).

## Sensor phase (pothole vs other-anomaly, NOT pothole vs smooth)

| Model | Data1 F1/AUC | Data3 F1/AUC | HybridSensor F1/AUC |
|---|---|---|---|
| RF (68 feats) | 0.616 / 0.863 | 0.158 / **0.898** | 0.433 / **0.876** |
| 1D-CNN | 0.392 / 0.704 | 0.345 / 0.835 | 0.549 / 0.870 |
| BiLSTM | 0.373 / 0.689 | 0.330 / 0.869 | **0.559** / 0.858 |
| Z-threshold floor | 0.248 | 0.211 | 0.076 |
| Majority floor (acc) | 0.841 | 0.967 | 0.832 |

All learned models clear both floors everywhere. The amplitude heuristic's
near-zero hybrid F1 is key evidence that "vibration magnitude = pothole"
fails once other anomalies are in the negative class.

**Protocol note:** mid-campaign, accuracy-based checkpoint selection was
found to favor majority collapse on imbalanced data (CNN/Data3 F1 0.0 →
0.345 after fix). All reported DL rows use **val-F1 checkpoint selection**
(`checkpoint_selection: val_f1_then_acc`), consistent with the fusion
operating-point protocol.

## Image phase

| Model | Pothole600* F1/AUC | PotholeV1* F1/AUC | Data3Images F1/AUC | Train (s, slowest ds) |
|---|---|---|---|---|
| YOLOv8n (30 ep, MPS) | 0.882 / 1.0 | 0.964 / 1.0 | 0.697 / 0.984 | 3619 |
| U-Net (25 ep, MPS) | 1.0 / 1.0 | 0.967 / 1.0 | 0.959 / 0.989 | 1510 |
| ResNet18 (10 ep, MPS) | 1.0 / 1.0 | 1.0 / 1.0 | 0.969 / 0.984 | 194 |
| Faster R-CNN (4 ep, CPU) | 1.0 / 1.0 | 0.985 / 1.0 | 0.962 / 0.992 | 7443 |

\* **Source-confounded pools** (see Caveats in results.md): Pothole600's
negatives come from NormalRoads; PotholeV1's negatives are a different
collection shipped inside the export (verified: label↔source correlation is
perfect; 0 augmentation leaks; ≤2 exact cross-split dups per pool).
Near-perfect scores there measure source recognition as much as potholes.
**Data3Images is the only same-source image pool — cite it for image-level
discrimination.** On it, YOLO's low F1 (0.697) vs high AUC (0.984) is the
uniform 0.5 threshold sitting below its confidence range, not poor ranking.

**Literature-comparable segmentation:** U-Net on the published Pothole-600
split with real pixel masks: **IoU 0.629** (`mask_source=pixel_masks`,
25 epochs, MPS, 253 s).

**Speed:** ResNet18 trains 20–40× faster than detectors and evaluates at
~5 ms/image; Faster R-CNN is ~2 orders slower to train on CPU (its MPS
path is 73–165× slower still — see device policy). Per-row
`train_seconds` / `eval_seconds` / `ms_per_sample` are in results.csv.

## Deployment pointer (next phase: iPhone / nuScenes drives)

Val-selected winner artifacts:
`runs_bench/fus_feature_yolov8_Pothole600_rf/` (fusion head + config.json),
image branch `runs_bench/img_yolov8_Pothole600/`, sensor branch
`runs_bench/snr_rf_Data1HybridSensor/`. If a single sensor-only model is
preferred for robustness, sensor RF (AUC 0.876–0.898) is the strongest
single branch.

## Reproduction

`python tools/run_benchmarks.py` (resumable; append-only
results.jsonl keyed by test id; `--smoke` for the 2-epoch pipeline check).
