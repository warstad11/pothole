# Methodology

This document specifies the experimental protocol the platform enforces, so
that numbers it produces can be defended in a paper. Read it together with
[DATASETS.md](DATASETS.md) (label semantics, provenance) and
[ARCHITECTURE.md](ARCHITECTURE.md) (where each rule is implemented).

## 1. Task definition

Binary classification/detection: **pothole (1) vs. not-pothole (0)**, where
the negative class deliberately includes *other road anomalies* — fatigue
("alligator") cracking, spalling, faulting, shoving, slippage cracks, speed
bumps, construction joints — alongside undamaged road.

This is the harder and more useful task. A vibration spike or rough texture
is evidence of *some* anomaly; the research question is whether the models
can tell potholes apart from everything else that shakes a phone or textures
an image. Earlier versions of the datasets labeled every anomaly positive,
which reduces the task to generic anomaly detection — results from that
labeling must never be compared to results under this definition.

Granularities:
- **Sensor**: per-window (1.0 s @ 100 Hz, 0.2 s stride) and per-file.
  A window/file is positive iff it overlaps any pothole-labeled sample.
- **Image**: image-level presence/absence (does the image contain ≥1
  pothole?), plus box-level mAP for the detectors.
- **Hybrid**: per paired sample (image + sensor recording of the same pass).

## 2. Label policy

One module — `app/services/labels.py` — makes every label decision:

1. **Images**: a YOLO-format annotation file is authoritative (non-empty →
   1, empty → 0). Filename/directory keywords are a fallback.
2. **Sensor recordings**: an embedded per-row `label` column is
   authoritative (it can be time-localized, e.g. Data 1). Path keywords are
   the fallback, broadcast to all rows.
3. **Unresolvable samples are SKIPPED**, never defaulted to 0 — an
   unlabeled sample treated as negative is label noise. Empty/NaN label
   cells parse to "unknown", not 0, so a dataset shipped with a blank label
   column falls through to path inference instead of becoming all-negative.

Keyword rules: only `pothole`/`positive` mean positive. Anomaly words
(`crack`, `bump`, `spall`, `fault`, `shoving`, `slippage`, `joint`,
`alligator`/`aligator`) are explicit negatives. Negation guards catch
`no_pothole`, `not_a_pothole`, `pothole_free`. The historical failure modes
this replaces (substring `"damage" in "undamaged"` flipping all negative
files positive; a filename override that force-relabeled entire datasets)
are documented in DATASETS.md and the git history.

## 3. Splits

**File-level, stratified, frozen 70/15/15** (`app/services/splits.py`):

- The unit of assignment is the *file* (= recording session / image), so
  overlapping windows from one recording can never straddle partitions.
- Stratified by file-level label so the rare positive class (e.g. 37/920
  files in Data 3) appears in every partition.
- Deterministic (seed recorded) and **persisted** to
  `split_manifest.json` in the dataset directory. Every stage — RF, CNN,
  image models, fusion — loads the same manifest. Files added later are
  excluded with a warning rather than re-rolled; deleting the manifest
  re-splits and invalidates all previously trained models.

Partition roles, enforced in code:
- **train** — parameter fitting only (incl. normalization statistics,
  feature scalers, class weights).
- **val** — checkpoint selection, hyperparameter/operating-point selection
  (fusion alpha/threshold), and model-pairing selection. Never reported.
  The selection **criterion is F1 on the positive class** (accuracy
  tie-break) for every trainable component — sensor CNN/BiLSTM checkpoints,
  the feature-fusion head, and the late-fusion operating point. Raw
  accuracy is not used: on an imbalanced partition (Data 3 is 96.7%
  negative) an all-negative epoch scores near-ceiling accuracy and
  systematically outranks genuine detectors. Each result records its rule
  as `checkpoint_selection: val_f1_then_acc`.
- **test** — `evaluate()` reports this partition and nothing else. Every
  metrics dict carries `eval_split`, the split strategy, and the seed.

For image datasets with explicit `train/val(valid)/test` folders, the
`test` folder is used for reporting when present (val is what checkpoint
selection consumed — e.g. Ultralytics selects `best.pt` by val fitness).
Datasets without any split structure get a persisted random split
(`val_indices.json`); evaluation *refuses* to run when the persisted split
is missing rather than scoring training data.

Single-file sensor datasets use a contiguous **time-block** 70/15/15 split
with a one-window decorrelation gap between blocks (random window-level
splits leak: adjacent windows share 80% of their samples).

## 4. Leakage controls (explicit list)

| Risk | Control |
|---|---|
| Window overlap across partitions | file-level split unit; time-block split with gap for single files |
| Checkpoint selection on the reported set | select on val, report on test; best checkpoint reloaded before evaluate so saved weights = evaluated weights |
| Majority collapse rewarded by the selection metric | checkpoints and operating points selected by val **F1** (accuracy tie-break), never raw accuracy, on all imbalanced-capable components |
| Scaler/normalization fit on eval data | `StandardScaler` and sensor norm-stats fit on train partition only, persisted with the checkpoint |
| Class weights from eval data | computed from train partition only |
| Winner's curse across model pairings | pairing selected by val metric; test metrics reported for all pairings; `selection_basis` recorded |
| Stacking leakage (fusion features from component models that saw the rows) | fusion partitions follow the file-level manifest; whether component models trained on the same manifest is **verified and recorded** as `component_split_shared` per result — when false, fusion results are internally honest but component leakage cannot be excluded and must be disclosed |
| Stale artifacts | feature caches are versioned (`_v2`); pre-existing checkpoints in an output dir are deleted at train start; first epoch always checkpoints |
| Fabricated metrics | a failed/undefined computation yields `None` plus a `degraded`/`roc_undefined` flag — never a zero matrix or a diagonal ROC |

## 5. Metrics

**Image-level classification metrics** (all models, `metric_framing:
"image_level_classification"`): accuracy, precision, recall, F1, confusion
matrix, ROC/AUC over per-image scores. Per-image score = max detection
confidence (YOLO, uncensored at conf=0.001; FasterRCNN, score_thresh
lowered to 0.001 for evaluation), or max sigmoid pixel probability (UNet).
**Uniform decision threshold 0.5 across models** so P/R/F1 are comparable;
AUC is threshold-free. TN is well-defined under this framing (it is not for
box-level detection).

**Box-level detection**: mAP@0.5. YOLO's comes from the Ultralytics
validator (`map50_source: "ultralytics.val"`); FasterRCNN's from the
in-repo matcher (`internal.calculate_map`, greedy one-to-one matching,
VOC-2010+ all-point AP). These two definitions differ from each other and
from COCO's 101-point AP — the source is recorded per metric and the caveat
belongs in any cross-detector table.

**Segmentation**: on `data/image/Pothole600_Segmentation` (the original
Fan et al. Pothole-600 dataset with true pixel masks and its published
train/val/test split), `mean_iou` is genuine segmentation IoU and is
comparable to the Pothole-600 literature. On YOLO-format datasets the masks
are **box-derived pseudo-masks** (rasterized rectangles) and `mean_iou` only
measures agreement with rectangles. Every result records which applies via
`mask_source` (`pixel_masks` vs `box_pseudo_masks`) — never compare the two.
IoU is computed per-image over images with non-empty union, with
`negative_specificity` reported separately, and is never reported under the
`accuracy` key.

**Sensor metrics**: window-level accuracy/P/R/F1/CM/ROC-AUC on the test
partition.

**Reference baselines** (`tools/baselines_sensor.py`, same windows/labels/
manifest as the learned models): majority-class, and a Z-threshold amplitude
trigger (Mednis-style; threshold selected on val by F1). Measured floors —
Data 3 test: majority acc 0.967 / F1 0.000; Z-threshold F1 0.211, AUC 0.795.
Hybrid Data 1 test: Z-threshold AUC **0.248 (worse than chance)** — bumps
shake harder than potholes, so amplitude triggers actively fail at
pothole-vs-anomaly discrimination. Learned models must be reported against
these floors. The image side has an analogous baseline: a ResNet18
image-level classifier (same protocol as the detectors), answering "is
detection machinery necessary for the image-level task?" empirically.

**Class imbalance**: under the pothole-strict labels the positive class is
rare (3-4% of files in Data 3). Accuracy is dominated by negatives — an
all-negative classifier scores ≈0.97 — so **F1 and ROC-AUC are the headline
metrics**; accuracy is reported for completeness only. Reference point
(cleaned Data 3, file-level test split, seed 42): RF accuracy 0.970,
precision 1.00, recall 0.09, F1 0.17, AUC 0.856.

**Event-level (inference)**: contiguous supra-threshold score runs are
merged into one event per physical anomaly (peak time/score, extent);
greedy ±1 s NMS as a backstop. Optional post-hoc score gates
(`strict_filter`) are off by default and recorded in the result when used,
because they describe an operating point, not the model.

## 6. Fusion protocols

**Feature (intermediate) fusion** — per-file embeddings from each modality
are concatenated and a small MLP (`FusionNet`) is trained on them.
Embeddings: YOLO SPPF global-avg-pool (256-d native), FasterRCNN FPN
top-level via the model's own input transform (256-d native), UNet encoder
(512-d), 1D-CNN penultimate max-pooled over windows (128-d), RF max window
probability (1-d). Dimensions are native (no zero-padding) and recorded in
`config.json` beside every checkpoint; the inference engine validates
embedding dimensions against the loaded head and disables fusion loudly on
mismatch.

**Late (score) fusion** — `fused = α·P_image + (1−α)·P_sensor` over
per-file component probabilities produced by `extract_scores()` (max
detection confidence / max window probability). α and the decision
threshold are selected on the **val** partition by F1 (grid search,
α ∈ {0, 0.05, …, 1}), then reported on **test**. The selected (α, τ) are
persisted to `config.json`. There are no other learned parameters.

Note for comparisons: the RF's "embedding" *is* its predicted probability,
so feature fusion over RF features is closer to stacking than to
representation fusion; with `component_split_shared = false` the meta-
learner trains on partially in-sample component outputs (disclosed
limitation — see §8).

## 7. Reproducibility

- Global seeding (`app/core/reproducibility.py`): python `random`, numpy,
  torch (+CUDA when present); seeded `DataLoader` generators; RF gets an
  explicit `random_state`. Every persisted result records its seed.
- Split manifests, normalization statistics, scalers, fusion architecture
  configs, and selected operating points are all persisted beside
  checkpoints — a result can be traced to: data + manifest + seed + config.
- Residual nondeterminism: GPU kernels are not forced deterministic
  (`torch.use_deterministic_algorithms` is not set); Ultralytics training
  has its own internal augmentation RNG. For publication, report mean ± std
  over ≥3 seeds rather than relying on bit-exactness.
- **Compute device**: selected per architecture by `app/core/device.py`
  (CPU/accelerator parity verified by `tools/benchmark_device.py`; on Apple
  Silicon, MPS is used only where it is actually faster end-to-end —
  FasterRCNN and the small sensor/fusion heads stay on CPU). Training the
  same model on different devices produces different weight trajectories
  (floating-point accumulation order), so every result records its
  `device` — never mix devices within one reported comparison.

## 8. Threats to validity / known limitations

These are honest disclosures the platform records but cannot fix by itself:

1. **Component/fusion dataset provenance.** The hybrid datasets are
   physically separate copies from the component-training datasets. The
   worker verifies directory identity and records `component_split_shared`
   per fusion result; when false, fusion-vs-unimodal comparisons may be
   optimistic for the fusion side and must be caveated. The clean
   configuration for Data 1 (see DATASETS.md): train sensor components on
   `data/hybrid/Data 1 - Both/sensor` (shares the fusion manifest →
   flag true); image components trained on the standard image datasets are
   structurally disjoint from the hybrid images (extracted drive frames;
   zero content-hash overlap, verified 2026-06-10).
2. **Hybrid "Data 3 - not sync" cannot be paired**: image stems are
   Roboflow hashes of opaque numeric originals, sensor stems are category
   names, and there is no `dataset_index.json`. The pairing is not
   recoverable from anything on disk (only the original publisher's
   metadata could provide it). The job fails honestly; hybrid results are
   Data 1-only unless that metadata is obtained.
3. **Cross-model training-budget asymmetry.** YOLO trains with the full
   Ultralytics augmentation pipeline and its own input resolution (640);
   FasterRCNN and UNet train with no augmentation at 800/256 respectively,
   and default epochs differ per model type. Image-model comparisons
   measure *pipelines as configured*, not architectures under equalized
   budgets. Equalize epochs/augmentation or disclose.
4. **Pseudo-mask segmentation GT** (§5).
5. **Sensor amplitude vs. normalization.** The CNN now normalizes with
   train-partition global statistics (amplitude-preserving). Legacy
   checkpoints without `norm_stats.json` fall back to per-window z-scoring
   (amplitude-erasing) — flagged at load time; do not mix the two schemes
   in one comparison. The RF uses raw-unit statistical features; there is
   no unit standardization (g vs m/s²) across heterogeneous sources, so
   cross-dataset transfer results conflate physics with units.
6. **Small positive counts.** Data 3's test partition contains 5 pothole
   files (~33 positive windows). Single-split point estimates are noisy;
   report binomial/bootstrap CIs and multi-seed means.
7. **Time alignment** in the hybrid continuous pipeline assumes a constant
   clock offset (no drift model) and the alignment error is unquantified.
8. **Every metric produced before 2026-06-10** (relabel + split overhaul)
   measures the old anomaly-detection task with leaky splits, and every
   model trained before then is stale. Retrain everything; do not cite
   `runs/` history.
