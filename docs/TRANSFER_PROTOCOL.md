# Real-World Transfer-Gap Protocol (pre-registered)

Written **before** any review labeling began (2026-06-11). Changing
thresholds, checkpoints, matching rules, or label definitions after viewing
model outputs or labels invalidates the protocol and must be disclosed.

## 1. Question

How much does performance degrade when the benchmark campaign's frozen
models are applied to unseen platforms (iPhone dash recordings; nuScenes
scenes), and does hybrid fusion degrade less than its unimodal components?

## 2. Frozen artifacts (no retraining, no tuning)

The val-selected winner of the 2026-06-11 benchmark campaign and its own
two component branches — comparing fusion against its own branches isolates
the fusion contribution:

| Pipeline | Checkpoint | Test (in-distribution) |
|---|---|---|
| image-only | `runs_bench/img_yolov8_Pothole600/full_model.pt` (YOLOv8n, 30 ep) | F1 0.143 on hybrid drives* |
| sensor-only | `runs_bench/snr_rf_Data1HybridSensor/model.joblib` (RF, 68 features) | F1 0.605 / AUC 0.889 |
| hybrid | `runs_bench/fus_feature_yolov8_Pothole600_rf/` (MLP over [256-d YOLO SPPF emb ‖ RF max-window prob], scaler applied) | F1 0.622 / AUC 0.894 |

\* the image branch is known-weak on drive frames and was trained on the
source-confounded Pothole600 pool — disclosed, expected to transfer worst.

**Operating points (frozen):** decision threshold 0.5 on each pipeline's
probability/confidence output — the same uniform rule the benchmark
reported. Event extraction: platform segment-merge (`EventGenerator`,
default gap tolerance).

## 3. Inference procedure

One scoring pass per drive produces all three pipelines from the same
inputs (identical opportunity set by construction):

- Sensor: 1 s windows (100 samples @ 100 Hz, stride 20) → RF P(pothole).
- Vision: frames at 5 Hz → YOLO max box confidence (conf=0.001,
  uncensored) + 256-d SPPF embedding from the same forward pass.
- Hybrid: for each frame whose nearest sensor-window center is within
  0.2 s: fusion head on [embedding ‖ window prob] (train-fit scaler).
- All three timelines restricted to the overlap interval where both
  modalities exist. Drives missing either modality are excluded.
- Deployment-vs-training note: fusion was trained with the RF feature
  aggregated as max window probability per (short) recording; at drive
  scale the aligned single-window probability is used. Disclosed as a
  protocol approximation.

## 4. Review items

- Events from the three pipelines are grouped into **physical-anomaly
  items**: events within ±1.5 s of a group's anchor (the highest-scoring
  member) are the same item. Each item is labeled once; per-pipeline
  trigger provenance and scores are stored server-side.
- **Probes**: per drive, random uniform timestamps ≥2.5 s away from every
  event and probe, count = max(2, 30% of that drive's items). Probes
  estimate the background pothole rate in non-triggered time (the miss
  rate the trigger-only design cannot otherwise see).
- Every item gets a 4 s clip (2 s pre-context). GPS attached where the
  platform provides it (iPhone yes; nuScenes export has no position).

### 4a. Review sampling addendum (added before any labeling)

The full inventory (≈630 event items + ≈200 probes; sensor/hybrid fire
frequently at the frozen 0.5 threshold on real roads — itself a transfer
finding) exceeds practical labeling capacity. A **seeded uniform per-drive
subsample** (`tools/sample_review.py`, seed 1337, targets ≈200 events +
≈60 probes) is marked for review; only sampled items reach the blinded
queue. Uniform random sampling subsamples TP and FP together, so precision
estimates and the probe background rate remain unbiased; per-drive
fractions are recorded in the manifest. The full event inventory with
provenance is retained for event-rate statistics, which use ALL events,
not the sample. The script refuses to run if any label already exists.

## 5. Blinded labeling

- The review page (`/transfer`) shows ONLY the clip (and a still frame for
  nuScenes). No scores, no trigger source, no event-vs-probe distinction.
  Presentation order is a fixed random shuffle (seed 42).
- Labels: **pothole** (bowl-shaped hole in the pavement surface, any
  size), **not_pothole** (cracks, patches, manholes, expansion joints,
  speed bumps, debris, smooth road, shadows), **unsure** (cannot tell).
  Unsure is excluded from TP/FP symmetrically and reported.
- Multiple reviewers supported (independent queues per reviewer id).
  With ≥2 reviewers, Cohen's κ is reported on the overlap; the primary
  reviewer's labels feed the headline metrics; disagreements may be
  adjudicated only by a written rule decided before adjudication.

## 6. Metrics (computed by tools/transfer_metrics.py)

Per platform and pooled, per pipeline:
- **Precision** = TP/(TP+FP) with Wilson 95% CI; TP/FP counts; events;
  events/min.
- **Estimated misses** = probe background rate × (un-covered drive time /
  4 s); **estimated recall proxy** = TP/(TP + est. misses). Always labeled
  as probe-sampling estimates — never reported as measured recall.
- **Accuracy is intentionally not reported**: with no exhaustive negative
  set it is not computable from a trigger-review design.
- Pipeline comparison: exact McNemar (two-sided binomial on discordant
  pairs) over pothole-labeled items, hybrid-vs-image, hybrid-vs-sensor,
  sensor-vs-image.

## 7. Claims discipline

- Two platforms = N=2 transfer settings. Supported claim shape: "the
  transfer gap was smaller for hybrid on both unseen platforms" — not
  "hybrid generalizes across platforms."
- nuScenes scenes are short (~20 s) curated urban segments; expect few
  genuine potholes. If CIs overlap, report direction + CI, not
  superiority.
- All raw label records (item, reviewer, label, timestamp) are retained in
  `results/transfer/labels.jsonl` and published with the results.

## 7a. Protocol amendment — round-2 relabeling (2026-06-12)

After completing round 1 (260/266 items) and seeing the computed metrics,
the reviewer reported **self-perceived labeling inconsistency** and
requested a full relabel. Deviation handling:

- Round-1 labels, metrics, and findings are **archived intact** in
  `results/transfer/round1_archive/` — nothing is deleted.
- Round 2 relabels the SAME 266-item sample, same blinded queue, same
  randomized order, same label definitions (§5 — unchanged).
- Risk disclosed: the reviewer has seen aggregate round-1 results (not
  per-item provenance, which remains blinded). Headline metrics will use
  round-2 labels; round-1 vs round-2 **intra-rater agreement (Cohen's κ)**
  will be reported as a reliability measure.

## 8. Outputs

`results/transfer/`: `review_items.json` (items + provenance + manifest of
checkpoints/thresholds/parameters), `labels.jsonl` (append-only audit
log), `metrics.json` / `metrics.md` (final tables).
