# Datasets

This document records the provenance, composition, and **label semantics** of
every dataset in the platform. Label semantics matter more here than in most
projects: the central research question is distinguishing potholes from *other
road anomalies* that also produce vibration and visual texture, so what counts
as "positive" must be exact.

## Task definition

**Label 1 = pothole. Label 0 = everything else**, explicitly including other
road anomalies: fatigue ("alligator") cracking, spalls, faulting, shoving,
slippage cracks, speed bumps, and construction joints, as well as undamaged
road. A model that scores well here genuinely detects potholes; a model that
merely detects "rough road" will produce false positives on the anomaly
negatives and score poorly. This is deliberate — see
[METHODOLOGY.md](METHODOLOGY.md).

## Sensor datasets (`data/sensor/`)

### Data 3 - Just Sensor (920 files)

Per-file recordings of single anomaly passes. Columns:
`label, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, seconds_elapsed`
(gyro columns are present but all-zero in this source). The label column is
constant per file.

| Filename prefix      | Files | Label | Category                       |
|----------------------|------:|:-----:|--------------------------------|
| `Pothole*`           |    37 | **1** | Pothole                        |
| `FatigueCrack*`      |   292 |   0   | Fatigue cracking               |
| `Undamaged_*`        |   220 |   0   | Undamaged road                 |
| `Construction joint*`|   126 |   0   | Construction joint             |
| `Spall*`             |    98 |   0   | Spalling                       |
| `Aligator cracking*` |    75 |   0   | Alligator cracking (sic — the misspelling is in the source data) |
| `Faulting*`          |    43 |   0   | Faulting                       |
| `Bump*`              |    25 |   0   | Bump                           |
| `Shoving*`           |     3 |   0   | Shoving                        |
| `Slippage cracks*`   |     1 |   0   | Slippage cracking              |

**Relabeling history (important).** The dataset as originally distributed had
`label=1` for *every* anomaly category (700 of 920 files), i.e. it was an
anomaly-detection dataset, not a pothole dataset. Only 5.3% of its positives
were actual potholes. On 2026-06-10 the labels were corrected on disk to the
table above using `tools/relabel_sensor_data.py`; the full per-file change
record is in `relabel_manifest.json` inside the dataset directory, and the
original labels are recoverable from it (or from the filename rule:
original label = 0 iff `Undamaged_*`). **Any metrics computed before this
correction describe anomaly detection, not pothole detection, and must not be
compared to current numbers.**

The class imbalance (37 positive files / ~3.6%) is intrinsic to the corrected
task. Use F1, precision/recall, and ROC-AUC; accuracy is dominated by the
negative class (an all-negative classifier scores ~0.97).

### Data 1 (297 files)

Continuous drive segments with **per-row, time-localized labels**. Columns:
`time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, label`.

| Filename suffix       | Files | Label semantics                          |
|-----------------------|------:|------------------------------------------|
| `*_isolated_potholes` |    97 | label=1 rows during the pothole pass     |
| `*_isolated_bumps`    |    97 | label=0 throughout (bumps are negatives) |
| `*_all_anomalies`     |    97 | mixed 0/1 rows (potholes time-localized) |

**59 of the 297 files are empty (0 rows)** — an artifact of the original
export. They are skipped by all loaders (a file shorter than one window
cannot produce a sample) and reported by `tools/relabel_sensor_data.py`.

### Data 2 - Clean (6 files, ~127k rows)

Long continuous drives: `plain_road_{1..4}.csv` (negatives) and
`plain_road_potholes_{1,2}.csv` (contain potholes). Labels are resolved from
filenames via `app/services/labels.py` (the `pothole` keyword wins over
`plain`, by design).

### `data/sensor/OLD/`

Superseded copies and experiments. Not used by any pipeline; excluded from
training paths. Kept for provenance only — do not point jobs at it.

## Image datasets (`data/image/`)

All image datasets use YOLO format: `images/<split>/*.jpg` +
`labels/<split>/*.txt`, where a label file lists `class cx cy w h` boxes and
an empty/absent label file means "no pothole". Label files are the
authoritative source for image labels; filename keywords are only a fallback
(`app/services/labels.py`).

| Dataset | Images | Notes |
|---|---:|---|
| `RealPothole600_Converted` | 600 | train/val split folders |
| `NormalRoads` | 600 | negative (no-pothole) images |
| `raw/Pothole.v1-raw.yolov8` | 1,265 | has train/val/test folders |
| `raw/Data 3 - Images` | 2,600 | image counterpart of Data 3 |
| `Pothole600_Segmentation` | 600 | **true pixel masks** (Fan et al. Pothole-600; train 240 / val 180 / test 180, published split) — the segmentation benchmark |

## Hybrid datasets (`data/hybrid/`)

Paired image + sensor samples, matched **by file stem** (e.g. `images/42_Pothole_7.jpg`
↔ `sensor/42_Pothole_7.csv`). `dataset_index.json`, when present, is the
authoritative stem → label map.

### Data 1 - Both — USABLE (verified 2026-06-10)

1,002 image/sensor pairs with **100% stem intersection** and a complete
`dataset_index.json`. Composition: 171 Pothole / 231 Bump / 600 Normal.
Index labels are pothole-strict (171 positives, 831 negatives) — the index
was regenerated on 2026-06-10 because the original generator counted Bump
as positive (231 mislabeled samples, 57% of the old positive class); the
sensor CSVs' own label columns were always correct and now agree with the
index.

Provenance matters for leakage analysis: these pairs were **extracted from
the Data 1 drive sessions** (`data/image/raw/Data 1`, video frames at
annotation events + matching sensor windows). Consequences:
- The images are NOT copies of the image-training datasets
  (RealPothole600 / NormalRoads / Pothole.v1) — verified by content hash:
  zero overlap. Image components trained on those datasets cannot have
  seen hybrid samples.
- The sensor recordings come from the SAME physical drives as
  `data/sensor/Data 1` (different extracts of sessions 1–103). A sensor
  component trained on `data/sensor/Data 1` may therefore share drives with
  hybrid test rows. **Train sensor components on
  `data/hybrid/Data 1 - Both/sensor` itself** — its `split_manifest.json`
  (702/151/149, created 2026-06-10) is then shared with the fusion stage
  and the worker records `component_split_shared=true`.

### Data 3 - not sync — NOT USABLE for fusion

Image stems are Roboflow hashes of opaque numeric originals (`108.jpg`),
sensor stems are category names (`Aligator cracking10_15-Feb-2021`), and no
`dataset_index.json` exists. The pairing is not recoverable from anything
on disk; it would require the original dataset publisher's image↔recording
metadata. Hybrid jobs on it fail honestly (no matching stems). Its sensor
copy WAS relabeled pothole-strict (2026-06-10) for hygiene.

## Split manifests

Every dataset directory used for training acquires a `split_manifest.json`
on first use: a deterministic, label-stratified, **file-level** 70/15/15
train/val/test assignment (`app/services/splits.py`). The manifest freezes
membership: files added later are *excluded* (with a warning) rather than
re-rolled, because re-splitting would invalidate previously trained models.
Delete the manifest to re-split — and retrain everything if you do.

All pipeline stages (sensor models, image models, fusion) consume the same
manifest, which is what makes the held-out test partition meaningful across
the whole platform. See [METHODOLOGY.md](METHODOLOGY.md#splits).

## Adding a new dataset

1. Place files following the layouts above (sensor CSVs need either a
   `label` column or unambiguous filenames; images need YOLO label files).
2. If labels encode "any anomaly" rather than "pothole", relabel first —
   extend `tools/relabel_sensor_data.py` with the category map.
3. The first training job creates the split manifest; check the printed
   train/val/test counts and the per-class balance before trusting results.
4. Record the dataset's provenance (collection vehicle, phone model, mount,
   sampling rate, labeling procedure) here — reviewers will ask.
