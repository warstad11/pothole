# Results in Plain Language

Two big experiments. One happy result, one humbling result. Both are real.

## Experiment 1: The lab benchmark (94 tests)

I trained every model with the same rules: fixed train/validation/test
splits, the model never sees its test data during training, and the test
score is the only score that counts. The full tables are in
[results/benchmarks/SUMMARY.md](results/benchmarks/SUMMARY.md).

**Sensor models** (does the road *feel* like a pothole?). Remember:
cracks and bumps must be called "not a pothole," which is hard.

| Model | Best F1 score | Notes |
|---|---|---|
| Random Forest (68 hand-built features) | 0.616 | The most reliable sensor model |
| 1D-CNN (deep learning) | 0.549 | |
| BiLSTM (deep learning) | 0.559 | |
| "Just react to big shakes" baseline | 0.248 | The bar every model must beat |

The baseline matters: a dumb rule that fires on every big shake gets an
F1 of only 0.248, and on one dataset it was actually **worse than
guessing** — because speed bumps shake a car *harder* than potholes do.
Real learning is needed.

**Image models** (does the road *look* like a pothole?). On the one
dataset where positives and negatives come from the same camera
(Data3Images), the models scored F1 0.70–0.97. On two other datasets the
scores look perfect, but that is partly a trick of the data — the
pothole photos and the plain-road photos came from different cameras, so
a model can cheat by recognizing the camera style. I flagged this in the
results instead of bragging about fake 100% scores.

**Hybrid (camera + sensor fused):**

| Setup | F1 | AUC |
|---|---|---|
| Best camera-only on paired drive data | 0.543 | 0.807 |
| Best sensor-only | 0.605 | 0.889 |
| **Best hybrid (picked fairly on validation data)** | **0.622** | **0.894** |

Hybrid wins, but the edge is small. Out of 54 fusion combinations, only
5 beat the best single sensor.

## Experiment 2: The real world (transfer test)

I froze the winning models — no retraining, no tweaking — and ran them
on 6 real iPhone drives and 10 nuScenes self-driving-car scenes. The
models flagged moments that looked like potholes. I then labeled 266 of
those clips **blind**: I could not see which model flagged each clip or
how confident it was. Hidden "probe" clips (random quiet moments) were
mixed in to catch potholes the models missed. I labeled everything twice
to check my own consistency.

| Model | Precision on real drives | What that means |
|---|---|---|
| Camera only | 0.00 | It found zero real potholes |
| Sensor only | ~0.11 | About 1 in 9 alerts was a real pothole |
| Hybrid | ~0.08 | No better than the sensor alone |

Three honest takeaways:

1. **Everything degraded a lot.** Models that looked good in the lab
   fired constantly on real roads, mostly on bumps and rough patches.
2. **The hybrid advantage vanished.** The camera model went blind on
   dashcam video (it was trained on close-up pothole photos), and a
   fusion of "good sensor + blind camera" is no better than the sensor
   alone. I even tested requiring some camera signal before trusting a
   hybrid alert: it made things *worse*, because the camera's opinion
   carried no information.
3. **Higher confidence did not mean more correct.** The models' most
   confident alerts were almost all wrong. That means you cannot fix
   this by raising the alert threshold. The models need better training
   data, not better settings.

Full analysis with statistics (confidence intervals, paired tests, my
labeling consistency scores): [results/transfer/FINDINGS.md](results/transfer/FINDINGS.md).

## Why the negative result matters

It would have been easy to stop after Experiment 1 and say "hybrid
wins!" The real-world test showed that claim does not survive contact
with an actual car yet. That is the most useful thing this project
found, and it points at exactly what to build next: an image model
trained on real dashcam footage. If you are a student looking for a
project, that is the open problem — see the ideas in the
[README](README.md).
