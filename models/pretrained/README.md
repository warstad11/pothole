# Pre-trained models

These are the exact frozen models from the benchmark campaign and the
real-world transfer study — the "winner" hybrid system and its two parts.
With these you can run detection on your own drives **without training
anything** (about 13 MB total).

| File | What it is | Trained on |
|---|---|---|
| `image_yolov8.pt` | YOLOv8n pothole detector (camera) | Pothole-600 + negatives, 30 epochs |
| `sensor_rf.joblib` | Random Forest (motion sensor, 68 features) | Paired drive data (Data 1) |
| `fusion_head.pth` | The hybrid fusion network | Outputs of the two models above |
| `scaler.pkl` | Feature scaler the fusion head needs | (fit on training data only) |
| `config.json` | Fusion input sizes (256 image + 1 sensor) | — |

## Use them

`tools/run_transfer.py` finds these files automatically when the
original training folders are not present (i.e., in a fresh clone):

```bash
# Put a drive in data/iphone/ (see docs/DATA_GUIDE.md), then:
python tools/run_transfer.py
# label your detections at http://localhost:8000/transfer
```

## Honest performance expectations

Read [RESULTS.md](../../RESULTS.md) before trusting these on your roads.
In our blinded real-world test, the sensor model's precision was about
0.11 and the camera model found zero potholes on dashcam video — the
camera model was trained on close-up pothole photos and does not
transfer well. These models are published as a **baseline to beat**, not
a finished product. If you train a better one (especially the camera
side), open an issue — we'd love to hear about it.

## Safety note

`.pt`, `.pth`, and `.joblib` files can run code when loaded (they use
Python pickle). Only load model files from sources you trust — these
were produced by the training code in this repo.
