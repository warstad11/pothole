"""
Central compute-device selection with PER-MODEL policy.

Measured on Apple M4 / torch 2.9.1 (tools/benchmark_device.py):

| Model              | MPS vs CPU (raw fwd)  | MPS vs CPU (end-to-end) | Parity   |
|--------------------|-----------------------|--------------------------|----------|
| U-Net (b4 256px)   | 4.7x faster           | wins (compute-bound)     | 6.4e-05  |
| YOLOv8n            | ~equal per image      | ~equal; exact parity     | 0.0      |
| 1D-CNN (b64)       | 5.2x faster           | **0.8x — slightly SLOWER** (dispatch overhead dominates tiny model) | 6.0e-08 |
| FusionNet          | tiny model            | same story as 1D-CNN     | 8.9e-08  |
| FasterRCNN         | **165x SLOWER eval, 73x SLOWER train** (op-fallback thrash) | catastrophic | n/a |

Policy: MPS for the compute-heavy image models (U-Net, YOLO); CPU for
FasterRCNN (running it on MPS is the "Mac optimization made everything
slower" failure mode) and for the tiny sensor/fusion heads where per-batch
dispatch latency outweighs the kernel speedup. The old CORRECTNESS bugs
that originally forced everything onto CPU (BatchNorm train/eval
inconsistency, nonzero deadlock, empty-target training crash) no longer
reproduce on torch 2.9 — verified with parity numbers by
tools/benchmark_device.py — so this is purely a throughput policy.

NOTE: training the same model on different devices yields different weight
trajectories (floating-point accumulation order), so the device is recorded
in every result — never mix devices within one reported comparison.

Overrides:
- POTHOLE_DEVICE=cpu|mps|cuda forces one device for every model.
- POTHOLE_MPS_ALLOW="cnn1d,fusion" (or "all") lifts denylist entries —
  re-measure with tools/benchmark_device.py first.

PYTORCH_ENABLE_MPS_FALLBACK=1 lets any op without an MPS kernel fall back
to CPU instead of raising. It must be set before torch initializes —
worker.py and main.py set it first thing; the setdefault here covers
direct module usage (tools, tests).
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Models whose MPS execution is slower than CPU end-to-end (measured —
# see the table above).
_MPS_DENYLIST = frozenset({"faster_rcnn", "cnn1d", "fusion"})


def resolve_device(model_key: str = None):
    """Return the torch.device to use.

    model_key: optional architecture tag ('faster_rcnn', 'unet', 'cnn1d',
    'fusion', 'yolo') consulted against the MPS denylist.
    """
    import torch

    override = os.environ.get("POTHOLE_DEVICE", "").strip().lower()
    if override:
        if override == "mps" and not torch.backends.mps.is_available():
            print("WARNING: POTHOLE_DEVICE=mps but MPS is unavailable; using cpu.")
            return torch.device("cpu")
        if override == "cuda" and not torch.cuda.is_available():
            print("WARNING: POTHOLE_DEVICE=cuda but CUDA is unavailable; using cpu.")
            return torch.device("cpu")
        return torch.device(override)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        allow = os.environ.get("POTHOLE_MPS_ALLOW", "").strip().lower()
        allowed = {a.strip() for a in allow.split(",") if a.strip()}
        lifted = allow == "all" or (model_key in allowed if model_key else False)
        if model_key in _MPS_DENYLIST and not lifted:
            print(f"[device] {model_key}: using CPU (measured slower on MPS for this "
                  f"architecture — see app/core/device.py; POTHOLE_MPS_ALLOW to override).")
            return torch.device("cpu")
        return torch.device("mps")

    return torch.device("cpu")


def device_str(model_key: str = None) -> str:
    """Device name for Ultralytics' `device=` argument and result records."""
    return str(resolve_device(model_key))
