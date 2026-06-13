"""CPU vs MPS/CUDA parity and speed benchmark for every model architecture.

Run this whenever torch is upgraded or results look suspicious:

    .venv/bin/python tools/benchmark_device.py

It verifies, for each architecture used by the platform, that the
accelerated device produces the SAME outputs as CPU (within float32
tolerance) — including the two failure modes that historically forced the
platform onto CPU (BatchNorm train/eval inconsistency; torchvision
detection ops on negative/empty-box samples) — and reports the speedup.

Exit code is non-zero if any parity check fails, so it can gate CI.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn

TOL = 2e-3  # float32 + different kernel orders; classification scores, not bits
failures = []


def check(name, cpu_out, dev_out, tol=TOL):
    err = float((cpu_out - dev_out).abs().max())
    ok = err < tol
    print(f"  parity {name:34s} max|diff|={err:.2e}  {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


def bench(fn, n=5, sync=None):
    fn()  # warmup
    if sync:
        sync()
    t0 = time.time()
    for _ in range(n):
        fn()
    if sync:
        sync()
    return (time.time() - t0) / n


def main():
    from app.core.device import resolve_device

    dev = resolve_device()
    if dev.type == "cpu":
        print("Accelerated device unavailable (or POTHOLE_DEVICE=cpu); nothing to compare.")
        return 0
    sync = torch.mps.synchronize if dev.type == "mps" else (
        torch.cuda.synchronize if dev.type == "cuda" else None)
    print(f"Comparing cpu vs {dev} | torch {torch.__version__}\n")
    torch.manual_seed(0)

    # ---------------- 1D-CNN (sensor) ----------------
    from app.services.models.sensor.dl import Simple1DCNN
    net = Simple1DCNN(3, 2)
    x = torch.randn(64, 3, 100)
    net.train(); _ = net(x)  # populate BN running stats
    net.eval()
    with torch.no_grad():
        cpu_out = torch.softmax(net(x), dim=1)
        dev_net = net.to(dev)
        dev_out = torch.softmax(dev_net(x.to(dev)), dim=1).cpu()
    check("1D-CNN softmax (post-BN-train)", cpu_out, dev_out)
    net_cpu = Simple1DCNN(3, 2); net_cpu.load_state_dict(dev_net.cpu().state_dict()); net_cpu.eval()
    dev_net = dev_net.to(dev)
    with torch.no_grad():
        t_cpu = bench(lambda: net_cpu(x))
        t_dev = bench(lambda: dev_net(x.to(dev)), sync=sync)
    print(f"  speed  1D-CNN batch64:           cpu {t_cpu*1e3:6.1f} ms | {dev.type} {t_dev*1e3:6.1f} ms ({t_cpu/t_dev:4.1f}x)\n")

    # ---------------- FusionNet (hybrid) ----------------
    from app.services.models.hybrid.fusion import FusionNet
    fnet = FusionNet(input_dim=384)
    xf = torch.randn(64, 384)
    fnet.train(); _ = fnet(xf)
    fnet.eval()
    with torch.no_grad():
        cpu_out = torch.softmax(fnet(xf), dim=1)
        fdev = fnet.to(dev)
        dev_out = torch.softmax(fdev(xf.to(dev)), dim=1).cpu()
    check("FusionNet softmax", cpu_out, dev_out)
    print()

    # ---------------- U-Net (segmentation) ----------------
    import segmentation_models_pytorch as smp
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    unet.eval()
    xi = torch.randn(4, 3, 256, 256)
    with torch.no_grad():
        cpu_out = torch.sigmoid(unet(xi))
        udev = unet.to(dev)
        dev_out = torch.sigmoid(udev(xi.to(dev))).cpu()
    check("U-Net sigmoid masks", cpu_out, dev_out, tol=5e-3)
    unet_cpu = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    unet_cpu.load_state_dict(udev.cpu().state_dict()); unet_cpu.eval(); udev = udev.to(dev)
    with torch.no_grad():
        t_cpu = bench(lambda: unet_cpu(xi), n=3)
        t_dev = bench(lambda: udev(xi.to(dev)), n=3, sync=sync)
    print(f"  speed  U-Net batch4 256px:       cpu {t_cpu*1e3:6.0f} ms | {dev.type} {t_dev*1e3:6.0f} ms ({t_cpu/t_dev:4.1f}x)\n")

    # ---------------- Faster R-CNN (detection) ----------------
    # OPT-IN (--frcnn): FasterRCNN is on the MPS denylist because detection
    # post-processing thrashes between MPS and CPU. Measured on M4/torch
    # 2.9.1: eval 131.7s vs 0.79s CPU (165x slower), train step 186.2s vs
    # 2.56s CPU (73x slower). Correctness is fine (BatchNorm parity 9.5e-07;
    # empty-box-target train step OK) — it is purely a performance trap.
    if "--frcnn" not in sys.argv:
        print("FasterRCNN MPS test skipped (extremely slow by design — it is "
              "CPU-pinned via the denylist). Run with --frcnn to re-measure.\n")
        if failures:
            print(f"PARITY FAILURES: {failures}")
            return 1
        print("All parity checks passed — accelerated device is safe to use.")
        return 0

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    frcnn = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    frcnn.eval()
    img = [torch.rand(3, 480, 640)]
    with torch.no_grad():
        cpu_res = frcnn(img)[0]
        fdev = frcnn.to(dev)
        dev_res = fdev([img[0].to(dev)])[0]
    n_cpu, n_dev = len(cpu_res["boxes"]), len(dev_res["boxes"])
    k = min(n_cpu, n_dev, 5)
    if k > 0:
        check("FRCNN top-5 scores", cpu_res["scores"][:k], dev_res["scores"][:k].cpu(), tol=5e-3)
        check("FRCNN top-5 boxes (px)", cpu_res["boxes"][:k], dev_res["boxes"][:k].cpu(), tol=1.0)
    print(f"  detections: cpu {n_cpu} vs {dev.type} {n_dev}")

    # The historical blocker: train step with an EMPTY-box (negative) target
    fdev.train()
    targets = [
        {"boxes": torch.zeros((0, 4)).to(dev), "labels": torch.zeros((0,), dtype=torch.int64).to(dev)},
        {"boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0]]).to(dev), "labels": torch.tensor([1]).to(dev)},
    ]
    imgs2 = [torch.rand(3, 480, 640).to(dev) for _ in range(2)]
    losses = fdev(imgs2, targets)
    total = sum(losses.values())
    total.backward()
    ok = bool(torch.isfinite(total).item())
    print(f"  FRCNN train step w/ empty-box target on {dev.type}: "
          f"{'OK' if ok else 'FAIL'} (loss={total.item():.3f})")
    if not ok:
        failures.append("FRCNN empty-target train step")
    fdev.eval()
    frcnn_cpu = fasterrcnn_resnet50_fpn(weights="DEFAULT"); frcnn_cpu.eval()
    with torch.no_grad():
        t_cpu = bench(lambda: frcnn_cpu(img), n=3)
        t_dev = bench(lambda: fdev([img[0].to(dev)]), n=3, sync=sync)
    print(f"  speed  FRCNN eval 480x640:       cpu {t_cpu*1e3:6.0f} ms | {dev.type} {t_dev*1e3:6.0f} ms ({t_cpu/t_dev:4.1f}x)\n")

    if failures:
        print(f"PARITY FAILURES: {failures}")
        print("Set POTHOLE_DEVICE=cpu until resolved.")
        return 1
    print("All parity checks passed — accelerated device is safe to use.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
