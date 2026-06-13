"""Real-world transfer-gap experiment runner.

Runs the FROZEN benchmark-campaign winner and its two component branches
over unlabeled real drives (iPhone sessions + nuScenes scenes), extracts
per-pipeline events from ONE shared score timeline (identical opportunity
set by construction), groups events across pipelines into physical-anomaly
review items, injects blind probe segments (random non-triggered times, for
miss-rate estimation), extracts clips, and writes a randomized BLINDED
review queue for the web UI.

Protocol: docs/TRANSFER_PROTOCOL.md (pre-registered — do not change
thresholds/checkpoints after looking at outputs).

Frozen artifacts (2026-06-11 campaign, val-selected winner):
  image  : runs_bench/img_yolov8_Pothole600/full_model.pt   (YOLOv8n, 30 ep)
  sensor : runs_bench/snr_rf_Data1HybridSensor/model.joblib (RF, 68 feats)
  hybrid : runs_bench/fus_feature_yolov8_Pothole600_rf/     (MLP + scaler)

Usage:
  .venv/bin/python tools/run_transfer.py            # all drives
  .venv/bin/python tools/run_transfer.py --drive data/iphone/<session>
"""
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from app.services.inference.events import EventGenerator
from app.services.inference.clips import ClipExtractor
from app.services.ingestion.alignment import DriveTimeline

SEED = 42
OUT = ROOT / "results" / "transfer"
OUT.mkdir(parents=True, exist_ok=True)

def _first_existing(*candidates):
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]  # load_models() reports it as missing

# Original campaign artifacts when present (full working tree); otherwise
# the published pre-trained bundle (fresh clone of the repo).
IMAGE_W = _first_existing(
    ROOT / "runs_bench" / "img_yolov8_Pothole600" / "full_model.pt",
    ROOT / "models" / "pretrained" / "image_yolov8.pt")
SENSOR_W = _first_existing(
    ROOT / "runs_bench" / "snr_rf_Data1HybridSensor" / "model.joblib",
    ROOT / "models" / "pretrained" / "sensor_rf.joblib")
FUSION_W = _first_existing(
    ROOT / "runs_bench" / "fus_feature_yolov8_Pothole600_rf" / "model.pth",
    ROOT / "models" / "pretrained" / "fusion_head.pth")
FUSION_DIR = FUSION_W.parent

# Frozen operating points — same uniform decision rule the benchmark
# campaign reported (0.5 on each pipeline's probability/confidence output).
THRESHOLDS = {"image": 0.5, "sensor": 0.5, "hybrid": 0.5}

VISION_HZ = 5.0          # frame sampling rate (matches platform inference)
WINDOW = 100             # sensor window samples (1 s @ 100 Hz)
STRIDE = 20              # sensor window stride
ALIGN_TOL = 0.2          # max |frame_t - window_center_t| for fusion pairing
MATCH_WINDOW = 1.5       # cross-pipeline events within +/- this = same anomaly
PROBE_FRACTION = 0.30    # probes per drive ~ 30% of triggered items
PROBE_MIN = 2            # at least this many probes per drive
PROBE_CLEARANCE = 2.5    # probes must be this far (s) from any event/probe
CLIP_DIR_NAME = "clips_transfer"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- #
# Model loading (frozen checkpoints)                                     #
# --------------------------------------------------------------------- #

def load_models():
    from app.services.models.image.yolo import YOLOWrapper
    from app.services.models.sensor import RandomForestWrapper
    from app.services.models.hybrid.fusion import FeatureFusionWrapper

    for p in (IMAGE_W, SENSOR_W, FUSION_W):
        if not p.exists():
            raise FileNotFoundError(f"frozen checkpoint missing: {p}")

    yolo = YOLOWrapper("transfer_image", {"output_dir": str(IMAGE_W.parent.parent)})
    yolo.load(IMAGE_W)

    rf = RandomForestWrapper("transfer_sensor", {"output_dir": str(SENSOR_W.parent)})
    rf.load(SENSOR_W)

    fusion = FeatureFusionWrapper("transfer_fusion", {"output_dir": str(FUSION_DIR)})
    fusion.load(FUSION_W)

    return yolo, rf, fusion


# --------------------------------------------------------------------- #
# Drive discovery                                                        #
# --------------------------------------------------------------------- #

def find_video(drive: Path):
    cands = sorted(drive.glob("*.mp4")) + sorted((drive / "Camera").glob("*.mp4")) \
        if (drive / "Camera").exists() else sorted(drive.glob("*.mp4"))
    return cands[0] if cands else None


def discover_drives():
    drives = []
    iphone_base = ROOT / "data" / "iphone"
    if iphone_base.exists():
        for d in sorted(p for p in iphone_base.iterdir() if p.is_dir()):
            drives.append(("iphone", d))
    nusc_base = ROOT / "data" / "nuscenes"
    if nusc_base.exists():
        for d in sorted(nusc_base.glob("scene-*")):
            if d.is_dir():
                drives.append(("nuscenes", d))
    return drives


def ensure_processed(platform, drive):
    """sensor.csv + video must exist; run ingestion if missing."""
    if not (drive / "sensor.csv").exists():
        if platform == "iphone":
            from app.services.ingestion.iphone import iPhoneIngestionService
            iPhoneIngestionService().process_session(drive)
        else:
            from app.services.ingestion.nuscenes import NuScenesIngestionService
            NuScenesIngestionService().process_scene(drive)
    video = find_video(drive)
    if video is None or not (drive / "sensor.csv").exists():
        return None
    return video


# --------------------------------------------------------------------- #
# Score timelines (one pass; three pipelines)                            #
# --------------------------------------------------------------------- #

def sensor_timeline(rf, drive):
    """RF window probabilities. Returns (times_video, probs, lat, lon)."""
    df = pd.read_csv(drive / "sensor.csv")
    cols = ["accel_x", "accel_y", "accel_z"]
    if not all(c in df.columns for c in cols):
        raise ValueError(f"{drive.name}: sensor.csv missing accel columns")
    timeline = DriveTimeline.load(drive) \
        if (drive / "alignment.json").exists() else None

    data = df[cols].to_numpy(dtype=np.float64)
    tsec = df["seconds_elapsed"].to_numpy(dtype=np.float64)
    has_gps = {"latitude", "longitude"}.issubset(df.columns)

    times, probs, lats, lons = [], [], [], []
    feats = []
    centers = []
    n = (len(data) - WINDOW) // STRIDE + 1
    for k in range(max(n, 0)):
        a, b = k * STRIDE, k * STRIDE + WINDOW
        w = data[a:b]
        feats.append(rf.predict_embedding(w))
        centers.append(a + WINDOW // 2)
    if not feats:
        return np.array([]), np.array([]), [], []
    X = np.asarray(feats)
    P = rf.model.predict_proba(X)[:, 1]
    for c, p in zip(centers, P):
        t_rel = float(tsec[c])
        t_vid = timeline.rel_to_video(t_rel) if timeline else t_rel
        times.append(t_vid)
        probs.append(float(p))
        if has_gps:
            lats.append(float(df["latitude"].iloc[c]))
            lons.append(float(df["longitude"].iloc[c]))
    return np.asarray(times), np.asarray(probs), lats, lons


def vision_timeline(yolo, video):
    """One forward pass per sampled frame with the SPPF hook registered:
    returns (times, max_conf_scores, embeddings)."""
    import cv2
    feature_box = {}

    def hook(module, inp, out):
        if out.dim() == 4:
            feature_box["v"] = out.mean(dim=[2, 3]).detach().cpu().numpy()
        else:
            feature_box["v"] = out.detach().cpu().numpy()

    target = yolo.model.model.model[9]
    handle = target.register_forward_hook(hook)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps / VISION_HZ)), 1)
    times, scores, embs = [], [], []
    idx = 0
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    idx += 1
                    continue
                feature_box["v"] = None
                # conf=0.001 -> uncensored score (same as benchmark ROC rule)
                res = yolo.model(frame, verbose=False, conf=0.001)
                conf = 0.0
                if len(res) > 0 and len(res[0].boxes) > 0:
                    conf = float(res[0].boxes.conf.max().cpu().item())
                v = feature_box.get("v")
                if v is None:
                    idx += 1
                    continue
                emb = v[0].flatten() if v.ndim > 1 else v.flatten()
                times.append(idx / fps)
                scores.append(conf)
                embs.append(emb.astype(np.float32))
            idx += 1
    finally:
        cap.release()
        handle.remove()
    return np.asarray(times), np.asarray(scores), embs


def fused_timeline(fusion, v_times, v_embs, s_times, s_probs):
    """Fusion probability at each vision timestamp whose nearest sensor
    window center is within ALIGN_TOL. Returns (times, probs, idx_pairs)."""
    if len(s_times) == 0 or len(v_times) == 0:
        return np.array([]), np.array([]), []
    times, probs, pairs = [], [], []
    for i, t in enumerate(v_times):
        j = int(np.argmin(np.abs(s_times - t)))
        if abs(s_times[j] - t) > ALIGN_TOL:
            continue
        p = fusion.predict((v_embs[i], np.array([s_probs[j]], dtype=np.float32)))
        times.append(float(t))
        probs.append(float(p))
        pairs.append((i, j))
    return np.asarray(times), np.asarray(probs), pairs


# --------------------------------------------------------------------- #
# Events, grouping, probes                                               #
# --------------------------------------------------------------------- #

def restrict(times, scores, lo, hi):
    m = (times >= lo) & (times <= hi)
    return times[m], scores[m]


def pipeline_events(times, scores, threshold):
    if len(times) == 0:
        return []
    return EventGenerator.generate_events(times, scores, threshold=threshold)


def group_items(events_by_pipe):
    """Greedy time-grouping of the union of pipeline events into physical-
    anomaly review items (events within MATCH_WINDOW of a group's anchor
    join that group)."""
    pool = []
    for pipe, evts in events_by_pipe.items():
        for e in evts:
            pool.append((float(e["time"]), pipe, e))
    pool.sort(key=lambda x: x[0])
    groups = []
    for t, pipe, e in pool:
        placed = False
        for g in groups:
            if abs(t - g["anchor"]) <= MATCH_WINDOW:
                g["members"].append((pipe, e))
                # anchor follows the highest-scoring member
                if e["score"] > g["peak_score"]:
                    g["anchor"], g["peak_score"] = t, float(e["score"])
                placed = True
                break
        if not placed:
            groups.append({"anchor": t, "peak_score": float(e["score"]),
                           "members": [(pipe, e)]})
    return groups


def sample_probes(rng, lo, hi, event_times, count):
    """Uniform random times >= PROBE_CLEARANCE away from every event and
    every previously placed probe."""
    probes, attempts = [], 0
    while len(probes) < count and attempts < 4000:
        attempts += 1
        t = rng.uniform(lo + 2.0, max(hi - 2.0, lo + 2.0))
        if all(abs(t - et) >= PROBE_CLEARANCE for et in event_times) and \
           all(abs(t - pt) >= PROBE_CLEARANCE for pt in probes):
            probes.append(round(t, 2))
    return probes


def interp_gps(t, s_times, lats, lons):
    if not lats or len(s_times) == 0:
        return None, None
    j = int(np.argmin(np.abs(s_times - t)))
    return lats[j], lons[j]


# --------------------------------------------------------------------- #
# Main                                                                   #
# --------------------------------------------------------------------- #

def main():
    only = None
    if "--drive" in sys.argv:
        only = Path(sys.argv[sys.argv.index("--drive") + 1]).resolve()

    rng = random.Random(SEED)
    yolo, rf, fusion = load_models()
    log("frozen checkpoints loaded "
        f"(fusion input_dim={fusion.input_dim}, scaler={'yes' if fusion.scaler else 'NO'})")
    if not fusion.scaler:
        raise RuntimeError("fusion scaler missing — deployment would be uncalibrated")

    items, drive_meta = [], []
    item_id = 0

    for platform, drive in discover_drives():
        if only and drive.resolve() != only:
            continue
        try:
            video = ensure_processed(platform, drive)
            if video is None:
                log(f"SKIP {drive.name}: missing video or sensor.csv (opportunity set requires both)")
                continue
            log(f"DRIVE {platform}/{drive.name}: scoring…")

            s_times, s_probs, lats, lons = sensor_timeline(rf, drive)
            v_times, v_scores, v_embs = vision_timeline(yolo, video)
            h_times, h_probs, _ = fused_timeline(fusion, v_times, v_embs, s_times, s_probs)

            if len(s_times) == 0 or len(v_times) == 0:
                log(f"SKIP {drive.name}: empty modality timeline")
                continue

            # identical opportunity set: overlap of both modalities
            lo = max(s_times.min(), v_times.min())
            hi = min(s_times.max(), v_times.max())
            if hi - lo < 5.0:
                log(f"SKIP {drive.name}: overlap {hi-lo:.1f}s too short")
                continue

            evts = {}
            for pipe, (tt, ss) in {
                "image": restrict(v_times, v_scores, lo, hi),
                "sensor": restrict(s_times, s_probs, lo, hi),
                "hybrid": restrict(h_times, h_probs, lo, hi),
            }.items():
                evts[pipe] = pipeline_events(tt, ss, THRESHOLDS[pipe])

            groups = group_items(evts)
            all_event_times = [g["anchor"] for g in groups]
            n_probes = max(PROBE_MIN, int(round(PROBE_FRACTION * len(groups))))
            probes = sample_probes(rng, lo, hi, all_event_times, n_probes)

            clip_dir = drive / CLIP_DIR_NAME
            clip_dir.mkdir(exist_ok=True)
            url_prefix = f"/static/{platform}/{drive.name}"

            def make_item(kind, t, members=None):
                nonlocal item_id
                item_id += 1
                iid = f"T{item_id:04d}"
                clip_path = clip_dir / f"{iid}.mp4"
                ClipExtractor.extract_clip(video, t, clip_path)
                lat, lon = interp_gps(t, s_times, lats, lons)
                prov = {}
                if members:
                    for pipe, e in members:
                        prov[pipe] = {
                            "triggered": True, "score": float(e["score"]),
                            "time": float(e["time"]),
                            "t_start": float(e.get("t_start", e["time"])),
                            "t_end": float(e.get("t_end", e["time"]))}
                for pipe in ("image", "sensor", "hybrid"):
                    prov.setdefault(pipe, {"triggered": False})
                img_url = None
                if platform == "nuscenes":
                    try:
                        from app.services.ingestion.nuscenes import NuScenesIngestionService
                        img_url = NuScenesIngestionService.get_image_url(drive, t)
                    except Exception:
                        pass
                return {
                    "id": iid, "kind": kind, "platform": platform,
                    "drive": drive.name, "time": round(float(t), 2),
                    "clip_url": f"{url_prefix}/{CLIP_DIR_NAME}/{iid}.mp4"
                                if clip_path.exists() else None,
                    "image_url": img_url, "gps": {"lat": lat, "lon": lon},
                    "provenance": prov,
                }

            for g in groups:
                items.append(make_item("event", g["anchor"], g["members"]))
            for t in probes:
                items.append(make_item("probe", t))

            n_clipless = sum(1 for it in items if it["drive"] == drive.name
                             and it["clip_url"] is None)
            drive_meta.append({
                "platform": platform, "drive": drive.name,
                "video": video.name, "interval": [round(float(lo), 2), round(float(hi), 2)],
                "duration_s": round(float(hi - lo), 1),
                "events": {p: len(e) for p, e in evts.items()},
                "review_items": len(groups), "probes": len(probes),
                "clips_failed": n_clipless,
                "gps_available": bool(lats),
            })
            log(f"DONE {drive.name}: image={len(evts['image'])} "
                f"sensor={len(evts['sensor'])} hybrid={len(evts['hybrid'])} "
                f"-> {len(groups)} items + {len(probes)} probes"
                + (f" ({n_clipless} clips FAILED)" if n_clipless else ""))
        except Exception as e:
            log(f"ERROR {drive.name}: {e}\n{traceback.format_exc()}")

    # Blinded, randomized review order (fixed seed; probe/event interleaved)
    order = list(range(len(items)))
    rng.shuffle(order)
    for rank, idx in enumerate(order):
        items[idx]["review_order"] = rank

    manifest = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": SEED,
        "checkpoints": {"image": str(IMAGE_W.relative_to(ROOT)),
                        "sensor": str(SENSOR_W.relative_to(ROOT)),
                        "hybrid": str(FUSION_W.relative_to(ROOT))},
        "thresholds": THRESHOLDS,
        "vision_hz": VISION_HZ, "window": WINDOW, "stride": STRIDE,
        "align_tol_s": ALIGN_TOL, "match_window_s": MATCH_WINDOW,
        "probe_fraction": PROBE_FRACTION, "probe_clearance_s": PROBE_CLEARANCE,
        "drives": drive_meta,
        "totals": {
            "items": sum(1 for i in items if i["kind"] == "event"),
            "probes": sum(1 for i in items if i["kind"] == "probe"),
        },
    }
    (OUT / "review_items.json").write_text(json.dumps(
        {"manifest": manifest, "items": items}, indent=1))
    log(f"WROTE {OUT/'review_items.json'} — "
        f"{manifest['totals']['items']} events + {manifest['totals']['probes']} probes "
        f"across {len(drive_meta)} drives")
    log("TRANSFER RUN COMPLETE")


if __name__ == "__main__":
    main()
