import os
# Must be set before torch initializes (see app/core/device.py)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import subprocess
import sys

from app.core.config import settings
from app.core.database import create_db_and_tables, engine
from app.core.jobs import job_manager
from app.models.db import Job, JobStatus, Event, ReviewSession, ReviewLabel
from sqlmodel import Session, select

app = FastAPI(title=settings.PROJECT_NAME)

# Mount iPhone data for video playback (More specific first)
if (settings.DATA_DIR / "iphone").exists():
    app.mount("/static/iphone", StaticFiles(directory=settings.DATA_DIR / "iphone"), name="iphone_data")

# Mount nuScenes data
if (settings.DATA_DIR / "nuscenes").exists():
    app.mount("/static/nuscenes", StaticFiles(directory=settings.DATA_DIR / "nuscenes"), name="nuscenes_data")

# Mount general static files (index.html, etc)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/runs", StaticFiles(directory="runs"), name="runs")

# Initialize DB on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()
    # Ensure worker is running?
    # For now, we'll let the user run it manually or use a start script, 
    # but strictly per "re-entrant" reqs we might want to manage it here.
    # We will assume external management for the single process constraint in Phase 1 
    # but a simple way to dev is to run it alongside.

@app.get("/")
def read_root():
    return FileResponse("static/index.html")


# --------------------------------------------------------------------- #
# Transfer-gap blinded review (docs/TRANSFER_PROTOCOL.md)                #
# The queue is BLINDED: no scores, no trigger source, no event-vs-probe  #
# distinction reaches the client. Labels append to a JSONL audit log.    #
# --------------------------------------------------------------------- #
import json as _json
import time as _time
from pathlib import Path as _Path

_TRANSFER_DIR = _Path("results/transfer")


def _transfer_items():
    f = _TRANSFER_DIR / "review_items.json"
    if not f.exists():
        raise HTTPException(404, "run tools/run_transfer.py first")
    return _json.loads(f.read_text())


def _transfer_labels():
    f = _TRANSFER_DIR / "labels.jsonl"
    out = {}
    if f.exists():
        for line in f.read_text().splitlines():
            if line.strip():
                r = _json.loads(line)
                out.setdefault(r.get("reviewer", "user"), {})[r["item_id"]] = r["label"]
    return out


@app.get("/transfer")
def transfer_review_page():
    return FileResponse("static/transfer_review.html")


@app.get("/api/transfer/queue")
def transfer_queue(reviewer: str = "user"):
    data = _transfer_items()
    labeled = _transfer_labels().get(reviewer, {})
    items = sorted(data["items"], key=lambda i: i["review_order"])
    # Blinded payload: clip + optional still frame only. Items outside the
    # pre-registered review sample (tools/sample_review.py) are excluded;
    # if sampling was never applied, everything is in scope.
    queue = [{"id": it["id"],
              "clip_url": it["clip_url"],
              "image_url": it.get("image_url"),
              "labeled": it["id"] in labeled,
              "label": labeled.get(it["id"])}
             for it in items
             if (it["clip_url"] or it.get("image_url"))
             and it.get("in_review_sample", True)]
    return {"total": len(queue),
            "labeled": sum(1 for q in queue if q["labeled"]),
            "items": queue}


@app.post("/api/transfer/label")
def transfer_label(payload: dict):
    item_id = payload.get("item_id")
    label = payload.get("label")
    if label not in ("pothole", "not_pothole", "unsure"):
        raise HTTPException(400, "label must be pothole | not_pothole | unsure")
    data = _transfer_items()
    if not any(it["id"] == item_id for it in data["items"]):
        raise HTTPException(404, f"unknown item {item_id}")
    _TRANSFER_DIR.mkdir(parents=True, exist_ok=True)
    rec = {"item_id": item_id, "label": label,
           "reviewer": payload.get("reviewer", "user"),
           "notes": payload.get("notes"),
           "ts": _time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(_TRANSFER_DIR / "labels.jsonl", "a") as f:
        f.write(_json.dumps(rec) + "\n")
    return {"ok": True}


@app.get("/api/transfer/metrics")
def transfer_metrics(reviewer: str = None):
    """Compute + persist metrics from the labels collected so far.
    Safe to call mid-review; numbers cover labeled items only."""
    import sys as _sys
    if "." not in _sys.path:
        _sys.path.insert(0, ".")
    from tools.transfer_metrics import compute, to_markdown
    m = compute(reviewer)
    (_TRANSFER_DIR / "metrics.json").write_text(_json.dumps(m, indent=1))
    (_TRANSFER_DIR / "metrics.md").write_text(to_markdown(m))
    return m

from pydantic import BaseModel
from typing import Dict, Any

class JobRequest(BaseModel):
    task_type: str
    args: Dict[str, Any] = {}

# API Endpoints
@app.post("/api/jobs")
def create_job(job_req: JobRequest):
    job = job_manager.submit_job(job_req.task_type, job_req.args)
    return job

@app.get("/api/jobs", response_model=List[Job])
def list_jobs(limit: int = 200):
    return job_manager.list_jobs(limit)

@app.get("/api/datasets/image")
def list_image_datasets():
    """List available image datasets (>= 30 images)."""
    from app.core.config import settings
    
    datasets = []
    raw_path = settings.DATA_DIR / "image" / "raw"
    if raw_path.exists():
        for d in raw_path.iterdir():
            if d.is_dir() and (d / "data.yaml").exists():
                # Count images
                count = 0
                # Check different possible structures
                for img_dir in [
                    d / "train" / "images",
                    d / "valid" / "images",
                    d / "images" / "train",
                    d / "images" / "val"
                ]:
                    if img_dir.exists():
                        count += len(list(img_dir.glob("*")))
                
                if count == 0:
                     count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                
                if count >= 30:
                    datasets.append(d.name)
    return sorted(datasets)

@app.get("/api/datasets/sensor")
def list_sensor_datasets():
    """List available sensor datasets (>= 10 events, must have both classes)."""
    from app.core.config import settings
    from app.core.utils import get_dataset_stats
    
    datasets = []
    # Check all in sensor dir (Data 2, 4, 5 etc)
    sensor_path = settings.DATA_DIR / "sensor"
    if sensor_path.exists():
        for d in sensor_path.iterdir():
            if d.is_dir() and d.name not in ["raw", "processed", "valid"]:
                try:
                    p, n = get_dataset_stats(d)
                    # Enforce binary classification validity
                    if p > 0 and n > 0 and (p + n) >= 5:
                        datasets.append(f"sensor/{d.name}")
                except:
                    pass
    
    # Also include raw if it exists separately
    raw_path = sensor_path / "raw"
    if raw_path.exists():
        for d in raw_path.iterdir():
            if d.is_dir():
                try:
                    p, n = get_dataset_stats(d)
                    if p > 0 and n > 0 and (p + n) >= 5:
                        datasets.append(f"sensor/raw/{d.name}")
                except:
                    pass
                    
    # Check top-level processed (unified)
    proc_path = settings.DATA_DIR / "processed" / "sensor"
    if proc_path.exists():
        for f in proc_path.iterdir():
            if f.is_file() and f.suffix == ".csv":
                # Assuming processed CSVs are valid train/val splits or single files
                datasets.append(f"processed/sensor/{f.name}")
    
    return sorted(datasets)

@app.get("/api/datasets/hybrid")
def list_hybrid_datasets():
    """List available hybrid datasets (>= 10 events)."""
    from app.core.config import settings
    from app.core.utils import get_dataset_stats
    
    datasets = []
    hybrid_path = settings.DATA_DIR / "hybrid"
    if hybrid_path.exists():
        for d in hybrid_path.iterdir():
            if d.is_dir() and d.name != "processed":
                # Explicit Blacklist for known invalid datasets
                if "dataset_2020a" in d.name.lower() or "not sync" in d.name.lower():
                    continue
                    
                try:
                    p, n = get_dataset_stats(d)
                    if p > 0 and n > 0 and (p + n) >= 5:
                        datasets.append(f"hybrid/{d.name}")
                except:
                    pass
                
    return sorted(datasets)

@app.get("/api/sessions")
def list_iphone_sessions():
    """List available iPhone driving sessions."""
    from app.services.ingestion.iphone import iPhoneIngestionService
    return iPhoneIngestionService.get_sessions(settings.DATA_DIR / "iphone")

@app.get("/api/nuscenes/sessions")
def list_nuscenes_sessions():
    """List available nuScenes driving scenes."""
    from app.services.ingestion.nuscenes import NuScenesIngestionService
    return NuScenesIngestionService.get_scenes(settings.DATA_DIR / "nuscenes")

@app.get("/api/models/candidates")
def list_model_candidates(type: str):
    """List valid models for hybrid experiments (completed, dataset size >= 100)."""
    if type not in ["image", "sensor"]:
        raise HTTPException(status_code=400, detail="Invalid model type")
    
    task_type = f"train_{type}"
    
    with Session(engine) as session:
        statement = select(Job).where(
            Job.task_type == task_type,
            Job.status == JobStatus.COMPLETED
        ).order_by(Job.created_at.desc())
        
        jobs = session.exec(statement).all()
        candidates = []
        
        for j in jobs:
            res = j.args.get("result", {})
            metrics = res.get("metrics", {})
            dataset_path = res.get("dataset")
            
            # Check dataset size
            if dataset_path:
                try:
                    from pathlib import Path
                    from app.core.utils import get_dataset_stats
                    p, n = get_dataset_stats(Path(dataset_path))
                    if (p + n) < 100:
                        continue
                except:
                    continue
            else:
                # If no dataset path, assume invalid or legacy
                continue
                
            algo = res.get("algorithm", j.args.get("model", "Unknown"))
            
            candidates.append({
                "id": str(j.id),
                "name": f"{algo} (Acc: {metrics.get('accuracy', 0):.2f}) - {p+n} samples",
                "accuracy": metrics.get("accuracy", 0.0),
                "algorithm": algo
            })
            
        return candidates

@app.get("/api/models/hybrid/top")
def list_top_hybrid_models():
    """List top 3 hybrid models based on accuracy (with sample size filter)."""
    with Session(engine) as session:
        # Find completed hybrid training jobs
        statement = select(Job).where(
            Job.task_type == "train_hybrid",
            Job.status == JobStatus.COMPLETED
        )
        jobs = session.exec(statement).all()
        
        candidates = []
        for j in jobs:
            res = j.args.get("result", {})
            if not res or "permutations" not in res:
                continue
            
            # Check permutations
            for p in res["permutations"]:
                # Filter by dataset size (>= 100 events)
                # Requirement: "trained models with sufficient trained size... (>= 100 events)"
                # The dataset info is in the job result.
                dataset_path = res.get("dataset")
                if dataset_path:
                    try:
                        from pathlib import Path
                        from app.core.utils import get_dataset_stats
                        potholes, normals = get_dataset_stats(Path(dataset_path))
                        total_events = potholes + normals
                        
                        if total_events < 100:
                            continue
                    except Exception as e:
                        print(f"Error checking stats for {dataset_path}: {e}")
                        continue
                
                fusion_type = j.args.get("fusion_type", "feature").title()
                candidates.append({
                    "id": f"{j.id}_{p['image_model']}_{p['sensor_model']}",
                    "name": f"{fusion_type}: {p['image_model']} + {p['sensor_model']}",
                    "accuracy": p["metrics"]["accuracy"],
                    "job_id": j.id,
                    "image_model": p['image_model'],
                    "sensor_model": p['sensor_model']
                })
        
        # Sort by accuracy desc
        candidates.sort(key=lambda x: x["accuracy"], reverse=True)
        return candidates[:3]

@app.post("/api/inference/run")
def run_inference(req: JobRequest):
    """Trigger inference on a session."""
    # Custom handler to ensure we process session first
    if req.task_type == "run_inference_iphone":
         # Check if session needs processing
         from app.services.ingestion.iphone import iPhoneIngestionService
         from pathlib import Path
         session_path = Path(req.args.get("session_path"))
         if not iPhoneIngestionService.process_session(session_path):
             raise HTTPException(status_code=400, detail="Failed to process session data")
             
         # Map to standard inference task
         # We need to adapt the worker to handle 'run_inference_iphone' or generic 'run_inference'
         # Let's use generic 'run_inference' but point to the sensor.csv we just made
         
         # The worker's run_inference expects 'drive_path' and 'model_path' (or implicit)
         # We want to use specific models.
         # For prototype, we will pass the model details in args and let worker handle it.
         return job_manager.submit_job("run_inference", {
             "drive_path": str(session_path),
             "model_config": req.args.get("model_config"), # {image_model:..., sensor_model:...}
             "session_id": req.args.get("session_id")
         })
         
    if req.task_type == "run_inference_nuscenes":
         # Check if scene needs processing
         from app.services.ingestion.nuscenes import NuScenesIngestionService
         from pathlib import Path
         scene_path = Path(req.args.get("session_path"))
         if not NuScenesIngestionService.process_scene(scene_path):
             raise HTTPException(status_code=400, detail="Failed to process nuScenes data")
             
         return job_manager.submit_job("run_inference", {
             "drive_path": str(scene_path),
             "model_config": req.args.get("model_config"),
             "session_id": req.args.get("session_id")
         })

    return job_manager.submit_job(req.task_type, req.args)

@app.get("/api/datasets")
def list_datasets():
    """Legacy endpoint for backward compatibility."""
    return list_image_datasets()

@app.get("/api/jobs/{job_id}", response_model=Job)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# Review API
from app.services.review import ReviewManager
review_manager = ReviewManager()

@app.get("/api/reviews/history/{folder_name:path}")
def get_review_history(folder_name: str):
    """Find the most recent completed inference and any associated review session."""
    with Session(engine) as session:
        # Find latest completed inference job for this folder
        # We need to handle both full path and basename comparison
        from pathlib import Path
        folder_base = Path(folder_name).name
        
        statement = select(Job).where(
            Job.task_type == "run_inference",
            Job.status == JobStatus.COMPLETED
        ).order_by(Job.completed_at.desc())
        
        jobs = session.exec(statement).all()
        target_job = None
        for j in jobs:
            dp = j.args.get("drive_path", "")
            if folder_base in dp or folder_name in dp:
                target_job = j
                break
        
        if not target_job:
            return {"status": "none"}
            
        # Find existing review session
        rev_statement = select(ReviewSession).where(ReviewSession.run_id == str(target_job.id))
        rev_session = session.exec(rev_statement).first()
        
        return {
            "status": "found",
            "job_id": target_job.id,
            "session_name": target_job.args.get("session_id", folder_base),
            "video_url": target_job.args.get("result", {}).get("video_url"), # if worker stashed it
            "review_session_id": rev_session.id if rev_session else None
        }

@app.post("/api/reviews/start")
def start_review_session(run_id: str):
    # Check if a session already exists for this run_id
    with Session(engine) as session:
        existing = session.exec(select(ReviewSession).where(ReviewSession.run_id == run_id)).first()
        if existing:
            return {"session_id": existing.id}
            
    session = review_manager.create_session(run_id)
    return {"session_id": session.id}

@app.get("/api/reviews/{session_id}/next")
def get_next_event(session_id: int):
    event = review_manager.get_next_event(session_id)
    if not event:
        return {"status": "done"}
    
    # Use stored video_url from DB
    video_url = event.video_url or f"/static/clips/{event.run_id}/{event.id}.mp4"
    
    return {
        "event_id": event.id,
        "video_url": video_url,
        "score": event.score,
        "time": event.time
    }

@app.post("/api/reviews/events/{event_id}/label")
def submit_label(session_id: int, event_id: int, is_pothole: bool):
    review_manager.submit_label(session_id, event_id, is_pothole)
    return review_manager.get_stats(session_id)

@app.get("/api/reviews/{session_id}/events")
def list_review_events(session_id: int):
    """List all events for a review session with their label status."""
    with Session(engine) as session:
        review_session = session.get(ReviewSession, session_id)
        if not review_session:
            return []
            
        events = session.exec(select(Event).where(Event.run_id == review_session.run_id).order_by(Event.time)).all()
        labels = session.exec(select(ReviewLabel).where(ReviewLabel.session_id == session_id)).all()
        label_map = {l.event_id: l.is_pothole for l in labels}
        
        results = []
        for e in events:
            # Need video url base. Assuming we know it from session, or we can stash it in Event?
            # Event doesn't have video info. But we can construct if we knew the session path. 
            # In 'run_inference', we passed session_path. But Job args are in Job table.
            # We can lookup Job via run_id (which is job.id).
            
            status = "unlabeled"
            if e.id in label_map:
                status = "pothole" if label_map[e.id] else "false_positive"
                
            results.append({
                "id": e.id,
                "time": e.time,
                "score": e.score,
                "status": status,
                "video_url": e.video_url,
                "source": e.trigger_source
            })
        return results

@app.get("/api/metrics/manual")
def get_manual_metrics():
    """Get aggregate metrics across all manual reviews (Section F), split by source."""
    with Session(engine) as session:
        # Map jobs to sources for fast lookup
        jobs = session.exec(select(Job).where(Job.task_type == "run_inference")).all()
        source_map = {}
        rf_job_ids = set()
        
        for j in jobs:
            dp = j.args.get("drive_path", "")
            source_map[str(j.id)] = "nuscenes" if "nuscenes" in dp.lower() else "iphone"
            
            # Identify Random Forest jobs to exclude from hybrid
            try:
                mc = j.args.get("model_config", {})
                sm = mc.get("sensor_model", "").lower()
                # Check for known RF identifiers
                if "random_forest" in sm or "rf" in sm:
                    rf_job_ids.add(str(j.id))
            except: 
                pass

        # Join labels with events
        statement = select(ReviewLabel, Event).join(Event, ReviewLabel.event_id == Event.id)
        results = session.exec(statement).all()
        
        # Initialize stats structure with TN and FN
        def init_stats():
            return {
                "hybrid": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "image": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
                "sensor": {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
            }
            
        sources = {"combined": init_stats(), "iphone": init_stats(), "nuscenes": init_stats()}
        
        # Threshold constants
        IMG_THRESH = 0.20
        SNR_THRESH = 0.70
        HYB_THRESH = 0.30 # Standard trigger for hybrid
        
        for label, event in results:
            src_key = source_map.get(event.run_id, "iphone")
            
            for key in ["combined", src_key]:
                s = sources[key]
                
                # Evaluation Logic:
                # TP: Predicted Pothole (score >= thresh) AND Labelled Pothole (is_pothole)
                # FP: Predicted Pothole (score >= thresh) AND Labelled FP (!is_pothole)
                # FN: Predicted NOT Pothole (score < thresh) AND Labelled Pothole (is_pothole)
                # TN: Predicted NOT Pothole (score < thresh) AND Labelled FP (!is_pothole)
                
                def update_mod(m_key, score, thresh):
                    # EXCLUSION: Skip hybrid update if this is a Random Forest job
                    if m_key == "hybrid" and str(event.run_id) in rf_job_ids:
                        return
                        
                    pred = score >= thresh
                    if pred and label.is_pothole: s[m_key]["tp"] += 1
                    elif pred and not label.is_pothole: s[m_key]["fp"] += 1
                    elif not pred and label.is_pothole: s[m_key]["fn"] += 1
                    else: s[m_key]["tn"] += 1

                update_mod("hybrid", event.score, HYB_THRESH)
                update_mod("image", event.v_score, IMG_THRESH)
                update_mod("sensor", event.s_score, SNR_THRESH)

        def calc(s):
            total = s["tp"] + s["fp"] + s["tn"] + s["fn"]
            prec_total = s["tp"] + s["fp"]
            prec = s["tp"] / prec_total if prec_total > 0 else 0.0
            acc = (s["tp"] + s["tn"]) / total if total > 0 else 0.0
            return {
                "precision": prec, 
                "accuracy": acc,
                "total": total, 
                "tp": s["tp"], 
                "fp": s["fp"],
                "tn": s["tn"],
                "fn": s["fn"]
            }

        # Prepare final output
        out = {
            "hybrid": calc(sources["combined"]["hybrid"]),
            "image": calc(sources["combined"]["image"]),
            "sensor": calc(sources["combined"]["sensor"]),
            "iphone": {k: calc(v) for k, v in sources["iphone"].items()},
            "nuscenes": {k: calc(v) for k, v in sources["nuscenes"].items()}
        }
        return out

@app.get("/api/analytics/research")
def get_research_analytics():
    """Aggregate all training results for research analysis."""
    with Session(engine) as session:
        # Fetch relevant jobs and filter by status case-insensitively
        statement = select(Job).where(
            Job.task_type.in_(["train_image", "train_sensor", "train_hybrid"])
        )
        all_jobs = session.exec(statement).all()
        jobs = [j for j in all_jobs if j.status.lower() == "completed"]
        
        flat_results = []
        for j in jobs:
            res = j.args.get("result", {})
            dataset = res.get("dataset", "Unknown")
            
            # Hybrid jobs contain multiple permutations
            if j.task_type == "train_hybrid" and "permutations" in res:
                fusion_type = j.args.get("fusion_type", "feature").title()
                for p in res["permutations"]:
                    # FILTER: Exclude Random Forest tests (Section F consistency)
                    if "random_forest" in p['sensor_model'] or "rf" in p['sensor_model']:
                        continue
                        
                    # FILTER: Exclude hybrid tests with < 70% accuracy as requested
                    # This ensures consistency charts aren't skewed by failed experiments
                    acc = p["metrics"].get("accuracy", 0)
                    if acc < 0.70:
                        continue
                        
                    flat_results.append({
                        "id": f"{j.id}_{p['image_model']}_{p['sensor_model']}",
                        "modality": "Hybrid",
                        "algorithm": f"{fusion_type}: {p['image_model']} + {p['sensor_model']}",
                        "fusion_type": fusion_type,
                        "dataset": _normalize_ds_name(dataset),
                        "accuracy": p["metrics"].get("accuracy", 0),
                        "precision": p["metrics"].get("precision", 0),
                        "recall": p["metrics"].get("recall", 0),
                        "f1": p["metrics"].get("f1", 0),
                        "confusion_matrix": p["metrics"].get("confusion_matrix", [[0,0],[0,0]]),
                        "samples": res.get("summary", {}).get("samples", 100), # Default if missing
                        "timestamp": res.get("timestamp")
                    })
            else:
                metrics = res.get("metrics", {})
                modality = "Image" if j.task_type == "train_image" else "1D-CNN"
                
                # FILTER: Exclude < 10% accuracy tests (garbage runs) per user request
                acc = metrics.get("accuracy", 0)
                if acc < 0.10:
                    continue
                    
                flat_results.append({
                    "id": str(j.id),
                    "modality": modality,
                    "algorithm": res.get("algorithm", j.args.get("model", "Unknown")),
                    "dataset": _normalize_ds_name(dataset),
                    "accuracy": metrics.get("accuracy", 0),
                    "precision": metrics.get("precision", 0),
                    "recall": metrics.get("recall", 0),
                    "f1": metrics.get("f1", 0),
                    "confusion_matrix": metrics.get("confusion_matrix", [[0,0],[0,0]]),
                    "samples": res.get("summary", {}).get("samples", 100),
                    "timestamp": res.get("timestamp")
                })
        
        highlights_data = _calculate_research_highlights(flat_results)
        return {
            "all_results": flat_results,
            "modality_summary": _calculate_modality_summary(flat_results),
            "manual_metrics": get_manual_metrics(),
            "dataset_gains": highlights_data.get("dataset_gains", []),
            "perspective_summary": highlights_data.get("perspective_summary", {}),
            "latency_estimates": highlights_data.get("latency_estimates", {})
        }

def _calculate_modality_summary(results):
    import math
    summary = {}
    for r in results:
        m = r["modality"]
        if m not in summary:
            summary[m] = {
                "count": 0, "acc_sum": 0, "f1_sum": 0, "best_acc": 0, 
                "acc_values": [], 
                "metrics_list": [], # Store all metrics for scatter plots
                "agg_confusion_matrix": [[0,0],[0,0]] # TN, FP, FN, TP
            }
        
        summary[m]["count"] += 1
        summary[m]["acc_sum"] += r["accuracy"]
        summary[m]["f1_sum"] += r["f1"]
        summary[m]["best_acc"] = max(summary[m]["best_acc"], r["accuracy"])
        summary[m]["acc_values"].append(r["accuracy"])
        
        # Store detailed metrics
        summary[m]["metrics_list"].append({
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "dataset_size": r.get("samples", 100)
        })
        
        # Aggregate Confusion Matrix
        cm = r.get("confusion_matrix", [[0,0],[0,0]])
        try:
            # Check dimensions
            if len(cm) == 2 and len(cm[0]) == 2:
                summary[m]["agg_confusion_matrix"][0][0] += cm[0][0] # TN
                summary[m]["agg_confusion_matrix"][0][1] += cm[0][1] # FP
                summary[m]["agg_confusion_matrix"][1][0] += cm[1][0] # FN
                summary[m]["agg_confusion_matrix"][1][1] += cm[1][1] # TP
        except: pass
        
    for m in summary:
        avg = summary[m]["acc_sum"] / summary[m]["count"]
        summary[m]["avg_accuracy"] = avg
        summary[m]["avg_f1"] = summary[m]["f1_sum"] / summary[m]["count"]
        
        # Calculate Standard Deviation (Consistency)
        if summary[m]["count"] > 1:
            variance = sum((x - avg) ** 2 for x in summary[m]["acc_values"]) / (summary[m]["count"] - 1)
            summary[m]["std_dev"] = math.sqrt(variance)
        else:
            summary[m]["std_dev"] = 0.0
        
        del summary[m]["acc_values"] # Cleanup
        
    return summary

def _normalize_ds_name(ds_path):
    """Convert 'data/image/raw/Data 1' -> 'Data 1'"""
    original = str(ds_path)
    lower_p = original.lower()
    
    # Try to strip common prefixes (case-insensitive)
    for prefix in ["data/image/raw/", "data/sensor/", "data/hybrid/", "data/"]:
        if lower_p.startswith(prefix):
             # Return the slice from original string to preserve case
             return original[len(prefix):]
             
    # Standardize common labels (keep Title Case for UI)
    if "data 1" in lower_p: return "Data 1"
    if "data 3" in lower_p: return "Data 3"
    if "data 4" in lower_p: return "Data 4"
    if "data 5" in lower_p: return "Data 5"
    
    return original.split('/')[-1].strip()

def _classify_dataset_perspective(ds_name):
    """Classify dataset as 'Vehicle' (Realistic) or 'Handheld' (Proxy/Web)."""
    ds = ds_name.lower()
    if "data 1" in ds or "data 3" in ds or "data 4" in ds or "data 5" in ds:
        return "Vehicle"
    return "Handheld"

def _estimate_latency(modality):
    """Estimate inference latency in ms based on modality."""
    if modality == "Image": return 35  # ~30fps equivalent
    if modality == "1D-CNN": return 3   # Very fast signal processing
    if modality == "Sensor": return 3   # Legacy fallback
    if modality == "Hybrid": return 40  # Overhead of alignment
    return 0

def _calculate_research_highlights(results):
    """Identify specific datasets where fusion provided the most gain and perspective gaps."""
    by_dataset = {}
    for r in results:
        ds = _normalize_ds_name(r["dataset"])
        if ds == "unknown": continue
        
        mod = r["modality"]
        if ds not in by_dataset:
            by_dataset[ds] = {"Image": 0, "1D-CNN": 0, "Hybrid": 0, "Perspective": _classify_dataset_perspective(ds)}
        by_dataset[ds][mod] = max(by_dataset[ds][mod], r["accuracy"])
        
    highlights = []
    
    # Track perspective agg
    perspective_stats = {"Vehicle": {"Image": [], "Hybrid": []}, "Handheld": {"Image": [], "Hybrid": []}}

    for ds, accuracies in by_dataset.items():
        persp = accuracies["Perspective"]
        if accuracies["Image"] > 0: perspective_stats[persp]["Image"].append(accuracies["Image"])
        if accuracies["Hybrid"] > 0: perspective_stats[persp]["Hybrid"].append(accuracies["Hybrid"])

        if accuracies["Hybrid"] > 0 and (accuracies["Image"] > 0 or accuracies["1D-CNN"] > 0):
            best_single = max(accuracies["Image"], accuracies["1D-CNN"])
            delta = accuracies["Hybrid"] - best_single
            highlights.append({
                "dataset": ds,
                "perspective": persp,
                "image_acc": accuracies["Image"],
                "sensor_acc": accuracies["1D-CNN"],
                "hybrid_acc": accuracies["Hybrid"],
                "fusion_gain": delta
            })
    
    # Calculate Perspective Statistics
    perspective_summary = {}
    for p_name, metrics in perspective_stats.items():
        perspective_summary[p_name] = {
            "avg_image": sum(metrics["Image"])/len(metrics["Image"]) if metrics["Image"] else 0,
            "avg_hybrid": sum(metrics["Hybrid"])/len(metrics["Hybrid"]) if metrics["Hybrid"] else 0
        }

    # Sort by gain to show the "biggest wins"
    highlights.sort(key=lambda x: x["fusion_gain"], reverse=True)
    
    return {
        "dataset_gains": highlights,
        "perspective_summary": perspective_summary,
        "latency_estimates": {
            "Image": _estimate_latency("Image"),
            "1D-CNN": _estimate_latency("1D-CNN"),
            "Hybrid": _estimate_latency("Hybrid")
        }
    }

@app.get("/api/analytics/export")
def export_research_excel():
    """Generate a comprehensive Excel export of all research metrics and charts."""
    import pandas as pd
    import io
    from fastapi.responses import StreamingResponse
    
    # 1. Gather Data
    res = get_research_analytics()
    all_results = res["all_results"]
    modality_summary = res["modality_summary"]
    perspective_summary = res["perspective_summary"]
    latency_estimates = res["latency_estimates"]

    # 2. Create DataFrames
    df_raw = pd.DataFrame(all_results)
    
    summary_rows = []
    for mod, metrics in modality_summary.items():
        summary_rows.append({
            "Modality": mod,
            "Avg Accuracy": metrics["avg_accuracy"],
            "Avg F1": metrics["avg_f1"],
            "Consistency (StdDev)": metrics["std_dev"],
            "Samples": metrics["count"]
        })
    df_summary = pd.DataFrame(summary_rows)

    persp_rows = []
    for persp, metrics in perspective_summary.items():
        persp_rows.append({
            "Perspective": persp,
            "Image Avg Accuracy": metrics["avg_image"],
            "Hybrid Avg Accuracy": metrics["avg_hybrid"],
            "Reality Gap": metrics["avg_image"] - metrics["avg_hybrid"] if metrics["avg_hybrid"] > 0 else 0
        })
    df_persp = pd.DataFrame(persp_rows)

    eff_rows = []
    for mod in modality_summary.keys():
        eff_rows.append({
            "Modality": mod,
            "Latency (ms)": latency_estimates.get(mod, 0),
            "Accuracy (%)": modality_summary[mod]["avg_accuracy"] * 100
        })
    df_efficiency = pd.DataFrame(eff_rows)

    # 3. Write to Excel with Charts
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Sheet 1: Raw Data
        df_raw.to_excel(writer, sheet_name='Raw Experiments', index=False)
        
        # Sheet 2: Modality Summary + Chart
        if not df_summary.empty:
            df_summary.to_excel(writer, sheet_name='Modality Analysis', index=False)
            workbook = writer.book
            worksheet = writer.sheets['Modality Analysis']
            
            chart_mod = workbook.add_chart({'type': 'column'})
            chart_mod.add_series({
                'name':       'Average Accuracy',
                'categories': ['Modality Analysis', 1, 0, len(df_summary), 0],
                'values':     ['Modality Analysis', 1, 1, len(df_summary), 1],
                'data_labels': {'value': True, 'num_format': '0%'},
            })
            chart_mod.set_title({'name': 'Performance by Modality'})
            chart_mod.set_y_axis({'name': 'Accuracy', 'min': 0, 'max': 1})
            worksheet.insert_chart('G2', chart_mod)

        # Sheet 3: Perspective Analysis + Chart
        if not df_persp.empty:
            df_persp.to_excel(writer, sheet_name='Perspective Gap', index=False)
            worksheet_p = writer.sheets['Perspective Gap']
            chart_p = workbook.add_chart({'type': 'bar'})
            chart_p.add_series({
                'name': 'Image (Proxy)',
                'categories': ['Perspective Gap', 1, 0, len(df_persp), 0],
                'values': ['Perspective Gap', 1, 1, len(df_persp), 1],
            })
            chart_p.add_series({
                'name': 'Hybrid (Real)',
                'categories': ['Perspective Gap', 1, 0, len(df_persp), 0],
                'values': ['Perspective Gap', 1, 2, len(df_persp), 2],
            })
            chart_p.set_title({'name': 'Handheld vs Vehicle Perspective'})
            worksheet_p.insert_chart('F2', chart_p)

        # Sheet 4: Efficiency Frontier + Chart
        if not df_efficiency.empty:
            df_efficiency.to_excel(writer, sheet_name='Efficiency', index=False)
            worksheet_e = writer.sheets['Efficiency']
            chart_e = workbook.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
            chart_e.add_series({
                'name': 'Efficiency Frontier',
                'categories': ['Efficiency', 1, 1, len(df_efficiency), 1],
                'values':     ['Efficiency', 1, 2, len(df_efficiency), 2],
            })
            chart_e.set_title({'name': 'Speed vs Accuracy Trade-off'})
            chart_e.set_x_axis({'name': 'Latency (ms)'})
            chart_e.set_y_axis({'name': 'Accuracy (%)'})
            worksheet_e.insert_chart('F2', chart_e)

    output.seek(0)
    from fastapi import Response
    return Response(
        content=output.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=pothole_research_package.xlsx"}
    )
def get_status():
    return {"status": "ok", "version": "0.1.0"}
