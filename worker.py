import time
import sys
import os
import signal
import traceback

# Must be set before torch initializes: lets ops without an MPS kernel fall
# back to CPU instead of raising (see app/core/device.py).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np

# Dependency Check
try:
    import pandas as pd
    from ultralytics import YOLO
    import yaml
except ImportError as e:
    print("\n" + "="*60)
    print("CRITICAL ERROR: Missing dependencies.")
    print(f"Failed to import: {e.name}")
    print("Please ensure you are running in the virtual environment.")
    print("Usage: ./run_worker.sh")
    print("="*60 + "\n")
    sys.exit(1)

import yaml
import subprocess

import pandas as pd
from sqlmodel import Session, select, desc
from datetime import datetime, timezone
from app.models.db import Job, JobStatus, Event, Run
from app.core.database import engine
from app.core.config import settings

# Import Services
from app.services.ingestion.validators import DatasetValidator
from app.services.ingestion.translators import YOLOTranslator, COCOTranslator
from app.services.ingestion.sensors import sensor_parser
from app.services.models.image import YOLOWrapper, FasterRCNNWrapper, UNetWrapper
from app.services.models.sensor import RandomForestWrapper, CNN1DWrapper
from app.services.models.hybrid import FeatureFusionWrapper
from pathlib import Path
import json
import shutil
from app.core.utils import get_dataset_stats

def get_best_model(algo: str, task_type: str, dataset_filter: str = None):
    """
    Find the best *VALID* completed job for a given algo/task.
    Prioritize recent jobs with valid 'artifacts' paths.
    Optionally filter by dataset name (case-insensitive partial match).
    """
    # Normalize algo name once, without mutating the parameter across loop iterations
    norm_algo = "random_forest" if algo == "rf" else algo

    with Session(engine) as session:
        # Get all completed jobs of this type, newest first
        jobs = session.exec(
            select(Job)
            .where(Job.task_type == task_type)
            .where(Job.status == JobStatus.COMPLETED)
            .order_by(desc(Job.id))
        ).all()

        valid_jobs = []
        for j in jobs:
            # Check algo match
            j_algo = j.args.get("model", j.args.get("model_type", "")) or j.args.get("algorithm")

            # Normalize job algo
            if j_algo == "rf": j_algo = "random_forest"

            if norm_algo.lower() not in str(j_algo).lower() and str(j_algo).lower() not in norm_algo.lower():
                 continue
            
            # Check Dataset Filter
            if dataset_filter:
                j_ds = j.args.get("dataset_path") or j.args.get("dataset")
                if not j_ds or dataset_filter.lower() not in str(j_ds).lower():
                    continue

            # Check for artifacts metadata
            res = j.args.get("result", {})
            summary = res.get("summary", {})
            artifacts_path = summary.get("artifacts")
            
            if not artifacts_path:
                continue
                
            # Extract metric (accuracy). May legitimately be None for
            # degraded runs — rank those last instead of crashing.
            metrics = res.get("metrics", {})
            acc = metrics.get("accuracy") or 0.0

            valid_jobs.append((acc, j.id, j))
            
        if not valid_jobs:
            return None
            
        # Sort by Accuracy Descending, then ID Descending
        # Max by accuracy
        best_job = max(valid_jobs, key=lambda x: x[0])[2]
        return best_job



def handle_task(job: Job):
    print(f"Processing job {job.id}: {job.task_type}")
    
    if job.task_type == "train_image":
        # Args: model (yolov8, faster_rcnn, unet), dataset_path, epochs, hyperparams
        model_type = job.args.get("model", "yolov8")
        dataset_path = Path(job.args.get("dataset_path", "data/image/raw/dummy_yolo"))
        epochs = int(job.args.get("epochs", 3))
        
        run_config = {
            "output_dir": f"runs/{job.id}",
        }
        
        model = None
        save_name = "full_model.pt"
        params = {}

        if "yolo" in model_type:
            # Fix data.yaml paths to be absolute to avoid CWD issues
            yaml_path = dataset_path / "data.yaml"
            if yaml_path.exists():
                with open(yaml_path, 'r') as f:
                    data_conf = yaml.safe_load(f)
                
                needs_update = False
                for key in ['train', 'val', 'test']:
                    if key in data_conf and isinstance(data_conf[key], str):
                        p = Path(data_conf[key])
                        # If path doesn't exist or is not absolute, try to fix it
                        if not p.exists() or not p.is_absolute():
                            # 1. Try relative to dataset_path as-is
                            test_path = (dataset_path / p).resolve()
                            if not test_path.exists():
                                # 2. Try standard YOLO structure: dataset_path/key/images
                                test_path = (dataset_path / key / 'images').resolve()
                                if not test_path.exists() and key == 'val':
                                    # Try 'valid' alias
                                    test_path = (dataset_path / 'valid' / 'images').resolve()
                            
                            if test_path.exists():
                                data_conf[key] = str(test_path)
                                needs_update = True
                            else:
                                print(f"Warning: Could not resolve YOLO path for {key} in {dataset_path}")
                
                if needs_update:
                    print(f"Updating data.yaml with absolute paths for YOLO...")
                    with open(yaml_path, 'w') as f:
                        yaml.dump(data_conf, f)

        if "yolo" in model_type:
            # ... already resolved data.yaml above
            model = YOLOWrapper(model_id=str(job.id), config={**run_config, "model_name": "yolov8n.pt"})
            save_name = "full_model.pt"
            params = {"model_type": "YOLOv8n", "imgsz": 640}
            
            # Real training metrics
            # Use single_cls=True to handle cases where labels have different class IDs (e.g. 0 and 1) 
            # but we only focus on potholes. This fixes "Label class 1 exceeds dataset class count 1"
            train_results = model.train(dataset_path, epochs=epochs, single_cls=True)
            model.save()
            
            # Only YOLO models are persisted for the real-world validation pipeline
            if (model.output_dir / save_name).exists():
                shutil.copy(model.output_dir / save_name, settings.LATEST_IMAGE_MODEL)
                print(f"Persisted latest YOLO model to {settings.LATEST_IMAGE_MODEL}")
        
        elif "faster_rcnn" in model_type:
            model = FasterRCNNWrapper(model_id=str(job.id), config=run_config)
            save_name = "faster_rcnn.pth"
            params = {"model_type": "Faster R-CNN", "backbone": "resnet50"}
            model.train(dataset_path, epochs=epochs)
            model.save()
            
        elif "unet" in model_type:
            model = UNetWrapper(model_id=str(job.id), config=run_config)
            save_name = "unet.pth"
            params = {"model_type": "U-Net", "encoder": "resnet34"}
            model.train(dataset_path, epochs=epochs)
            model.save()

        elif "resnet" in model_type or "classifier" in model_type:
            # Plain image-level classification baseline — answers "why use
            # detectors?" empirically for the image-level framing.
            from app.services.models.image import ResNet18ClassifierWrapper
            model = ResNet18ClassifierWrapper(model_id=str(job.id), config=run_config)
            save_name = "resnet18.pth"
            params = {"model_type": "ResNet18 classifier", "imgsz": 224}
            model.train(dataset_path, epochs=epochs)
            model.save()
            
        if not model:
            return {"status": "failed", "error": f"Unsupported model type: {model_type}"}

        # Extract metrics from validation...

        # Extract metrics from Ultralytics results
        # results is a list of Results objects or a single Results object?
        # Ultralytics train returns None in some versions, or the results object. 
        # Safer to run validation separately or read from CSV if needed, 
        # but let's try to extract from the returned object first or validate.
        
        # Explicit validation to get clean metrics
        val_metrics = model.evaluate(dataset_path)

        # Pass metrics through verbatim. Missing values stay None — a missing
        # confusion matrix / ROC must never be replaced with a zero matrix or
        # a fabricated diagonal, and image-level accuracy must never be
        # silently substituted with map50 (different metrics).
        metrics = {
            "precision": val_metrics.get("precision"),
            "recall": val_metrics.get("recall"),
            "f1": val_metrics.get("f1"),
            "accuracy": val_metrics.get("accuracy"),
            "map50": val_metrics.get("map50"),
            "mean_iou": val_metrics.get("mean_iou"),
            "confusion_matrix": val_metrics.get("confusion_matrix"),
            "roc_curve": val_metrics.get("roc_curve"),
            "roc_auc": val_metrics.get("roc_auc"),
            "metric_framing": val_metrics.get("metric_framing"),
            "eval_split": val_metrics.get("eval_split"),
        }
        if metrics["confusion_matrix"] is None:
            metrics["degraded"] = True
        if metrics["roc_curve"] is None:
            # Legitimately undefined for a single-class eval set — flag the
            # reason without marking the whole run degraded.
            metrics["roc_undefined"] = True

        parameters = {**params, "epochs": epochs, "dataset": str(dataset_path), "seed": 42}
        
        # Enrich with metadata
        result = {
            "status": "completed",
            "algorithm": model_type,
            "dataset": str(dataset_path),
            "timestamp": datetime.utcnow().isoformat(),
            "parameters": parameters,
            "metrics": metrics,
            "summary": {
                "epochs": epochs,
                "artifacts": str(model.output_dir)
            }
        }
        
        # Save metrics to json
        with open(Path(model.output_dir) / "metrics.json", "w") as f:
            json.dump(result, f)
            
        return result
            
    elif job.task_type == "train_sensor":
        # Args: model (rf, cnn), dataset_path (folder of csvs or preprocessed)
        model_type = job.args.get("model", "rf")
        dataset_path = Path(job.args.get("dataset_path", "data/sensor/raw/dummy_sensor"))
        
        run_config = {
            "output_dir": f"runs/{job.id}",
            "model_name": model_type
        }
        
        
        if model_type == "random_forest" or model_type == "rf":
            model = RandomForestWrapper(str(job.id), run_config)
            model.train(dataset_path)
            metrics_res = model.evaluate(dataset_path)
            model.save()
            
            if (model.output_dir / "model.joblib").exists():
                # RF is a joblib pickle — never deploy it under the .pth name
                # (the inference engine dispatches loaders by file suffix).
                rf_target = settings.LATEST_SENSOR_MODEL.with_suffix(".joblib")
                shutil.copy(model.output_dir / "model.joblib", rf_target)
                print(f"Persisted latest sensor model to {rf_target}")
            
            parameters = {
                "n_estimators": model.n_estimators,
                "seed": getattr(model, "seed", 42),
                "dataset": Path(dataset_path).name
            }

            # Pass metrics through verbatim; never substitute fabricated
            # zero matrices / diagonal ROC curves for missing values.
            metrics = {
                "precision": metrics_res.get("precision"),
                "recall": metrics_res.get("recall"),
                "f1": metrics_res.get("f1"),
                "accuracy": metrics_res.get("accuracy"),
                "confusion_matrix": metrics_res.get("confusion_matrix"),
                "roc_curve": metrics_res.get("roc_curve"),
                "roc_auc": metrics_res.get("roc_auc"),
                "eval_split": metrics_res.get("eval_split"),
                "split": metrics_res.get("split"),
            }
            if metrics_res.get("error"):
                metrics["degraded"] = True
                metrics["error"] = metrics_res["error"]

            return {
                "status": "completed",
                "algorithm": "Random Forest",
                "dataset": str(dataset_path),
                "timestamp": datetime.utcnow().isoformat(),
                "parameters": parameters,
                "metrics": metrics,
                "summary": {
                    "type": "Classical", 
                    "samples": metrics_res.get("samples", 0),
                    "artifacts": str(model.output_dir)
                }
            }
            
        elif model_type in ["cnn", "lstm", "1d-cnn"]:
            # Deep Learning: 'lstm' trains a REAL bidirectional LSTM (it used
            # to silently train the CNN — reporting it as an LSTM would have
            # been a fabricated architecture claim).
            arch = "lstm" if model_type == "lstm" else "cnn"
            model = CNN1DWrapper(str(job.id), {**run_config, "arch": arch})
            train_res = model.train(dataset_path, epochs=20) # Use real training
            
            # Evaluate on dataset (or holdout if implemented in evaluate)
            metrics_res = model.evaluate(dataset_path)
            model.save()
            
            if (model.output_dir / "model.pth").exists():
                shutil.copy(model.output_dir / "model.pth", settings.LATEST_SENSOR_MODEL)
                print(f"Persisted latest sensor model to {settings.LATEST_SENSOR_MODEL}")
            
            # Real parameters
            parameters = {
                "architecture": "BiLSTM" if arch == "lstm" else "1D-CNN",
                "layers": 3,
                "epochs": 20,
                "hidden_dim": 64,
                "seed": getattr(model, "seed", 42),
                "dataset": Path(dataset_path).name,
                "best_val_acc": train_res.get("val_acc", 0.0) if train_res else 0.0
            }

            # Pass metrics through verbatim; missing values stay None.
            metrics = {
                "precision": metrics_res.get("precision"),
                "recall": metrics_res.get("recall"),
                "f1": metrics_res.get("f1"),
                "accuracy": metrics_res.get("accuracy"),
                "confusion_matrix": metrics_res.get("confusion_matrix"),
                "roc_curve": metrics_res.get("roc_curve"),
                "roc_auc": metrics_res.get("roc_auc"),
                "eval_split": metrics_res.get("eval_split"),
                "split": metrics_res.get("split"),
            }
            if metrics_res.get("error"):
                metrics["degraded"] = True
                metrics["error"] = metrics_res["error"]

            return {
                "status": "completed",
                "algorithm": "BiLSTM" if arch == "lstm" else "1D-CNN",
                "dataset": str(dataset_path),
                "timestamp": datetime.utcnow().isoformat(),
                "parameters": parameters,
                "metrics": metrics,
                "summary": {
                    "type": "DL", 
                    "samples": metrics_res.get("samples", 0),
                    "artifacts": str(model.output_dir)
                }
            }

    elif job.task_type == "train_hybrid":
        dataset_path = Path(job.args.get("dataset_path", "data/dummy_hybrid"))
        fusion_type = job.args.get("fusion_type", "feature")
        
        results = []
        potholes, normals = get_dataset_stats(dataset_path)
        
        # Match the hybrid job's dataset tag (e.g. "Data 3")
        ds_filter = None
        current_ds = job.args.get("dataset_path", "")
        if "data 3" in current_ds.lower():
            ds_filter = "Data 3"
        elif "data 1" in current_ds.lower():
            ds_filter = "Data 1"
        
        print(f"Hybrid Training Objective: Evaluate 6 combinations (3 image x 2 sensor) for {ds_filter or 'All'}", flush=True)

        if "image_algo" in job.args:
            image_algos = [job.args["image_algo"]]
        else:
            image_algos = ["yolov8", "faster_rcnn", "unet"]
            
        if "sensor_algo" in job.args:
            sensor_algos = [job.args["sensor_algo"]]
        else:
            sensor_algos = ["cnn", "random_forest"]

        # Load dataset_index.json for authoritative labels and explicit pairing
        index_path = dataset_path / "dataset_index.json"
        dataset_index = {}  # stem -> {label, image, sensor}
        if index_path.exists():
            with open(index_path) as f:
                idx_data = json.load(f)
            if isinstance(idx_data, dict) and "samples" not in idx_data:
                # Format: {"images/stem.jpg": {"sensor_path": "...", "label": N}, ...}
                for img_key, meta in idx_data.items():
                    stem = Path(img_key).stem
                    dataset_index[stem] = meta
            elif isinstance(idx_data, dict) and "samples" in idx_data:
                # Format: {"samples": [{image, sensor_path, label}, ...]}
                for entry in idx_data["samples"]:
                    stem = Path(entry.get("image", "")).stem
                    if stem:
                        dataset_index[stem] = entry
            elif isinstance(idx_data, list):
                for entry in idx_data:
                    stem = Path(entry.get("image", entry.get("stem", ""))).stem
                    if stem:
                        dataset_index[stem] = entry
            print(f"Loaded dataset_index.json: {len(dataset_index)} paired samples", flush=True)
        else:
            print("WARNING: No dataset_index.json found. Falling back to filename-based label inference.", flush=True)

        # Late fusion consumes per-file component PROBABILITIES
        # (extract_scores); feature fusion consumes embeddings
        # (extract_features). Same alignment/split machinery for both.
        use_scores = (fusion_type == "late")

        # Caches to avoid redundant extraction
        img_feats_cache = {} # algo -> (feats, labels, stems)
        snr_feats_cache = {} # algo -> (feats, labels, stems)
        
        # Persistent Cache Setup
        cache_base = settings.DATA_DIR / "cache" / "features"
        cache_base.mkdir(parents=True, exist_ok=True)
        
        def get_cache_path(algo, model_id, sub_folder):
            # _v2: cache format versioned so pre-overhaul caches (zero-padded
            # 512-dim embeddings, stem-desynced rows) can never be reused.
            safe_ds = dataset_path.name.replace(" ", "_")
            return cache_base / f"{safe_ds}_{algo}_{model_id}_{sub_folder}_v2.npz"

        
        permutations = []
        
        # Standardize Hybrid Dataset Structure (Symlinks)
        abs_ds_path = dataset_path.resolve()
        for src_name, target_name in [("Image_data", "images"), ("Motion_data", "sensor")]:
            if (abs_ds_path / src_name).exists():
                target = abs_ds_path / target_name
                if target.exists() or target.is_symlink(): target.unlink()
                try: os.symlink(abs_ds_path / src_name, target)
                except Exception as e: print(f"Warning: {e}")

        for img_algo in image_algos:
            for snr_algo in sensor_algos:
                print(f"\n--- evaluating combination: {img_algo} + {snr_algo} ---", flush=True)
                
                # 1. Get Models (Override Check)
                img_best = None
                if "image_job_id" in job.args:
                    with Session(engine) as session:
                        img_best = session.get(Job, job.args["image_job_id"])
                        if img_best: session.expunge(img_best)
                else:
                    # No silent cross-dataset fallback: a component trained on a
                    # different dataset invalidates the "best model for this
                    # dataset" claim AND breaks the shared split manifest.
                    img_best = get_best_model(img_algo, "train_image", dataset_filter=ds_filter)
                    if not img_best and ds_filter:
                        print(f"No {img_algo} model trained on '{ds_filter}' — skipping "
                              f"(train one on this dataset first; cross-dataset fallback removed).", flush=True)

                snr_best = None
                if "sensor_job_id" in job.args:
                    with Session(engine) as session:
                        snr_best = session.get(Job, job.args["sensor_job_id"])
                        if snr_best: session.expunge(snr_best)
                else:
                    snr_best = get_best_model(snr_algo, "train_sensor", dataset_filter=ds_filter)
                    if not snr_best and ds_filter:
                        print(f"No {snr_algo} model trained on '{ds_filter}' — skipping "
                              f"(train one on this dataset first; cross-dataset fallback removed).", flush=True)
                
                if not img_best or not snr_best:
                    print(f"Skipping {img_algo}+{snr_algo}: missing component models", flush=True)
                    continue

                # 2. Extract Features (with caching)
                if img_algo not in img_feats_cache:
                    print(f"Initializing {img_algo} for extraction...")
                    if img_algo == "yolov8": wrapper = YOLOWrapper(str(img_best.id), img_best.args)
                    elif img_algo == "faster_rcnn": wrapper = FasterRCNNWrapper(str(img_best.id), img_best.args)
                    else: wrapper = UNetWrapper(str(img_best.id), img_best.args)
                    
                    # Try to load weights
                    weights_names = {"yolov8": "full_model.pt", "faster_rcnn": "faster_rcnn.pth", "unet": "unet.pth"}
                    w_path = Path(img_best.args.get("result", {}).get("summary", {}).get("artifacts", "runs/0")) / weights_names[img_algo]
                    if w_path.exists():
                         print(f"Loading weights from {w_path}")
                         wrapper.load(w_path)
                    
                    # Check persistent cache first (scores and features are
                    # cached separately — they are different artifacts)
                    c_path = get_cache_path(img_algo, img_best.id, "images_scores" if use_scores else "images")
                    if c_path.exists():
                        print(f"Cache Hit: Loading {img_algo} {'scores' if use_scores else 'features'} from {c_path}")
                        c_data = np.load(c_path, allow_pickle=True)
                        cached_stems = list(c_data['stems']) if 'stems' in c_data else []
                        img_feats_cache[img_algo] = (c_data['features'], c_data['labels'], cached_stems)
                    else:
                        print(f"Cache Miss: Extracting {img_algo} {'scores' if use_scores else 'features'}...")
                        if use_scores:
                            feats, lbls, stems = wrapper.extract_scores(dataset_path / "images")
                        else:
                            feats, lbls, stems = wrapper.extract_features(dataset_path / "images")
                        img_feats_cache[img_algo] = (feats, lbls, stems)
                        # Save to cache (include stems for alignment)
                        np.savez(c_path, features=feats, labels=lbls, stems=np.array(stems))

                
                if snr_algo not in snr_feats_cache:
                    print(f"Initializing {snr_algo} for extraction...")
                    if snr_algo == "cnn": wrapper = CNN1DWrapper(str(snr_best.id), snr_best.args)
                    else: wrapper = RandomForestWrapper(str(snr_best.id), snr_best.args)
                    
                    weights_names = {"cnn": "model.pth", "random_forest": "model.joblib"}
                    w_path = Path(snr_best.args.get("result", {}).get("summary", {}).get("artifacts", "runs/0")) / weights_names[snr_algo]
                    if w_path.exists():
                         print(f"Loading weights from {w_path}")
                         wrapper.load(w_path)
                    
                    # Check persistent cache
                    c_path = get_cache_path(snr_algo, snr_best.id, "sensor_scores" if use_scores else "sensor")
                    if c_path.exists():
                        print(f"Cache Hit: Loading {snr_algo} {'scores' if use_scores else 'features'} from {c_path}")
                        c_data = np.load(c_path, allow_pickle=True)
                        cached_stems = list(c_data['stems']) if 'stems' in c_data else []
                        snr_feats_cache[snr_algo] = (c_data['features'], c_data['labels'], cached_stems)
                    else:
                        print(f"Cache Miss: Extracting {snr_algo} {'scores' if use_scores else 'features'}...")
                        if use_scores:
                            feats, lbls, stems = wrapper.extract_scores(dataset_path / "sensor")
                        else:
                            feats, lbls, stems = wrapper.extract_features(dataset_path / "sensor")
                        snr_feats_cache[snr_algo] = (feats, lbls, stems)
                        np.savez(c_path, features=feats, labels=lbls, stems=np.array(stems))


                img_feats, img_labels, img_stems = img_feats_cache[img_algo]
                snr_feats, snr_labels, snr_stems = snr_feats_cache[snr_algo]

                if img_feats is None or snr_feats is None or len(img_feats) == 0 or len(snr_feats) == 0:
                    print(f"Skipping {img_algo}+{snr_algo}: feature extraction failed")
                    continue

                # 3. Align by stem and use authoritative index labels
                job_perm_dir = Path(f"runs/{job.id}_{img_algo}_{snr_algo}")
                job_perm_dir.mkdir(parents=True, exist_ok=True)

                # Build stem -> row-index lookup for each modality
                img_stem_map = {s: i for i, s in enumerate(img_stems)} if img_stems else {}
                snr_stem_map = {s: i for i, s in enumerate(snr_stems)} if snr_stems else {}

                # Determine paired stems: intersection of both modalities.
                # Positional pairing of two independently-scanned directories
                # is essentially arbitrary — refuse instead of misaligning.
                if not img_stems or not snr_stems:
                    print(f"Skipping {img_algo}+{snr_algo}: feature cache has no stem "
                          f"info (old format). Delete the cache files under "
                          f"{cache_base} and re-run to regenerate with stems.", flush=True)
                    continue
                common_stems = [s for s in img_stems if s in snr_stem_map]

                if len(common_stems) == 0:
                    print(f"Skipping {img_algo}+{snr_algo}: no matching stems between image and sensor")
                    continue

                # Gather aligned features in stem order
                aligned_img = np.array([img_feats[img_stem_map[s]] for s in common_stems])
                aligned_snr = np.array([snr_feats[snr_stem_map[s]] for s in common_stems])

                # Use dataset_index labels (authoritative) if available, else fallback to model-inferred
                if dataset_index:
                    aligned_labels = np.array([
                        dataset_index[s]["label"] if s in dataset_index else img_labels[img_stem_map[s]]
                        for s in common_stems
                    ], dtype=np.int64)
                    n_from_index = sum(1 for s in common_stems if s in dataset_index)
                    print(f"Labels: {n_from_index}/{len(common_stems)} from dataset_index.json", flush=True)
                else:
                    aligned_labels = np.array([img_labels[img_stem_map[s]] for s in common_stems])

                print(f"Aligned {len(common_stems)} paired samples by stem (img={len(img_stems)}, snr={len(snr_stems)})", flush=True)

                # Assign each paired sample to train/val/test via a file-level
                # split manifest. This makes the fusion stage INTERNALLY
                # honest (train/val/test are disjoint files; scaler, class
                # weights, and selection use train/val only).
                #
                # Whether the COMPONENT models also never saw the fusion test
                # rows depends on where they were trained: only if the
                # component sensor job trained on this same directory (same
                # manifest) does that stronger guarantee hold. We check and
                # record it honestly instead of assuming it.
                from app.services.splits import get_or_create_split, stem_partition_map
                manifest_dir = (dataset_path / "sensor") if (dataset_path / "sensor").exists() else dataset_path
                snr_train_ds = Path(str(snr_best.args.get("dataset_path", "")))
                try:
                    component_split_shared = (snr_train_ds.exists()
                                              and manifest_dir.resolve() == snr_train_ds.resolve())
                except OSError:
                    component_split_shared = False
                if not component_split_shared:
                    print(f"WARNING: sensor component (job {snr_best.id}) trained on "
                          f"'{snr_train_ds}', not on this hybrid dataset's sensor dir. "
                          f"The fusion test partition is honest w.r.t. the fusion head, "
                          f"but component models may have seen these files — recorded as "
                          f"component_split_shared=false.", flush=True)

                stem_label_lookup = {s: int(l) for s, l in zip(common_stems, aligned_labels)}
                manifest = get_or_create_split(
                    manifest_dir, [Path(f"{s}.csv") for s in common_stems], seed=42,
                    label_fn=lambda p: stem_label_lookup.get(p.stem))
                part_map = stem_partition_map(manifest)
                split_assignment = [part_map.get(s) for s in common_stems]
                # Keep only rows with a partition AND a valid binary label
                # (a -1 from a malformed index entry would crash CrossEntropy).
                kept = [i for i, p in enumerate(split_assignment)
                        if p is not None and int(aligned_labels[i]) in (0, 1)]
                if len(kept) < len(common_stems):
                    print(f"Excluding {len(common_stems) - len(kept)} stems "
                          f"(not in manifest or invalid label).", flush=True)

                # Save aligned features/scores (+ stem & split) for the wrapper
                data = {}
                if use_scores:
                    # 1-D component probabilities -> scores.csv for LateFusion
                    out_csv = job_perm_dir / "scores.csv"
                    data["img_score"] = aligned_img[kept].reshape(-1)
                    data["snr_score"] = aligned_snr[kept].reshape(-1)
                else:
                    out_csv = job_perm_dir / "features.csv"
                    for i in range(aligned_img.shape[1]): data[f"img_{i}"] = aligned_img[kept][:, i]
                    for i in range(aligned_snr.shape[1]): data[f"snr_{i}"] = aligned_snr[kept][:, i]
                data["label"] = aligned_labels[kept]
                data["stem"] = [common_stems[i] for i in kept]
                data["split"] = [split_assignment[i] for i in kept]
                pd.DataFrame(data).to_csv(out_csv, index=False)
                
                # Train
                from app.services.models.hybrid.fusion import FeatureFusionWrapper, LateFusionWrapper
                run_config = {
                    "output_dir": str(job_perm_dir),
                    "input_dim_image": aligned_img.shape[1],
                    "input_dim_sensor": aligned_snr.shape[1],
                    "fusion_type": fusion_type
                }
                
                if fusion_type == "late":
                    # Score-level fusion: alpha and threshold are selected on
                    # the val partition by F1, reported on test.
                    fusion_model = LateFusionWrapper(f"{job.id}_{img_algo}_{snr_algo}", run_config)
                    print(f"Selecting Late Fusion (alpha, threshold) for {img_algo}+{snr_algo}...")
                else:
                    fusion_model = FeatureFusionWrapper(f"{job.id}_{img_algo}_{snr_algo}", run_config)
                    print(f"Training Feature Fusion head for {img_algo}+{snr_algo}...")

                train_res = fusion_model.train(job_perm_dir, epochs=15) or {}
                if train_res.get("status") == "failed":
                    # Never evaluate an untrained model — random-init weights on
                    # unscaled features produce real-looking garbage metrics.
                    print(f"Skipping {img_algo}+{snr_algo}: fusion training failed "
                          f"({train_res.get('reason')})", flush=True)
                    continue
                metrics = fusion_model.evaluate(job_perm_dir)

                permutations.append({
                    "image_model": img_algo,
                    "sensor_model": snr_algo,
                    # selection metric: val F1 for both fusion types
                    # (feature: checkpoint selection; late: operating point)
                    "selection_score": train_res.get("val_f1", train_res.get("val_acc")),
                    "selection_metric": "val_f1",
                    "val_acc": train_res.get("val_acc"),
                    "metrics": metrics,                    # held-out test metrics
                    "component_split_shared": component_split_shared,
                    "image_job_id": img_best.id,
                    "sensor_job_id": snr_best.id
                })

        if not permutations:
            return {"status": "failed", "error": "No valid model combinations could be trained. Check component models."}

        # Select the best pairing on the VALIDATION metric, report its TEST
        # metrics. Selecting by test accuracy and reporting that same max
        # would be a multiple-comparisons (winner's curse) bias.
        best_p = max(permutations, key=lambda x: x.get("selection_score") or 0)

        # Deploy the winning fusion head so run_inference can use it
        if fusion_type != "late":
            best_dir = Path(f"runs/{job.id}_{best_p['image_model']}_{best_p['sensor_model']}")
            if (best_dir / "model.pth").exists():
                settings.LATEST_HYBRID_MODEL.parent.mkdir(parents=True, exist_ok=True)
                for fname in ("model.pth", "scaler.pkl", "config.json"):
                    src = best_dir / fname
                    if src.exists():
                        shutil.copy(src, settings.LATEST_HYBRID_MODEL.parent / fname)
                print(f"Persisted best fusion head ({best_p['image_model']}+"
                      f"{best_p['sensor_model']}) to {settings.LATEST_HYBRID_MODEL.parent}", flush=True)

        return {
            "status": "completed",
            "algorithm": f"Multi-Model Hybrid Evaluation ({fusion_type})",
            "dataset": str(dataset_path),
            "timestamp": datetime.utcnow().isoformat(),
            "fusion_type": fusion_type,
            "permutations": permutations,
            "summary": {
                "permutations_evaluated": len(permutations),
                "selection_basis": best_p.get("selection_metric"),
                "best_pairing": f"{best_p['image_model']} + {best_p['sensor_model']}",
                "best_pairing_selection_score": best_p.get("selection_score"),
                "best_pairing_test_accuracy": best_p["metrics"].get("accuracy"),
                "best_pairing_test_f1": best_p["metrics"].get("f1")
            }
        }

    elif job.task_type == "run_inference":
        # Args: drive_path, model_path
        drive_path = Path(job.args.get("drive_path", "data/dummy_drive"))
        
        from app.services.inference import InferenceEngine, EventGenerator, ClipExtractor
        
        # Resolve model path: Check args, then fallback to latest canonical models
        model_path = job.args.get("model_path")
        if model_path:
            model_path = Path(model_path)
        else:
            if settings.LATEST_HYBRID_MODEL.exists():
                model_path = settings.LATEST_HYBRID_MODEL
            elif settings.LATEST_SENSOR_MODEL.exists():
                model_path = settings.LATEST_SENSOR_MODEL
            elif settings.LATEST_IMAGE_MODEL.exists():
                model_path = settings.LATEST_IMAGE_MODEL
        
        if model_path:
            print(f"Inference using model: {model_path}")
        else:
            print("Warning: No model found for inference, using defaults.")

        # Support model_config arg
        model_config = job.args.get("model_config")
        inf_engine = InferenceEngine(model_path=model_path, model_config=model_config) 
        preds = inf_engine.process_drive(drive_path)
        
        # Events (Thresholded at 0.75 normally, 0.5 for NuScenes)
        default_thresh = 0.5 if "nuscenes" in str(drive_path).lower() else 0.75
        threshold = job.args.get("threshold", default_thresh)
        times = [p['time'] for p in preds]
        scores = [p['score'] for p in preds]
        v_scores = [p.get('v_score', 0.0) for p in preds]
        s_scores = [p.get('s_score', 0.0) for p in preds]
        
        # generate_events merges contiguous supra-threshold runs into one
        # event per physical anomaly; apply_nms kept as a safety net.
        raw_events = EventGenerator.generate_events(times, scores, threshold=threshold, v_scores=v_scores, s_scores=s_scores)
        final_events = EventGenerator.apply_nms(raw_events)
        initial_count = len(final_events)

        # Optional conjunctive gate. OFF by default: it overrides the model's
        # decision rule and suppresses modality-exclusive detections, so any
        # results produced with it describe the heuristic, not the model. If
        # enabled, the operating point is recorded in the job result.
        strict_filter = job.args.get("strict_filter", False)
        if strict_filter:
            final_events = [
                e for e in final_events
                if e.get('v_score', 0.0) >= 0.15
                and e.get('s_score', 0.0) >= 0.15
                and e.get('score', 0.0) >= 0.8
            ]
            print(f"STRICT FILTER ON: {initial_count} -> {len(final_events)} events (v>=0.15, s>=0.15, score>=0.8)")

        print(f"Detected {len(final_events)} events over threshold {threshold} (segment-merged).")
        
        # Save events in a distinct block to avoid StaleDataError during long I/O
        db_events = []
        with Session(engine) as session:
            # Create or update Run entry so the UI can find these results
            run = session.get(Run, str(job.id))
            if not run:
                run = Run(
                    id=str(job.id),
                    config=job.args,
                    metrics={"events_found": len(final_events)},
                    status="completed"
                )
                session.add(run)
            else:
                run.status = "completed"
                run.metrics = {"events_found": len(final_events)}
                session.add(run)
            
            # Clean up existing events if re-running
            session.exec(select(Event).where(Event.run_id == str(job.id))).all() # Just to be sure? Actually simpler to just DELETE
            from sqlmodel import delete
            session.exec(delete(Event).where(Event.run_id == str(job.id)))
            
            for evt in final_events:
                vs = evt.get('v_score', 0.0)
                ss = evt.get('s_score', 0.0)
                
                # Determine primary trigger
                if "nuscenes" in str(drive_path).lower():
                    # NuScenes Specific Logic
                    # Filter: both must be contributing
                    if vs <= 0 or ss <= 0:
                        continue
                        
                    # Labeling: simple threshold
                    if evt['score'] > 0.7:
                        source = "hybrid"
                    else:
                        source = "vision" if vs > ss else "sensor"
                else:
                    # iPhone Logic (Standard)
                    if vs >= 0.10 and ss >= 0.4:
                        source = "hybrid" 
                    elif vs >= 0.30:
                        source = "vision"
                    elif ss >= 0.5:
                        source = "sensor"
                    else:
                        # Fallback for weak signals: label based on dominant score
                        if vs > ss:
                            source = "vision"
                        else:
                            source = "sensor"
                    
                db_event = Event(
                    run_id=str(job.id),
                    time=evt['time'],
                    score=evt['score'],
                    v_score=vs,
                    s_score=ss,
                    trigger_source=source
                )
                session.add(db_event)
                db_events.append(db_event)
            
            session.commit()
            # Re-fetch or refresh to get IDs safely
            for e in db_events: session.refresh(e)
            print(f"Successfully persisted {len(db_events)} events for run {job.id}", flush=True)
            
        # Clip Extraction (Performed outside the main event creation transaction)
        clips_dir = drive_path / "clips"
        clips_dir.mkdir(exist_ok=True)
        
        # Find video path
        video_path = drive_path / "video.mp4"
        if not video_path.exists():
            # Check root or Camera folder for any mp4
            video_files = list(drive_path.glob("*.mp4")) + list((drive_path / "Camera").glob("*.mp4"))
            if video_files:
                video_path = video_files[0]
        
        if video_path.exists() and db_events:
            print(f"Extracting clips from {video_path}...")
            # Determine static prefix for video_url
            url_prefix = "/static/iphone"
            rel_drive = ""
            try:
                abs_drive = drive_path.resolve()
                if "nuscenes" in str(abs_drive):
                    url_prefix = "/static/nuscenes"
                    abs_base = (settings.DATA_DIR / "nuscenes").resolve()
                else:
                    abs_base = (settings.DATA_DIR / "iphone").resolve()
                rel_drive = str(abs_drive.relative_to(abs_base))
            except Exception as e:
                print(f"Path mapping error: {e}")

            # Re-open session to update video URLs
            with Session(engine) as session:
                for i, evt in enumerate(db_events[:100]):
                    output_file = f"event_{evt.id}.mp4"
                    try:
                        ClipExtractor.extract_clip(
                            video_path=video_path, 
                            time=evt.time, 
                            output_path=clips_dir / output_file,
                            duration=4.0,
                            pre_context=2.0
                        )
                        
                        if rel_drive:
                            # Re-fetch to ensure we are attached to current session
                            db_evt = session.get(Event, evt.id)
                            if db_evt:
                                db_evt.video_url = f"{url_prefix}/{rel_drive}/clips/{output_file}"
                                
                                # nuScenes Specific: Add direct image link for preview
                                if url_prefix == "/static/nuscenes":
                                    from app.services.ingestion.nuscenes import NuScenesIngestionService
                                    img_url = NuScenesIngestionService.get_image_url(drive_path, evt.time)
                                    if img_url:
                                        db_evt.image_url = img_url
                                        
                                session.add(db_evt)
                    except Exception as e:
                        print(f"Clip extraction failed for event {evt.id}: {e}")
                
                session.commit()
        elif not video_path.exists():
            print(f"Warning: No video found in {drive_path}, skipping clip extraction.")
            
        return {"status": "inference_complete", "events_found": len(final_events),
                "threshold": threshold,
                "strict_filter": bool(strict_filter),
                "strict_filter_operating_point": {"v_score": 0.15, "s_score": 0.15, "score": 0.8} if strict_filter else None}

    elif job.task_type == "validate_datasets":
        results = {
            "image": {},
            "sensor": {},
            "hybrid": {}
        }
        
        # Helper for recursive directory scanning
        def scan_recursive(base_path: Path):
            if not base_path.exists():
                return []
            dirs = []
            for item in base_path.rglob("*"):
                if item.is_dir():
                    dirs.append(item)
            return dirs

        # Check Image Datasets
        img_dir = settings.DATA_DIR / "image"
        for ds in scan_recursive(img_dir):
            is_yolo, msg = DatasetValidator.validate_image_dataset(ds, "yolo")
            if is_yolo:
                results["image"][str(ds.relative_to(img_dir))] = {"valid": is_yolo, "msg": msg}
        
        # Check Sensor Datasets
        sensor_dir = settings.DATA_DIR / "sensor"
        found_sensor_dirs = []
        for ds in scan_recursive(sensor_dir):
            is_log, msg = DatasetValidator.validate_sensor_dataset(ds, "sensorlogger")
            is_just, j_msg = DatasetValidator.validate_just_sensor(ds)
            
            if is_log:
                found_sensor_dirs.append((ds, is_log, msg))
            elif is_just:
                found_sensor_dirs.append((ds, is_just, j_msg))
            else:
                # Also check for our unified format
                csvs = list(ds.glob("*.csv"))
                if csvs:
                    try:
                        df = pd.read_csv(csvs[0], nrows=0)
                        if 'accel_x' in df.columns and 'label' in df.columns:
                            results["sensor"][str(ds.relative_to(sensor_dir))] = {"valid": True, "msg": "✅ Unified Sensor Format"}
                    except: pass
        
        # Grouping Logic for Sensor Datasets
        grouped = {}
        for ds_path, valid, msg in found_sensor_dirs:
            rel_path = ds_path.relative_to(sensor_dir)
            parts = rel_path.parts
            
            # If it's nested (e.g. Data 4/Road Anomalies/1. Bump), target the highest reasonable grouping
            # Usually the first part is the dataset name (Data 4, Data 5, etc.)
            if len(parts) > 1:
                top_parent = parts[0]
                if top_parent not in grouped: grouped[top_parent] = []
                grouped[top_parent].append((rel_path, valid, msg))
            else:
                results["sensor"][str(rel_path)] = {"valid": valid, "msg": msg}
                
        for parent, members in grouped.items():
            # Heuristic: Group if there are multiple sub-datasets, 
            # OR if it's a known large dataset directory (Data 2, 4, 5)
            is_known_set = any(x in parent for x in ["Data 2", "Data 4", "data 5", "Data 5"])
            
            if is_known_set or len(members) > 3:
                # Consolidate into one entry
                results["sensor"][parent] = {"valid": True, "msg": f"✅ Grouped Dataset ({len(members)} sub-items)"}
            else:
                # Keep separate
                for rel_p, valid, msg in members:
                    results["sensor"][str(rel_p)] = {"valid": valid, "msg": msg}

        # Check Hybrid Datasets
        hybrid_dir = settings.DATA_DIR / "hybrid"
        found_hybrid_dirs = []
        for ds in scan_recursive(hybrid_dir):
            is_valid, msg = DatasetValidator.validate_hybrid_dataset(ds)
            if is_valid:
                found_hybrid_dirs.append((ds, is_valid, msg))

        # Grouping Logic for Hybrid Datasets
        grouped_hybrid = {}
        for ds_path, valid, msg in found_hybrid_dirs:
            rel_path = ds_path.relative_to(hybrid_dir)
            parts = rel_path.parts
            
            if len(parts) > 1:
                top_parent = parts[0]
                if top_parent not in grouped_hybrid: grouped_hybrid[top_parent] = []
                grouped_hybrid[top_parent].append((rel_path, valid, msg))
            else:
                results["hybrid"][str(rel_path)] = {"valid": valid, "msg": msg}
                
        for parent, members in grouped_hybrid.items():
            is_known_set = any(x in parent for x in ["Data 1", "Drive", "DriveTimeline"])
            if is_known_set or len(members) > 3:
                results["hybrid"][parent] = {"valid": True, "msg": f"✅ Grouped Hybrid Dataset ({len(members)} sub-items)"}
            else:
                for rel_p, valid, msg in members:
                    results["hybrid"][str(rel_p)] = {"valid": valid, "msg": msg}

        # Processed cache
        processed_dir = settings.DATA_DIR / "processed" / "sensor"
        if processed_dir.exists():
            results["sensor"]["processed"] = {"valid": True, "msg": f"✅ Processed cache: {len(list(processed_dir.glob('*.csv')))} files"}
                    
        return results

    elif job.task_type == "run_tests":
        results = {}
        
        # 1. Run Pytest for logic and API
        print("Running pytest...")
        py_res = subprocess.run([sys.executable, "-m", "pytest", "app/tests"], capture_output=True, text=True)
        results["pytest"] = {
            "status": "passed" if py_res.returncode == 0 else "failed",
            "output": py_res.stdout + py_res.stderr
        }
        
        # 2. Run Algorithm Simulation Scripts (Phase Scripts)
        print("Running algorithm simulation tests...")
        for i in range(1, 7):
            script = f"verify_phase{i}.py"
            if os.path.exists(script):
                print(f"Executing {script}...")
                s_res = subprocess.run([sys.executable, script], capture_output=True, text=True)
                results[f"phase_{i}"] = {
                    "status": "passed" if s_res.returncode == 0 else "failed",
                    "output": s_res.stdout + s_res.stderr
                }
        
        total_passed = sum(1 for v in results.values() if v["status"] == "passed")
        return {
            "status": "completed",
            "summary": f"Passed {total_passed}/{len(results)} test modules.",
            "details": results
        }

    elif job.task_type == "process_dataset":
        # Args: path, type, output_name
        return {"status": "processed (mock)"}

    # Fallback
    time.sleep(2)
    return {"message": f"Task {job.task_type} completed (stub)"}

def run_worker():
    print("WORKER STARTUP - FLUSHED CACHE")
    print(f"Worker started with PID {os.getpid()}")
    while True:
        try:
            with Session(engine) as session:
                # Find queued job
                statement = select(Job).where(Job.status == JobStatus.QUEUED).limit(1)
                job = session.exec(statement).first()

                if job:
                    # Mark as running
                    job.status = JobStatus.RUNNING
                    job.started_at = datetime.now(timezone.utc)
                    job.pid = os.getpid()
                    session.add(job)
                    session.commit()
                    session.refresh(job)
                    
                    print(f"Picked up job {job.id}", flush=True)

                    try:
                        # Execute task
                        result = handle_task(job)
                        
                        job.status = JobStatus.COMPLETED
                        job.completed_at = datetime.now(timezone.utc)
                        
                        # Fix for JSON mutation tracking: 
                        # SQLAlchemy doesn't track changes inside dicts unless re-assigned
                        new_args = dict(job.args)
                        new_args["result"] = result
                        job.args = new_args
                        
                    except Exception as e:
                        print(f"Job {job.id} failed: {e}")
                        job.status = JobStatus.FAILED
                        job.error_message = str(e) + "\n" + traceback.format_exc()
                        job.completed_at = datetime.now(timezone.utc)
                    
                    session.add(job)
                    session.commit()
                else:
                    # No jobs, sleep
                    time.sleep(settings.WORKER_POLL_INTERVAL)
        except Exception as e:
            print(f"Worker main loop error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    from app.core.database import create_db_and_tables
    create_db_and_tables()
    run_worker()
