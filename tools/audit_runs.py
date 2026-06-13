
import os
import json
from pathlib import Path

def audit_runs():
    runs_dir = Path("runs")
    
    to_retrain = []
    to_recompute = []
    valid_but_unseeded = []
    
    for run_path in sorted(list(runs_dir.iterdir())):
        if not run_path.is_dir(): continue
        metrics_path = run_path / "metrics.json"
        if not metrics_path.exists(): continue
        
        try:
            with open(metrics_path, 'r') as f:
                data = json.load(f)
            
            algo = data.get("algorithm", "").lower()
            dataset = data.get("dataset", "")
            
            # Hybrid Models (Fusion) -> RETRAIN
            # Reason: Feature Dimensionality changed (RF now 128-dim) + Late Fusion Logic
            if "fusion" in algo or "hybrid" in algo or "wrapper" in algo.lower() and "late" in algo.lower():
                to_retrain.append(f"{run_path.name} ({algo})")
                
            # Hybrid Job sub-folders (e.g. 96_yolov8_cnn) -> depends
            # The folders like `96_yolov8_cnn` are usually sub-runs.
            # If they are just folders, we might skip or check parent.
            
            # Faster R-CNN -> RETRAIN
            # Reason: Strict Split Validation added used to prevent leakage. Old runs likely leaked.
            elif "faster_rcnn" in algo or "rcnn" in algo:
                to_retrain.append(f"{run_path.name} ({algo})")
                
            # YOLO -> RECOMPUTE
            elif "yolo" in algo:
                to_recompute.append(f"{run_path.name} ({algo})")
                
            # U-Net -> RECOMPUTE
            elif "unet" in algo:
                to_recompute.append(f"{run_path.name} ({algo})")
                
            # Sensor Models (RF / CNN) -> VALID (But Unseeded)
            elif "random_forest" in algo or "cnn" in algo or "dl" in algo:
                 valid_but_unseeded.append(f"{run_path.name} ({algo})")
                 
        except:
            pass
            
    print("## RETRAIN REQUIRED (Critical Structural/Data Fixes)")
    for x in to_retrain: print(f"- {x}")
    
    print("\n## RECOMPUTE REQUIRED (Metric Fixes Only)")
    for x in to_recompute: print(f"- {x}")
    
    print("\n## OPTIONAL RETRAIN (Reproducibility Only)")
    for x in valid_but_unseeded: print(f"- {x}")

if __name__ == "__main__":
    audit_runs()
