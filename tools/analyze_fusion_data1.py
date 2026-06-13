
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from app.services.models.image.torchvision import FasterRCNNWrapper, collate_fn, PotholeDetectionDataset
from app.services.models.sensor.classical import RandomForestWrapper
import numpy as np
import pandas as pd
import json
import joblib

def run_analysis():
    # Paths
    data_root = Path("data/hybrid/Data 1 - Both")
    json_path = data_root / "dataset_index.json"
    image_dir = data_root / "images"
    sensor_dir = data_root / "sensor"
    
    # Models
    rcnn_path = Path("runs/55/faster_rcnn.pth") # Check full_model.pt too
    if not rcnn_path.exists(): rcnn_path = Path("runs/55/full_model.pt")
    
    rf_path = Path("runs/2/model.joblib")
    
    if not rcnn_path.exists() or not rf_path.exists():
        print("Models not found.")
        return

    # Load Labels
    print(f"Loading Index: {json_path}")
    with open(json_path, 'r') as f:
        index = json.load(f)
        
    keys = sorted(list(index.keys()))
    print(f"Total Index Entries: {len(keys)}")
    
    # 1. Faster R-CNN Setup
    print("Loading Faster R-CNN...")
    rcnn_config = {"num_classes": 2, "backbone": "resnet50"}
    rcnn = FasterRCNNWrapper("eval_rcnn", rcnn_config)
    rcnn.load(rcnn_path)
    rcnn.model.eval()
    device = torch.device("cpu")
    rcnn.model.to(device)
    
    # 2. RF Setup
    print("Loading Random Forest...")
    rf_config = {"n_estimators": 100}
    rf = RandomForestWrapper("eval_rf", rf_config)
    rf.load(rf_path)
    
    # buffers
    y_true = []
    y_rcnn = [] # probs
    y_rf = []   # probs
    
    print("Starting Aligned Inference...")
    
    # Batch R-CNN Inference? No, single loop to keep alignment simple and robust
    # Optimizable, but explicit is safer for alignment
    
    for i, key in enumerate(keys):
        item = index[key]
        img_rel = key # "images/file.jpg"
        sensor_rel = item['sensor_path']
        label = item['label']
        
        img_full = data_root / img_rel
        sensor_full = data_root / sensor_rel
        
        if not img_full.exists() or not sensor_full.exists():
            continue
            
        if i % 50 == 0: print(f"Processing {i}/{len(keys)}...", flush=True)

        # --- R-CNN Pred ---
        # Load Image using Dataset logic (just manual here for speed)
        import cv2
        img0 = cv2.imread(str(img_full))
        if img0 is None: continue
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            res = rcnn.model(img_t)[0]
            scores = res['scores'].cpu().numpy()
            # Image Prob = Max Score of any box
            p_rcnn = 0.0
            if len(scores) > 0:
                p_rcnn = float(np.max(scores))
        
        # --- RF Pred ---
        # Load Sensor
        p_rf = 0.0
        try:
            # Reimplement RF load logic quickly
            # It expects windowing.
            df = None
            if sensor_full.suffix == '.csv':
                df = pd.read_csv(sensor_full)
            else:
                 # txt fallback
                 df = pd.read_csv(sensor_full, header=None)
                 if df.shape[1] == 4:
                     df.columns = ['label', 'accel_x', 'accel_y', 'accel_z']
                     df['gyro_x']=0; df['gyro_y']=0; df['gyro_z']=0
                 else:
                     df.rename(columns={0:'label'}, inplace=True)
            
            # Label might be missing in bare file, add dummy
            if 'label' not in df.columns: df['label'] = 0 
            
            # Extract
            X_w, _ = rf._create_windows_and_extract_features(df)
            if not X_w.empty:
                probs = rf.model.predict_proba(X_w)[:, 1]
                p_rf = float(np.max(probs))
            else:
                p_rf = 0.0 # No windows?
                
        except Exception as e:
            # print(f"RF Error {sensor_rel}: {e}")
            p_rf = 0.0

        y_true.append(label)
        y_rcnn.append(p_rcnn)
        y_rf.append(p_rf)

    # Metrics Calc
    y_true = np.array(y_true)
    y_rcnn = np.array(y_rcnn)
    y_rf = np.array(y_rf)
    
    def report(name, y_prob):
        y_pred = (y_prob > 0.5).astype(int)
        
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        acc = (tp + tn) / len(y_true)
        f1 = 2*prec*rec/(prec+rec+1e-6)
        
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1:        {f1:.4f}")
        print(f"CM: [[{tn}, {fp}], [{fn}, {tp}]]")

    report("Faster R-CNN (Data 3 -> Data 1)", y_rcnn)
    report("Random Forest (Data 3 -> Data 1)", y_rf)
    
    # Late Fusion
    y_late = (y_rcnn + y_rf) / 2
    report("Late Fusion (Average)", y_late)
    
    # Feature Fusion (Simulated for RF as Stacking)
    # Since we can't train, we can't do true feature fusion. 
    # But for RF, "Feature Fusion" == "Late Fusion" usually. 
    # We could optimize weights?
    # Let's trust Late Fusion.

if __name__ == "__main__":
    run_analysis()
