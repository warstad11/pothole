
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from app.services.models.image.torchvision import FasterRCNNWrapper
from app.services.models.sensor.dl import CNN1DWrapper, SensorWindowDataset
import numpy as np
import pandas as pd
import json

def run_analysis():
    # Paths
    data_root = Path("data/hybrid/Data 1 - Both")
    json_path = data_root / "dataset_index.json"
    
    # Models
    rcnn_path = Path("runs/55/faster_rcnn.pth") 
    if not rcnn_path.exists(): rcnn_path = Path("runs/55/full_model.pt")
    
    cnn_path = Path("runs/4/model.pth")
    
    if not rcnn_path.exists() or not cnn_path.exists():
        print(f"Models not found: RCNN={rcnn_path.exists()}, CNN={cnn_path.exists()}")
        return

    # Load Labels
    print(f"Loading Index: {json_path}")
    with open(json_path, 'r') as f:
        index = json.load(f)
        
    keys = sorted(list(index.keys()))
    print(f"Total Index Entries: {len(keys)}")
    
    device = torch.device("cpu") # Stability
    
    # 1. Faster R-CNN Setup
    print("Loading Faster R-CNN...")
    rcnn_config = {"num_classes": 2, "backbone": "resnet50"}
    rcnn = FasterRCNNWrapper("eval_rcnn", rcnn_config)
    rcnn.load(rcnn_path)
    rcnn.model.eval()
    rcnn.model.to(device)
    
    # 2. 1D-CNN Setup
    print("Loading 1D-CNN...")
    # Config from Job 4
    cnn_config = {
        "architecture": "1D-CNN", 
        "layers": 3, 
        "input_channels": 3, 
        "num_classes": 2,
        "window_size": 100,
        "stride": 20
    }
    cnn = CNN1DWrapper("eval_cnn", cnn_config)
    # Force device before load to map storage
    cnn.device = device 
    cnn.model.to(device)
    cnn.load(cnn_path)
    cnn.model.eval()
    
    # buffers
    y_true = []
    y_rcnn = [] # probs
    y_cnn = []   # probs
    
    print("Starting Aligned Inference...")
    
    for i, key in enumerate(keys):
        item = index[key]
        img_rel = key 
        sensor_rel = item['sensor_path']
        label = item['label']
        
        img_full = data_root / img_rel
        sensor_full = data_root / sensor_rel
        
        if not img_full.exists() or not sensor_full.exists():
            continue
            
        if i % 50 == 0: print(f"Processing {i}/{len(keys)}...", flush=True)

        # --- R-CNN Pred ---
        import cv2
        img0 = cv2.imread(str(img_full))
        if img0 is None: continue
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            res = rcnn.model(img_t)[0]
            scores = res['scores'].cpu().numpy()
            p_rcnn = 0.0
            if len(scores) > 0:
                p_rcnn = float(np.max(scores))
        
        # --- CNN Pred ---
        p_cnn = 0.0
        try:
            # We use the wrapper's _load_data logic but restricted to one file
            # _load_data returns a list of DataFrames.
            # We can mock a list [sensor_full] for it? 
            # Actually _load_data takes a directory.
            # Let's manually reuse the cleaning logic
            
            # Simple Load
            df = pd.read_csv(sensor_full)
            # Basic cleanup (renaming/padding handled by wrapper helper if we could use it, 
            # but it is bound to directory. Let's do a quick adapt)
            
            # Ensure 3 cols: Ax, Ay, Az
            valid_cols = []
            for c in df.columns:
                 if any(x in str(c).lower() for x in ['ax', 'ay', 'az', 'acc']):
                     valid_cols.append(c)
            
            if not valid_cols and df.shape[1] >= 3:
                # Headerless?
                valid_cols = df.columns[:3]
            
            if len(valid_cols) < 3:
                # Fallback
                 p_cnn = 0.0
            else:
                final_df = df[valid_cols].copy()
                # Normalize Z-Score
                for c in final_df.columns:
                    std = final_df[c].std()
                    if std > 1e-6:
                        final_df[c] = (final_df[c] - final_df[c].mean()) / std
                        final_df[c] = final_df[c].clip(-10.0, 10.0)
                    else:
                        final_df[c] = 0.0
                
                # Pad if needed (unlikely if we took 3)
                # ...
                
                # Windowing
                windows = []
                data_arr = final_df.values
                win_size = cnn.window_size
                stride = cnn.stride
                
                if len(data_arr) >= win_size:
                    for w_i in range(0, len(data_arr) - win_size, stride):
                        window = data_arr[w_i : w_i + win_size]
                        # Transpose to (Channels, Time)
                        window = window.T 
                        windows.append(window)
                
                if windows:
                    # Batch Predictions
                    windows_t = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
                    # (B, C, T)
                    with torch.no_grad():
                        outputs = cnn.model(windows_t)
                        # Softmax
                        probs = torch.softmax(outputs, dim=1)[:, 1] # Class 1
                        # Max Pool
                        p_cnn = float(probs.max().item())
                else:
                    p_cnn = 0.0

        except Exception as e:
            # print(f"CNN Error: {e}")
            p_cnn = 0.0

        y_true.append(label)
        y_rcnn.append(p_rcnn)
        y_cnn.append(p_cnn)

    # Metrics
    y_true = np.array(y_true)
    y_rcnn = np.array(y_rcnn)
    y_cnn = np.array(y_cnn)
    
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
    report("1D-CNN (Data 3 -> Data 1)", y_cnn)
    
    # Late Fusion
    y_late = (y_rcnn + y_cnn) / 2
    report("Late Fusion (Average)", y_late)

if __name__ == "__main__":
    run_analysis()
