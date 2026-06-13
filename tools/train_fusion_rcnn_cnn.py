
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import shutil
import random

# Add app to path
sys.path.append(".")

from app.services.models.image.torchvision import FasterRCNNWrapper
from app.services.models.sensor.dl import CNN1DWrapper
from app.services.models.hybrid.fusion import FeatureFusionWrapper
from app.core.config import settings

def main():
    # 1. Config
    RCNN_JOB_ID = "55" # Faster R-CNN (Data 3)
    CNN_JOB_ID = "4"   # 1D-CNN (Data 3)
    
    # Paths
    rcnn_weights = Path(f"runs/{RCNN_JOB_ID}/faster_rcnn.pth")
    if not rcnn_weights.exists():
        rcnn_weights = Path("runs/train_faster_rcnn_55_data3/faster_rcnn.pth")
        
    cnn_weights = Path(f"runs/{CNN_JOB_ID}/model.pth")
    if not cnn_weights.exists():
        cnn_weights = Path("runs/train_1dcnn_4_data3/model.pth")
        
    original_dataset_path = Path("data/hybrid/Data 1 - Both")
    
    output_dir = Path("runs/fusion_rcnn_cnn_data1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Prepare Balanced Subset (To speed up extraction)
    subset_dir = output_dir / "temp_subset"
    if subset_dir.exists(): shutil.rmtree(subset_dir)
    subset_dir.mkdir()
    (subset_dir / "images").mkdir()
    (subset_dir / "sensor").mkdir()
    
    print(f"Preparing Balanced Subset from {original_dataset_path}...")
    with open(original_dataset_path / "dataset_index.json", 'r') as f:
        index = json.load(f)
        
    # Group by label
    positives = []
    negatives = []
    
    for img_rel, meta in index.items():
        if meta['label'] == 1:
            positives.append(img_rel)
        else:
            negatives.append(img_rel)
            
    print(f"Original: Pos={len(positives)}, Neg={len(negatives)}")
    
    # Sample
    # Take all positives
    selected_pos = positives
    # Take equal negatives
    random.seed(42)
    selected_neg = random.sample(negatives, len(selected_pos))
    
    selected = selected_pos + selected_neg
    print(f"Selected: {len(selected)} samples ({len(selected_pos)} Pos, {len(selected_neg)} Neg)")
    
    # Link files
    for img_rel in selected:
        # Source Paths
        src_img = original_dataset_path / img_rel
        src_snr = original_dataset_path / index[img_rel]['sensor_path']
        
        # Dest Paths (keep filenames)
        dst_img = subset_dir / "images" / src_img.name
        dst_snr = subset_dir / "sensor" / src_snr.name
        
        # Symlink (or copy if symlink fails)
        try:
            dst_img.symlink_to(src_img.resolve())
            dst_snr.symlink_to(src_snr.resolve())
        except OSError:
             shutil.copy(src_img, dst_img)
             shutil.copy(src_snr, dst_snr)
             
    dataset_path = subset_dir # Use subset for extraction
    
    print(f"--- Starting Fusion Training (R-CNN + 1D-CNN) ---")
    
    if not rcnn_weights.exists() or not cnn_weights.exists():
        print("Error: Missing model weights.")
        return

    # 3. Extract Features
    features_csv = output_dir / "features.csv"
    rcnn_checkpoint = output_dir / "rcnn_features.npz"
    
    img_feats = None
    img_labels = None
    
    # ALWAYS Extract for the subset (don't rely on cached CSV since subset changes)
    if True: 
        print("Extracting features (Subset)...")
        
        # Check for R-CNN Checkpoint
        if rcnn_checkpoint.exists():
            print(f"Loading cached R-CNN features from {rcnn_checkpoint}...")
            data = np.load(rcnn_checkpoint)
            img_feats = data['feats']
            img_labels = data['labels']
        else:
            # Load R-CNN
            print("Loading Faster R-CNN...")
            rcnn = FasterRCNNWrapper("rcnn_extractor", {"num_classes": 2})
            rcnn.load(rcnn_weights)
            
            print("Extracting R-CNN features...")
            img_feats, img_labels = rcnn.extract_features(dataset_path / "images")
            
            print(f"Saving R-CNN features to {rcnn_checkpoint}...")
            np.savez(rcnn_checkpoint, feats=img_feats, labels=img_labels)
            
            del rcnn
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Load 1D-CNN
        print("Loading 1D-CNN...")
        # FIX: Job 4 was trained on 3 channels (Accel Only), even if data has 6.
        cnn = CNN1DWrapper("cnn_extractor", {
            "input_channels": 3, 
            "window_size": 200,
            "hidden_dims": [64, 128, 256],
            "num_classes": 2,
            "dataset_type": "sensor"
        })
        cnn.load(cnn_weights)
        
        print("Extracting Sensor Features...")
        snr_feats, snr_labels = cnn.extract_features(dataset_path / "sensor")
        
        del cnn
        
        if len(img_feats) != len(snr_feats):
            print(f"CRITICAL ERROR: Feature mismatch! Image={len(img_feats)}, Sensor={len(snr_feats)}")
            return
            
        data = {}
        for i in range(img_feats.shape[1]):
            data[f"img_{i}"] = img_feats[:, i]
        for i in range(snr_feats.shape[1]):
            data[f"snr_{i}"] = snr_feats[:, i]
            
        data['label'] = img_labels
        
        df = pd.DataFrame(data)
        df.to_csv(features_csv, index=False)
        print("Features saved.")

    # 4. Train Fusion Model
    print("Initializing Feature Fusion Model...")
    fusion = FeatureFusionWrapper("fusion_rcnn_cnn", {
        "input_dim_image": 512, 
        "input_dim_sensor": 128 
    })
    
    fusion.output_dir = output_dir
    
    print("Starting Training...")
    stats = fusion.train(output_dir, epochs=50) # Increased epochs for small data
    print(f"Training Complete. Train Acc: {stats.get('train_acc', 0):.4f}")
    
    # 5. Evaluate (on the SUBSET - as a proxy)
    # Ideally should eval on remaining data. But for fusion, we just want to see if it learns.
    # We used 80% for train. 20% validation.
    # fusion.evaluate handles splitting if passed a dataset. 
    # But FeatureFusionWrapper.evaluate() loads the whole dataset passed.
    # The 'dataset_path' contains features.csv.
    # If we call evaluate(output_dir), it loads the features.csv we just saved.
    # So it evaluates on (Train+Val).
    
    metrics = fusion.evaluate(output_dir)
    
    print("\n--- Final Results (Data 1 Hybrid Subset) ---")
    print(f"Accuracy:  {metrics.get('accuracy', 0):.4f}")
    print(f"Precision: {metrics.get('precision', 0):.4f}")
    print(f"Recall:    {metrics.get('recall', 0):.4f}")
    print(f"F1 Score:  {metrics.get('f1', 0):.4f}")
    print(f"Confusion Matrix: {metrics.get('confusion_matrix')}")
    
    # Cleanup
    shutil.rmtree(subset_dir)

if __name__ == "__main__":
    main()
