
import sys
import cv2
import pandas as pd
import numpy as np
from pathlib import Path

# Add app to path
sys.path.append(".")
from app.services.inference.engine import InferenceEngine

def main():
    print("--- TESTING DEPLOYMENT PIPELINE ---")
    
    # 1. Initialize Engine
    print("Initializing Engine...")
    engine = InferenceEngine()
    
    if engine.fusion_model is None:
        print("FAIL: Fusion Model not loaded!")
        return
    else:
        print("PASS: Fusion Model loaded.")

    # 2. Load Data
    img_path = Path("data/hybrid/Data 1 - Both/images/100_Normal_0.jpg")
    csv_path = Path("data/hybrid/Data 1 - Both/sensor/100_Normal_0.csv")
    
    if not img_path.exists() or not csv_path.exists():
        print("FAIL: Test files missing.")
        return
        
    img = cv2.imread(str(img_path))
    df = pd.read_csv(csv_path)
    
    print(f"Image Shape: {img.shape}")
    print(f"Sensor Rows: {len(df)}")
    
    # 3. Test Feature Extraction
    print("\n--- Testing Feature Extraction ---")
    
    # YOLO
    print("Extracting YOLO embedding...")
    yolo_emb = engine.image_model.predict_embedding(img)
    print(f"YOLO Emb Shape: {yolo_emb.shape}, Mean: {yolo_emb.mean():.4f}")
    if yolo_emb.shape != (512,):
        print(f"FAIL: YOLO dim mismatch {yolo_emb.shape}")
        
    # CNN
    print("Extracting CNN embedding...")
    # CNNWrapper expects window or df?
    # predict_embedding takes window (numpy or tensor).
    # We need to prep window from DF.
    cols = ['accel_x', 'accel_y', 'accel_z']
    # Check if cols exist, rename if needed.
    # Data 1 CSVs usually have 'Ax', 'Ay', 'Az' or similar.
    # Let's check first row to be safe or rely on logic inside DL?
    # But predict_embedding is low level. 
    # Let's inspect DF cols
    print(f"CSV Columns: {df.columns.tolist()}")
    
    # Simplest: conversion logic
    # Data 1 usually: timestamp, Ax, Ay, Az ...
    # Let's try to grab 3 numeric cols
    data_cols = [c for c in df.columns if c not in ['timestamp', 'label', 'time', 'index']]
    if len(data_cols) >= 3:
        arr = df[data_cols[:3]].values.astype(np.float32)
    else:
        print("FAIL: Not enough sensor cols")
        arr = np.zeros((100, 3), dtype=np.float32)
        
    # Pad/Crop to 100
    if len(arr) > 100: arr = arr[:100]
    elif len(arr) < 100: 
        pad = np.zeros((100 - len(arr), 3))
        arr = np.concatenate([arr, pad])
        
    cnn_emb = engine.sensor_model.predict_embedding(arr)
    print(f"CNN Emb Shape: {cnn_emb.shape}, Mean: {cnn_emb.mean():.4f}")
    if cnn_emb.shape != (128,): # 128 dim
        print(f"FAIL: CNN dim mismatch {cnn_emb.shape}")
        
    # 4. Test Fusion
    print("\n--- Testing Fusion Prediction ---")
    prob = engine.fusion_model.predict((yolo_emb, cnn_emb))
    print(f"Fusion Probability: {prob:.4f}")
    
    if 0.0 <= prob <= 1.0:
        print("PASS: Valid Probability")
    else:
        print(f"FAIL: Invalid Probability {prob}")
        
    # 5. Check Scaler
    if engine.fusion_model.scaler:
        print("PASS: Scaler is active.")
    else:
        print("FAIL: Scaler missing")

if __name__ == "__main__":
    main()
