
import os
from pathlib import Path
import pandas as pd
import glob

DATA_ROOT = Path("data")

IMAGE_DATASETS = [
    DATA_ROOT / "image/raw/Data 3 - Images",
    DATA_ROOT / "image/raw/RealPothole600_Converted",
    DATA_ROOT / "image/raw/Pothole.v1-raw.yolov8",
    DATA_ROOT / "image/raw/NormalRoads"
]

SENSOR_DATASETS = [
    DATA_ROOT / "sensor/Data 3 - Just Sensor",
    DATA_ROOT / "sensor/Data 1 - Hybrid (Sensor)", # Verify path name
    DATA_ROOT / "sensor" # Scan root too just in case
]

def audit_image_yolo(path):
    if not path.exists():
        return f"{path.name:<30}: NOT FOUND"
    
    # Smarter YOLO detection: look for 'labels' or 'train/labels'
    labels_dirs = list(path.rglob("labels"))
    images_dirs = list(path.rglob("images"))
    
    if not labels_dirs:
        # Fallback: maybe flattened?
        return f"{path.name:<30}: No 'labels' dir found"
        
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    total_imgs = 0
    pos_count = 0
    neg_count = 0
    
    # Iterate all images found
    all_img_files = []
    for img_dir in images_dirs:
        for f in img_dir.rglob("*"):
             if f.suffix.lower() in img_exts and not f.name.startswith("._"):
                 all_img_files.append(f)
                 
    total_imgs = len(all_img_files)
    
    # Iterate all label files found
    all_txt_files = []
    for lbl_dir in labels_dirs:
        for f in lbl_dir.rglob("*.txt"):
            if not f.name.startswith("._") and f.name != "classes.txt":
                 all_txt_files.append(f)

    # Map stems
    txt_map = {f.stem: f for f in all_txt_files}
    
    for img_f in all_img_files:
        stem = img_f.stem
        if stem in txt_map:
            # Check content
            try:
                with open(txt_map[stem], 'r') as f:
                    content = f.read().strip()
                if content:
                    pos_count += 1
                else:
                    neg_count += 1
            except: pass
        else:
            # Implicit negative
            neg_count += 1
            
    balance = 0.0
    if total_imgs > 0:
        balance = pos_count / total_imgs

    return f"{path.name:<30} | Total: {total_imgs:<5} | Pos: {pos_count:<5} | Neg: {neg_count:<5} | Balance: {balance:.1%}"

def audit_sensor_csv(path):
    if not path.exists():
        # Check if it's a specific folder inside
        return f"{path}: NOT FOUND"
        
    files = list(path.rglob("*.csv")) + list(path.rglob("*.txt"))
    # Filter out metadata
    files = [f for f in files if f.name not in ['metadata.json', 'train.csv', 'val.csv', 'test.csv']]
    
    if not files:
        return f"{path.name:<30} | No CSV/TXT files found"

    pos_count = 0
    neg_count = 0
    
    for f in files:
        try:
            # Heuristic count based on label column or filename
            # For strict counting, we should read the file. 
            # But reading 5000 files is slow.
            # Let's use filename heuristic which we trust for splitting.
            
            # Filename heuristic from classical.py
            name_str = f.name.lower()
            parent_str = f.parent.name.lower()
            
            # Check filename first
            if any(x in name_str for x in ['pothole', 'positive', 'damage', 'cracking', 'alligator', 'spall', 'faulting', 'shoving']):
                pos_count += 1
            elif any(x in name_str for x in ['normal', 'plain', 'negative', 'undamaged', 'regular', 'road', 'joint', 'bump']):
                neg_count += 1
            else:
                # Parent dir check
                if any(x in parent_str for x in ['pothole', 'positive', 'damage']):
                    pos_count += 1
                elif any(x in parent_str for x in ['normal', 'plain', 'negative', 'undamaged', 'regular', 'road', 'joint']):
                    neg_count += 1
        except: pass
        
    total = pos_count + neg_count
    return f"{path.name:<30} | Total: {total:<5} | Pos: {pos_count:<5} | Neg: {neg_count:<5} | Balance: {pos_count/(total+1e-6):.1%}"

print(f"{'DATASET':<30} | {'TOTAL':<5} | {'POS':<5} | {'NEG':<5} | {'BALANCE (Pos %)'}")
print("-" * 80)

print("--- IMAGE DATASETS ---")
for p in IMAGE_DATASETS:
    print(audit_image_yolo(p))

print("\n--- SENSOR DATASETS ---")
# Explicitly list the ones we care about logic-wise
# "Data 3 - Just Sensor"
# "Data 1 - Hybrid (Sensor)" ?? Let's find real name
sensor_root = DATA_ROOT / "sensor"
if sensor_root.exists():
    dirs = [d for d in sensor_root.iterdir() if d.is_dir()]
    for d in dirs:
        print(audit_sensor_csv(d))
