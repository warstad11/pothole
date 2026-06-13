import hashlib
from pathlib import Path
import os

DATASETS = [
    "data/image/raw/Data 3 - Images",
    "data/image/raw/Pothole.v1-raw.yolov8",
    "data/sensor/Data 3 - Just Sensor"
]

def hash_file(path):
    try:
        if path.is_file():
            return hashlib.md5(path.read_bytes()).hexdigest()
    except:
         return None
    return None

def check_image_dataset(path_str):
    print(f"\nChecking {path_str}...")
    base = Path(path_str)
    
    # YOLO structure assumption: images/train, images/val OR train/images, val/images
    train_dir = base / "train" 
    val_dir = base / "valid"
    if not val_dir.exists(): val_dir = base / "val"
    
    # Adjust for 'images' subdir if needed
    if (train_dir / "images").exists(): train_dir = train_dir / "images"
    if (val_dir / "images").exists(): val_dir = val_dir / "images"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"  [!] Missing folders: Train={train_dir.exists()}, Val={val_dir.exists()}")
        # Check alternate YOLO struct
        if (base / "images" / "train").exists():
            train_dir = base / "images" / "train"
            val_dir = base / "images" / "val"
        else:
            return

    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")
    
    train_hashes = set()
    train_files = list(train_dir.rglob("*.*"))
    print(f"  Hashing {len(train_files)} training files...")
    for f in train_files:
        h = hash_file(f)
        if h: train_hashes.add(h)
        
    val_files = list(val_dir.rglob("*.*"))
    print(f"  Hashing {len(val_files)} validation files...")
    leakage_count = 0
    for f in val_files:
        h = hash_file(f)
        if h and h in train_hashes:
            leakage_count += 1
            if leakage_count <= 5:
                print(f"    [LEAK] {f.name} matches a training file!")
                
    if leakage_count > 0:
        print(f"  [CRITICAL] Found {leakage_count} overlapping files (Data Leakage!)")
    else:
        print("  [OK] No file content overlap found.")

def check_sensor_dataset(path_str):
    print(f"\nChecking Sensor {path_str}...")
    # Sensor usually is one folder? Or has split?
    # If standard folder, split implies random split in code.
    # Looking for duplicate CSVs
    base = Path(path_str)
    files = list(base.rglob("*.csv"))
    hashes = {}
    dupes = 0
    for f in files:
        h = hash_file(f)
        if h:
            if h in hashes:
                dupes += 1
                if dupes <= 5:
                    print(f"    [DUPE] {f.name} is duplicate of {hashes[h]}")
            else:
                hashes[h] = f.name
    
    if dupes > 0:
        print(f"  [WARNING] Found {dupes} duplicate files within dataset.")
    else:
        print("  [OK] No duplicates found.")

if __name__ == "__main__":
    for d in DATASETS:
        if "sensor" in d.lower():
            check_sensor_dataset(d)
        else:
            check_image_dataset(d)
