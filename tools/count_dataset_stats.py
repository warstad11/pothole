
import os
from pathlib import Path
import yaml
import glob
import sys

def count_yolo(path, name):
    # Detect structure
    train_path = path / "train"
    val_path = path / "val"
    
    # Handle RealPothole structure (images/train)
    if not train_path.exists() and (path / "images" / "train").exists():
        train_path = path / "images" / "train"
        val_path = path / "images" / "val"
    
    # Handle Pothole.v1 valid alias
    if not val_path.exists() and (path / "valid").exists():
        val_path = path / "valid"
        
    splits = [("Train", train_path), ("Val", val_path)]
    
    print(f"\n--- {name} ---")
    
    for split_name, split_path in splits:
        if not split_path.exists():
            print(f"{split_name}: Not Found at {split_path}")
            continue
            
        # Find images
        # Should check 'images' subdir if split_path is root of split
        img_dir = split_path
        if (split_path / "images").exists():
            img_dir = split_path / "images"
            
        lbl_dir = split_path
        if (split_path / "labels").exists():
            lbl_dir = split_path / "labels"
        elif (split_path.parent / "labels" / split_path.name).exists():
             # Parallel structure: dataset/images/train -> dataset/labels/train
             lbl_dir = split_path.parent / "labels" / split_path.name
        elif (path / "labels" / split_name.lower()).exists():
             lbl_dir = path / "labels" / split_name.lower()
             
        # Glob images
        images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        
        pos = 0
        neg = 0
        
        for img in images:
            # Find label
            lbl_file = lbl_dir / img.with_suffix(".txt").name
            
            is_pos = False
            if lbl_file.exists():
                with open(lbl_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        is_pos = True
            
            if is_pos:
                pos += 1
            else:
                neg += 1
                
        print(f"{split_name}: {len(images)} files (Pos: {pos}, Neg: {neg})")

def count_sensor(path, name):
    print(f"\n--- {name} ---")
    files = list(path.glob("*.csv"))
    
    pos = 0
    neg = 0
    
    # Keywords from classical.py (Patched)
    keywords = ['pothole', 'positive', 'damage', 'crack', 'bump', 'shoving', 'spall', 'faulting', 'alligator', 'cracking']
    
    for f in files:
        fname = f.name.lower()
        # Logic: any keyword matches
        if any(k in fname for k in keywords):
            pos += 1
        else:
            neg += 1
            
    print(f"Total: {len(files)} files (Pos: {pos}, Neg: {neg})")
    print(f"Note: Dynamic Split (80% Train, 20% Val) used in code.")

def main():
    base_img = Path("data/image/raw")
    base_sensor = Path("data/sensor")
    
    # Image Datasets
    count_yolo(base_img / "Data 3 - Images", "Data 3 - Images")
    count_yolo(base_img / "RealPothole600_Converted", "RealPothole600")
    count_yolo(base_img / "Pothole.v1-raw.yolov8", "Pothole.v1")
    
    # Sensor Datasets
    count_sensor(base_sensor / "Data 1", "Data 1")
    count_sensor(base_sensor / "Data 2 - Clean", "Data 2")
    count_sensor(base_sensor / "Data 3 - Just Sensor", "Data 3")

if __name__ == "__main__":
    main()
