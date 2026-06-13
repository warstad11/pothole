
import os
from pathlib import Path
import glob

def count_split(dataset_path: Path, split_name: str):
    # Try standard YOLO paths
    # 1. dataset/split_name/images
    # 2. dataset/images/split_name
    # 3. dataset/split_name (flat)
    
    img_dir = None
    lbl_dir = None
    
    candidates = [
        (dataset_path / split_name / "images", dataset_path / split_name / "labels"),
        (dataset_path / "images" / split_name, dataset_path / "labels" / split_name),
        (dataset_path / split_name, dataset_path / split_name) # Flat?
    ]
    
    for cand_img, cand_lbl in candidates:
        if cand_img.exists():
            img_dir = cand_img
            lbl_dir = cand_lbl
            break
            
    if not img_dir:
        print(f"[{dataset_path.name}] {split_name}: NOT FOUND")
        return

    # Count images
    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    pos = 0
    neg = 0
    
    for img_path in images:
        # Find Label
        is_pos = False
        
        # Check standard YOLO label locations relative to image
        # 1. Same dir (flat)
        # 2. Parallel labels folder (labels/val/img.txt)
        
        possible_labels = []
        # Case A: Same dir
        possible_labels.append(img_path.with_suffix(".txt"))
        
        # Case B: Parallel ../labels/splitname/img.txt
        # If img_path is .../images/val/img.jpg, we want .../labels/val/img.txt
        if "images" in img_path.parts:
             # easy logic: replace 'images' with 'labels' in path logic
             try:
                 parts = list(img_path.parts)
                 idx = parts.index("images") 
                 parts[idx] = "labels"
                 parallel_path = Path(*parts).with_suffix(".txt")
                 possible_labels.append(parallel_path)
             except: pass
        
        # Case C: explicit lbl_dir matched above
        if lbl_dir and lbl_dir.exists():
            possible_labels.append(lbl_dir / img_path.with_suffix(".txt").name)

        label_file = None
        for p in possible_labels:
            if p.exists():
                label_file = p
                break
        
        if label_file:
            # Check content
            try:
                with open(label_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        is_pos = True
            except: pass
        else:
             # Heuristic check for "NormalRoads" or negative names if needed?
             # YOLO assumes missing txt = negative.
             pass
             
        if is_pos:
            pos += 1
        else:
            neg += 1
            
    print(f"[{dataset_path.name}] {split_name.upper()} | Total: {len(images)} | Pos: {pos} | Neg: {neg} | Balance: {pos/len(images)*100:.1f}% Pos")

def scan_all():
    base = Path("data/image/raw")
    datasets = [
        base / "Data 3 - Images",
        base / "RealPothole600_Converted",
        base / "Pothole.v1-raw.yolov8"
    ]
    
    print("-" * 80)
    print("VALIDATION SET AUDIT")
    print("-" * 80)
    
    for ds in datasets:
        # Check 'val', 'valid', 'test'
        count_split(ds, "val")
        count_split(ds, "valid")
        count_split(ds, "test")
        print("-" * 40)

if __name__ == "__main__":
    scan_all()
