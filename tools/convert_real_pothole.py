
import cv2
import numpy as np
import shutil
from pathlib import Path
import random
import yaml

def convert_real_pothole_dataset():
    """
    Converts RealPothole600 (segmentation masks) to YOLO format (bboxes).
    Splits into train/val (80/20).
    """
    base_dir = Path("/Users/tstading/gemini/antigravity/scratch/pothole1223/data/image/raw/OLD/RealPothole600")
    source_train_img = base_dir / "training" / "rgb"
    source_train_lbl = base_dir / "training" / "label"
    
    # We might also have 'testing' or 'validation' folders in the raw download?
    # Checked earlier: testing, training, validation folders exist.
    # Let's aggregate ALL of them and then do a clean split, or honor existing structure?
    # Usually "RealPothole600" raw has split, but let's confirm sizes.
    # Earlier listing showed training/label has files.
    
    output_dir = Path("/Users/tstading/gemini/antigravity/scratch/pothole1223/data/image/RealPothole600_Converted")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        
    (output_dir / "images" / "train").mkdir(parents=True)
    (output_dir / "images" / "val").mkdir(parents=True)
    (output_dir / "labels" / "train").mkdir(parents=True)
    (output_dir / "labels" / "val").mkdir(parents=True)
    
    # Gather all pairs
    all_files = []
    
    # scan all subdirs
    for split in ["training", "validation", "testing"]:
        img_dir = base_dir / split / "rgb"
        lbl_dir = base_dir / split / "label"
        
        if not img_dir.exists(): continue
        
        images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
        print(f"Found {len(images)} images in {split}")
        
        for p in images:
            # Match label
            # Assuming same name
            l = lbl_dir / p.name # usually png
            if not l.exists():
                # try replacing extension?
                l = lbl_dir / p.with_suffix('.png').name
            
            if l.exists():
                all_files.append((p, l))
            else:
                print(f"Warning: No label for {p}")
                
    print(f"Total valid pairs: {len(all_files)}")
    
    # Sort files to ensure sequential order (vital for video frames)
    # Removing random.shuffle to prevent frame leakage
    all_files.sort(key=lambda x: x[0].name) # Ensure testing_0000, 0001 order
    
    # Sequential Split (80/20)
    split_idx = int(0.8 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    
    print(f"Sequential Split: {len(train_files)} Train (Frames 0-{split_idx-1}), {len(val_files)} Val (Frames {split_idx}-End)")
    
    def process(pair_list, split_name):
        print(f"Processing {split_name} ({len(pair_list)} files)...")
        for img_path, mask_path in pair_list:
            # Fix Overwrite Issue: Prepend source split to filename
            # img_path parent is 'rgb', parent.parent is 'training'/'testing' etc
            source_split = img_path.parent.parent.name
            new_name = f"{source_split}_{img_path.name}"
            
            # Copy Image
            shutil.copy(img_path, output_dir / "images" / split_name / new_name)
            
            # Process Mask to Box
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            h, w = mask.shape
            
            # Find contours
            # Pothole is likely 255 (white) or non-zero
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            yolo_lines = []
            for cnt in contours:
                x, y, bw, bh = cv2.boundingRect(cnt)
                
                # Filter small noise
                if bw < 5 or bh < 5: continue
                
                # Normalize
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                
                # Class 0 = pothole
                yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                
            # Write label file (matching new name)
            txt_name = Path(new_name).with_suffix(".txt").name
            with open(output_dir / "labels" / split_name / txt_name, "w") as f:
                f.write("\n".join(yolo_lines))
                
    process(train_files, "train")
    process(val_files, "val")
    
    # Write data.yaml
    yaml_content = {
        "path": str(output_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["pothole"]
    }
    
    with open(output_dir / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f)
        
    print(f"Conversion complete. Data saved to {output_dir}")

if __name__ == "__main__":
    convert_real_pothole_dataset()
