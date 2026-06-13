
import os
from pathlib import Path

def undo_augmentation():
    # Source of truth for what was added
    source = Path("data/image/NormalRoads")
    if not source.exists():
        print(f"Error: {source} not found. Cannot determine which files to remove.")
        return
        
    negatives = set([f.name for f in source.glob("*.png")])
    if not negatives:
        print("Warning: No files found in source. Nothing to remove?")
        # Maybe check if I should list *.png just in case glob fails?
        # But source should be populated from previous step.
        pass
        
    print(f"Identified {len(negatives)} filenames to remove.")
    
    # Targets
    target_real = Path("data/image/raw/RealPothole600_Converted")
    target_v1 = Path("data/image/raw/Pothole.v1-raw.yolov8")
    
    # Paths to scan (Image dirs)
    # RealPothole: images/train, images/val
    # Pothole.v1: train/images, val/images
    
    # For each target directory, we also need to find corresponding labels to delete
    # RealPothole labels: labels/train, labels/val (parallel to images)
    # Pothole.v1 labels: train/labels, val/labels (parallel to images)
    
    targets = [
        # (Image Dir, Label Dir)
        (target_real / "images/train", target_real / "labels/train"),
        (target_real / "images/val", target_real / "labels/val"),
        (target_v1 / "train/images", target_v1 / "train/labels"),
        (target_v1 / "val/images", target_v1 / "val/labels")
    ]
    
    removed_count = 0
    
    for img_dir, lbl_dir in targets:
        if not img_dir.exists():
            continue
            
        print(f"Scanning {img_dir}...")
        
        # Iterate files in img_dir
        # If filename in negatives set, delete it and its label
        
        # Safe iteration: list first
        current_files = list(img_dir.glob("*"))
        
        for f in current_files:
            if f.name in negatives:
                # Found a match
                # Delete Image
                f.unlink()
                removed_count += 1
                
                # Delete Label
                # Label name = image name with .txt suffix
                # My augment script created .txt labels corresponding to .png images
                txt_name = f.with_suffix(".txt").name
                lbl_file = lbl_dir / txt_name
                
                if lbl_file.exists():
                    lbl_file.unlink()
                    
    print(f"Undo Complete. Removed {removed_count} images (and their labels) across all datasets.")

if __name__ == "__main__":
    undo_augmentation()
