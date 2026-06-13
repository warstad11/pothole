import shutil
from pathlib import Path
import random
import os

def fix_sequential_split(dataset_root: Path):
    print(f"Fixing split for {dataset_root}...")
    
    # 1. Gather ALL images and labels
    all_images = []
    all_labels = [] # We need to keep pairs together
    
    # Check structure
    # Existing: dataset/train/images, dataset/val/images
    
    # Temp storage
    temp_dir = dataset_root / "temp_merge"
    if temp_dir.exists(): shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    
    # Move everything to temp to sort
    # We must scan both train and val
    for split in ["train", "val"]:
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        
        if not img_dir.exists(): continue
        
        for img in list(img_dir.glob("*")):
            shutil.move(str(img), str(temp_dir / img.name))
            
        if lbl_dir.exists():
            for lbl in list(lbl_dir.glob("*")):
                shutil.move(str(lbl), str(temp_dir / lbl.name))
                
    # Now all files in temp_merge
    files = list(temp_dir.glob("*"))
    images = [f for f in files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    # Sort
    # Try numeric sort for img-XXX
    # Try timestamp sort for Adachi
    def sort_key(f):
        name = f.stem
        # Extract number if possible
        import re
        nums = re.findall(r'\d+', name)
        if nums:
            # Join all nums to handle timestamps or sequences
            return int("".join(nums))
        return name
        
    images.sort(key=sort_key)
    
    print(f"Found {len(images)} total images. First 5: {[f.name for f in images[:5]]}")
    
    # Split 80/20
    split_idx = int(0.8 * len(images))
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    print(f"Split: {len(train_imgs)} Train, {len(val_imgs)} Val")
    
    # Re-distribute
    # Ensure dirs exist (cleared or empty)
    for split in ["train", "val"]:
        (dataset_root / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_root / split / "labels").mkdir(parents=True, exist_ok=True)
        
    def move_files(file_list, target_split):
        for img in file_list:
            # Move Image
            dest_img = dataset_root / target_split / "images" / img.name
            shutil.move(str(img), str(dest_img))
            
            # Move Label (if exists)
            lbl_name = img.with_suffix(".txt").name
            lbl_src = temp_dir / lbl_name
            if lbl_src.exists():
                dest_lbl = dataset_root / target_split / "labels" / lbl_name
                shutil.move(str(lbl_src), str(dest_lbl))
                
    move_files(train_imgs, "train")
    move_files(val_imgs, "val")
    
    shutil.rmtree(temp_dir)
    print("Fix Complete.")

if __name__ == "__main__":
    base = Path("/Users/tstading/gemini/antigravity/scratch/pothole1223/data/image/raw")
    
    # Fix Data 3
    fix_sequential_split(base / "Data 3 - Images")
    
    # Fix Pothole.v1
    fix_sequential_split(base / "Pothole.v1-raw.yolov8")
