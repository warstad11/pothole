
import shutil
from pathlib import Path
import os
import random

def augment():
    source = Path("data/image/NormalRoads")
    negatives = sorted(list(source.glob("*.png")))
    total = len(negatives)
    
    print(f"Found {total} negative samples in {source}")
    
    # Stratified Split Logic
    # Filenames start with type: asphalt_..., paved_..., upaved_...
    groups = {"asphalt": [], "paved": [], "upaved": []}
    
    for img in negatives:
        if img.name.startswith("asphalt"):
            groups["asphalt"].append(img)
        elif img.name.startswith("paved"):
            groups["paved"].append(img)
        elif img.name.startswith("upaved"):
            groups["upaved"].append(img)
        else:
            print(f"Warning: Unknown type for {img.name}, skipping stratification...")
            # Fallback or add to one bucket?
            # Looking at sample script, these should be the only types.
    
    train_negs = []
    val_negs = []
    
    print("\nStratified Split Details:")
    for type_name, items in groups.items():
        count = len(items)
        split_idx = int(0.8 * count)
        t = items[:split_idx]
        v = items[split_idx:]
        train_negs.extend(t)
        val_negs.extend(v)
        print(f"  {type_name.capitalize()}: {len(t)} Train, {len(v)} Val (Total {count})")
        
    random.shuffle(train_negs)
    random.shuffle(val_negs)
    
    print(f"\nTotal Split: {len(train_negs)} Train, {len(val_negs)} Val")
    
    # Targets
    target_real = Path("data/image/raw/RealPothole600_Converted")
    target_v1 = Path("data/image/raw/Pothole.v1-raw.yolov8")
    
    # RealPothole Logic (images/train, labels/train)
    print(f"\nAugmenting RealPothole600 at {target_real}...")
    copy_files(train_negs, target_real / "images/train", target_real / "labels/train")
    copy_files(val_negs, target_real / "images/val", target_real / "labels/val")
    
    # Pothole.v1 Logic (train/images, train/labels)
    print(f"\nAugmenting Pothole.v1 at {target_v1}...")
    copy_files(train_negs, target_v1 / "train/images", target_v1 / "train/labels")
    copy_files(val_negs, target_v1 / "val/images", target_v1 / "val/labels")

def copy_files(images, img_dest, lbl_dest):
    if not img_dest.exists() or not lbl_dest.exists():
        print(f"Warning: Destination {img_dest} or {lbl_dest} does not exist. Skipping.")
        return
        
    print(f"  -> Copying {len(images)} to {img_dest}...")
    for img in images:
        # Copy Image
        shutil.copy(img, img_dest / img.name)
        
        # Create Empty Label
        # Name: image_name.txt (replace .png with .txt)
        txt_name = img.with_suffix(".txt").name
        lbl_path = lbl_dest / txt_name
        
        # Overwrite/Create empty
        with open(lbl_path, 'w') as f:
            f.write("") # Empty for negative

if __name__ == "__main__":
    augment()
