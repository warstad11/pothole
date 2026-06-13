
import os
import shutil
import random
from pathlib import Path

def augment_data3():
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
            
    # Target counts
    TARGET_TRAIN = 520
    TARGET_VAL = 80
    TOTAL_TARGET = TARGET_TRAIN + TARGET_VAL
    
    if total < TOTAL_TARGET:
        print(f"Warning: Available negatives ({total}) < Requested ({TOTAL_TARGET}). Using all.")
        
    # Calculate ratios based on counts
    train_negs = []
    val_negs = []
    
    print("\nStratified Split Details:")
    for type_name, items in groups.items():
        count = len(items)
        if count == 0: continue
        
        # Proportional split
        # global ratio = 520 / 600 = 0.866
        split_ratio = TARGET_TRAIN / TOTAL_TARGET
        train_count = int(count * split_ratio)
        
        # Adjust for rounding errors if needed?
        # Simply ensure we don't exceed what we have.
        val_count = count - train_count
        
        t = items[:train_count]
        v = items[train_count:]
        train_negs.extend(t)
        val_negs.extend(v)
        print(f"  {type_name.capitalize()}: {len(t)} Train, {len(v)} Val (Total {count})")
        
    # Shuffle
    random.seed(42)
    random.shuffle(train_negs)
    random.shuffle(val_negs)
    
    print(f"\nTotal Split: {len(train_negs)} Train, {len(val_negs)} Val")
    
    # Target 1: Data 3 - Images (Split Structure)
    t1_root = Path("data/image/raw/Data 3 - Images")
    print(f"\nAugmenting {t1_root}...")
    copy_files(train_negs, t1_root / "train/images", t1_root / "train/labels")
    copy_files(val_negs, t1_root / "val/images", t1_root / "val/labels")
    
    # Target 2: Data 3 - not sync (Flat Structure)
    t2_root = Path("data/hybrid/Data 3 - not sync")
    print(f"\nAugmenting {t2_root} (Flat Structure - Adding All)...")
    # Add both train and val sets to the shared folder
    all_negs = train_negs + val_negs
    copy_files(all_negs, t2_root / "images", t2_root / "labels")

def copy_files(images, img_dest, lbl_dest):
    if not img_dest.exists() or not lbl_dest.exists():
        print(f"Warning: Destination {img_dest} or {lbl_dest} does not exist. Skipping.")
        return
        
    print(f"  -> Copying {len(images)} to {img_dest}...")
    for img in images:
        # Copy Image
        shutil.copy(img, img_dest / img.name)
        
        # Create Empty Label
        txt_name = img.with_suffix(".txt").name
        lbl_path = lbl_dest / txt_name
        
        with open(lbl_path, 'w') as f:
            f.write("") # Empty for negative

if __name__ == "__main__":
    augment_data3()
