
import shutil
from pathlib import Path
import os
import random

def convert_cls_to_yolo(input_dir: Path, output_dir: Path):
    """
    Converts a Classification dataset (folders as classes) to YOLOv8 Detection format.
    Assumes:
    - 'normal' / 'negative' -> No pothole (empty label file)
    - 'pothole' / 'potholes' / 'positive' -> Pothole (full image bounding box)
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Define Splits
    for split in ["train", "valid"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Iterate categories
    cats = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found categories: {[c.name for c in cats]}")
    
    # We only have 1 class for detection: "pothole" (id 0)
    # If category is "normal", we produce no labels.
    # If category is "pothole", we produce label "0 0.5 0.5 1.0 1.0"
    
    files_processed = 0
    
    for cat in cats:
        cat_name = cat.name.lower()
        is_pothole = cat_name in ["pothole", "potholes", "positive"]
        
        images = list(cat.glob("*.jpg")) + list(cat.glob("*.png")) + list(cat.glob("*.jpeg"))
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        for pool, split_name in [(train_imgs, "train"), (val_imgs, "valid")]:
            for img_path in pool:
                # Copy Image
                dest_img = output_dir / split_name / "images" / img_path.name
                shutil.copy(img_path, dest_img)
                
                # Create Label
                label_path = output_dir / split_name / "labels" / (img_path.stem + ".txt")
                if is_pothole:
                    # ID x_center y_center width height (normalized)
                    # We assume full image is the object for classification datasets
                    with open(label_path, "w") as f:
                        f.write("0 0.5 0.5 1.0 1.0")
                else:
                    # Empty file = background / no object
                    label_path.touch()
                    
                files_processed += 1
                
    # Create data.yaml
    yaml_content = f"""
names:
- pothole
nc: 1
train: {str((output_dir / 'train' / 'images').absolute())}
val: {str((output_dir / 'valid' / 'images').absolute())}
    """
    
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"Merged {files_processed} images into YOLO format at: {output_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        input_root = Path(sys.argv[1])
        output_root = Path(sys.argv[2])
    else:
        # Default
        input_root = Path("data/image/raw/Pothole-600")
        output_root = Path("data/image/raw/Pothole-600_YOLO")
        
    convert_cls_to_yolo(input_root, output_root)
