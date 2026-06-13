
import json
import shutil
from pathlib import Path
import os
import tqdm

def convert_coco_to_yolo(input_dir: Path, output_dir: Path):
    """
    Converts a dataset from COCO folder structure (train/valid/test with _annotations.coco.json)
    to YOLOv8 structure (train/valid/test with images/labels folders).
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    splits = ["train", "valid", "test"]
    
    # We need to map category IDs to 0-indexed YOLO class IDs
    # We'll learn this from the first split we find
    cat_id_map = {} 
    
    for split in splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            continue
            
        json_file = split_dir / "_annotations.coco.json"
        if not json_file.exists():
            print(f"Skipping {split}, no JSON found.")
            continue
            
        print(f"Processing {split}...")
        
        # Create output dirs
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        
        with open(json_file, "r") as f:
            data = json.load(f)
            
        # Build category map if empty
        if not cat_id_map:
            # We assume single class "pothole" usually, but let's be generic
            # Sort categories by ID to ensure consistent mapping
            cats = sorted(data["categories"], key=lambda x: x["id"])
            for i, cat in enumerate(cats):
                cat_id_map[cat["id"]] = i
            print(f"Category Mapping: {cat_id_map}")
            
        # Index annotations by image_id
        img_annots = {}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_annots:
                img_annots[img_id] = []
            img_annots[img_id].append(ann)
            
        # Process Images
        for img in data["images"]:
            img_id = img["id"]
            file_name = img["file_name"]
            
            # Source file
            src_path = split_dir / file_name
            if not src_path.exists():
                print(f"Warning: Image {file_name} not found.")
                continue
                
            # Copy Image
            dst_img_path = output_dir / split / "images" / file_name
            shutil.copy(src_path, dst_img_path)
            
            # Create Label File
            img_w = img["width"]
            img_h = img["height"]
            
            label_lines = []
            if img_id in img_annots:
                for ann in img_annots[img_id]:
                    cat_id = ann["category_id"]
                    if cat_id not in cat_id_map:
                        continue
                        
                    cls_idx = cat_id_map[cat_id]
                    bbox = ann["bbox"] # [x, y, w, h] (top-left)
                    
                    # Convert to center, normalized
                    x, y, w, h = bbox
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    
                    # Clamp
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    
                    label_lines.append(f"{cls_idx} {cx} {cy} {nw} {nh}")
            
            # Write label file (same basename as image, .txt extension)
            label_name = Path(file_name).stem + ".txt"
            with open(output_dir / split / "labels" / label_name, "w") as f:
                f.write("\n".join(label_lines))
                
    # Create data.yaml
    yaml_content = f"""
names:
{os.linesep.join([f"- {c['name']}" for c in cats])}
nc: {len(cats)}
train: {str((output_dir / 'train' / 'images').absolute())}
val: {str((output_dir / 'valid' / 'images').absolute())}
test: {str((output_dir / 'test' / 'images').absolute())}
    """
    
    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    print(f"Conversion complete! YOLO dataset ready at: {output_dir}")

if __name__ == "__main__":
    input_root = Path("data/image/raw/PotholeRobo.coco")
    output_root = Path("data/image/raw/Pothole_YOLO")
    convert_coco_to_yolo(input_root, output_root)
