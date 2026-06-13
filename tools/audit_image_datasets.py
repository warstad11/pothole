
import os
from pathlib import Path
import yaml

def audit_yolo_dataset(dataset_path: Path):
    print(f"\n--- Auditing: {dataset_path.name} ---")
    
    # Check structure
    has_yaml = (dataset_path / "data.yaml").exists()
    
    # Directories
    img_dirs = [dataset_path / "train/images", dataset_path / "valid/images", dataset_path / "images/train", dataset_path / "images/val"]
    lbl_dirs = [dataset_path / "train/labels", dataset_path / "valid/labels", dataset_path / "labels/train", dataset_path / "labels/val"]
    
    total_images = 0
    total_labels = 0
    positives = 0
    negatives = 0
    
    train_count = 0
    val_count = 0
    
    # 1. Collect all images
    all_images = set()
    for d in img_dirs:
        if d.exists():
            for f in d.glob("*"):
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    # Store relative path or just stem to match labels
                    # Be careful if train/val have same filenames (unlikely but possible)
                    # Let's verify unique check
                    all_images.add(f.stem)
                    if "train" in str(d): train_count += 1
                    else: val_count += 1
    
    total_images = len(all_images)
    
    # 2. Collect and parse labels
    labeled_stems = set()
    empty_stems = set()
    
    for d in lbl_dirs:
        if d.exists():
            for f in d.glob("*.txt"):
                if f.name == "classes.txt": continue
                
                # Check content
                try:
                    content = f.read_text().strip()
                    if content:
                        labeled_stems.add(f.stem)
                    else:
                        empty_stems.add(f.stem)
                except:
                    pass

    # 3. Correlate
    # Positives: Images that have a non-empty label file
    # Negatives: Images that have NO label file OR an empty label file
    
    real_positives = 0
    real_negatives = 0
    
    for img_stem in all_images:
        if img_stem in labeled_stems:
            real_positives += 1
        else:
            real_negatives += 1
            
    print(f"Total Images: {total_images} (Train: {train_count}, Val: {val_count})")
    print(f"Positives (Images with Labels): {real_positives}")
    print(f"Negatives (Background Images):  {real_negatives}")
    
    if total_images > 0:
        ratio = real_positives / total_images
        print(f"Positive Ratio: {ratio:.2%}")
    
    return {
        "name": dataset_path.name,
        "total": total_images,
        "pos": real_positives,
        "neg": real_negatives
    }

def main():
    base_dir = Path("data/image/raw")
    if not base_dir.exists():
        print("data/image/raw does not exist.")
        return

    results = []
    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            try:
                # Basic check if it looks like a dataset
                if (d / "data.yaml").exists() or list(d.glob("*/images")):
                    stats = audit_yolo_dataset(d)
                    results.append(stats)
            except Exception as e:
                print(f"Error auditing {d.name}: {e}")

    print("\n" + "="*60)
    print(f"{'Dataset':<30} {'Total':<10} {'Pos':<10} {'Neg':<10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<30} {r['total']:<10} {r['pos']:<10} {r['neg']:<10}")
    print("="*60)

if __name__ == "__main__":
    main()
