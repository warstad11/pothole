
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.labels import infer_label_from_path  # noqa: E402

def generate_index():
    base_dir = Path("data/hybrid/Data 1 - Both")
    img_dir = base_dir / "images"
    sensor_dir = base_dir / "sensor"
    output_path = base_dir / "dataset_index.json"
    
    if not base_dir.exists():
        print(f"Error: {base_dir} does not exist.")
        return

    index = {}
    
    # Scan images
    images = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    print(f"Found {len(images)} images.")
    
    matched_count = 0
    
    for img_path in images:
        # Expected sensor filename: same name but .csv
        sensor_name = img_path.with_suffix(".csv").name
        sensor_path = sensor_dir / sensor_name
        
        if sensor_path.exists():
            # Store relative paths
            rel_img = f"images/{img_path.name}"
            rel_sensor = f"sensor/{sensor_name}"
            
            # Pothole-strict labels via the central module: Bump is NOT a
            # pothole (the old rule counted it positive, poisoning 231 of
            # the 402 positive samples in this index).
            label = infer_label_from_path(img_path)
            if label is None:
                print(f"Warning: cannot infer label for {img_path.name}; skipping")
                continue

            index[rel_img] = {
                "sensor_path": rel_sensor,
                "label": int(label)
            }
            matched_count += 1
        else:
            print(f"Warning: No sensor file for {img_path.name}")
            
    print(f"Matched {matched_count} pairs.")
    
    with open(output_path, "w") as f:
        json.dump(index, f, indent=2)
        
    print(f"Index written to {output_path}")

if __name__ == "__main__":
    generate_index()
