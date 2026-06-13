
import re
from pathlib import Path
import json

def check_alignment(hybrid_path: Path):
    print(f"Checking alignment in {hybrid_path}...")
    
    sensor_path = hybrid_path / "sensor"
    images_path = hybrid_path / "images/images"
    
    if not sensor_path.exists() or not images_path.exists():
        print("❌ Missing sensor or images directory")
        return

    # 1. Index Sensor Files
    sensor_map = {} # ID -> Path
    
    # Regex for sensor: Try to find the integer ID
    # Pattern 1: ..._(\d+)_isolated_...
    # Pattern 2: ..._(\d+)_-_Normal_Road...
    # Pattern 3: ..._(\d+)_combined...
    
    print("Indexing Sensor Files...")
    for f in sensor_path.glob("*.csv"):
        # Heuristic: look for standalone number segments
        # Split by underscore and dashes
        parts = re.split(r'[_\-]', f.name)
        
        # Find parts that are digits
        ids = [p for p in parts if p.isdigit()]
        
        # Valid IDs in this dataset seem to be 1-3 digits
        # Heuristic: The last purely numeric part before file extension or 'isolated' is likely the ID
        # Let's try to extract ID from specific patterns seen in ls
        
        assigned_id = None
        
        # Normal Road: Normal_Road_Normal_Road_37_-_Normal_Road.csv -> 37
        if "Normal_Road" in f.name:
            match = re.search(r'_(\d+)_-_Normal', f.name)
            if match: assigned_id = match.group(1)
            
        # Anomalies: Road_Anomalies_..._10_isolated... -> 10
        if not assigned_id:
            # Look for _(\d+)_isolated
            match = re.search(r'_(\d+)_isolated', f.name)
            if match: assigned_id = match.group(1)
            
        # Fallback
        if not assigned_id:
             # Look for _(\d+)_combined
            match = re.search(r'_(\d+)_combined', f.name)
            if match: assigned_id = match.group(1)
            
        if assigned_id:
            sensor_map[assigned_id] = f
            # print(f"  Mapped {f.name} -> {assigned_id}")
        else:
            print(f"  ⚠️ Could not determine ID for {f.name}")

    print(f"Found {len(sensor_map)} unique sensor IDs.")

    # 2. Check Images
    image_counts = {"matched": 0, "unmatched": 0}
    unmatched_ids = set()
    
    print("Checking Images...")
    # Iterate train and val
    for img_file in images_path.glob("**/*.jpg"):
        # Format: {ID}_{Time}.jpg e.g. 10_0p171.jpg
        name = img_file.name
        if "_" not in name:
            continue
            
        img_id = name.split("_")[0]
        
        if img_id in sensor_map:
            image_counts["matched"] += 1
        else:
            image_counts["unmatched"] += 1
            unmatched_ids.add(img_id)
            if len(unmatched_ids) < 10:
                print(f"  ❌ Unmatched Image: {name} (ID: {img_id})")

    print("\nResults:")
    print(f"  Matched Images: {image_counts['matched']}")
    print(f"  Unmatched Images: {image_counts['unmatched']}")
    
    if image_counts["unmatched"] > 0:
        print(f"  Unmatched IDs examples: {list(unmatched_ids)[:10]}")
        print("  ⚠️ Data is NOT fully aligned.")
    else:
        print("  ✅ Data is fully textually aligned ( filenames match IDs ).")
        
        # 3. Generate Index?
        # If aligned, we might want to generate a `dataset_index.json` to help loading
        index = {}
        for img_file in images_path.glob("**/*.jpg"):
             img_id = img_file.name.split("_")[0]
             # Parse timestamp: 0p171 -> 0.171
             ts_str = img_file.stem.split("_")[1].replace("p", ".")
             try:
                 ts = float(ts_str)
                 sensor_file = sensor_map[img_id]
                 index[str(img_file.relative_to(hybrid_path))] = {
                     "sensor_path": str(sensor_file.relative_to(hybrid_path)),
                     "timestamp": ts,
                     "event_id": img_id
                 }
             except:
                 pass
        
        with open(hybrid_path / "dataset_index.json", "w") as f:
            json.dump(index, f, indent=2)
        print(f"  ✅ Generated dataset_index.json with {len(index)} entries.")

if __name__ == "__main__":
    check_alignment(Path("data/hybrid/Data 1"))
