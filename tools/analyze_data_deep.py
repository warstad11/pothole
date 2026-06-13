
import os
import csv

DATA_DIR = "/Users/tstading/gemini/antigravity/scratch/pothole1223/data"

def analyze_sensor_file(filepath):
    """
    Returns (row_count, event_count)
    event_count: 1 if file exists (basic), but we can try to find label changes.
    """
    row_count = 0
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if not lines:
                return 0, 0
            # Assume 1 header
            row_count = len(lines) - 1
            if row_count < 0: row_count = 0
            
            # Simple heuristic: 1 file = 1 event for this dataset context
            # unless we want to parse labels. Let's return 1 file = 1 event.
            return row_count, 1
    except:
        return 0, 0

def analyze_hybrid(path):
    csv_count = 0
    img_count = 0
    total_rows = 0
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith('.csv'):
                csv_count += 1
                r, _ = analyze_sensor_file(os.path.join(root, file))
                total_rows += r
            elif file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_count += 1
                
    return csv_count, img_count, total_rows

print("--- Deep Data Analysis ---")

# 1. Image
print("\n[Images - By Type]")
image_counts = {'jpg': 0, 'png': 0, 'other': 0}
img_dir = os.path.join(DATA_DIR, 'image')
for root, dirs, files in os.walk(img_dir):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            image_counts['jpg'] += 1
        elif ext == '.png':
            image_counts['png'] += 1
        elif ext in ['.bmp', '.tiff', '.gif']: # just in case
            image_counts['other'] += 1
            
print(f"  JPG/JPEG: {image_counts['jpg']}")
print(f"  PNG: {image_counts['png']}")
print(f"  Other: {image_counts['other']}")
print(f"  Total Image Files: {sum(image_counts.values())}")


# 2. Sensor
print("\n[Sensor - Deep Count]")
sensor_root = os.path.join(DATA_DIR, 'sensor')
# Iterate over immediate subdirs to group by "Data X"
if os.path.exists(sensor_root):
    for item in sorted(os.listdir(sensor_root)):
        path = os.path.join(sensor_root, item)
        if os.path.isdir(path):
            file_count = 0
            total_rows = 0
            # Walk strictly this folder
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.lower().endswith('.csv') and 'manifest' not in file:
                        r, e = analyze_sensor_file(os.path.join(root, file))
                        file_count += 1
                        total_rows += r
            
            if file_count > 0:
                print(f"  {item}:")
                print(f"    Files (Events): {file_count}")
                print(f"    Total Data Points (Rows): {total_rows}")


# 3. Hybrid
print("\n[Hybrid - Composition]")
hybrid_root = os.path.join(DATA_DIR, 'hybrid')
if os.path.exists(hybrid_root):
    for item in sorted(os.listdir(hybrid_root)):
        path = os.path.join(hybrid_root, item)
        if os.path.isdir(path):
            c, i, r = analyze_hybrid(path)
            print(f"  {item}:")
            print(f"    Sensor Files (.csv): {c}")
            print(f"    Image Files: {i}")
            print(f"    Total Sensor Rows: {r}")

# 4. NuScenes (re-verify samples)
print("\n[NuScenes - Detail]")
nuscenes_path = os.path.join(DATA_DIR, 'nuscenes')
if os.path.exists(nuscenes_path):
    total_files = 0
    scenes = [d for d in os.listdir(nuscenes_path) if d.startswith('scene-')]
    print(f"  Total files across {len(scenes)} scenes:")
    for scene in sorted(scenes):
        scene_dir = os.path.join(nuscenes_path, scene)
        files = [f for f in os.listdir(scene_dir) if os.path.isfile(os.path.join(scene_dir, f))]
        # Should distinguish between JSON labels or binary data if needed
        # NuScenes usually has json tokens or similar.
        print(f"    {scene}: {len(files)} files")

