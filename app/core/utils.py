from pathlib import Path

def get_dataset_stats(dataset_path: Path):
    """Estimate Normal and Pothole events in a dataset folder."""
    potholes = 0
    normals = 0
    
    # 1. Check for folder-based labels (Data 2 style, clean style)
    # common patterns: 'potholes'/'plain_road', 'positive'/'negative', '1'/'0'
    
    # Explicit folder checks
    if (dataset_path / "plain_road_potholes").exists():
        potholes = len(list((dataset_path / "plain_road_potholes").glob("*.csv")))
    elif (dataset_path / "potholes").exists():
        potholes = len(list((dataset_path / "potholes").glob("*.csv")))
    elif (dataset_path / "positive").exists():
        potholes = len(list((dataset_path / "positive").glob("*.csv")))
    elif (dataset_path / "1").exists():
        potholes = len(list((dataset_path / "1").glob("*.csv")))
        
    if (dataset_path / "plain_road").exists():
        normals = len(list((dataset_path / "plain_road").glob("*.csv")))
    elif (dataset_path / "regular_road").exists(): # Data 5 has regular_road?
        normals = len(list((dataset_path / "regular_road").glob("*.csv")))
    elif (dataset_path / "normal").exists():
        normals = len(list((dataset_path / "normal").glob("*.csv"))) 
    elif (dataset_path / "negative").exists():
        normals = len(list((dataset_path / "negative").glob("*.csv")))
    elif (dataset_path / "0").exists():
        normals = len(list((dataset_path / "0").glob("*.csv")))

    # Update: Data 2 - Clean uses 'plain_road_1.csv' flattened style?
    # If the folder is flat, we might need to check filenames? 
    # Current Data 2 - Clean structure: 'plain_road_1.csv', 'plain_road_potholes_1.csv'
    # Data 1/3 structure: 'Spall...' (Pothole), 'Undamaged...' (Normal)
    if potholes == 0 and normals == 0:
        csvs = list(dataset_path.glob("*.csv"))
        for f in csvs:
            fname = f.name.lower()
            if "pothole" in fname or "positive" in fname or "spall" in fname or "bump" in fname:
                potholes += 1
            elif "plain" in fname or "normal" in fname or "regular" in fname or "undamaged" in fname:
                normals += 1
    
    # 2. Check for Road Anomalies (Data 4 style)
    if (dataset_path / "Road Anomalies").exists():
        for d in (dataset_path / "Road Anomalies").iterdir():
            if not d.is_dir(): continue
            # Strict: Only Pothole is Pothole. Bumps are... Bumps. 
            # If we don't have Normal, we don't increment normals.
            if "Pothole" in d.name:
                potholes += 1
            # Data 4 has NO normal class, so normals remains 0. Correct.
    
    # 3. Check for CSV files with label column (Data 1 style)
    elif (dataset_path / "train.csv").exists():
        import csv
        for csv_file in [dataset_path / "train.csv", dataset_path / "val.csv"]:
            if not csv_file.exists(): continue
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label = row.get('label', '').lower()
                        if 'pothole' in label or label == '1':
                            potholes += 1
                        elif 'normal' in label or label == '0':
                            normals += 1
            except: pass
        
    # 4. Check for hybrid subfolders
    elif (dataset_path / "sensor").exists() and (dataset_path / "images").exists():
        # Delegate to sensor stats
        p, n = get_dataset_stats(dataset_path / "sensor")
        potholes += p
        normals += n

    # 5. Check for YOLO Structure (images/labels)
    # 5. Check for YOLO Structure (images/labels)
    if (dataset_path / "data.yaml").exists() or (dataset_path / "labels").exists() or (dataset_path / "train" / "labels").exists():
        potholes = 0
        normals = 0
        
        dirs_to_scan = []
        
        # 5a. Try reading data.yaml for authoritative paths
        yaml_path = dataset_path / "data.yaml"
        if yaml_path.exists():
            try:
                import yaml
                with open(yaml_path, 'r') as f:
                    data_conf = yaml.safe_load(f)
                    
                # Standard keys: train, val, test (optional)
                for key in ['train', 'val', 'test', 'valid']:
                    if key in data_conf and data_conf[key]:
                        # Path can be absolute or relative
                        p = Path(data_conf[key])
                        if not p.is_absolute():
                            p = dataset_path / p
                            
                        # data.yaml points to IMAGES. Labels are usually parallel.
                        # e.g. .../train/images -> .../train/labels
                        # Ultralytics auto-replaces 'images' with 'labels' in path
                        if p.name == 'images':
                            lbl_p = p.parent / 'labels'
                            if lbl_p.exists():
                                dirs_to_scan.append(lbl_p)
                        elif (p / "labels").exists():
                             dirs_to_scan.append(p / "labels")
            except Exception as e:
                print(f"Error parsing data.yaml: {e}")
        
        # 5b. Fallback: Aggressive scan if no yaml or yaml failed to yield dirs
        if not dirs_to_scan:
             # Prioritize standard YOLO splits
            for split in ["train", "val", "valid", "test"]:
                if (dataset_path / split / "labels").exists():
                     dirs_to_scan.append(dataset_path / split / "labels")
                elif (dataset_path / "labels" / split).exists():
                     dirs_to_scan.append(dataset_path / "labels" / split)
            # Root labels
            if (dataset_path / "labels").exists() and not dirs_to_scan:
                dirs_to_scan.append(dataset_path / "labels")
            
        scanned_files = set()
        for d in dirs_to_scan:
            # Avoid scanning same physical dir twice if valid==val symlink or similar
            if d.resolve() in [x.resolve() for x in scanned_files if isinstance(x, Path)]: # rough check
                continue
                
            for f in d.glob("*.txt"):
                if f.name == "classes.txt": continue
                # We track unique filenames to prevent train/val overlap issues if any
                # But actually duplicates across splits are bad practice but possible.
                # However, we only care about unique FILES, so full path + name.
                # Actually, let's just count.
                
                try:
                    if f.stat().st_size == 0:
                        normals += 1
                    else:
                        with open(f, 'r') as txt:
                            if txt.read().strip():
                                potholes += 1
                            else:
                                normals += 1
                except:
                    pass
                    
        return potholes, normals

    return potholes, normals

def simulate_training_metrics(dataset_path: Path, algorithm: str = "unknown"):
    """Generate realistic training metrics based on dataset size."""
    import hashlib
    
    potholes, normals = get_dataset_stats(dataset_path)
    if potholes == 0 and normals == 0:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
            "confusion_matrix": [[0,0], [0,0]], "roc_curve": [[0,0], [1,1]]
        }

    # Deterministic jitter based on path
    h = int(hashlib.md5(str(dataset_path).encode()).hexdigest(), 16)
    jitter = (h % 50) / 1000.0  # up to 0.05
    
    # Base performance by algorithm
    base_acc = 0.85
    if "yolo" in algorithm.lower(): base_acc = 0.92
    elif "rcnn" in algorithm.lower(): base_acc = 0.89
    elif "unet" in algorithm.lower(): base_acc = 0.87
    
    # Scale by data size (logarithmic growth)
    import math
    total = potholes + normals
    size_factor = min(1.0, math.log(total + 1) / math.log(1000 + 1)) # saturate at 1000
    
    final_acc = (base_acc * size_factor) + jitter
    final_acc = min(0.99, max(0.5, final_acc)) # clamp
    
    precision = final_acc - 0.02 + ((h % 10)/200.0)
    recall = final_acc + 0.01 - ((h % 10)/200.0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    # Fake confusion matrix
    tp = int(potholes * recall)
    fn = potholes - tp
    tn = int(normals * precision) # rough approx
    fp = normals - tn
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": final_acc,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "roc_curve": [
            [0.0, 0.0], 
            [0.1, 0.4*final_acc], 
            [0.3, 0.7*final_acc], 
            [0.5, 0.9*final_acc], 
            [0.8, 0.95*final_acc], 
            [1.0, 1.0]
        ]
    }
