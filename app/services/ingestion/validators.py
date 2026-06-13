import os
from pathlib import Path
from typing import List, Dict, Tuple

class DatasetValidator:
    @staticmethod
    @staticmethod
    def validate_image_dataset(path: str, format: str) -> Tuple[bool, str]:
        path = Path(path)
        if not path.exists():
            return False, f"❌ Root path missing: {path}"
        
        if format.lower() == "yolo":
            # Strict YOLOv8 Validation
            # 1. Check for data.yaml OR strict train/val structure
            errors = []
            has_yaml = (path / "data.yaml").exists()
            
            # Check for splits
            found_splits = []
            for split in ["train", "valid", "test"]:
                split_path = path / split
                if split_path.exists() and split_path.is_dir():
                    # Check for images and labels subdirs
                    if not (split_path / "images").exists():
                        errors.append(f"❌ Missing 'images' folder in {split}")
                    else:
                        img_count = len(list((split_path / "images").glob("*")))
                        if img_count == 0:
                            errors.append(f"⚠️ '{split}/images' is empty")
                        else:
                            found_splits.append(split)
                            
                    if not (split_path / "labels").exists():
                        errors.append(f"❌ Missing 'labels' folder in {split}")
                elif split == "train" or split == "valid":
                     # Train/Val are usually required
                     if not has_yaml: # strict structure mode
                         errors.append(f"❌ Missing required split folder: '{split}'")

            if not has_yaml and not found_splits:
                 # Check for Classification Format (heuristic)
                 subdirs = [x.name for x in path.iterdir() if x.is_dir()]
                 common_classes = {"normal", "pothole", "potholes", "positive", "negative"}
                 if any(c in subdirs for c in common_classes):
                     return False, f"⚠️ Classification Format detected (Found: {subdirs}). Needs conversion to YOLO."
                 
                 return False, "❌ Invalid YOLO format. Need 'data.yaml' OR 'train/valid' folders with 'images/labels'."
            
            if errors:
                return False, "\n".join(errors)
                
            return True, f"✅ Valid YOLOv8 (Found items in: {', '.join(found_splits)})"
            
        return False, f"Unknown format: {format}"

    @staticmethod
    def validate_sensor_dataset(path: str, format: str) -> Tuple[bool, str]:
        path = Path(path)
        if not path.exists():
            return False, f"❌ Root path missing: {path}"

        if format.lower() == "sensorlogger":
            # Strict SensorLogger Validation
            required_files = {
                "Accelerometer.csv": ["time", "seconds_elapsed", "z", "y", "x"],
                "Gyroscope.csv": ["time", "seconds_elapsed", "z", "y", "x"]
            }
            
            import pandas as pd
            errors = []
            
            for fname, required_cols in required_files.items():
                fpath = path / fname
                if not fpath.exists():
                    errors.append(f"❌ Missing critical file: {fname}")
                    continue
                
                try:
                    # Read just header
                    df = pd.read_csv(fpath, nrows=0)
                    missing_cols = [c for c in required_cols if c not in df.columns]
                    if missing_cols:
                         errors.append(f"❌ {fname} missing columns: {missing_cols}")
                except Exception as e:
                    errors.append(f"❌ Could not read {fname}: {e}")

            if errors:
                return False, "\n".join(errors)
                
            return True, "✅ Valid SensorLogger Dataset"
            
        return False, f"Unknown format: {format}"

    @staticmethod
    def validate_just_sensor(path: Path) -> Tuple[bool, str]:
        """Validates Data 2 style CSV format: Time,Gx,Gy,Gz,Ax,Ay,Az or subsets"""
        if not path.is_dir():
            return False, "Not a directory"
            
        csvs = list(path.glob("*.csv"))
        if not csvs:
            return False, "No CSV files found"
            
        import pandas as pd
        # Flexible check: at least one accelerometer axis
        acc_cols = {"Ax", "Ay", "Az", "acc_x", "acc_y", "acc_z", "accelerometer_x", "accelerometer_y", "accelerometer_z"}
        
        try:
            df = pd.read_csv(csvs[0], nrows=0)
            if any(cols in df.columns for cols in acc_cols):
                return True, f"✅ Sensor CSV Format ({len(csvs)} files)"
            return False, f"Missing accelerometer columns. Found: {list(df.columns)}"
        except Exception as e:
            return False, f"Read error: {e}"

    @staticmethod
    def validate_hybrid_dataset(path: str, format: str = "standard") -> Tuple[bool, str]:
        path = Path(path)
        if not path.exists():
            return False, f"❌ Root path missing"
            
        errors = []
        is_raw_discovery = False
        
        # 1. Check Alignment (Optional for raw discovery)
        align_file = path / "alignment.json"
        if not align_file.exists():
             is_raw_discovery = True # If missing alignment, we assume it's a raw folder being discovered
        else:
             import json
             try:
                 with open(align_file) as f:
                     data = json.load(f)
                     if "segments" not in data and "sync_points" not in data:
                         errors.append("❌ 'alignment.json' invalid schema")
             except Exception as e:
                 errors.append(f"❌ 'alignment.json' corrupted: {e}")
        
        # 2. Check Visual Data
        # Search recursively for video files
        video_files = list(path.rglob("*.mp4")) + list(path.rglob("*.mov")) + list(path.rglob("*.m4v"))
        # Recognition of split structures or standard folders
        has_images = (
            (path / "images").exists() or 
            (path / "frames").exists() or 
            (path / "Image_data").exists() or
            (path / "camera").exists() or
            (path / "Camera").exists()
        )
        has_video = len(video_files) > 0 or has_images
        
        # 3. Check Sensor Data
        # Check for SensorLogger files, unified sensor files, or split structures
        # For .txt files, we exclude 'labels' directories and small files to avoid YOLO label false positives
        sensor_txt_files = [
            f for f in path.rglob("*.txt") 
            if "label" not in str(f).lower() and f.stat().st_size > 500
        ]
        has_sensor = (
            (path / "Accelerometer.csv").exists() or 
            (path / "sensor").exists() or 
            (path / "1.csv").exists() or
            (path / "Motion_data").exists() or
            len(list(path.glob("*Accelerometer*.csv"))) > 0 or
            len(sensor_txt_files) > 10 
        )
        
        if not has_video:
            errors.append("❌ Missing visual data")
        if not has_sensor:
            errors.append("❌ Missing sensor data")
            
        if errors:
            return False, "\n".join(errors)
            
        msg = "✅ Valid Hybrid Dataset"
        if is_raw_discovery:
            msg += " (Raw/Unprocessed)"
            
        return True, msg
