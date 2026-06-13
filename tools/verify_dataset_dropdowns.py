import sys
import os
from pathlib import Path

# Add root to python path to emulate main.py context
sys.path.append(os.getcwd())

# Mock strict config to avoid DB connection issues if import main fails
class MockSettings:
    DATA_DIR = Path("data")
    PROJECT_NAME = "Test"

# We cannot easily import main because it creates FastAPI app and connects DB on import
# So we will copy the logic directly to verify it works as intended.
# This ensures we are testing the LOGIC, not the APP STATE.

from app.core.utils import get_dataset_stats

def list_sensor_datasets():
    print("Testing Sensor Datasets...")
    datasets = []
    sensor_path = Path("data/sensor")
    if sensor_path.exists():
        for d in sensor_path.iterdir():
            if d.is_dir() and d.name not in ["raw", "processed", "valid", "OLD"]:
                try:
                    p, n = get_dataset_stats(d)
                    status = "VALID"
                    if not (p > 0 and n > 0 and (p + n) >= 5):
                        status = f"INVALID (p={p}, n={n})"
                    
                    print(f"  Checking {d.name}: {status}")
                    
                    if p > 0 and n > 0 and (p + n) >= 5:
                        datasets.append(f"sensor/{d.name}")
                except Exception as e:
                    print(f"  Error {d.name}: {e}")
                    pass
    return sorted(datasets)

def list_hybrid_datasets():
    print("\nTesting Hybrid Datasets...")
    datasets = []
    hybrid_path = Path("data/hybrid")
    if hybrid_path.exists():
        for d in hybrid_path.iterdir():
            if d.is_dir() and d.name not in ["processed", "OLD"]:
                # Explicit Blacklist
                if "dataset_2020a" in d.name.lower() or "not sync" in d.name.lower():
                    print(f"  Checking {d.name}: BLACKLISTED")
                    continue
                    
                try:
                    p, n = get_dataset_stats(d)
                    status = "VALID"
                    if not (p > 0 and n > 0 and (p + n) >= 10):
                        status = f"INVALID (p={p}, n={n})"
                    
                    print(f"  Checking {d.name}: {status}")

                    if p > 0 and n > 0 and (p + n) >= 10:
                        datasets.append(f"hybrid/{d.name}")
                except Exception as e:
                    print(f"  Error {d.name}: {e}")
                    pass
    return sorted(datasets)

def list_image_datasets():
    print("\nTesting Image Datasets...")
    datasets = []
    raw_path = Path("data/image/raw")
    if raw_path.exists():
        for d in raw_path.iterdir():
            if d.is_dir() and (d / "data.yaml").exists():
                count = 0
                # Check different possible structures
                for img_dir in [
                    d / "train" / "images",
                    d / "valid" / "images",
                    d / "images" / "train",
                    d / "images" / "val"
                ]:
                    if img_dir.exists():
                        count += len(list(img_dir.glob("*")))
                
                if count == 0:
                     count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                
                state = "VALID"
                if count < 30: state = f"TOO SMALL ({count})"
                print(f"  Checking {d.name}: {state}")
                
                if count >= 30:
                    datasets.append(d.name)
            elif d.is_dir():
                print(f"  Checking {d.name}: INVALID (Missing data.yaml)")
                
    return sorted(datasets)

if __name__ == "__main__":
    sensors = list_sensor_datasets()
    print(f"-> Allowed Sensor Dropdown: {sensors}")
    
    hybrids = list_hybrid_datasets()
    print(f"-> Allowed Hybrid Dropdown: {hybrids}")
    
    images = list_image_datasets()
    print(f"-> Allowed Image Dropdown: {images}")
