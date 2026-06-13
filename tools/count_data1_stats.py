
import os
from pathlib import Path

def count_stats(base_path):
    base = Path(base_path)
    
    modes = ["images", "sensor"]
    
    print(f"Analyzing {base}...")
    
    for mode in modes:
        mode_path = base / mode
        if not mode_path.exists():
            print(f"  {mode}: NOT FOUND")
            continue
            
        # Recursive scan for files
        files = []
        if mode == "images":
            exts = {".jpg", ".jpeg", ".png"}
        else:
            exts = {".csv", ".txt"}
            
        for f in mode_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in exts:
                files.append(f)
        
        pos = 0
        neg = 0
        unknown = 0
        
        for f in files:
            name_check = f"{f.parent.name} {f.name}".lower()
            
            # Label Logic (mirrors codebase)
            if any(x in name_check for x in ['pothole', 'positive', 'damage', 'cracking', 'bump']):
                pos += 1
            elif any(x in name_check for x in ['normal', 'plain', 'negative', 'undamaged', 'regular', 'road', 'joint']):
                neg += 1
            else:
                unknown += 1
                # print(f"    Unknown label: {f.name}")

        print(f"  {mode.upper()}:")
        print(f"    Total: {len(files)}")
        print(f"    Positive (Pothole): {pos}")
        print(f"    Negative (Normal):  {neg}")
        if unknown > 0:
            print(f"    Unknown:            {unknown}")

if __name__ == "__main__":
    count_stats("data/hybrid/Data 1 - Both")
