
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

def convert_roadsens(input_dir: Path, output_dir: Path):
    print(f"Converting data from {input_dir} to {output_dir}...")
    
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Supported input columns -> Output columns
    col_map = {
        "seconds_elapsed": "time",
        "accelerometer_x": "accel_x",
        "accelerometer_y": "accel_y",
        "accelerometer_z": "accel_z",
        "gyroscope_x": "gyro_x",
        "gyroscope_y": "gyro_y",
        "gyroscope_z": "gyro_z"
    }
    
    count = 0
    
    # Process all CSVs
    for csv_file in input_dir.rglob("*.csv"):
        try:
            # Skip summary stats
            if "statistics" in csv_file.name: continue
            
            print(f"Processing {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            # Check for required columns
            if not all(c in df.columns for c in col_map.keys()):
                print(f"Skipping {csv_file.name}: Missing sensor columns")
                continue
                
            # Rename
            out_df = df.rename(columns=col_map)[list(col_map.values())]
            
            # Determine Labels
            if "annotation_text" in df.columns:
                out_df['label'] = df['annotation_text'].apply(lambda x: 1 if str(x).lower() == "pothole" else 0)
            else:
                # Fallback to directory/filename
                name_ref = f"{csv_file.parent.name} {csv_file.name}".lower()
                if "pothole" in name_ref or "anomalies" in name_ref:
                    out_df['label'] = 1
                else:
                    out_df['label'] = 0
                    
            # Save
            # Create a unique filename based on parent structure to avoid collisions
            rel_path = csv_file.relative_to(input_dir)
            safe_name = str(rel_path).replace("/", "_").replace(" ", "_")
            out_df.to_csv(output_dir / safe_name, index=False)
            count += 1
            
        except Exception as e:
            print(f"Error converting {csv_file}: {e}")
            
    print(f"Conversion complete. Validated {count} files in {output_dir}")

if __name__ == "__main__":
    base_dir = Path("data/Data1_test")
    target_dir = Path("data/sensor/Data1_converted")
    convert_roadsens(base_dir, target_dir)
