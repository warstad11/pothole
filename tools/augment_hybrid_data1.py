
import os
import shutil
import random
import pandas as pd
import numpy as np
from pathlib import Path

def augment_hybrid():
    # 1. Setup Paths
    source_imgs = Path("data/image/NormalRoads")
    source_sensor_dir = Path("data/sensor/Data 2 - Clean")
    
    dest_imgs = Path("data/hybrid/Data 1 - Both/images")
    dest_sensor = Path("data/hybrid/Data 1 - Both/sensor")
    
    # 2. Load Sensor Source Buffer
    print("Loading sensor source files...")
    sensor_buffer = []
    
    # Use plain_road_*.csv (exclude potholes)
    sensor_files = list(source_sensor_dir.glob("plain_road_*.csv"))
    sensor_files = [f for f in sensor_files if "pothole" not in f.name.lower()]
    
    print(f"Found {len(sensor_files)} sensor source files: {[f.name for f in sensor_files]}")
    
    # Load and concat
    cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    for f in sensor_files:
        try:
            # Attempt to read. Might have headers or not.
            # Data 2 usually has headers? previous ls showed large sizes.
            # Let's assume standard pd.read_csv works, maybe check columns
            df = pd.read_csv(f)
            
            # Normalize columns if needed (assuming user said they exist)
            # User provided format: seconds_elapsed, accel_x...
            # But raw files in Data 2 might be different.
            # Let's hope they have these columns or similar.
            
            # If columns missing, try to map by index (assuming 6 axis + time?)
            # But relying on names is safer.
            
            # Filter just the sensor columns to create our noise pool
            # We will ignore time and label from source, and synthesize them.
            
            # Check for generic names or specific
            valid_cols = [c for c in df.columns if any(x in c.lower() for x in ['acc', 'gyro'])]
            if len(valid_cols) < 6:
                print(f"Warning: {f.name} has overlapping/insufficient columns: {valid_cols}. Skipping.")
                continue
                
            sensor_buffer.append(df[valid_cols])
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    if not sensor_buffer:
        print("Error: No sensor data loaded. Cannot synthesize.")
        return
        
    master_sensor = pd.concat(sensor_buffer, ignore_index=True)
    print(f"Loaded {len(master_sensor)} rows of sensor noise.")
    
    # 3. Load Images
    images = sorted(list(source_imgs.glob("*.png")))
    total_imgs = len(images)
    print(f"Found {total_imgs} images to add.")
    
    # 4. Augmentation Loop
    # Target: 600 pairs.
    # Start ID: 100
    
    start_id = 100
    sample_rate = 0.00538257
    num_rows = 372 # ~2 seconds
    
    print("Starting augmentation...")
    
    for i, img_path in enumerate(images):
        current_id = start_id + i
        base_name = f"{current_id}_Normal_{i}"
        
        # A. Copy Image
        new_img_name = f"{base_name}.jpg" # Convert or rename? User said .jpg convention.
        # But source is .png.
        # I should probably convert to .jpg to match dataset convention 
        # OR just rename ext if I trust it.
        # Safest: Copy as is, but PIL convert if strictly needed.
        # User example: "#_normal_#.jpg"
        # I'll convert to JPG to be perfectly compliant.
        
        from PIL import Image
        with Image.open(img_path) as im:
            rgb_im = im.convert('RGB')
            rgb_im.save(dest_imgs / new_img_name, quality=95)
            
        # B. Synthesize Sensor
        # Random start index in master_sensor
        max_start = len(master_sensor) - num_rows
        if max_start < 0:
            print("Error: Not enough sensor data for a single sample!")
            break
            
        rand_idx = random.randint(0, max_start)
        chunk = master_sensor.iloc[rand_idx : rand_idx + num_rows].copy()
        
        # Reset Time
        chunk['seconds_elapsed'] = [j * sample_rate for j in range(len(chunk))]
        chunk['label'] = 0
        
        # Reorder columns matches user req
        out_cols = ['seconds_elapsed', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
        
        # Ensure chunk has these columns (mapping if needed)
        # We loaded 'valid_cols' earlier. 
        # If headers differ in source, we simply rename the first 6 we found to the standard 6.
        # heuristic: sort valid_cols and map to a_x, a_y, a_z, g_x, g_y, g_z?
        # A bit risky. Let's trust they align or simply rename by position.
        
        curr_cols = chunk.columns.tolist()
        # Remove our new cols from list
        curr_cols = [c for c in curr_cols if c not in ['seconds_elapsed', 'label']]
        
        # Map first 6 to standard
        rename_map = {}
        targets = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
        
        # Try to match names smart
        for tc in targets:
            match = next((c for c in curr_cols if tc in c.lower()), None) # exact-ish match
            if not match:
                # fallback: 'Ax' -> accel_x
                short = tc.split('_')[0][:1] + tc.split('_')[1] # ax
                match = next((c for c in curr_cols if short in c.lower() or tc.split('_')[1] in c.lower()), None)
            
            if match:
                rename_map[match] = tc
                
        chunk.rename(columns=rename_map, inplace=True)
        
        # Fill missing if any
        for tc in targets:
            if tc not in chunk.columns:
                chunk[tc] = 0.0
                
        # Final Select
        final_df = chunk[out_cols]
        
        # Save CSV
        new_csv_name = f"{base_name}.csv"
        final_df.to_csv(dest_sensor / new_csv_name, index=False, float_format='%.6f')
        
    print(f"Done. Created {len(images)} pairs in Data 1.")

if __name__ == "__main__":
    augment_hybrid()
