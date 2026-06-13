
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
import shutil
import os

def process_data1_hybrid():
    print("--- Processing Data 1 for Hybrid (Extraction) ---")
    
    source_dir = Path("data/image/raw/Data 1")
    output_dir = Path("data/hybrid/Data 1 - Both")
    
    # Reset Output
    if output_dir.exists():
        print(f"Removing existing {output_dir}...")
        shutil.rmtree(output_dir)
    
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "sensor").mkdir(parents=True, exist_ok=True)
    
    sessions = sorted([p for p in source_dir.iterdir() if p.is_dir() and (p / "Annotation.csv").exists()])
    print(f"Found {len(sessions)} sessions.")
    
    # Shared sensor columns we need (standardized)
    # The output CSV should have: time, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, label
    
    total_extracted = 0
    
    for sess in sessions:
        try:
            # 1. Load Annotations
            anno_df = pd.read_csv(sess / "Annotation.csv")
            
            # 2. Load Sensor Data
            # We need Accelerometer and Gyroscope. 
            # Note: "Annotation.csv" 'time' is usually nanoseconds or similar. Check example.
            # Example Data 1/1/Annotation: 1753513579461000000 (ns) -> 1.7535...e9 (sec)?
            # Video timestamps? 
            # "seconds_elapsed" column in Annotation.csv is relative to start?
            
            # Let's verify sensor CSV format. Usually they have a 'time' column OR 'seconds_elapsed'.
            # Data 1 example: "time,seconds_elapsed,z,y,x"
            
            acc_path = sess / "Accelerometer.csv"
            gyro_path = sess / "Gyroscope.csv"
            
            if not acc_path.exists() or not gyro_path.exists():
                print(f"Skipping {sess.name}: Missing Acc/Gyro CSVs")
                continue
                
            acc_df = pd.read_csv(acc_path)
            gyro_df = pd.read_csv(gyro_path)
            
            # Merge logic: Interpolate to align? Or just take nearest?
            # Creating a unified sensor df is complex.
            # Simpler: Create a unified DataFrame on the Union of timestamps or resample.
            # Let's assume 100Hz (0.01s).
            
            # Common time base: seconds_elapsed
            # Check if columns exist
            if 'seconds_elapsed' not in acc_df.columns:
                # Try 'time'
                pass
            
            # Rename for standard format
            # accelerometer usually z,y,x columns based on previous `head`.
            # Standard: accel_x, accel_y, accel_z
            # Data 1 headers seen in verify: "time,seconds_elapsed,z,y,x"
            
            # Normalize Headers
            def normalize(df, prefix):
                df.columns = [c.lower() for c in df.columns]
                rename_map = {}
                for c in df.columns:
                    if c == 'x': rename_map[c] = f'{prefix}_x'
                    elif c == 'y': rename_map[c] = f'{prefix}_y'
                    elif c == 'z': rename_map[c] = f'{prefix}_z'
                df = df.rename(columns=rename_map)
                return df

            acc_df = normalize(acc_df, 'accel')
            gyro_df = normalize(gyro_df, 'gyro')
            
            # Merge on seconds_elapsed nearest
            # Using pandas merge_asof
            acc_df = acc_df.sort_values('seconds_elapsed')
            gyro_df = gyro_df.sort_values('seconds_elapsed')
            
            merged_sensor = pd.merge_asof(
                acc_df, gyro_df, 
                on='seconds_elapsed', 
                direction='nearest',
                tolerance=0.02 # 20ms tolerance
            )
            
            # 3. Load Video
            vid_files = list((sess / "Camera").glob("*.mp4"))
            if not vid_files:
                print(f"Skipping {sess.name}: No video")
                continue
            video_path = vid_files[0]
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            
            # 4. Iterate Events
            for idx, row in anno_df.iterrows():
                label_text = row['text'] # Pothole, Bump
                sec = row['seconds_elapsed']
                
                # Filter Label
                # Pothole -> 1
                # Bump -> 0 (Treat as negative/normal for Pothole detection? Or 1 for Anomaly?)
                # User said: "identify positive or negative... fusion... combine"
                # If target is "Pothole Detection", then Bump is negative (0).
                # If target is "Anomaly Detection", Bump is positive (1).
                # Given project name "Pothole", assume Pothole=1.
                
                label_val = 1 if 'pothole' in label_text.lower() else 0
                
                # Window Extraction
                # +/- 1.0 second (2 seconds total)
                window_sec = 2.0
                start_t = sec - (window_sec / 2)
                end_t = sec + (window_sec / 2)
                
                # Image Extraction
                # Frame at exactly 'sec'
                frame_id = int(sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                
                if not ret:
                    # Video might be shorter than sensor log or sync issue
                    # Try timestamp seek
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    ret, frame = cap.read()
                
                if ret:
                    base_name = f"{sess.name}_{label_text}_{int(sec*1000)}"
                    base_name = base_name.replace(" ", "_").replace("(", "").replace(")", "")
                    
                    # Save Image
                    cv2.imwrite(str(output_dir / "images" / f"{base_name}.jpg"), frame)
                    
                    # Save Sensor Window
                    # Filter merged_sensor by time
                    mask = (merged_sensor['seconds_elapsed'] >= start_t) & (merged_sensor['seconds_elapsed'] <= end_t)
                    window_df = merged_sensor[mask].copy()
                    
                    # Add label column
                    window_df['label'] = label_val
                    
                    # Ensure required cols
                    valid_cols = ['seconds_elapsed', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']
                    # Fill missing if any
                    for c in valid_cols:
                        if c not in window_df.columns:
                            window_df[c] = 0.0
                            
                    window_df = window_df[valid_cols]
                    window_df.to_csv(output_dir / "sensor" / f"{base_name}.csv", index=False)
                    
                    total_extracted += 1
            
            cap.release()
            
        except Exception as e:
            print(f"Error processing {sess.name}: {e}")

    print(f"--- Processing Complete ---")
    print(f"Total Extracted Events: {total_extracted}")
    print(f"Data saved to {output_dir}")

if __name__ == "__main__":
    process_data1_hybrid()
