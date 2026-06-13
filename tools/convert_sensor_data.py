import pandas as pd
from pathlib import Path
import os

def convert_just_sensor():
    # Data 2 - Just Sensor: plain_road (label 0), plain_road_potholes (label 1)
    base_path = Path("data/sensor/Data 2 - Just Sensor")
    output_dir = Path("data/processed/sensor")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mapping = {
        "plain_road": 0,
        "plain_road_potholes": 1
    }
    
    for subdir, label in mapping.items():
        data_path = base_path / subdir
        if not data_path.exists(): continue
        
        for csv_file in data_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            # Just Sensor Format: Time,Gx,Gy,Gz,Ax,Ay,Az
            # Normalized: accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, label
            df_norm = pd.DataFrame()
            df_norm['accel_x'] = df['Ax']
            df_norm['accel_y'] = df['Ay']
            df_norm['accel_z'] = df['Az']
            df_norm['gyro_x'] = df['Gx']
            df_norm['gyro_y'] = df['Gy']
            df_norm['gyro_z'] = df['Gz']
            df_norm['label'] = label
            
            output_path = output_dir / f"just_sensor_{subdir}_{csv_file.name}"
            df_norm.to_csv(output_path, index=False)
            print(f"Converted {csv_file} -> {output_path}")

def convert_hybrid():
    # Data 1 - Both: Road Anomalies (label 1?), Normal Road (label 0?)
    # Based on listing, we have "Raw Data" folders with 1.csv inside them
    # And separate Annotation.csv
    base_path = Path("data/hybrid/Data 1 - Both/Raw Data")
    output_dir = Path("data/processed/sensor")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for drive_dir in base_path.iterdir():
        if not drive_dir.is_dir(): continue
        
        csv_file = drive_dir / "1.csv"
        annot_file = drive_dir / "Annotation.csv"
        
        if not csv_file.exists(): continue
        
        df = pd.read_csv(csv_file)
        # Hybrid 1.csv format (seen in view_file):
        # seconds_elapsed, accelerometer_z, accelerometer_y, accelerometer_x, gravity_z, ..., gyroscope_z, gyroscope_y, gyroscope_x, ...
        
        df_norm = pd.DataFrame()
        # Note: mapping based on typical coordinate systems and file column names
        df_norm['accel_x'] = df['accelerometer_x']
        df_norm['accel_y'] = df['accelerometer_y']
        df_norm['accel_z'] = df['accelerometer_z']
        df_norm['gyro_x'] = df['gyroscope_x']
        df_norm['gyro_y'] = df['gyroscope_y']
        df_norm['gyro_z'] = df['gyroscope_z']
        
        # Merge annotations if present
        df_norm['label'] = 0
        if annot_file.exists():
            annots = pd.read_csv(annot_file)
            # Sample annots have 'seconds_elapsed' and 'text' (e.g. 'Pothole')
            for _, row in annots.iterrows():
                if row['text'].lower() == 'pothole':
                    # Find closest time in df
                    # This is naive but works for a single point annotation
                    idx = (df['seconds_elapsed'] - row['seconds_elapsed']).abs().idxmin()
                    # Apply label to a window around this index? 
                    # Prompt from prev session: "±0.5 seconds tolerance"
                    # 100Hz = 0.01 step. ±0.5s = ±50 samples.
                    start_idx = max(0, idx - 50)
                    end_idx = min(len(df_norm), idx + 51)
                    df_norm.loc[start_idx:end_idx, 'label'] = 1
        
        # Clean up NaNs from alignment/interpolation if any
        df_norm = df_norm.dropna()
        
        output_path = output_dir / f"hybrid_{drive_dir.name}.csv"
        df_norm.to_csv(output_path, index=False)
        print(f"Converted {csv_file} -> {output_path}")

if __name__ == "__main__":
    print("Starting sensor data conversion...")
    convert_just_sensor()
    convert_hybrid()
    print("Conversion complete.")
