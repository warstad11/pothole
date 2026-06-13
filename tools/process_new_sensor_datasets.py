import pandas as pd
import os
import glob

def process_data_2():
    source_root = "data/sensor/Data 2 - Just Sensor"
    dest_root = "data/sensor/Data 2 - Clean"
    
    if not os.path.exists(source_root):
        print(f"Source not found: {source_root}")
        return

    os.makedirs(dest_root, exist_ok=True)
    
    # Define mapping: Folder Name -> Label
    # implicit: plain_road = 0? plain_road_potholes = 1?
    # Let's verify standard: usually 0 is negative (normal), 1 is positive (pothole).
    class_map = {
        "plain_road": 0,
        "plain_road_potholes": 1
    }
    
    # Data 2 columns: Time,Gx,Gy,Gz,Ax,Ay,Az
    # Standard Target: label,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
    
    files_processed = 0
    
    for folder, label in class_map.items():
        search_path = os.path.join(source_root, folder, "*.csv")
        files = glob.glob(search_path)
        
        for f in files:
            try:
                df = pd.read_csv(f)
                
                # Check expected columns
                required_cols = ['Time', 'Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az']
                if not all(col in df.columns for col in required_cols):
                    print(f"Skipping {f}: Missing columns. Found {df.columns.tolist()}")
                    continue
                
                # Rename and Reorder
                # Note: Data 2 seems to be in G's or m/s^2? 
                # Sample: Ax~2.7, Az~10.8. Likely m/s^2 (gravity ~9.8).
                # Gyro: ~0.05. Rad/s or Deg/s? Usually Rad/s if small.
                
                df_out = pd.DataFrame()
                df_out['label'] = label
                df_out['accel_x'] = df['Ax']
                df_out['accel_y'] = df['Ay']
                df_out['accel_z'] = df['Az']
                df_out['gyro_x'] = df['Gx']
                df_out['gyro_y'] = df['Gy']
                df_out['gyro_z'] = df['Gz']
                # df['Time'] is timestamp string. We usually need relative seconds_elapsed
                # But our models might just ignore time?
                # Let's try to parse time to relative if possible, else 0.01 step
                # Time format: 2019-01-28 22:05:48:548
                # It has milliseconds.
                
                try:
                    # Fix format '22:05:48:548' -> '22:05:48.548'
                    # The sample showed colon before millis
                    # 2019-01-28 22:05:48:548
                    times = df['Time'].astype(str).str.replace(r':(\d{3})$', r'.\1', regex=True)
                    dt = pd.to_datetime(times)
                    start_t = dt.iloc[0]
                    df_out['seconds_elapsed'] = (dt - start_t).dt.total_seconds()
                except Exception as e:
                    print(f"Time parsing failed for {f}: {e}.Using index step.")
                    # Fallback: assume 100Hz? sample has diff 2s between lines?
                    # Sample: 48:548, 48:548... wait.
                    # Sample showed DUPLICATE timestamps for many rows.
                    # This implies burst or fast sampling.
                    # For safety, let's just use index * 0.01 (100hz assumption) or just keep 0
                    df_out['seconds_elapsed'] = df.index * 0.01

                out_name = f"{folder}_{os.path.basename(f)}"
                out_path = os.path.join(dest_root, out_name)
                df_out.to_csv(out_path, index=False)
                files_processed += 1
                
            except Exception as e:
                print(f"Error processing {f}: {e}")

    print(f"Data 2 Processing Complete. {files_processed} files saved to {dest_root}")

if __name__ == "__main__":
    process_data_2()
