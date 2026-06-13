
import pandas as pd
import shutil
from pathlib import Path
import os

def process_2020():
    print("--- Processing Dataset 2020 ---")
    src_base = Path("data/hybrid/Dataset_2020")
    dst_base = Path("data/hybrid/Dataset_2020a")
    
    if dst_base.exists(): shutil.rmtree(dst_base)
    dst_base.mkdir(parents=True)
    
    (dst_base / "images").mkdir()
    (dst_base / "labels").mkdir()
    (dst_base / "sensor").mkdir()
    
    # 1. Process Images
    print("Processing Images...")
    src_img_dir = src_base / "Image_data" / "images"
    src_lbl_dir = src_base / "Image_data" / "labels"
    
    img_count = 0
    if src_img_dir.exists():
        for img_p in list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")):
            # Copy Image
            shutil.copy(img_p, dst_base / "images" / img_p.name)
            
            # Copy Label
            txt_name = img_p.with_suffix(".txt").name
            txt_src = src_lbl_dir / txt_name
            if txt_src.exists():
                shutil.copy(txt_src, dst_base / "labels" / txt_name)
            else:
                # Create empty label (Negative)
                (dst_base / "labels" / txt_name).touch()
            img_count += 1
            
    print(f"Processed {img_count} images.")

    # 2. Process Sensor
    print("Processing Sensor Data...")
    # Source: Motion_data/extracted_txt
    src_motion = src_base / "Motion_data" / "extracted_txt"
    sensor_count = 0
    
    if src_motion.exists():
        for txt_p in src_motion.glob("*.txt"):
            try:
                # Read headerless TXT: col1=label, c2=ax, c3=ay, c4=az
                # Readme says: "first column... labeled 1 or 0"
                # "second, third, fourth... acceleration x, y, z"
                df = pd.read_csv(txt_p, header=None)
                
                # Check dim
                if df.shape[1] < 4:
                    print(f"Skipping {txt_p.name}: Not enough columns {df.shape}")
                    continue
                    
                # Rename
                # Assuming first 4 cols are relevant
                data = {
                    "label": df.iloc[:, 0],
                    "accel_x": df.iloc[:, 1],
                    "accel_y": df.iloc[:, 2],
                    "accel_z": df.iloc[:, 3],
                    # Pad Gyro with 0.0
                    "gyro_x": 0.0,
                    "gyro_y": 0.0,
                    "gyro_z": 0.0,
                    # Fake time (100Hz assumed)
                    "seconds_elapsed": [i*0.01 for i in range(len(df))]
                }
                
                new_df = pd.DataFrame(data)
                
                # Save as CSV
                out_name = txt_p.with_suffix(".csv").name
                new_df.to_csv(dst_base / "sensor" / out_name, index=False)
                sensor_count += 1
                
            except Exception as e:
                print(f"Error converting {txt_p.name}: {e}")
                
    print(f"Processed {sensor_count} sensor files.")
    
    # 3. Create data.yaml for Images
    yaml_content = f"""
path: {(dst_base).resolve()}
train: images
val: images
names:
  0: Crack
  1: Pothole
    """
    with open(dst_base / "data.yaml", "w") as f:
        f.write(yaml_content)
        
    print("Created data.yaml")
    
    # 4. Hybrid Alignment Check
    print("\n--- Hybrid Alignment Check ---")
    # Check if filenames overlap
    img_names = set(p.stem for p in (dst_base / "images").glob("*"))
    snr_names = set(p.stem for p in (dst_base / "sensor").glob("*"))
    
    common = img_names.intersection(snr_names)
    print(f"Images: {len(img_names)}")
    print(f"Sensor: {len(snr_names)}")
    print(f"Common Filenames (Pairs): {len(common)}")
    
    if len(common) == 0:
        print("❌ NO PAIRS FOUND.")
        print("   This confirms datasets are disjoint (Image vs Sensor).")
        print("   Hybrid Testing is NOT POSSIBLE with this dataset.")
    else:
        print(f"✅ Found {len(common)} pairs.")

if __name__ == "__main__":
    process_2020()
