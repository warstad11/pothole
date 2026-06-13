
import os
import subprocess
import json

DATA_DIR = "/Users/tstading/gemini/antigravity/scratch/pothole1223/data"

def get_video_duration(filepath):
    """Returns duration in seconds."""
    cmd = [
        "ffprobe", 
        "-v", "error", 
        "-show_entries", "format=duration", 
        "-of", "default=noprint_wrappers=1:nokey=1", 
        filepath
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except Exception as e:
        # print(f"Error reading {filepath}: {e}")
        return 0.0

def analyze_folder_media(path, group_name_prefix=""):
    print(f"\n[{group_name_prefix}]")
    if not os.path.exists(path):
        print("  Path not found.")
        return

    # Iterate immediate subdirectories (Scenes or Trips)
    items = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    
    total_csv_imu = 0
    total_video_sec = 0
    
    for item in items:
        item_path = os.path.join(path, item)
        csv_count_imu = 0
        video_sec = 0
        video_count = 0
        
        # Specific search for known IMU filenames
        # iPhone: Compass.csv, maybe others?
        # NuScenes: index_cam_front_to_imu.csv
        
        for root, dirs, files in os.walk(item_path):
            for file in files:
                if file.endswith('.csv'):
                    if 'Compass.csv' in file or 'imu' in file.lower() or 'accelerometer' in file.lower():
                        csv_count_imu += 1
                elif file.endswith('.mp4'):
                    dur = get_video_duration(os.path.join(root, file))
                    video_sec += dur
                    video_count += 1
        
        if csv_count_imu > 0 or video_count > 0:
            print(f"  {item}:")
            if csv_count_imu > 0:
                print(f"    IMU/Accel CSVs: {csv_count_imu}")
            if video_count > 0:
                print(f"    Videos: {video_count} files ({video_sec/60:.2f} minutes)")
            
            total_csv_imu += csv_count_imu
            total_video_sec += video_sec

    print(f"  TOTAL {group_name_prefix} IMU CSVs: {total_csv_imu}")
    print(f"  TOTAL {group_name_prefix} VIDEO MINUTES: {total_video_sec/60:.2f}")

print("--- Media Analysis ---")

analyze_folder_media(os.path.join(DATA_DIR, 'iphone'), "iPhone Data")
analyze_folder_media(os.path.join(DATA_DIR, 'nuscenes'), "NuScenes Data")
