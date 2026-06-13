
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import subprocess
import os
from app.core.config import SENSOR_RESAMPLE_HZ

class NuScenesIngestionService:
    """Service to ingest and format nuScenes data (similar to iPhone structure)."""

    @staticmethod
    def process_scene(scene_path: Path) -> bool:
        """
        Processes a nuScenes scene directory.
        1. Resamples IMU data to SENSOR_RESAMPLE_HZ and saves as 'sensor.csv'.
        2. Generates 'video.mp4' from front camera frames using FFmpeg.
        3. Extracts metadata.
        """
        if not scene_path.exists():
            return False
            
        print(f"Processing nuScenes scene: {scene_path.name}")
        
        # 1. IMU Processing
        imu_csv = scene_path / "imu" / "ms_imu.csv"
        if not imu_csv.exists():
            print(f"Missing IMU data in {scene_path}")
            return False
            
        try:
            df_imu = pd.read_csv(imu_csv)
            # nuScenes time is in microseconds (utime)
            df_imu['seconds_elapsed'] = (df_imu['utime'] - df_imu['utime'].min()) / 1_000_000.0
            
            t_min = df_imu['seconds_elapsed'].min()
            t_max = df_imu['seconds_elapsed'].max()
            
            if t_max <= t_min:
                return False
                
            # Target the canonical resample rate (consistent across all ingestion)
            target_time = np.arange(t_min, t_max, 1.0 / SENSOR_RESAMPLE_HZ)
            
            final_data = {'seconds_elapsed': target_time}
            
            # Map columns and interpolate
            # ax, ay, az -> accel_x, y, z
            # gx, gy, gz -> gyro_x, y, z
            mapping = {
                'ax': 'accel_x', 'ay': 'accel_y', 'az': 'accel_z',
                'gx': 'gyro_x', 'gy': 'gyro_y', 'gz': 'gyro_z'
            }
            
            for src, dst in mapping.items():
                if src in df_imu.columns:
                    final_data[dst] = np.interp(target_time, df_imu['seconds_elapsed'], df_imu[src])
            
            # Save processed sensor data
            final_df = pd.DataFrame(final_data)
            final_df.to_csv(scene_path / "sensor.csv", index=False)
            
        except Exception as e:
            print(f"Error processing IMU: {e}")
            return False
            
        # 2. Video Generation (FFmpeg)
        # Frame filenames are utimes (microseconds) — derive the true frame
        # rate from the median timestamp delta instead of assuming 20 fps.
        video_path = scene_path / "video.mp4"
        if not video_path.exists():
            frames_dir = scene_path / "frames_cam_front"
            if frames_dir.exists():
                try:
                    fps = 20.0
                    utimes = []
                    for p in sorted(frames_dir.glob("*.jpg")):
                        try:
                            utimes.append(int(p.stem))
                        except ValueError:
                            pass
                    if len(utimes) >= 2:
                        utimes.sort()
                        median_dt_us = float(np.median(np.diff(utimes)))
                        if median_dt_us > 0:
                            fps = 1_000_000.0 / median_dt_us
                            print(f"Derived frame rate {fps:.3f} fps from frame timestamps.")
                        else:
                            print("WARNING: Non-positive timestamp deltas; falling back to 20 fps.")
                    else:
                        print("WARNING: Fewer than 2 parseable frame timestamps; falling back to 20 fps.")
                    cmd = [
                        "ffmpeg", "-y",
                        "-framerate", f"{fps:.6f}",
                        "-pattern_type", "glob", "-i", f"{frames_dir}/*.jpg",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
                        str(video_path)
                    ]
                    # Note: pattern_type glob might not work on all shells, but should in subprocess
                    subprocess.run(cmd, check=True, capture_output=True)
                except Exception as e:
                    print(f"FFmpeg error: {e}")
                    # Fallback to empty if needed, but video is required for Section E UI
                    
        # 3. Alignment (Create dummy alignment for now, scale=1.0, offset=0.0)
        from app.services.ingestion.alignment import aligner
        timeline = aligner.align_drive(0.0, 1.0)
        timeline.save(scene_path)
        
        return True

    @staticmethod
    def get_scenes(base_path: Path) -> List[Dict]:
        """List available nuScenes scenes."""
        scenes = []
        if not base_path.exists():
            return []
            
        for d in base_path.iterdir():
            if d.is_dir() and d.name.startswith("scene-"):
                # Check if processed
                is_processed = (d / "sensor.csv").exists() and (d / "video.mp4").exists()
                
                # Determine Video URL (relative to DATA_DIR/nuscenes)
                video_url = None
                if (d / "video.mp4").exists():
                    video_url = f"/static/nuscenes/{d.name}/video.mp4"
                
                scenes.append({
                    "id": d.name,
                    "name": d.name,
                    "path": str(d),
                    "processed": is_processed,
                    "video_url": video_url
                })
                
        return sorted(scenes, key=lambda x: x['id'])

    @staticmethod
    def get_image_url(scene_path: Path, time: float) -> Optional[str]:
        """Finds the closest front camera image for a given elapsed time."""
        frames_dir = scene_path / "frames_cam_front"
        imu_csv = scene_path / "imu" / "ms_imu.csv"
        
        if not frames_dir.exists() or not imu_csv.exists():
            return None
            
        try:
            # 1. Get min_utime from IMU (read the whole column — the CSV is
            # not guaranteed to be sorted, so the first row may not be the min)
            df_imu = pd.read_csv(imu_csv, usecols=['utime'])
            min_utime = df_imu['utime'].min()
            
            target_utime = int(min_utime + (time * 1_000_000))
            
            # 2. Find closest JPG
            all_jpgs = list(frames_dir.glob("*.jpg"))
            if not all_jpgs:
                return None
                
            jpg_utimes = [int(p.stem) for p in all_jpgs]
            closest_utime = min(jpg_utimes, key=lambda x: abs(x - target_utime))
            
            return f"/static/nuscenes/{scene_path.name}/frames_cam_front/{closest_utime}.jpg"
            
        except Exception as e:
            print(f"Error finding image: {e}")
            return None
