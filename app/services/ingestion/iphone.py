
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import json
import shutil
from app.core.config import SENSOR_RESAMPLE_HZ

class iPhoneIngestionService:
    """Service to ingest and format iPhone SensorLogger data."""

    REQUIRED_FILES = ["Accelerometer.csv", "Gyroscope.csv", "Location.csv"]

    @staticmethod
    def _to_seconds(values) -> np.ndarray:
        """Detects the time unit from magnitude and converts to seconds.

        SensorLogger 'time' is UNIX nanoseconds; 'seconds_elapsed' is already
        seconds. Building an np.arange grid over raw nanoseconds would OOM,
        so any epoch-scale column is converted here.
        """
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return v
        m = np.nanmax(np.abs(v))
        if m > 1e17:    # nanoseconds
            return v / 1e9
        if m > 1e14:    # microseconds
            return v / 1e6
        if m > 1e11:    # milliseconds
            return v / 1e3
        return v        # already seconds

    @staticmethod
    def process_session(session_path: Path) -> bool:
        """
        Processes a raw iPhone session directory.
        1. Checks if all required files exist.
        2. Merges and resamples sensor data to SENSOR_RESAMPLE_HZ.
        3. Saves 'sensor.csv' in the session directory.
        4. Extracts video metadata if possible.
        """
        if not session_path.exists():
            return False
            
        print(f"Processing iPhone session: {session_path.name}")
        
        # 1. Load Data
        dfs = {}
        try:
            # Flexible file matching
            files = list(session_path.glob("*.csv"))
            for fname in iPhoneIngestionService.REQUIRED_FILES:
                match = [f for f in files if f.name.lower() == fname.lower()]
                if not match:
                    print(f"Missing required file: {fname}")
                    # If Location is missing, we can survive, but need Accel/Gyro
                    if fname in ["Accelerometer.csv", "Gyroscope.csv"]:
                        return False
                    continue
                dfs[fname.split(".")[0]] = pd.read_csv(match[0])
        except Exception as e:
            print(f"Error reading CSVs: {e}")
            return False

        if "Accelerometer" not in dfs or "Gyroscope" not in dfs:
            return False

        acc = dfs["Accelerometer"]
        gyro = dfs["Gyroscope"]
        loc = dfs.get("Location", pd.DataFrame())
        
        # 2. Alignment & Resampling
        # Handle different column names (time vs seconds_elapsed)
        def get_time_col(df):
            for c in ['seconds_elapsed', 'time', 'Timestamp']:
                if c in df.columns: return c
            return None
            
        acc_t = get_time_col(acc)
        gyro_t = get_time_col(gyro)

        if not acc_t or not gyro_t:
            print("Missing time column in sensor data")
            return False

        # Normalize time columns to seconds (the 'time' fallback column is in
        # UNIX nanoseconds) and re-zero epoch-scale values to a COMMON start so
        # the streams stay mutually aligned, like seconds_elapsed.
        acc = acc.sort_values(acc_t).copy()
        gyro = gyro.sort_values(gyro_t).copy()
        acc[acc_t] = iPhoneIngestionService._to_seconds(acc[acc_t].values)
        gyro[gyro_t] = iPhoneIngestionService._to_seconds(gyro[gyro_t].values)

        loc_t = get_time_col(loc) if not loc.empty else None
        if loc_t:
            # np.interp requires monotonically increasing xp — sort location too
            loc = loc.sort_values(loc_t).copy()
            loc[loc_t] = iPhoneIngestionService._to_seconds(loc[loc_t].values)

        t0_candidates = [acc[acc_t].min(), gyro[gyro_t].min()]
        if loc_t:
            t0_candidates.append(loc[loc_t].min())
        t0 = min(t0_candidates)
        if t0 > 1e6:  # absolute epoch timestamps -> make them relative
            acc[acc_t] = acc[acc_t] - t0
            gyro[gyro_t] = gyro[gyro_t] - t0
            if loc_t:
                loc[loc_t] = loc[loc_t] - t0

        # Define common time axis at the canonical resample rate
        # (NOTE: unified at SENSOR_RESAMPLE_HZ=100; this path previously used 50Hz)
        t_min = max(acc[acc_t].min(), gyro[gyro_t].min())
        t_max = min(acc[acc_t].max(), gyro[gyro_t].max())

        if t_max <= t_min:
             print("Invalid time range for session")
             return False

        target_time = np.arange(t_min, t_max, 1.0 / SENSOR_RESAMPLE_HZ)
        
        def interpolate_stream(source_df, t_col, cols, prefix):
            source_df = source_df.sort_values(t_col)
            data = {'seconds_elapsed': target_time}
            for col in cols:
                # Case-insensitive column match
                match = [c for c in source_df.columns if c.lower() == col.lower()]
                if match:
                    data[f"{prefix}_{col.lower()}"] = np.interp(target_time, source_df[t_col], source_df[match[0]])
            return pd.DataFrame(data)

        # Process Accel/Gyro
        df_acc = interpolate_stream(acc, acc_t, ['x', 'y', 'z'], "accel")
        df_gyro = interpolate_stream(gyro, gyro_t, ['x', 'y', 'z'], "gyro")
        
        # Merge
        final_df = df_acc.copy()
        for col in ['gyro_x', 'gyro_y', 'gyro_z']:
            if col in df_gyro.columns:
                final_df[col] = df_gyro[col]
        
        # Location (loc is already sorted by loc_t and converted to seconds above)
        if not loc.empty and loc_t:
            for col in ['latitude', 'longitude', 'speed']:
                match = [c for c in loc.columns if c.lower() == col.lower()]
                if match:
                    final_df[col] = np.interp(target_time, loc[loc_t], loc[match[0]])

        # 3. Save
        output_path = session_path / "sensor.csv"
        final_df.to_csv(output_path, index=False)
        
        # 4. Alignment
        from app.services.ingestion.alignment import aligner
        timeline = aligner.align_drive(0.0, 0.0) # Simple default
        timeline.save(session_path)
        
        return True

    @staticmethod
    def get_sessions(base_path: Path) -> List[Dict]:
        """List and auto-process available iPhone sessions."""
        sessions = []
        if not base_path.exists():
            return []
            
        for d in base_path.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                # Auto-process if not yet done
                if not (d / "sensor.csv").exists():
                    try:
                        iPhoneIngestionService.process_session(d)
                    except: pass
                
                is_processed = (d / "sensor.csv").exists()
                
                # Check for video
                video_url = None
                # Flexible video discovery
                video_files = list(d.glob("*.mp4")) + list((d / "Camera").glob("*.mp4"))
                if video_files:
                    v = video_files[0]
                    # Determine URL
                    rel_path = v.relative_to(base_path)
                    video_url = f"/static/iphone/{rel_path}"
                
                sessions.append({
                    "id": d.name,
                    "name": d.name.replace("_", " "),
                    "path": str(d),
                    "processed": is_processed,
                    "video_url": video_url
                })
        return sorted(sessions, key=lambda x: x['id'], reverse=True)
