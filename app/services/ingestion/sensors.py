import pandas as pd
from pathlib import Path
from typing import Dict, Any
from app.core.config import SENSOR_RESAMPLE_HZ

class SensorParser:
    def parse_sensor_logger(self, path: Path) -> pd.DataFrame:
        """Reads SensorLogger CSVs and merges them into a single 100Hz dataframe."""
        # Simple implementation: Read Accel and Gyro, merge on nearest timestamp
        try:
            accel = pd.read_csv(path / "Accelerometer.csv")
            gyro = pd.read_csv(path / "Gyroscope.csv")
            
            # Standardization: Ensure column names map to canonical x,y,z
            # SensorLogger usually has: time, seconds_elapsed, z, y, x (or similar)
            # We will just look for seconds_elapsed
            
            if 'seconds_elapsed' not in accel.columns:
                raise ValueError("Accelerometer.csv missing 'seconds_elapsed'")
                
            # Rename for merge
            accel = accel.add_prefix('accel_')
            gyro = gyro.add_prefix('gyro_')
            
            # Rename time col back to shared
            accel = accel.rename(columns={'accel_seconds_elapsed': 'seconds_elapsed', 'accel_time': 'time'})
            gyro = gyro.rename(columns={'gyro_seconds_elapsed': 'seconds_elapsed', 'gyro_time': 'time'})
            
            # Merge on seconds_elapsed (approximate)
            # For robust sampling, we should reindex to a fixed 100Hz grid
            
            # Create 100Hz target index
            start_t = max(accel['seconds_elapsed'].min(), gyro['seconds_elapsed'].min())
            end_t = min(accel['seconds_elapsed'].max(), gyro['seconds_elapsed'].max())
            
            import numpy as np
            target_time = np.arange(start_t, end_t, 1.0 / SENSOR_RESAMPLE_HZ)

            def resample(df, name_prefix):
                # interp1d or pandas resample
                df = df.sort_values('seconds_elapsed')
                df = df.set_index('seconds_elapsed')
                # Remove duplicate indices if any
                df = df[~df.index.duplicated(keep='first')]

                # Reindex and interpolate.
                # method='index' interpolates against the actual time index;
                # 'linear' would assume equally-spaced samples, which raw
                # sensor timestamps are not.
                df_interp = df.reindex(df.index.union(target_time)).interpolate(method='index')
                return df_interp.reindex(target_time).add_prefix(name_prefix)

            # Note: real SensorLogger data might need cleaning (dropping non-numeric cols before interp)
            numeric_cols_accel = accel.select_dtypes(include=[np.number]).columns
            numeric_cols_gyro = gyro.select_dtypes(include=[np.number]).columns

            accel_res = resample(accel[numeric_cols_accel], "")
            gyro_res = resample(gyro[numeric_cols_gyro], "")

            # Both frames carry a raw 'time' column after the rename above —
            # keep only the accel copy to avoid duplicate columns in concat.
            gyro_res = gyro_res.drop(columns=['time'], errors='ignore')
            merged = pd.concat([accel_res, gyro_res], axis=1)
            # De-dupe any remaining duplicated column names defensively
            merged = merged.loc[:, ~merged.columns.duplicated(keep='first')]
            # Interpolation cannot fill target timestamps before the first /
            # after the last real sample — drop all-NaN rows.
            merged = merged.dropna(how='all')
            return merged.reset_index().rename(columns={'index': 'seconds_elapsed'})

        except Exception as e:
            print(f"Error parsing sensor logger: {e}")
            raise e

sensor_parser = SensorParser()
