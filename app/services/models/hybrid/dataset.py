import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List
import pandas as pd
import numpy as np
import json
import cv2
from app.core.config import SENSOR_RESAMPLE_HZ
from app.services.ingestion.alignment import DriveTimeline


# Canonical, ordered sensor channel layout. Accelerometer channels are
# mandatory; gyroscope channels are appended ONLY when every sensor file in
# the dataset provides all three (so the channel count is identical across
# all samples).
REQUIRED_CHANNELS = ['accel_x', 'accel_y', 'accel_z']
OPTIONAL_CHANNELS = ['gyro_x', 'gyro_y', 'gyro_z']

# Common naming variants normalized to the canonical channel names.
_CHANNEL_RENAMES = {
    'accelerometer_x': 'accel_x', 'accelerometer_y': 'accel_y', 'accelerometer_z': 'accel_z',
    'gyroscope_x': 'gyro_x', 'gyroscope_y': 'gyro_y', 'gyroscope_z': 'gyro_z',
}


class HybridDataset(Dataset):
    def __init__(self, drive_paths: List[Path], window_size: int = 100, stride: int = 10,
                 image_size: int = 224):
        self.samples = []
        self.window_size = window_size
        self.image_size = image_size

        # Fix a single ordered channel list up-front so every sample has the
        # same channel count/ordering (raises on files missing accel_x/y/z).
        self.channels = self._resolve_channels(drive_paths)
        print(f"[HybridDataset] Using ordered sensor channels: {self.channels}")

        for d_path in drive_paths:
            # --- Event Mode (dataset_index.json) ---
            index_path = d_path / "dataset_index.json"
            if index_path.exists():
                self._load_event_mode(d_path, index_path)
                continue

            # --- Continuous Mode (alignment.json + sensor.csv) ---
            if (d_path / "alignment.json").exists():
                self._load_continuous_mode(d_path, stride)

        # Labels must be valid class indices for CrossEntropyLoss. Drop any
        # sample whose label is not 0/1 (e.g. the -1 "unknown" placeholder).
        n_before = len(self.samples)
        self.samples = [s for s in self.samples if s["label"] in (0, 1)]
        n_dropped = n_before - len(self.samples)
        if n_dropped > 0:
            print(f"[HybridDataset] Dropped {n_dropped} samples with invalid labels "
                  f"(not 0/1); {len(self.samples)} samples remain.")

    # ------------------------------------------------------------------ #
    # Channel resolution: scan all sensor CSV headers before loading
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_channels(drive_paths: List[Path]) -> List[str]:
        csv_paths = []
        for d_path in drive_paths:
            index_path = d_path / "dataset_index.json"
            if index_path.exists():
                try:
                    with open(index_path, "r") as f:
                        index = json.load(f)
                except Exception:
                    continue
                for rel_img_path, meta in index.items():
                    sensor_path = d_path / meta["sensor_path"]
                    image_path = d_path / rel_img_path
                    if sensor_path.exists() and image_path.exists():
                        csv_paths.append(sensor_path)
            elif (d_path / "alignment.json").exists() and (d_path / "sensor.csv").exists():
                csv_paths.append(d_path / "sensor.csv")

        all_have_gyro = bool(csv_paths)
        for p in csv_paths:
            try:
                cols = list(pd.read_csv(p, nrows=0).columns)
            except Exception:
                # Unreadable files are skipped during loading as well
                continue
            cols = [_CHANNEL_RENAMES.get(c, c) for c in cols]
            missing = [c for c in REQUIRED_CHANNELS if c not in cols]
            if missing:
                raise ValueError(
                    f"Sensor file {p} is missing required channels {missing}. "
                    f"Every sensor CSV must provide {REQUIRED_CHANNELS} "
                    f"(or recognized variants {sorted(_CHANNEL_RENAMES.keys())})."
                )
            if not all(c in cols for c in OPTIONAL_CHANNELS):
                all_have_gyro = False

        channels = list(REQUIRED_CHANNELS)
        if all_have_gyro and csv_paths:
            channels += OPTIONAL_CHANNELS
        return channels

    # ------------------------------------------------------------------ #
    # Event Mode: each image maps to a short sensor snippet via an index
    # ------------------------------------------------------------------ #
    def _load_event_mode(self, d_path: Path, index_path: Path):
        with open(index_path, "r") as f:
            index = json.load(f)

        for rel_img_path, meta in index.items():
            sensor_path = d_path / meta["sensor_path"]
            image_path = d_path / rel_img_path

            if not sensor_path.exists() or not image_path.exists():
                continue

            # The authoritative label comes from dataset_index.json, NOT
            # from the CSV's internal label column.  The internal CSV labels
            # are unreliable for some categories (e.g. Bump files are marked
            # label=0 internally but are positive road-damage events).
            index_label = meta.get("label", -1)

            try:
                sensor_df = pd.read_csv(sensor_path).rename(columns=_CHANNEL_RENAMES)
                # Fixed, ordered channel selection (validated in _resolve_channels)
                sensor_vals = sensor_df[self.channels].values.astype(np.float32)

                length = len(sensor_vals)
                if length < self.window_size:
                    pad_len = self.window_size - length
                    sensor_window = np.pad(sensor_vals, ((0, pad_len), (0, 0)),
                                           mode='constant')
                else:
                    start = (length - self.window_size) // 2
                    sensor_window = sensor_vals[start:start + self.window_size]

                self.samples.append({
                    "sensor_window": sensor_window,
                    "image_path": str(image_path),
                    "label": index_label,
                })
            except Exception:
                continue

    # ------------------------------------------------------------------ #
    # Continuous Mode: sliding window over sensor.csv, aligned to frames
    # ------------------------------------------------------------------ #
    def _load_continuous_mode(self, d_path: Path, stride: int):
        timeline = DriveTimeline.load(d_path)

        try:
            sensor_df = pd.read_csv(d_path / "sensor.csv")
        except Exception:
            return
        sensor_df = sensor_df.rename(columns=_CHANNEL_RENAMES)

        missing = [c for c in self.channels if c not in sensor_df.columns]
        if missing:
            raise ValueError(
                f"{d_path / 'sensor.csv'} is missing required sensor channels "
                f"{missing}; expected ordered channels {self.channels}."
            )

        sensor_vals = sensor_df[self.channels].values.astype(np.float32)
        has_labels = 'label' in sensor_df.columns
        labels = sensor_df['label'].values if has_labels else None
        times = (sensor_df['seconds_elapsed'].values
                 if 'seconds_elapsed' in sensor_df.columns
                 else np.arange(len(sensor_vals)) * (1.0 / SENSOR_RESAMPLE_HZ))

        # Build an index of available frame images sorted by name
        images_dir = d_path / "images"
        if not images_dir.exists():
            return

        frame_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        )
        if not frame_files:
            return

        # Extract numeric time from filenames when possible (e.g. frame_0.123.jpg)
        frame_times = []
        failed_idx = []
        for i, fp in enumerate(frame_files):
            stem = fp.stem
            parts = stem.rsplit("_", 1)
            try:
                frame_times.append(float(parts[-1]))
            except ValueError:
                frame_times.append(None)
                failed_idx.append(i)

        fps, video_duration = self._video_metadata(d_path)

        if failed_idx:
            # Only frames whose OWN timestamp failed to parse get a fallback,
            # derived from video metadata (frame index / fps) — frame times are
            # in VIDEO time, so spacing them over the sensor duration would be
            # the wrong coordinate frame.
            if fps is None or fps <= 0:
                bad = frame_files[failed_idx[0]]
                raise ValueError(
                    f"Cannot determine frame time for '{bad}': the filename "
                    f"timestamp failed to parse and no video metadata (fps) is "
                    f"available in {d_path}. Refusing to fabricate frame times."
                )
            print(f"[HybridDataset] {len(failed_idx)} frame filename(s) in {d_path} "
                  f"lack parseable timestamps; deriving their times from "
                  f"frame_index / fps ({fps:.3f}).")
            for i in failed_idx:
                frame_times[i] = i / fps

        frame_times_arr = np.array(frame_times, dtype=float)

        # Sanity check: a parsed "timestamp" sequence of 0,1,2,... while the
        # video is shorter than len(frames) seconds is a frame INDEX, not
        # seconds. Detect and convert (or raise if fps is unknown).
        if not failed_idx and len(frame_times_arr) > 1:
            integer_like = (
                np.allclose(frame_times_arr, np.round(frame_times_arr))
                and np.allclose(np.diff(np.round(frame_times_arr)), 1.0)
            )
            if integer_like and video_duration is not None and video_duration < len(frame_files):
                if fps and fps > 0:
                    print(f"[HybridDataset] Frame filename 'timestamps' in {d_path} "
                          f"look like frame indices (0,1,2,... over a "
                          f"{video_duration:.1f}s video); converting via index / fps "
                          f"({fps:.3f}).")
                    frame_times_arr = frame_times_arr / fps
                else:
                    raise ValueError(
                        f"Frame filename 'timestamps' in {d_path} look like frame "
                        f"indices (0,1,2,... over a {video_duration:.1f}s video) but "
                        f"fps is unavailable to convert them to seconds."
                    )

        num_windows = (len(sensor_vals) - self.window_size) // stride + 1
        for i in range(num_windows):
            start = i * stride
            end = start + self.window_size
            center_idx = start + self.window_size // 2
            t_center = float(times[center_idx])

            # Map sensor time to video time via timeline
            t_video = timeline.rel_to_video(t_center)
            if t_video < 0:
                continue

            # Find nearest frame
            diffs = np.abs(frame_times_arr - t_video)
            nearest_idx = int(np.argmin(diffs))
            # Only accept if within 0.5s of a real frame
            if diffs[nearest_idx] > 0.5:
                continue

            image_path = frame_files[nearest_idx]
            label = int(np.max(labels[start:end])) if has_labels else -1

            self.samples.append({
                "sensor_window": sensor_vals[start:end],
                "image_path": str(image_path),
                "label": label,
            })

    # ------------------------------------------------------------------ #
    @staticmethod
    def _video_metadata(d_path: Path):
        """Returns (fps, duration_seconds) from the drive's video, or (None, None)."""
        candidates = [d_path / "video.mp4"]
        candidates += sorted(d_path.glob("*.mp4"))
        camera_dir = d_path / "Camera"
        if camera_dir.exists():
            candidates += sorted(camera_dir.glob("*.mp4"))

        for vp in candidates:
            if not vp.exists():
                continue
            cap = cv2.VideoCapture(str(vp))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            finally:
                cap.release()
            if fps and fps > 0:
                duration = (n_frames / fps) if n_frames and n_frames > 0 else None
                return float(fps), duration
        return None, None

    # ------------------------------------------------------------------ #
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Load and preprocess the actual image
        img = cv2.imread(item['image_path'])
        if img is None:
            img_tensor = torch.zeros((3, self.image_size, self.image_size),
                                     dtype=torch.float32)
        else:
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (C, H, W)

        # Sensor tensor — transpose to (Channels, Length) for Conv1d
        sensor_window = item['sensor_window']
        sensor_tensor = torch.from_numpy(sensor_window).float().transpose(1, 0)

        label = torch.tensor(item['label'], dtype=torch.long)
        return img_tensor, sensor_tensor, label
