import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

class SensorWindowDataset(Dataset):
    def __init__(self, drives: List[pd.DataFrame], window_size: int = 100, stride: int = 20,
                 normalize: bool = True,
                 norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        """
        drives: List of dataframes, each representing a continuous drive.
                Expected cols: [...sensors..., 'label'] (if training)
        window_size: Number of samples (e.g. 100 @ 100Hz = 1.0s)
        stride: Step size (e.g. 20 @ 100Hz = 0.2s)
        norm_stats: (mean, std) per channel, computed on the TRAIN partition
                and persisted with the checkpoint. Preferred: it preserves
                amplitude information (the most discriminative cue between
                potholes and milder anomalies) and is identical at training
                and deployment time.
        normalize: fallback per-window, per-channel z-score used only when
                no norm_stats are provided (legacy models). Per-window
                scaling erases amplitude — see METHODOLOGY.md.
        """
        self.samples = []
        self.labels = []
        self.normalize = normalize
        self.norm_stats = norm_stats

        for drive in drives:
            # A window is positive iff it overlaps any pothole-labeled sample.
            has_labels = 'label' in drive.columns
            values = drive.drop(columns=['label', 'time', 'seconds_elapsed'], errors='ignore').values.astype(np.float32)

            num_windows = (len(values) - window_size) // stride + 1

            for i in range(num_windows):
                start = i * stride
                end = start + window_size
                window = values[start:end]

                self.samples.append(window)

                if has_labels:
                    window_labels = pd.to_numeric(
                        pd.Series(drive['label'].values[start:end]), errors='coerce'
                    ).fillna(0).values
                    self.labels.append(int(np.max(window_labels) >= 0.5))
                else:
                    self.labels.append(-1) # Unknown

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # PyTorch Conv1d expects (Batch, Channels, Time)
        x = self.samples[idx].copy() # (Time, Channels)
        y = self.labels[idx]

        # Sanitize BEFORE normalizing: a single NaN cell must not poison the
        # window statistics (NaN std would zero the whole channel).
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        if self.norm_stats is not None:
            mean, std = self.norm_stats
            std_safe = np.where(std > 1e-6, std, 1.0)
            x = (x - mean) / std_safe
        elif self.normalize:
            for c in range(x.shape[1]):
                std = x[:, c].std()
                if std > 1e-6:
                    x[:, c] = (x[:, c] - x[:, c].mean()) / std
                else:
                    x[:, c] = 0.0

        # Transpose for PyTorch Conv1d: (C, L)
        x = x.transpose(1, 0)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
