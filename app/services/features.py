import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from typing import Dict, List


def extract_features(window: pd.DataFrame) -> Dict[str, float]:
    """Compute time-domain and frequency-domain features for a sensor window.

    Expected columns: accel_x, accel_y, accel_z.
    Automatically adds accel_mag (acceleration magnitude) if absent.
    """
    features: Dict[str, float] = {}

    # Only accelerometer axes are available in the sensor hardware
    axes = ['accel_x', 'accel_y', 'accel_z']

    if 'accel_mag' not in window.columns:
        window = window.copy()
        window['accel_mag'] = np.sqrt(
            window['accel_x']**2 + window['accel_y']**2 + window['accel_z']**2
        )
    axes.append('accel_mag')

    for axis in axes:
        if axis not in window.columns:
            continue

        data = window[axis].values.astype(np.float64)
        n = len(data)
        if n == 0:
            continue

        # ---- Time domain (basic) ----
        features[f'{axis}_mean'] = float(np.mean(data))
        features[f'{axis}_std'] = float(np.std(data))
        features[f'{axis}_max'] = float(np.max(data))
        features[f'{axis}_min'] = float(np.min(data))
        features[f'{axis}_range'] = float(np.ptp(data))
        features[f'{axis}_rms'] = float(np.sqrt(np.mean(data**2)))
        features[f'{axis}_median'] = float(np.median(data))

        # ---- Time domain (higher-order statistics) ----
        features[f'{axis}_kurtosis'] = float(sp_stats.kurtosis(data, fisher=True))
        features[f'{axis}_skewness'] = float(sp_stats.skew(data))

        # Zero-crossing rate
        zero_crossings = np.sum(np.diff(np.sign(data - np.mean(data))) != 0)
        features[f'{axis}_zcr'] = float(zero_crossings / max(n - 1, 1))

        # Jerk (first derivative) statistics
        if n > 1:
            jerk = np.diff(data)
            features[f'{axis}_jerk_mean'] = float(np.mean(np.abs(jerk)))
            features[f'{axis}_jerk_std'] = float(np.std(jerk))
            features[f'{axis}_jerk_max'] = float(np.max(np.abs(jerk)))
        else:
            features[f'{axis}_jerk_mean'] = 0.0
            features[f'{axis}_jerk_std'] = 0.0
            features[f'{axis}_jerk_max'] = 0.0

        # ---- Frequency domain ----
        fft_vals = np.fft.rfft(data)
        fft_mag = np.abs(fft_vals)

        # Total spectral energy
        fft_energy = float(np.sum(fft_mag**2) / n)
        features[f'{axis}_energy'] = fft_energy

        # Dominant frequency (index of peak magnitude, excluding DC)
        if len(fft_mag) > 1:
            dom_idx = int(np.argmax(fft_mag[1:])) + 1
            features[f'{axis}_dom_freq_idx'] = float(dom_idx)
            features[f'{axis}_dom_freq_mag'] = float(fft_mag[dom_idx])
        else:
            features[f'{axis}_dom_freq_idx'] = 0.0
            features[f'{axis}_dom_freq_mag'] = 0.0

        # Spectral entropy
        psd = fft_mag**2
        psd_sum = psd.sum()
        if psd_sum > 0:
            psd_norm = psd / psd_sum
            psd_norm = psd_norm[psd_norm > 0]
            features[f'{axis}_spectral_entropy'] = float(-np.sum(psd_norm * np.log2(psd_norm)))
        else:
            features[f'{axis}_spectral_entropy'] = 0.0

    return features


def extract_features_batch(windows: List[pd.DataFrame]) -> pd.DataFrame:
    """Process a list of windows into a feature matrix."""
    return pd.DataFrame([extract_features(w) for w in windows])


def feature_names() -> List[str]:
    """Canonical, deterministic feature column order.

    Training (classical.py), fusion embeddings (predict_embedding), and the
    inference engine all order columns by this list, so a model trained on
    these features can never be fed a permuted vector at deployment.
    """
    dummy = pd.DataFrame({
        'accel_x': np.zeros(8), 'accel_y': np.zeros(8), 'accel_z': np.zeros(8),
    })
    return list(extract_features(dummy).keys())


def extract_features_array(window: np.ndarray) -> np.ndarray:
    """Feature vector (canonical order) from a raw (N, 3) accel window."""
    df = pd.DataFrame(np.asarray(window)[:, :3],
                      columns=['accel_x', 'accel_y', 'accel_z'])
    feats = extract_features(df)
    return np.array([feats[k] for k in feature_names()], dtype=np.float32)
