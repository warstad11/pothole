from app.services.models.base import BaseModel
from app.core.reproducibility import set_global_seed
from app.services.labels import sensor_labels_for_df, infer_label_from_path
from app.services.splits import get_or_create_split, partition_files
from app.services.features import extract_features as compute_window_features
from app.services.features import feature_names, extract_features_array
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Windowing parameters shared with the 1D-CNN pipeline (see sensor/dataset.py).
# 100 samples @ 100 Hz = 1.0 s windows, 0.2 s stride.
WINDOW_SIZE = 100
STRIDE = 20

RESERVED_NAMES = ['train.csv', 'val.csv', 'test.csv', 'metadata.json',
                  'split_manifest.json', 'relabel_manifest.json']


class RandomForestWrapper(BaseModel):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.n_estimators = config.get("n_estimators", 100)
        self.seed = config.get("seed", 42)
        # Explicit random_state: determinism must not depend on a global
        # np.random.seed call happening first.
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            random_state=self.seed)

    # ------------------------------------------------------------------ #
    # Loading & labeling                                                  #
    # ------------------------------------------------------------------ #

    def _scan_files(self, dataset_path: Path):
        files = sorted(list(dataset_path.glob("*.csv")) + list(dataset_path.glob("*.txt")) +
                       list(dataset_path.rglob("*.csv")) + list(dataset_path.rglob("*.txt")))
        # 'labels' dirs hold YOLO annotation txts, not sensor recordings
        return sorted(set(f for f in files if f.is_file()
                          and f.name not in RESERVED_NAMES
                          and 'labels' not in f.parts))

    def _load_file(self, f: Path) -> Optional[pd.DataFrame]:
        """Load one recording and resolve per-row labels via the central
        labels module (embedded label column first, path keywords second).
        Returns None when the file is unlabelable — callers skip it."""
        if f.suffix == '.txt':
            # Headerless extracted windows: col 0 = label
            df = pd.read_csv(f, header=None)
            if df.shape[1] == 4:
                df.columns = ['label', 'accel_x', 'accel_y', 'accel_z']
            else:
                df.rename(columns={0: 'label'}, inplace=True)
        else:
            df = pd.read_csv(f)

        if df.empty:
            return None

        labels = sensor_labels_for_df(f, df)
        if labels is None:
            print(f"Skipping {f.name}: no label column and no path keyword match.")
            return None
        # Rows whose label cell couldn't be parsed are treated as background
        # (0) rather than guessed positive; whole-file ambiguity was already
        # handled by sensor_labels_for_df returning None.
        df['label'] = labels.fillna(0.0)
        return df

    def _file_level_label(self, f: Path) -> Optional[int]:
        """File-level label for split stratification (path keywords only —
        cheap, no CSV read needed for the common datasets)."""
        return infer_label_from_path(f)

    # ------------------------------------------------------------------ #
    # Windowing                                                           #
    # ------------------------------------------------------------------ #

    def _create_windows_and_extract_features(self, df: pd.DataFrame,
                                             window_size=WINDOW_SIZE,
                                             stride=STRIDE) -> (pd.DataFrame, np.array):
        """Converts a raw time-series DF into windowed feature vectors via the
        centralized extractor (app/services/features.py): time-domain stats,
        higher-order moments, jerk, zero-crossing rate, and frequency-domain
        features (spectral energy, dominant frequency, spectral entropy) per
        axis plus the acceleration magnitude.

        Window count formula matches the 1D-CNN dataset exactly:
        n = (len - window_size) // stride + 1.
        """
        if df.empty or 'label' not in df.columns:
            return pd.DataFrame(), np.array([])

        # Strict 3-axis: accelerometer only
        all_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ['label', 'session', 't_center', 'time', 'Time', 'index', 'timestamp']]
        sensor_cols = [c for c in all_cols
                       if any(x in str(c).lower() for x in ['acc', 'ax', 'ay', 'az'])]
        if not sensor_cols:
            # Headerless files have integer column names — mirror dl.py's
            # fallback (first 3 numeric non-label columns) so both sensor
            # models ingest the same data from the same file.
            sensor_cols = all_cols[:3]
        if not sensor_cols:
            return pd.DataFrame(), np.array([])

        # NaN-safe: a single NaN cell must not produce NaN features (which
        # crash RandomForest.fit) — same sanitization as the CNN dataset.
        data_arr = np.nan_to_num(df[sensor_cols[:3]].values.astype(np.float64),
                                 nan=0.0, posinf=1.0, neginf=-1.0)
        labels_arr = pd.to_numeric(df['label'], errors='coerce').fillna(0).values

        num_samples = len(df)
        if num_samples < window_size:
            return pd.DataFrame(), np.array([])

        canonical = feature_names()
        X_windows, y_windows = [], []
        # Inclusive of the final full window — same count as the CNN dataset.
        for i in range(0, num_samples - window_size + 1, stride):
            window_data = data_arr[i: i + window_size]
            window_label = labels_arr[i: i + window_size]

            wdf = pd.DataFrame(window_data[:, :3],
                               columns=['accel_x', 'accel_y', 'accel_z'])
            X_windows.append(compute_window_features(wdf))

            # A window is positive iff it overlaps any pothole-labeled sample.
            y_windows.append(1 if np.nanmax(window_label) >= 0.5 else 0)

        df_out = pd.DataFrame(X_windows)
        if not df_out.empty:
            # Canonical column order (must match predict_embedding / engine)
            df_out = df_out[[c for c in canonical if c in df_out.columns]]

        return df_out, np.array(y_windows)

    def _process_files(self, file_list):
        X_list, y_list = [], []
        for f in file_list:
            try:
                df = self._load_file(f)
                if df is None:
                    continue
                X_w, y_w = self._create_windows_and_extract_features(df)
                if not X_w.empty:
                    X_list.append(X_w)
                    y_list.append(y_w)
            except Exception as e:
                print(f"Error processing {f.name}: {e}")
        if not X_list:
            return pd.DataFrame(), np.array([])
        return pd.concat(X_list, ignore_index=True), np.concatenate(y_list)

    # ------------------------------------------------------------------ #
    # Train / evaluate                                                    #
    # ------------------------------------------------------------------ #

    def train(self, dataset_path: Path, epochs: int = 0):
        set_global_seed(self.seed)
        print(f"Training Random Forest on {dataset_path}...")

        all_files = self._scan_files(dataset_path)

        if not all_files:
            # Single train.csv pool: contiguous TIME-BLOCK split (random
            # window-level splits leak — adjacent windows share 80% of their
            # samples). A gap of one window between blocks prevents overlap.
            df = pd.read_csv(dataset_path / "train.csv") if (dataset_path / "train.csv").exists() else pd.DataFrame()
            if df.empty:
                print("No data found.")
                return
            labels = sensor_labels_for_df(dataset_path / "train.csv", df)
            if labels is None:
                print("train.csv has no usable labels; aborting.")
                return
            df['label'] = labels.fillna(0.0)
            n = len(df)
            # Each of val/test must hold at least one full window after the
            # decorrelation gap; refuse tiny datasets up front instead of
            # training "successfully" and failing at evaluate().
            if int(0.15 * n) - WINDOW_SIZE < WINDOW_SIZE:
                raise RuntimeError(
                    f"train.csv has only {n} rows — too small for a "
                    f"70/15/15 time-block split with {WINDOW_SIZE}-sample "
                    f"windows (need >= {int(WINDOW_SIZE * 2 / 0.15) + 1}).")
            i_train = int(0.70 * n)
            i_val = int(0.85 * n)
            parts = {
                'train': df.iloc[:i_train],
                'val': df.iloc[i_train + WINDOW_SIZE: i_val],
                'test': df.iloc[i_val + WINDOW_SIZE:],
            }
            self.X_train, self.y_train = self._create_windows_and_extract_features(parts['train'])
            self.X_val, self.y_val = self._create_windows_and_extract_features(parts['val'])
            X_test, y_test = self._create_windows_and_extract_features(parts['test'])
            self.test_data_cache = (X_test, y_test)
            self.split_info = {"strategy": "time-block 70/15/15 (single file)", "seed": self.seed}
            self.model.fit(self.X_train, self.y_train)
            print(f"Model fitted on {len(self.y_train)} windows (time-block split).")
            return

        # File-level split shared with every other pipeline stage.
        manifest = get_or_create_split(dataset_path, all_files, seed=self.seed,
                                       label_fn=self._file_level_label)
        parts = partition_files(all_files, manifest)
        print(f"Split (shared manifest): {len(parts['train'])} train / "
              f"{len(parts['val'])} val / {len(parts['test'])} test files.")

        print("Processing Train Files...")
        self.X_train, self.y_train = self._process_files(parts['train'])
        print("Processing Val Files...")
        self.X_val, self.y_val = self._process_files(parts['val'])
        print("Processing Test Files...")
        X_test, y_test = self._process_files(parts['test'])
        self.test_data_cache = (X_test, y_test)
        self.split_info = {
            "strategy": manifest.get("strategy", "file-level"),
            "seed": manifest.get("seed", self.seed),
            "manifest": str(Path(dataset_path) / "split_manifest.json"),
        }

        if self.X_train.empty:
            # Refuse to silently fall back to a leaky split. Empty train
            # after a file split means the dataset itself is unusable.
            raise RuntimeError(
                f"Train partition produced no windows for {dataset_path}; "
                f"refusing to fall back to a window-level random split (leaky).")

        print(f"Train Windows: {len(self.y_train)} | Val: {len(self.y_val)} | Test: {len(y_test)}")
        self.model.fit(self.X_train, self.y_train)

    def evaluate(self, dataset_path: Path) -> Dict[str, Any]:
        """Reports metrics on the held-out TEST partition (never used for
        fitting or selection)."""
        if hasattr(self, 'test_data_cache'):
            print("Using held-out test split from training...")
            X_test, y_test = self.test_data_cache
            split_info = getattr(self, 'split_info', {})
        else:
            # Standalone evaluation of a loaded model: re-derive the test
            # partition from the persisted manifest (deterministic).
            all_files = self._scan_files(dataset_path)
            if not all_files and (dataset_path / "train.csv").exists():
                # Single-file dataset: mirror train()'s deterministic
                # time-block split and evaluate its test block.
                df = pd.read_csv(dataset_path / "train.csv")
                labels = sensor_labels_for_df(dataset_path / "train.csv", df)
                if labels is None:
                    return {"accuracy": None, "error": "train.csv has no usable labels"}
                df['label'] = labels.fillna(0.0)
                n = len(df)
                X_test, y_test = self._create_windows_and_extract_features(
                    df.iloc[int(0.85 * n) + WINDOW_SIZE:])
                self.test_data_cache = (X_test, y_test)
                self.split_info = {"strategy": "time-block 70/15/15 (single file)",
                                   "seed": self.seed}
                return self.evaluate(dataset_path)
            if not all_files:
                print("No files to evaluate.")
                return {"accuracy": None, "error": "no data"}
            manifest = get_or_create_split(dataset_path, all_files, seed=self.seed,
                                           label_fn=self._file_level_label)
            parts = partition_files(all_files, manifest)
            print(f"Evaluating on {len(parts['test'])} manifest test files...")
            X_test, y_test = self._process_files(parts['test'])
            split_info = {"strategy": manifest.get("strategy"),
                          "seed": manifest.get("seed"),
                          "manifest": str(Path(dataset_path) / "split_manifest.json")}

        if len(y_test) == 0:
            return {"accuracy": None, "error": "empty test partition"}

        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, roc_curve, auc)

        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        if len(set(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            roc_auc = float(auc(fpr, tpr))
            if len(roc_list) > 100:
                indices = np.linspace(0, len(roc_list) - 1, 100).astype(int)
                roc_list = [roc_list[i] for i in indices]
        else:
            # ROC is undefined for a single-class test set — report None,
            # never a fabricated diagonal.
            roc_list = None
            roc_auc = None
            print("WARNING: test partition has a single class; ROC undefined.")

        metrics = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm.tolist(),
            "roc_curve": roc_list,
            "roc_auc": roc_auc,
            "samples": int(len(y_test)),
            "eval_split": "test",
            "split": split_info,
            "seed": self.seed,
        }

        import json
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    # ------------------------------------------------------------------ #
    # Prediction / feature extraction                                     #
    # ------------------------------------------------------------------ #

    def predict(self, input_data: Any) -> Any:
        # Expects feature dataframe
        return self.model.predict(input_data)

    def predict_embedding(self, window_data: np.ndarray) -> np.ndarray:
        """
        Extracts the canonical feature vector from a raw window — the SAME
        features, in the SAME order, the model trains on (the inference
        engine feeds this straight into predict_proba).
        Input: window_data (N, >=3) - Accel X, Y, Z (extra channels ignored)
        Output: (len(feature_names()),) vector
        """
        window_data = np.asarray(window_data, dtype=np.float64)
        if window_data.ndim != 2 or window_data.shape[1] < 3:
            raise ValueError(
                f"predict_embedding needs (N, >=3) accel channels, got {window_data.shape}")
        window_data = np.nan_to_num(window_data, nan=0.0, posinf=1.0, neginf=-1.0)
        return extract_features_array(window_data)

    def extract_features(self, dataset_path: Path):
        """Extracts one fusion feature per FILE: the max window probability
        under this (already-trained) model.

        NOTE for downstream consumers: these are in-sample for files the
        model trained on. The fusion stage MUST assign train/val/test by the
        shared split manifest so its reported metrics come only from files
        this model never saw.
        """
        files = self._scan_files(dataset_path)
        if not files:
            print("No sensor files found for extraction.")
            return None, None, []

        print(f"Extracting features from {len(files)} files (Aggregating per file)...")

        agg_features, agg_labels, agg_stems = [], [], []
        skipped = 0

        for f in files:
            try:
                df = self._load_file(f)
                if df is None:
                    skipped += 1
                    continue

                # File-level label: a drive containing any pothole-labeled
                # sample is a pothole file.
                label = int(pd.to_numeric(df['label'], errors='coerce').fillna(0).max() >= 0.5)

                X_win, _ = self._create_windows_and_extract_features(df)
                if X_win.empty:
                    # Too short to form a single window — skip it entirely
                    # rather than fabricating a 0.0-probability feature.
                    skipped += 1
                    continue

                probs = self.model.predict_proba(X_win)[:, 1]
                file_prob = float(np.max(probs))

                # Atomic append: all three lists stay aligned.
                agg_features.append([file_prob])
                agg_labels.append(label)
                agg_stems.append(f.stem)

            except Exception as e:
                print(f"Failed to process {f}: {e}")
                skipped += 1

        if skipped:
            print(f"extract_features: skipped {skipped}/{len(files)} files "
                  f"(unlabelable, empty, or too short).")
        assert len(agg_features) == len(agg_labels) == len(agg_stems), \
            "feature/label/stem misalignment"

        return (np.array(agg_features, dtype=np.float32),
                np.array(agg_labels, dtype=np.int64),
                agg_stems)

    def extract_scores(self, dataset_path: Path):
        """One pothole probability per file for the late-fusion benchmark.

        Identical to extract_features (whose single feature already IS the
        max window probability), flattened to a 1-D score array.
        """
        feats, labels, stems = self.extract_features(dataset_path)
        if feats is None:
            return None, None, []
        return feats.reshape(-1).astype(np.float32), labels, stems

    def proba(self, input_data: Any) -> Any:
        return self.model.predict_proba(input_data)

    def save(self):
        joblib.dump(self.model, self.output_dir / "model.joblib")

    def load(self, weights_path: Path):
        self.model = joblib.load(weights_path)
