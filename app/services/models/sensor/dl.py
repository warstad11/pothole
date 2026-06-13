from app.services.models.base import BaseModel
from app.core.reproducibility import set_global_seed, get_torch_generator
from app.services.labels import sensor_labels_for_df, infer_label_from_path
from app.services.splits import get_or_create_split, partition_files
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from .dataset import SensorWindowDataset
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

class Simple1DCNN(nn.Module):
    def __init__(self, input_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

        # Backward compatibility flag
        # If weights are loaded without BN keys, we disable BN usage
        self.use_bn = True

    def forward(self, x):
        # x: (Batch, Channels, Time)
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = self.pool(self.relu(x))

        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.pool(self.relu(x))

        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

    def get_embedding(self, x):
        # Return 128-dim vector before FC
        x = self.conv1(x)
        if self.use_bn: x = self.bn1(x)
        x = self.pool(self.relu(x))

        x = self.conv2(x)
        if self.use_bn: x = self.bn2(x)
        x = self.pool(self.relu(x))

        return self.global_pool(x).squeeze(-1)

class SimpleLSTM(nn.Module):
    """Bidirectional LSTM over raw sensor windows.

    Input (B, C, T) — same layout the dataset emits for the CNN — is
    transposed to (B, T, C) internally. The mean-pooled hidden states give a
    128-dim embedding (64 hidden x 2 directions), matching the CNN's
    embedding size so fusion treats both identically.
    """
    def __init__(self, input_channels, num_classes, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=0.3 if layers > 1 else 0.0,
                            bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # (B, C, T) -> (B, T, C)
        out, _ = self.lstm(x.transpose(1, 2))
        return self.fc(out.mean(dim=1))

    def get_embedding(self, x):
        out, _ = self.lstm(x.transpose(1, 2))
        return out.mean(dim=1)  # (B, 128)


class CNN1DWrapper(BaseModel):
    """Sensor sequence model wrapper. config['arch'] selects the network:
    'cnn' (default, Simple1DCNN) or 'lstm' (SimpleLSTM). Everything else —
    windowing, normalization, splits, evaluation protocol — is identical, so
    CNN-vs-LSTM comparisons differ only in architecture."""
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.input_channels = config.get("input_channels", 3) # Default 3: accel_x/y/z
        self.num_classes = config.get("num_classes", 2)
        self.seed = config.get("seed", 42)
        self.arch = config.get("arch", "cnn").lower()

        if self.arch == "lstm":
            self.model = SimpleLSTM(self.input_channels, self.num_classes)
        else:
            self.model = Simple1DCNN(self.input_channels, self.num_classes)
        # Central device selection (cuda > mps > cpu; POTHOLE_DEVICE
        # overrides). The BatchNorm consistency issue that forced CPU no
        # longer reproduces on torch 2.9 — verified by
        # tools/benchmark_device.py (1D-CNN: ~5x faster on Apple MPS).
        from app.core.device import resolve_device
        self.device = resolve_device("cnn1d")

        self.model.to(self.device)
        self.window_size = config.get("window_size", 100)
        self.stride = config.get("stride", 20)
        # Per-channel (mean, std) computed on the TRAIN partition and saved
        # beside the checkpoint; preserves amplitude information.
        self.norm_stats = None

    def _load_data(self, dataset_path: Path, return_paths: bool = False):
        """Load sensor CSVs. If return_paths=True, returns List[Tuple[Path, DataFrame]].

        Labels are resolved through app.services.labels.sensor_labels_for_df:
        an embedded label column wins (per-row, time-localized), path keywords
        are the fallback, and unlabelable files are skipped. There is NO
        filename override of embedded labels — the sensor datasets were
        cleaned (tools/relabel_sensor_data.py) so the columns are
        authoritative, and the old override mislabeled e.g. 'Undamaged_*'
        files as potholes ('damage' substring).
        """
        print(f"Loading sensor data from {dataset_path}")
        dfs = []
        if not dataset_path.exists():
            return dfs

        csv_files = sorted(list(dataset_path.glob("*.csv")))
        # Handle recursive or split files (train/val)
        if len(csv_files) == 0:
            # Avoid picking up YOLO labels in 'labels' directory
            all_files = list(dataset_path.rglob("*.csv")) + list(dataset_path.rglob("*.txt"))
            csv_files = sorted([f for f in all_files if 'labels' not in f.parts and f.stat().st_size > 0])
        csv_files = [f for f in csv_files if f.name not in
                     ('split_manifest.json', 'relabel_manifest.json')]

        for f in csv_files:
            try:
                try:
                    df = pd.read_csv(f)
                except (pd.errors.ParserError, UnicodeDecodeError):
                     # Fallback to whitespace delimiter
                    df = pd.read_csv(f, sep=r'\s+', engine='python')

                # Headerless heuristic: numeric column names mean the first
                # row was data
                try:
                    float(df.columns[0])
                    try:
                         df = pd.read_csv(f, header=None)
                    except (pd.errors.ParserError, UnicodeDecodeError):
                         df = pd.read_csv(f, sep=r'\s+', header=None, engine='python')
                except ValueError:
                    pass # Columns are strings, likely valid header

                if df.empty:
                    continue

                # Map a recognizable label column to the canonical 'label' name
                label_col = None
                possible_labels = ['label', 'class', 'target', 'accident', 'pothole']
                if isinstance(df.columns[0], int):
                     # Headerless: low-cardinality first column = label
                     if df[0].nunique() <= 5:
                         label_col = 0
                else:
                    for c in df.columns:
                        if str(c).lower() in possible_labels:
                            label_col = c
                            break
                if label_col is None:
                     obj_cols = df.select_dtypes(include=['object']).columns
                     for c in obj_cols:
                         if df[c].nunique() < 10:
                             label_col = c
                             break
                if label_col is not None and label_col != 'label':
                    df = df.rename(columns={label_col: 'label'})

                # Centralized label resolution (column first, path second)
                labels = sensor_labels_for_df(f, df)
                if labels is None:
                    print(f"Warning: No label found in {f.name}; skipping.")
                    continue
                resolved_labels = labels.fillna(0.0)
                df = df.drop(columns=['label'], errors='ignore')

                # Select Numeric Feature Columns
                exclude = ['label', 'time', 'timestamp', 'seconds_elapsed', 'id', 'index']
                feature_df = df.select_dtypes(include=[np.number])
                valid_cols = [c for c in feature_df.columns if str(c).lower() not in exclude]

                # STRICT 3-AXIS ACCELEROMETER ONLY enforcement
                acc_cols = []
                for c in valid_cols:
                    lower_c = str(c).lower()
                    if any(x in lower_c for x in ['acc', 'ax', 'ay', 'az']):
                        acc_cols.append(c)

                # specific sort for axis alignment order: x, y, z
                def axis_sort(cols):
                    x_c = [c for c in cols if 'x' in str(c).lower()]
                    y_c = [c for c in cols if 'y' in str(c).lower()]
                    z_c = [c for c in cols if 'z' in str(c).lower()]
                    if len(x_c)==1 and len(y_c)==1 and len(z_c)==1:
                        return x_c + y_c + z_c
                    if set(cols) == {'Ax', 'Ay', 'Az'}:
                        return ['Ax', 'Ay', 'Az']
                    return sorted(cols) # default sort

                acc_cols = axis_sort(acc_cols)

                # Force strictly 3 Accel cols
                if len(acc_cols) >= 3:
                     selected_cols = acc_cols[:3]
                elif len(acc_cols) > 0:
                     selected_cols = acc_cols # Pad below
                elif len(valid_cols) >= 3:
                    selected_cols = valid_cols[:3]
                else:
                    selected_cols = valid_cols

                if not selected_cols:
                    print(f"Warning: no sensor channels in {f.name}; skipping.")
                    continue

                final_df = df[selected_cols].copy()

                # NOTE: no per-file normalization here. Windows are z-scored
                # per-window inside SensorWindowDataset, matching exactly what
                # predict()/predict_embedding() do at inference time.

                final_df['label'] = resolved_labels.values

                # Pad if missing channels
                if len(selected_cols) < self.input_channels:
                    for i in range(self.input_channels - len(selected_cols)):
                        final_df[f"pad_{i}"] = 0.0

                dfs.append((f, final_df) if return_paths else final_df)
            except (pd.errors.EmptyDataError, IndexError):
                print(f"Skipping empty/invalid file {f}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"Error loading {f}: {e}")
        return dfs

    def train(self, dataset_path: Path, epochs: int = 50):
        set_global_seed(self.seed)

        drives_with_paths = self._load_data(dataset_path, return_paths=True)
        if len(drives_with_paths) == 0:
            print("No data found!")
            return {"status": "failed", "reason": "No CSV data found"}

        # Three-way file-level split from the shared manifest. The same
        # manifest is used by the RF, image, and fusion stages, so the test
        # partition is untouched by training AND checkpoint selection in
        # every pipeline. label_fn keeps the split STRATIFIED regardless of
        # which wrapper creates the manifest first.
        files = [p for p, _ in drives_with_paths]
        df_by_stem = {p.stem: d for p, d in drives_with_paths}
        manifest = get_or_create_split(dataset_path, files, seed=self.seed,
                                       label_fn=infer_label_from_path)
        parts = partition_files(files, manifest)

        train_drives = [df_by_stem[p.stem] for p in parts['train']]
        val_drives = [df_by_stem[p.stem] for p in parts['val']]
        test_drives = [df_by_stem[p.stem] for p in parts['test']]
        print(f"Data Split (shared manifest): {len(train_drives)} train / "
              f"{len(val_drives)} val / {len(test_drives)} test drives")

        # Per-channel normalization stats from the TRAIN partition ONLY,
        # persisted with the checkpoint and reused at inference. Global stats
        # preserve amplitude (a pothole jolt and gentle texture differ mainly
        # in amplitude); per-window z-scoring would erase it.
        train_values = [d.drop(columns=['label', 'time', 'seconds_elapsed'],
                               errors='ignore').values.astype(np.float32)
                        for d in train_drives]
        train_values = [v for v in train_values if len(v) > 0]
        if not train_values:
            print("Train Dataset empty after processing.")
            return {"status": "failed", "reason": "empty train partition"}
        all_train = np.nan_to_num(np.concatenate(train_values, axis=0),
                                  nan=0.0, posinf=1.0, neginf=-1.0)
        self.norm_stats = (all_train.mean(axis=0), all_train.std(axis=0))
        with open(self.output_dir / "norm_stats.json", "w") as f:
            json.dump({"mean": self.norm_stats[0].tolist(),
                       "std": self.norm_stats[1].tolist(),
                       "computed_on": "train partition"}, f, indent=2)

        train_ds = SensorWindowDataset(train_drives, window_size=self.window_size,
                                       stride=self.stride, norm_stats=self.norm_stats)
        val_ds = SensorWindowDataset(val_drives, window_size=self.window_size,
                                     stride=self.stride, norm_stats=self.norm_stats)
        test_ds = SensorWindowDataset(test_drives, window_size=self.window_size,
                                      stride=self.stride, norm_stats=self.norm_stats)

        if len(train_ds) == 0:
            print("Train Dataset empty after processing.")
            return {"status": "failed", "reason": "no train windows"}
        if len(val_ds) == 0:
            # Without val windows checkpoint selection is impossible and the
            # best-checkpoint reload below would risk reusing stale weights.
            return {"status": "failed", "reason": "empty val partition (no windows)"}

        # drop_last guards BatchNorm against a final train batch of size 1
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                  generator=get_torch_generator(self.seed),
                                  drop_last=len(train_ds) > 32)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

        # evaluate() reports on the held-out TEST partition only
        self.test_loader_cache = DataLoader(test_ds, batch_size=32, shuffle=False)
        self.split_info = {
            "strategy": manifest.get("strategy", "file-level"),
            "seed": manifest.get("seed", self.seed),
            "manifest": str(Path(dataset_path) / "split_manifest.json"),
        }

        # A stale checkpoint from a previous run in the same output_dir must
        # never be confused with this run's best checkpoint.
        stale_ckpt = self.output_dir / "model.pth"
        if stale_ckpt.exists():
            stale_ckpt.unlink()

        # Class weight from the TRAIN windows (the fixed [1, 2] guess badly
        # underweights potholes now that labels are pothole-strict).
        train_labels = np.array(train_ds.labels)
        pos = int((train_labels == 1).sum())
        neg = int((train_labels == 0).sum())
        if pos > 0 and neg > 0:
            weight = torch.tensor([1.0, neg / pos], dtype=torch.float32).to(self.device)
            print(f"Class weights from train windows: neg={neg}, pos={pos}, w_pos={neg/pos:.1f}")
        else:
            weight = None
            print(f"WARNING: single-class train partition (neg={neg}, pos={pos}).")
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        # Checkpoint selection: val F1 (positive class), accuracy tie-break.
        # Raw accuracy is unusable on imbalanced data — on Data 3 (96.7%
        # negative) an all-negative epoch scores 0.967 and outranks every
        # genuine attempt, so accuracy-selection systematically picks
        # majority collapse. F1 selection matches the fusion operating-point
        # protocol (val F1). Keys start at -1 so the first epoch always
        # checkpoints — THIS run's weights must persist, never none on disk.
        best_key = (-1.0, -1.0)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()

            # Validation (checkpoint selection ONLY — never reported)
            self.model.eval()
            correct = 0
            total = 0
            tp = fp = fn = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    tp += ((predicted == 1) & (labels == 1)).sum().item()
                    fp += ((predicted == 1) & (labels == 0)).sum().item()
                    fn += ((predicted == 0) & (labels == 1)).sum().item()

            val_acc = correct / total if total > 0 else 0
            val_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}, "
                  f"Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            if (val_f1, val_acc) > best_key:
                best_key = (val_f1, val_acc)
                self.save() # Checkpoint

        # The saved checkpoint (best val F1) is the model of record —
        # reload it so evaluate() measures the same weights that save()
        # persisted, not the final-epoch weights.
        best_ckpt = self.output_dir / "model.pth"
        if best_ckpt.exists():
            self.load(best_ckpt)

        return {"val_f1": best_key[0], "val_acc": best_key[1],
                "checkpoint_selection": "val_f1_then_acc"}

    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        """Reports metrics on the held-out TEST partition."""
        self.model.eval()

        if hasattr(self, 'test_loader_cache'):
            print("Using held-out test loader from training...")
            loader = self.test_loader_cache
            split_info = getattr(self, 'split_info', {})
        else:
            print("Re-deriving test partition from the split manifest...")
            drives_with_paths = self._load_data(dataset_path, return_paths=True)
            if not drives_with_paths:
                 return {"accuracy": None, "error": "no data"}
            files = [p for p, _ in drives_with_paths]
            df_by_stem = {p.stem: d for p, d in drives_with_paths}
            manifest = get_or_create_split(dataset_path, files, seed=self.seed,
                                           label_fn=infer_label_from_path)
            parts = partition_files(files, manifest)
            test_drives = [df_by_stem[p.stem] for p in parts['test']]
            test_ds = SensorWindowDataset(test_drives, window_size=self.window_size,
                                          stride=self.stride, norm_stats=self.norm_stats)
            if len(test_ds) == 0:
                 return {"accuracy": None, "error": "empty test partition"}
            loader = DataLoader(test_ds, batch_size=32, shuffle=False)
            split_info = {"strategy": manifest.get("strategy"),
                          "seed": manifest.get("seed"),
                          "manifest": str(Path(dataset_path) / "split_manifest.json")}

        all_preds = []
        all_labels = []
        all_probs = []

        # Evaluation runs on the configured device — CPU/MPS parity for this
        # architecture is verified by tools/benchmark_device.py (force CPU
        # globally with POTHOLE_DEVICE=cpu if a torch regression appears).
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        if not all_labels:
            return {"accuracy": None, "error": "no samples"}

        rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

        pothole_metrics = rep.get('1', rep.get('1.0', {}))

        # ROC from probability scores. Undefined (single-class) -> None,
        # never a fabricated diagonal.
        roc_list = None
        roc_auc = None
        if len(np.unique(all_labels)) >= 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_auc = float(auc(fpr, tpr))
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            roc_list = [[0.0 if np.isnan(x) else x for x in point] for point in roc_list]
        else:
            print("WARNING: test partition has a single class; ROC undefined.")

        def clean(val):
            if val is None or isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return 0.0
            return float(val)

        metrics = {
            "accuracy": clean(rep['accuracy']),
            "precision": clean(pothole_metrics.get('precision', 0.0)),
            "recall": clean(pothole_metrics.get('recall', 0.0)),
            "f1": clean(pothole_metrics.get('f1-score', 0.0)),
            "confusion_matrix": cm.tolist(),
            "roc_curve": roc_list,
            "roc_auc": roc_auc,
            "samples": len(all_labels),
            "eval_split": "test",
            "split": split_info,
            "seed": self.seed,
            "device": str(self.device),
        }

        # Save metrics.json
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def predict(self, input_data: Any) -> Any:
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, torch.Tensor):
               # For tensor, assume it's already normalized if coming from a batch loader,
               # but if it's a raw frame, we might need a stats-based norm.
               # Let's assume raw frame needs locally-calculated stats if it's (C, L)
               inputs = input_data.to(self.device).unsqueeze(0) if input_data.dim() == 2 else input_data.to(self.device)
            else:
               # Assume numpy (Time, Channels) -> (1, C, T)
               data = np.array(input_data, dtype=np.float32)
               data = self._normalize_window(data)
               inputs = torch.tensor(data).transpose(1,0).unsqueeze(0).to(self.device)

            outputs = self.model(inputs)
            probs = torch.softmax(outputs, dim=1)
            return probs.cpu().numpy()

    def _normalize_window(self, data: np.ndarray) -> np.ndarray:
        """Apply the SAME normalization used in training: train-partition
        global stats when available, per-window z-score as legacy fallback."""
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
        if self.norm_stats is not None:
            mean, std = self.norm_stats
            n = min(data.shape[1], len(mean))
            std_safe = np.where(std[:n] > 1e-6, std[:n], 1.0)
            data[:, :n] = (data[:, :n] - mean[:n]) / std_safe
            return data
        for i in range(data.shape[1]):
            mean = data[:, i].mean()
            std = data[:, i].std()
            if std > 1e-6:
                data[:, i] = (data[:, i] - mean) / std
            else:
                data[:, i] = 0.0
        return data

    def save(self):
        torch.save(self.model.state_dict(), self.output_dir / "model.pth")

    def load(self, weights_path: Path):
        print(f"Loading 1D-CNN from {weights_path}")
        # Restore the training-time normalization stats saved beside the
        # checkpoint; without them inference falls back to per-window
        # z-scoring (legacy models only).
        stats_path = Path(weights_path).parent / "norm_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                s = json.load(f)
            self.norm_stats = (np.array(s["mean"], dtype=np.float32),
                               np.array(s["std"], dtype=np.float32))
        else:
            print("WARNING: no norm_stats.json beside checkpoint — falling back "
                  "to per-window normalization (legacy model).")
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            # Architecture sanity: an LSTM checkpoint must not be silently
            # shoehorned into the CNN (or vice versa).
            is_lstm_ckpt = any(k.startswith("lstm.") for k in state_dict)
            if is_lstm_ckpt != (self.arch == "lstm"):
                raise ValueError(
                    f"checkpoint architecture ({'lstm' if is_lstm_ckpt else 'cnn'}) "
                    f"does not match wrapper arch '{self.arch}'")
            if self.arch != "lstm":
                # Check for bn keys to set backward compatibility
                has_bn = any("bn" in k for k in state_dict.keys())
                self.model.use_bn = has_bn
                if not has_bn:
                    print("WARNING: Loading legacy model without BatchNorm layers. Disabling BN.")

            self.model.load_state_dict(state_dict, strict=False)
            self.model.eval()
        except Exception as e:
            print(f"Error loading {weights_path}: {e}")
            raise e

    def predict_embedding(self, input_data: Any) -> np.ndarray:
        """
        Returns 128-dim embedding for single window or batch.
        """
        self.model.eval()
        with torch.no_grad():
             if isinstance(input_data, torch.Tensor):
                 inputs = input_data.to(self.device)
                 if inputs.dim() == 2: inputs = inputs.unsqueeze(0)
             else:
                 # Same normalization as predict()/training
                 data = np.array(input_data, dtype=np.float32)
                 if data.shape[1] < self.input_channels: # Pad
                     pad = np.zeros((data.shape[0], self.input_channels - data.shape[1]))
                     data = np.concatenate([data, pad], axis=1).astype(np.float32)
                 data = self._normalize_window(data)
                 inputs = torch.tensor(data).transpose(1,0).unsqueeze(0).to(self.device)

             emb = self.model.get_embedding(inputs)
             # Return flatten numpy
             return emb.cpu().numpy().flatten()

    def extract_scores(self, dataset_path: Path):
        """One pothole probability per file for the late-fusion benchmark:
        max softmax P(pothole) over the file's windows. Same skipping rules
        as extract_features. Returns (scores, labels, stems)."""
        self.model.eval()
        drives_with_paths = self._load_data(dataset_path, return_paths=True)
        if not drives_with_paths:
            return None, None, []

        scores, labels, stems = [], [], []
        skipped = 0
        for drive_path, drive_df in drives_with_paths:
            ds = SensorWindowDataset([drive_df], window_size=self.window_size, stride=self.stride, norm_stats=self.norm_stats)
            if len(ds) == 0 or 'label' not in drive_df.columns:
                skipped += 1
                continue
            loader = DataLoader(ds, batch_size=32, shuffle=False)
            window_probs = []
            with torch.no_grad():
                for inputs, _ in loader:
                    inputs = inputs.to(self.device)
                    probs = torch.softmax(self.model(inputs), dim=1)
                    window_probs.append(probs[:, 1].cpu().numpy())
            if not window_probs:
                skipped += 1
                continue
            file_score = float(np.max(np.concatenate(window_probs)))
            lbl = int(pd.to_numeric(drive_df['label'], errors='coerce').fillna(0).max() >= 0.5)

            scores.append(file_score)
            labels.append(lbl)
            stems.append(drive_path.stem)

        if skipped:
            print(f"extract_scores: skipped {skipped}/{len(drives_with_paths)} files.")
        assert len(scores) == len(labels) == len(stems)
        if not scores:
            return None, None, []
        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int64), stems

    def extract_features(self, dataset_path: Path):
        """One 128-dim embedding per FILE (max-pooled over window embeddings).

        Files too short to form a single window are SKIPPED (not zero-padded:
        the old placeholder rows desynced the stem list from the feature list
        and poisoned the fusion training table). Downstream pairing is by
        stem, so skipping is safe.
        """
        self.model.eval()
        drives_with_paths = self._load_data(dataset_path, return_paths=True)
        if not drives_with_paths: return None, None, []

        file_feats = []
        file_labels = []
        file_stems = []
        skipped = 0

        for drive_path, drive_df in drives_with_paths:
            ds = SensorWindowDataset([drive_df], window_size=self.window_size, stride=self.stride, norm_stats=self.norm_stats)

            if len(ds) == 0:
                skipped += 1
                continue

            loader = DataLoader(ds, batch_size=32, shuffle=False)

            drive_embeddings = []

            with torch.no_grad():
                for inputs, labels in loader:
                    inputs = inputs.to(self.device)
                    emb = self.model.get_embedding(inputs)
                    drive_embeddings.append(emb.cpu().numpy())

            if not drive_embeddings:
                skipped += 1
                continue

            # (N_windows, 128) -> max-pool -> (128,): strongest signal in file
            pooled = np.max(np.concatenate(drive_embeddings, axis=0), axis=0)

            # File-level label: any pothole-labeled row makes it a pothole
            # file (iloc[0] missed potholes in time-localized drives).
            if 'label' in drive_df.columns:
                lbl = int(pd.to_numeric(drive_df['label'], errors='coerce').fillna(0).max() >= 0.5)
            else:
                skipped += 1
                continue

            # Atomic append: the three lists must stay aligned.
            file_feats.append(pooled)
            file_labels.append(lbl)
            file_stems.append(drive_path.stem)

        if skipped:
            print(f"extract_features: skipped {skipped}/{len(drives_with_paths)} "
                  f"files (too short or unlabeled).")
        assert len(file_feats) == len(file_labels) == len(file_stems), \
            "feature/label/stem misalignment"
        if not file_feats:
            return None, None, []

        return np.vstack(file_feats), np.array(file_labels), file_stems
