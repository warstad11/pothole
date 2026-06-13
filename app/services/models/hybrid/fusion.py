
from app.services.models.base import BaseModel
from app.core.reproducibility import set_global_seed, get_torch_generator
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import json
import joblib

# --- Deployment Model Architecture ---
class FusionNet(nn.Module):
    def __init__(self, input_dim=640, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.fc(x)


class _FeatureCSVDataset(Dataset):
    """Loads pre-extracted, concatenated feature vectors."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- Feature Fusion Wrapper ---
class FeatureFusionWrapper(BaseModel):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.input_dim = config.get("input_dim", 640)
        self.seed = config.get("seed", 42)
        self.model = FusionNet(input_dim=self.input_dim)

        # Central device selection (cuda > mps > cpu; POTHOLE_DEVICE
        # overrides). BatchNorm parity on MPS verified by
        # tools/benchmark_device.py.
        from app.core.device import resolve_device
        self.device = resolve_device("fusion")
        self.model.to(self.device)

        self.scaler = None
        self._test_loader = None
        self._split_summary = None

    def load(self, weights_path: Path):
        print(f"Loading Fusion Model from {weights_path}")
        # Architecture from the saved config, not from path-name guessing
        cfg_path = weights_path.parent / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                saved_cfg = json.load(f)
            dim = saved_cfg.get("input_dim", self.input_dim)
            if dim != self.input_dim:
                self.input_dim = dim
                self.model = FusionNet(input_dim=dim).to(self.device)
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

        scaler_path = weights_path.parent / "scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded Fusion Scaler from {scaler_path}")
        else:
            print("Warning: No scaler found for fusion model. Inference may be degraded.")

    def predict(self, input_data: Any) -> Any:
        img_f, snr_f = input_data

        if img_f.ndim == 1: img_f = img_f[None, :]
        if snr_f.ndim == 1: snr_f = snr_f[None, :]

        cat_feats = np.concatenate([img_f, snr_f], axis=1)

        if self.scaler:
            cat_feats = self.scaler.transform(cat_feats)

        t_feats = torch.tensor(cat_feats, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(t_feats)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1].item())

    # ------------------------------------------------------------------ #
    # Data loading
    # ------------------------------------------------------------------ #
    def _load_partitions(self, dataset_path: Path):
        """Read features.csv and return per-partition (X, y) plus metadata.

        The CSV is produced by the hybrid worker and must contain
        ``img_*``/``snr_*`` feature columns and ``label``. A ``split`` column
        (train/val/test, assigned from a file-level split manifest) is
        strongly preferred: it keeps the fusion stage internally honest
        (disjoint files; scaler/weights/selection fit on train/val only).
        Whether the COMPONENT models also never saw the test rows depends on
        where they were trained — the worker records that separately as
        ``component_split_shared``. Without a split column we fall back to a
        deterministic stratified ROW split (weakest guarantee) — flagged in
        the returned metadata.
        """
        feat_csv = dataset_path / "features.csv"
        if not feat_csv.exists():
            return None

        df = pd.read_csv(feat_csv)
        meta_cols = {'label', 'split', 'stem'}
        feature_cols = [c for c in df.columns if c not in meta_cols]

        if 'split' in df.columns:
            split_meta = {"strategy": "shared file-level manifest (split column)"}
            df = df[df['split'].isin(['train', 'val', 'test'])]
            parts_df = {p: df[df['split'] == p] for p in ('train', 'val', 'test')}
        else:
            print("WARNING: features.csv has no 'split' column — falling back to a "
                  "stratified row split. Component-model leakage into val/test "
                  "cannot be ruled out for this run.")
            split_meta = {"strategy": "stratified row split (NO manifest — "
                                      "component leakage possible)"}
            rng = np.random.RandomState(self.seed)
            parts_idx = {'train': [], 'val': [], 'test': []}
            for lbl in np.unique(df['label'].values):
                idx = np.where(df['label'].values == lbl)[0]
                rng.shuffle(idx)
                n = len(idx)
                n_tr, n_va = int(0.70 * n), int(0.15 * n)
                parts_idx['train'].extend(idx[:n_tr])
                parts_idx['val'].extend(idx[n_tr:n_tr + n_va])
                parts_idx['test'].extend(idx[n_tr + n_va:])
            parts_df = {p: df.iloc[sorted(ix)] for p, ix in parts_idx.items()}

        out = {}
        for p, pdf in parts_df.items():
            X = pdf[feature_cols].values.astype(np.float32)
            y = pdf['label'].values.astype(np.int64)
            stems = pdf['stem'].tolist() if 'stem' in pdf.columns else None
            out[p] = (X, y, stems)
        split_meta["counts"] = {p: int(len(out[p][1])) for p in out}
        return out, feature_cols, split_meta

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, dataset_path: Path, epochs: int = 15):
        """Train FusionNet on a features.csv produced by the hybrid worker.

        Train fits weights, val selects the checkpoint, test is reported by
        evaluate() only.
        """
        set_global_seed(self.seed)

        loaded = self._load_partitions(dataset_path)
        if loaded is None:
            print(f"No features.csv in {dataset_path}. Skipping fusion training.")
            return {"status": "failed", "reason": "features.csv not found"}
        parts, feature_cols, split_meta = loaded
        (X_train, y_train, train_stems) = parts['train']
        (X_val, y_val, _) = parts['val']
        (X_test, y_test, test_stems) = parts['test']

        if len(y_train) == 0 or len(y_val) == 0:
            return {"status": "failed", "reason": "empty train or val partition"}

        # Scaler fit on TRAIN ONLY (fitting on all rows leaks val/test
        # statistics into preprocessing).
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val = self.scaler.transform(X_val).astype(np.float32)
        X_test = self.scaler.transform(X_test).astype(np.float32) if len(y_test) else X_test

        # Re-initialise model if feature dimension changed
        actual_dim = X_train.shape[1]
        if actual_dim != self.input_dim:
            print(f"Adjusting FusionNet input_dim {self.input_dim} -> {actual_dim}")
            self.input_dim = actual_dim
            self.model = FusionNet(input_dim=actual_dim).to(self.device)
        self._feature_cols = feature_cols

        # drop_last guards FusionNet's BatchNorm1d against a final train
        # batch of size 1 (which raises in train mode)
        train_loader = DataLoader(_FeatureCSVDataset(X_train, y_train), batch_size=32,
                                  shuffle=True, generator=get_torch_generator(self.seed),
                                  drop_last=len(y_train) > 32)
        val_loader = DataLoader(_FeatureCSVDataset(X_val, y_val), batch_size=32, shuffle=False)
        self._test_loader = (DataLoader(_FeatureCSVDataset(X_test, y_test), batch_size=32,
                                        shuffle=False) if len(y_test) else None)
        self._split_summary = {**split_meta, "seed": self.seed}

        # Class weights from TRAIN labels only
        pos_count = int(y_train.sum())
        neg_count = len(y_train) - pos_count
        if pos_count > 0 and neg_count > 0:
            weight = torch.tensor([1.0, neg_count / pos_count], dtype=torch.float32).to(self.device)
        else:
            weight = None

        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # A stale checkpoint from a previous run in the same output_dir must
        # never be reloaded as this run's "best" weights.
        stale_ckpt = self.output_dir / "model.pth"
        if stale_ckpt.exists():
            stale_ckpt.unlink()

        # Checkpoint selection: val F1 (positive class), accuracy tie-break —
        # raw accuracy favors majority collapse on imbalanced hybrid data
        # (~17% positive) and would outrank genuine detectors. Same protocol
        # as the sensor DL wrapper and the late-fusion operating point.
        # -1 so the first epoch always checkpoints.
        best_key = (-1.0, -1.0)
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            for feats, labels in train_loader:
                feats, labels = feats.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                logits = self.model(feats)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # Validation (checkpoint selection only — never reported)
            self.model.eval()
            correct, total = 0, 0
            tp = fp = fn = 0
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats, labels = feats.to(self.device), labels.to(self.device)
                    preds = self.model(feats).argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    tp += ((preds == 1) & (labels == 1)).sum().item()
                    fp += ((preds == 1) & (labels == 0)).sum().item()
                    fn += ((preds == 0) & (labels == 1)).sum().item()

            val_acc = correct / total if total > 0 else 0.0
            val_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            avg_loss = running_loss / max(len(train_loader), 1)
            print(f"Fusion Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  "
                  f"Val Acc: {val_acc:.4f}  Val F1: {val_f1:.4f}")

            if (val_f1, val_acc) > best_key:
                best_key = (val_f1, val_acc)
                self.save()

        # Reload the best checkpoint so evaluate() measures the saved model,
        # not the final-epoch weights.
        best_ckpt = self.output_dir / "model.pth"
        if best_ckpt.exists():
            self.model.load_state_dict(torch.load(best_ckpt, map_location=self.device))
            self.model.eval()

        # Persist split membership for auditability
        with open(self.output_dir / "split.json", "w") as f:
            json.dump({**self._split_summary,
                       "train_stems": train_stems, "test_stems": test_stems}, f, indent=2)

        return {"val_f1": best_key[0], "val_acc": best_key[1],
                "checkpoint_selection": "val_f1_then_acc"}

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        """Reports metrics on the held-out TEST partition."""
        self.model.eval()

        if self._test_loader is not None:
            loader = self._test_loader
            split_summary = self._split_summary
        else:
            loaded = self._load_partitions(dataset_path)
            if loaded is None:
                return {"accuracy": None, "error": "features.csv not found"}
            parts, _, split_meta = loaded
            X_test, y_test, _ = parts['test']
            if len(y_test) == 0:
                return {"accuracy": None, "error": "empty test partition"}
            scaler_missing = self.scaler is None
            if self.scaler:
                X_test = self.scaler.transform(X_test).astype(np.float32)
            else:
                print("WARNING: evaluating without the training scaler — features are "
                      "unscaled and metrics are unreliable (flagged as degraded).")
            loader = DataLoader(_FeatureCSVDataset(X_test, y_test), batch_size=32, shuffle=False)
            split_summary = {**split_meta, "seed": self.seed}
            if scaler_missing:
                split_summary = {**split_summary, "scaler_missing": True}

        if loader is None:
            return {"accuracy": None, "error": "empty test partition"}

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for feats, labels in loader:
                feats = feats.to(self.device)
                logits = self.model(feats)
                probs = torch.softmax(logits, dim=1)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        if not all_labels:
            return {"accuracy": None, "error": "no samples"}

        rep = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
        pothole_metrics = rep.get('1', rep.get('1.0', {}))

        # ROC from probabilities; undefined -> None (never a fabricated
        # diagonal that could end up in a figure).
        roc_list = None
        roc_auc = None
        if len(np.unique(all_labels)) >= 2:
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            roc_auc = float(auc(fpr, tpr))
        else:
            print("WARNING: test partition has a single class; ROC undefined.")

        def clean(val):
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
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
            "split": split_summary,
            "seed": self.seed,
            "device": str(self.device),
        }

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        return metrics

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "model.pth")
        if self.scaler:
            joblib.dump(self.scaler, self.output_dir / "scaler.pkl")
        # Architecture metadata so loaders (InferenceEngine) never have to
        # guess input_dim from the checkpoint's directory name.
        cfg = {
            "input_dim": self.input_dim,
            "image_dim": self.config.get("input_dim_image"),
            "sensor_dim": self.config.get("input_dim_sensor"),
            "seed": self.seed,
        }
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)


# --- Late Fusion Wrapper ---
class LateFusionWrapper(BaseModel):
    """Score-level fusion: fused = alpha * P_image + (1 - alpha) * P_sensor.

    Consumes a scores.csv produced by the hybrid worker via the wrappers'
    extract_scores() methods: columns img_score, snr_score, label, and
    (preferably) stem + split from the shared file-level manifest.

    Protocol: alpha and the decision threshold are selected on the VAL
    partition by F1 (accuracy is misleading under the class imbalance of the
    pothole-vs-other-anomalies task), then evaluate() reports the TEST
    partition only. There are no learned weights beyond (alpha, threshold);
    both are persisted to config.json for reproducibility.
    """

    ALPHA_GRID = [i / 20.0 for i in range(21)]            # 0.00 .. 1.00
    THRESHOLD_GRID = [i / 20.0 for i in range(1, 20)]     # 0.05 .. 0.95

    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.alpha = config.get("alpha", 0.5)
        self.threshold = config.get("threshold", 0.5)
        self.seed = config.get("seed", 42)

    # -------------------------------------------------------------- #
    def predict(self, input_data: Any) -> Any:
        img_prob = 0.0
        snr_prob = 0.0

        if isinstance(input_data, dict):
            img_prob = float(input_data.get('image_prob', 0.0))
            snr_prob = float(input_data.get('sensor_prob', 0.0))
        elif isinstance(input_data, (list, tuple)) and len(input_data) >= 2:
            img_prob = float(input_data[0])
            snr_prob = float(input_data[1])

        return self.alpha * img_prob + (1.0 - self.alpha) * snr_prob

    # -------------------------------------------------------------- #
    def _load_score_partitions(self, dataset_path: Path):
        csv_path = dataset_path / "scores.csv"
        if not csv_path.exists():
            return None
        df = pd.read_csv(csv_path)
        required = {'img_score', 'snr_score', 'label'}
        if not required.issubset(df.columns):
            raise ValueError(f"scores.csv must contain {required}, got {list(df.columns)}")

        if 'split' in df.columns:
            split_meta = {"strategy": "shared file-level manifest (split column)"}
            df = df[df['split'].isin(['train', 'val', 'test'])]
            parts = {p: df[df['split'] == p] for p in ('train', 'val', 'test')}
        else:
            print("WARNING: scores.csv has no 'split' column — falling back to a "
                  "stratified row split. Component-model leakage into val/test "
                  "cannot be ruled out for this run.")
            split_meta = {"strategy": "stratified row split (NO manifest)"}
            rng = np.random.RandomState(self.seed)
            parts_idx = {'train': [], 'val': [], 'test': []}
            for lbl in np.unique(df['label'].values):
                idx = np.where(df['label'].values == lbl)[0]
                rng.shuffle(idx)
                n = len(idx)
                n_tr, n_va = int(0.70 * n), int(0.15 * n)
                parts_idx['train'].extend(idx[:n_tr])
                parts_idx['val'].extend(idx[n_tr:n_tr + n_va])
                parts_idx['test'].extend(idx[n_tr + n_va:])
            parts = {p: df.iloc[sorted(ix)] for p, ix in parts_idx.items()}
        split_meta["counts"] = {p: int(len(parts[p])) for p in parts}
        return parts, split_meta

    @staticmethod
    def _fuse(df: pd.DataFrame, alpha: float) -> np.ndarray:
        return (alpha * df['img_score'].values
                + (1.0 - alpha) * df['snr_score'].values)

    # -------------------------------------------------------------- #
    def train(self, dataset_path: Path, epochs: int = 0):
        """Select (alpha, threshold) on the val partition by F1."""
        from sklearn.metrics import f1_score

        loaded = self._load_score_partitions(dataset_path)
        if loaded is None:
            print(f"No scores.csv in {dataset_path}. Skipping late-fusion training.")
            return {"status": "failed", "reason": "scores.csv not found"}
        parts, split_meta = loaded
        val_df = parts['val']
        if len(val_df) == 0:
            return {"status": "failed", "reason": "empty val partition"}

        y_val = val_df['label'].values.astype(int)
        best = (-1.0, self.alpha, self.threshold)
        for alpha in self.ALPHA_GRID:
            fused = self._fuse(val_df, alpha)
            for thr in self.THRESHOLD_GRID:
                f1 = f1_score(y_val, (fused >= thr).astype(int), zero_division=0)
                if f1 > best[0]:
                    best = (f1, alpha, thr)

        best_f1, self.alpha, self.threshold = best
        self._split_summary = {**split_meta, "seed": self.seed}
        self._parts_cache = parts
        print(f"Late fusion selected alpha={self.alpha:.2f}, threshold={self.threshold:.2f} "
              f"(val F1={best_f1:.4f})")
        self.save()
        return {"val_f1": float(best_f1), "alpha": self.alpha, "threshold": self.threshold}

    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        """Reports metrics on the held-out TEST partition."""
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, roc_curve, auc)

        if getattr(self, '_parts_cache', None) is not None:
            parts = self._parts_cache
            split_summary = self._split_summary
        else:
            loaded = self._load_score_partitions(dataset_path)
            if loaded is None:
                return {"accuracy": None, "error": "scores.csv not found"}
            parts, split_meta = loaded
            split_summary = {**split_meta, "seed": self.seed}

        test_df = parts['test']
        if len(test_df) == 0:
            return {"accuracy": None, "error": "empty test partition"}

        y_test = test_df['label'].values.astype(int)
        fused = self._fuse(test_df, self.alpha)
        y_pred = (fused >= self.threshold).astype(int)

        roc_list = None
        roc_auc = None
        if len(np.unique(y_test)) >= 2:
            fpr, tpr, _ = roc_curve(y_test, fused)
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            roc_auc = float(auc(fpr, tpr))
        else:
            print("WARNING: test partition has a single class; ROC undefined.")

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
            "roc_curve": roc_list,
            "roc_auc": roc_auc,
            "samples": int(len(y_test)),
            "alpha": self.alpha,
            "threshold": self.threshold,
            "eval_split": "test",
            "split": split_summary,
            "seed": self.seed,
        }

        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "config.json", "w") as f:
            json.dump({"fusion_type": "late", "alpha": self.alpha,
                       "threshold": self.threshold, "seed": self.seed,
                       "selection": "alpha+threshold by F1 on val partition"}, f, indent=2)

    def load(self, weights_path: Path):
        cfg_path = Path(weights_path)
        if cfg_path.is_dir():
            cfg_path = cfg_path / "config.json"
        if cfg_path.name.endswith('.json') and cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.alpha = cfg.get("alpha", self.alpha)
            self.threshold = cfg.get("threshold", self.threshold)
