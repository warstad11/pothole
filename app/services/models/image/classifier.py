"""Plain image-classification baseline (ResNet18, binary pothole/no-pothole).

Why this exists: the platform's image metrics are image-level
presence/absence classification, so the natural reviewer question is
"why use detectors at all?". This baseline answers it empirically — if the
detectors don't beat a vanilla classifier on the image-level task, their
localization machinery isn't earning its complexity for this framing.

Protocol matches the other image wrappers: labels via the centralized
resolver (annotation file → path keywords → skip), train on train/, select
the checkpoint on val/, evaluate() reports test/ when present (else val),
uniform 0.5 decision threshold, ROC None when undefined.
"""

from app.services.models.base import BaseModel
from app.core.reproducibility import set_global_seed, get_torch_generator
from app.core.device import resolve_device
from app.services.labels import infer_label_from_yolo_txt, infer_label_from_path
from pathlib import Path
from typing import Dict, Any
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class _ImageLevelDataset(Dataset):
    """Images + image-level binary labels from a YOLO-layout folder."""
    def __init__(self, root: Path, size=224):
        self.size = size
        candidates = sorted(list(root.rglob("*.jpg")) + list(root.rglob("*.png")))
        # Exclude annotation dirs; resolve labels up front and DROP
        # unlabelable images (never default to 0).
        self.samples = []
        skipped = 0
        for p in candidates:
            if {"label", "labels", "masks", "mask"} & set(p.parts):
                continue
            lbl = infer_label_from_yolo_txt(p)
            if lbl is None:
                lbl = infer_label_from_path(p)
            if lbl is None:
                skipped += 1
                continue
            self.samples.append((p, lbl))
        if skipped:
            print(f"Classifier dataset: skipped {skipped} unlabelable images in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, lbl = self.samples[idx]
        img = cv2.imread(str(p))
        if img is None:
            # Corrupt image: mark with label -1 so loops can exclude it
            return torch.zeros(3, self.size, self.size), torch.tensor(-1)
        img = cv2.resize(img, (self.size, self.size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        return torch.tensor(img).permute(2, 0, 1), torch.tensor(lbl, dtype=torch.long)


class ResNet18ClassifierWrapper(BaseModel):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.seed = config.get("seed", 42)
        self.model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.device = resolve_device("classifier")
        self.model.to(self.device)

    # -------------------------------------------------------------- #
    def _split_dirs(self, dataset_path: Path):
        def find(names):
            for n in names:
                for cand in (dataset_path / n, dataset_path / "images" / n):
                    if cand.exists():
                        return cand
            return None
        return (find(["train", "training"]),
                find(["val", "valid", "validation"]),
                find(["test", "testing"]))

    def train(self, dataset_path: Path, epochs: int = 10):
        set_global_seed(self.seed)
        train_dir, val_dir, test_dir = self._split_dirs(dataset_path)
        if train_dir is None or val_dir is None:
            raise RuntimeError(
                f"ResNet18 classifier needs explicit train/ and val/ folders in "
                f"{dataset_path} (no leaky random-split fallback).")

        train_ds = _ImageLevelDataset(train_dir)
        val_ds = _ImageLevelDataset(val_dir)
        if len(train_ds) == 0 or len(val_ds) == 0:
            return {"status": "failed", "reason": "empty train or val set"}
        print(f"Classifier split: {len(train_ds)} train / {len(val_ds)} val images")

        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True,
                                  generator=get_torch_generator(self.seed),
                                  drop_last=len(train_ds) > 16)
        val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

        # Class weights from the TRAIN labels only
        labels = np.array([l for _, l in train_ds.samples])
        pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
        weight = (torch.tensor([1.0, neg / pos], dtype=torch.float32).to(self.device)
                  if pos > 0 and neg > 0 else None)
        criterion = nn.CrossEntropyLoss(weight=weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        stale = self.output_dir / "resnet18.pth"
        if stale.exists():
            stale.unlink()

        best_acc = -1.0
        for epoch in range(epochs):
            self.model.train()
            running = 0.0
            for x, y in train_loader:
                valid = y >= 0
                if not valid.any():
                    continue
                x, y = x[valid].to(self.device), y[valid].to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(x), y)
                loss.backward()
                optimizer.step()
                running += loss.item()

            # Checkpoint selection on val (never reported)
            self.model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    valid = y >= 0
                    x, y = x[valid].to(self.device), y[valid].to(self.device)
                    preds = self.model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.numel()
            val_acc = correct / total if total else 0.0
            print(f"Classifier Epoch {epoch+1}/{epochs}  Loss: {running/max(len(train_loader),1):.4f}  Val Acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                self.save()

        # Reload best checkpoint so evaluate() measures the saved weights
        if (self.output_dir / "resnet18.pth").exists():
            self.load(self.output_dir / "resnet18.pth")
        return {"val_acc": best_acc}

    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, roc_curve, auc)
        train_dir, val_dir, test_dir = self._split_dirs(dataset_path)
        eval_dir, eval_split = (test_dir, "test") if test_dir else (val_dir, "val")
        if eval_dir is None:
            return {"accuracy": None, "error": "no val/test folder"}

        ds = _ImageLevelDataset(eval_dir)
        loader = DataLoader(ds, batch_size=16, shuffle=False)
        self.model.eval()

        y_true, y_scores = [], []
        skipped_corrupt = 0
        with torch.no_grad():
            for x, y in loader:
                valid = y >= 0
                skipped_corrupt += int((~valid).sum())
                if not valid.any():
                    continue
                x = x[valid].to(self.device)
                probs = torch.softmax(self.model(x), dim=1)[:, 1].cpu().numpy()
                y_true.extend(y[valid].numpy().tolist())
                y_scores.extend(probs.tolist())

        if not y_true:
            return {"accuracy": None, "error": "no evaluable samples"}

        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
        roc_list, roc_auc = None, None
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            roc_auc = float(auc(fpr, tpr))
            if len(roc_list) > 100:
                idx = np.linspace(0, len(roc_list) - 1, 100).astype(int)
                roc_list = [roc_list[i] for i in idx]
        else:
            print("WARNING: single-class eval set; ROC undefined.")

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
            "roc_curve": roc_list,
            "roc_auc": roc_auc,
            "decision_threshold": 0.5,
            "metric_framing": "image_level_classification",
            "eval_split": eval_split,
            "device": str(self.device),
            "seed": self.seed,
            "samples": len(y_true),
            "skipped_corrupt_images": skipped_corrupt,
        }
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return metrics

    def predict(self, input_data: Any) -> Any:
        """input: (B, 3, H, W) tensor already normalized — returns softmax probs."""
        self.model.eval()
        with torch.no_grad():
            return torch.softmax(self.model(input_data.to(self.device)), dim=1).cpu().numpy()

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "resnet18.pth")

    def load(self, weights_path: Path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()
