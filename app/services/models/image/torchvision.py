from app.services.models.base import BaseModel
from app.core.reproducibility import set_global_seed
from app.services.labels import infer_label_from_yolo_txt, infer_label_from_path
from app.services.metrics.detection import calculate_map
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import cv2
import glob # explicit glob

class PotholeDetectionDataset(Dataset):
    def __init__(self, root: Path, transforms=None):
        self.root = root
        self.transforms = transforms
        # Robustly finding images
        self.imgs = sorted(list(root.rglob("*.jpg")) + list(root.rglob("*.png")))
        
        # Determine labels dir
        # If structure is images/... labels/..., assume labels is sibling
        # Or if flat, same dir.
        
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        
        # Load Image
        image = cv2.imread(str(img_path))
        if image is None:
             # Return a blank 224x224 image with empty targets instead of
             # recursing, which can overflow the stack if many images are corrupt.
             # The "is_corrupt" flag lets train/evaluate SKIP these images
             # (a corrupt image is not a real negative sample).
             image = np.zeros((224, 224, 3), dtype=np.float32)
             image = torch.tensor(image).permute(2, 0, 1)
             target = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                       "labels": torch.zeros((0,), dtype=torch.int64),
                       "image_id": torch.tensor([idx]),
                       "area": torch.zeros((0,), dtype=torch.float32),
                       "iscrowd": torch.zeros((0,), dtype=torch.int64),
                       "is_corrupt": torch.tensor([1])}
             return image, target

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0 # Normalize 0-1
        
        # PyTorch expects (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1)
        
        # Load Label
        # Attempt to find corresponding txt file
        # Check adjacent 'labels' folder first
        label_path = None
        
        # Heuristic 1: ../labels/name.txt (YOLO standard)
        parents = list(img_path.parents)
        if len(parents) > 1 and (parents[1] / "labels").exists():
             label_path = parents[1] / "labels" / img_path.with_suffix(".txt").name
        
        # Heuristic 2: Same dir
        if not label_path or not label_path.exists():
             label_path = img_path.with_suffix(".txt")

        # Heuristic 3: YOLO content structure (images/split/img.jpg -> labels/split/img.txt)
        if (not label_path or not label_path.exists()) and "images" in img_path.parts:
            try:
                # Reconstruct path replacing 'images' with 'labels'
                parts = list(img_path.parts)
                # Find the last occurrence of 'images' in case of nested paths? 
                # Usually it's the one splitting dataset root and split folder.
                # Let's find index relative to root if possible, or just iterate.
                # Simple replace of the part "images"
                if "images" in parts:
                    idx = parts.index("images")
                    parts[idx] = "labels"
                    new_path = Path(*parts).with_suffix(".txt")
                    if new_path.exists():
                        label_path = new_path
            except (ValueError, TypeError):
                pass

        boxes = []
        labels = []
        
        height, width = image.shape[1], image.shape[2]

        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO format: class cx cy w h (normalized)
                        cls_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to XYXY (absolute)
                        x_min = (cx - w/2) * width
                        y_min = (cy - h/2) * height
                        x_max = (cx + w/2) * width
                        y_max = (cy + h/2) * height
                        
                        # Validate Box
                        # Faster R-CNN requires x_max > x_min and y_max > y_min
                        if x_max > x_min and y_max > y_min:
                            boxes.append([x_min, y_min, x_max, y_max])
                            labels.append(1) # Always class 1 (Pothole) for binary detection
                        else:
                            # Skip invalid box (likely width 0)
                            pass
        
        # Must return at least one box or handle "negative" image
        if len(boxes) == 0:
            # Negative sample (background)
            # Faster R-CNN handles empty targets? Yes, passing 0 boxes.
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["is_corrupt"] = torch.tensor([0])

        return image, target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

class FasterRCNNWrapper(BaseModel):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.num_classes = config.get("num_classes", 2) # Background + Pothole
        
        # Load model with pretrained backbone
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        # Replace head
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Central device selection (cuda > mps > cpu; POTHOLE_DEVICE overrides).
        # The historical MPS bugs (nonzero deadlock, empty-target crash) were
        # re-verified fixed on torch 2.9 — see tools/benchmark_device.py.
        from app.core.device import resolve_device
        self.device = resolve_device("faster_rcnn")

        self.model.to(self.device)
        # Using SGD as it's standard for R-CNN
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
        # Decay LR?
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.1)

    def _validate_split_structure(self, dataset_path: Path) -> tuple[Path, Path]:
        """Validate that dataset has proper train/val splits."""
        # Check for train directory
        train_candidates = [
            dataset_path / "train",
            dataset_path / "images" / "train"
        ]
        train_path = None
        for candidate in train_candidates:
            if candidate.exists():
                train_path = candidate
                break
        
        # Check for val directory
        val_candidates = [
            dataset_path / "val",
            dataset_path / "images" / "val",
            dataset_path / "valid",
            dataset_path / "images" / "valid"
        ]
        val_path = None
        for candidate in val_candidates:
            if candidate.exists():
                val_path = candidate
                break
        
        if train_path is None or val_path is None:
            raise ValueError(
                f"Dataset validation failed for {dataset_path}\n"
                f"Required structure:\n"
                f"  {dataset_path}/train/images/  OR  {dataset_path}/images/train/\n"
                f"  {dataset_path}/val/images/    OR  {dataset_path}/images/val/\n"
                f"Found: train={train_path}, val={val_path}"
            )
        
        return train_path, val_path

    def train(self, dataset_path: Path, epochs: int = 3):
        set_global_seed(42)
        print(f"Starting Faster R-CNN training on {self.device}...")
        
        # VALIDATE SPLITS BEFORE TRAINING
        train_root, val_root = self._validate_split_structure(dataset_path)
        print(f"Validated splits - Train: {train_root}, Val: {val_root}")
            
        print(f"Training on {train_root}...")
        dataset = PotholeDetectionDataset(train_root)
        if len(dataset) == 0:
            print(f"Error: No images found in {dataset_path}")
            return
            
        print(f"Found {len(dataset)} images.")
        
        # Seeded shuffle generator so batch order is reproducible
        g = torch.Generator()
        g.manual_seed(42)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn, generator=g)

        self.model.train()

        skipped_corrupt = 0
        for epoch in range(epochs):
            running_loss = 0.0
            for images, targets in loader:
                # Skip unreadable/corrupt images flagged by the dataset.
                # (Device-independent — the old MPS-only filtering made the
                # training data depend on the device, which is unacceptable.)
                valid_indices = [i for i, t in enumerate(targets)
                                 if not bool(t.get("is_corrupt", torch.tensor([0])).item())]
                if len(valid_indices) < len(targets):
                    skipped_corrupt += len(targets) - len(valid_indices)
                    images = [images[i] for i in valid_indices]
                    targets = [targets[i] for i in valid_indices]
                if len(images) == 0:
                    continue

                # Move to device (drop the bookkeeping key the model doesn't expect)
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items() if k != "is_corrupt"} for t in targets]

                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                self.optimizer.zero_grad()
                losses.backward()
                self.optimizer.step()

                running_loss += losses.item()

            self.lr_scheduler.step()
            print(f"HEARTBEAT: Faster R-CNN Epoch {epoch+1}/{epochs} Completed (Loss: {running_loss/len(loader):.4f})", flush=True)

        if skipped_corrupt > 0:
            print(f"Training skipped {skipped_corrupt} corrupt/unreadable image instances.", flush=True)
            
    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        self.model.eval()

        # Evaluation runs on the configured device. (The old "evaluate on
        # CPU to avoid the nonzero_mps deadlock" workaround targeted a torch
        # bug that no longer reproduces on torch 2.9 — verified by
        # tools/benchmark_device.py. Force CPU globally with
        # POTHOLE_DEVICE=cpu if it ever resurfaces.)
        eval_device = self.device

        # Prefer the held-out TEST split when the dataset declares one;
        # fall back to val (and record which one was used).
        eval_split = "val"
        eval_root = None
        for candidate, name in [(dataset_path / "test", "test"),
                                (dataset_path / "images" / "test", "test")]:
            if candidate.exists():
                eval_root = candidate
                eval_split = name
                break
        if eval_root is None:
            _, eval_root = self._validate_split_structure(dataset_path)

        print(f"Evaluating on {eval_root} ('{eval_split}' split)...")
        dataset = PotholeDetectionDataset(eval_root)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

        # Lower the box score threshold for evaluation so per-image max
        # scores (and the mAP PR tail) are uncensored — torchvision's default
        # box_score_thresh=0.05 silently zeroes weak detections.
        roi_heads = getattr(self.model, 'roi_heads', None)
        orig_score_thresh = getattr(roi_heads, 'score_thresh', None)
        if roi_heads is not None:
            roi_heads.score_thresh = 0.001

        # Image-level classification framing (same FRAMING as the YOLO
        # wrapper, with the same uniform 0.5 decision threshold): image
        # score = max detection score (0.0 if no detections),
        # y_pred = score >= 0.5, y_true = image has any GT box.
        # The old IoU-matched CM could never count FPs on positive images,
        # which inflated precision.
        y_true = []
        y_scores = []

        # Per-image collections for box-level mAP@0.5
        map_preds = []
        map_targets = []

        skipped_corrupt = 0

        with torch.no_grad():
             for images, targets in loader:
                 # Exclude unreadable/corrupt images from the metrics —
                 # a corrupt image is not a real negative sample.
                 valid_indices = [i for i, t in enumerate(targets)
                                  if not bool(t.get("is_corrupt", torch.tensor([0])).item())]
                 if len(valid_indices) < len(targets):
                     skipped_corrupt += len(targets) - len(valid_indices)
                     images = [images[i] for i in valid_indices]
                     targets = [targets[i] for i in valid_indices]
                 if len(images) == 0:
                     continue

                 images = list(image.to(eval_device) for image in images)
                 outputs = self.model(images)

                 for i, output in enumerate(outputs):
                     pred_boxes = output['boxes'].cpu().numpy()
                     pred_scores = output['scores'].cpu().numpy()
                     pred_labels = output['labels'].cpu().numpy()
                     gt_boxes = targets[i]['boxes'].numpy()
                     gt_labels = targets[i]['labels'].numpy()

                     has_gt = len(gt_boxes) > 0
                     max_score = float(max(pred_scores)) if len(pred_scores) > 0 else 0.0

                     y_true.append(1 if has_gt else 0)
                     y_scores.append(max_score)

                     map_preds.append({"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels})
                     map_targets.append({"boxes": gt_boxes, "labels": gt_labels})

        # Image-level confusion matrix from the single consistent rule (>=)
        y_pred = [1 if s >= 0.5 else 0 for s in y_scores]
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        cm = [[tn, fp], [fn, tp]]
        total_samples = len(y_true)
        accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0

        # Box-level mAP@0.5 via the shared metrics module
        map50 = calculate_map(map_preds, map_targets, iou_threshold=0.5)

        # Compute real ROC curve from per-image max scores (uncensored)
        from sklearn.metrics import roc_curve, auc
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            if len(roc_list) > 100:
                indices = np.linspace(0, len(roc_list) - 1, 100).astype(int)
                roc_list = [roc_list[i] for i in indices]
            roc_auc = float(auc(fpr, tpr))
        else:
            # ROC undefined for single-class y_true — never fabricate a diagonal
            roc_list = None
            roc_auc = None

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "map50": float(map50),
            "map50_source": "internal.calculate_map",
            # P/R/F1/accuracy are image-level presence/absence metrics,
            # NOT box-level detection metrics. map50 is box-level.
            "metric_framing": "image_level_classification",
            "decision_threshold": 0.5,
            "confusion_matrix": cm,
            "roc_curve": roc_list,
            "roc_auc": roc_auc,
            "eval_split": eval_split,
            "device": str(eval_device),
            "samples": total_samples,
            "skipped_corrupt_images": skipped_corrupt
        }

        # Save metrics.json
        import json
        with open(self.output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Restore the configured score threshold and device
        if roi_heads is not None and orig_score_thresh is not None:
            roi_heads.score_thresh = orig_score_thresh
        if eval_device != self.device:
            self.model.to(self.device)

        return metrics

    def predict(self, input_data: Any) -> Any:
        # Device policy lives in app/core/device.py (FasterRCNN stays on CPU
        # on Apple Silicon — MPS is dramatically slower for detection
        # post-processing, not a correctness issue).
        self.model.eval()
        input_data = [i.to(self.device) for i in input_data]

        with torch.no_grad():
            res = self.model(input_data)

        return [{k: v.cpu() for k, v in r.items()} for r in res]

    def extract_features(self, dataset_path: Path):
        """Extracts backbone features (native FPN dim, 256 for resnet50-fpn)
        for alignment/fusion."""
        # Scan images
        images = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
        if not images: return None, None, []

        print(f"Extracting Faster R-CNN Features for {len(images)} images...", flush=True)

        # Hook to capture features from the backbone
        # For ResNet50 FPN, the backbone is accessible
        self.captured_feats = None
        def hook(module, input, output):
             # output is a dict for FPN. Use most abstract level (usually '3')
             if isinstance(output, dict):
                  # Find highest level (smallest feature map)
                  keys = sorted(output.keys())
                  feat = output[keys[-1]]
             else:
                  feat = output
             
             # Pool to vector
             self.captured_feats = torch.mean(feat, dim=[2, 3]).detach().cpu().numpy()

        # Register hook on the backbone
        handle = self.model.backbone.register_forward_hook(hook)
        self.model.eval()
        
        features = []
        labels = []
        stems = []
        skipped = 0

        with torch.no_grad():
            for i, p in enumerate(images):
                # Reset before each forward so a failed pass can't reuse the
                # previous image's features.
                self.captured_feats = None
                try:
                    if i % 100 == 0:
                        print(f"Faster R-CNN Extraction Progress: {i}/{len(images)}", flush=True)

                    img_raw = cv2.imread(str(p))
                    if img_raw is None:
                        skipped += 1
                        continue
                    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_t = torch.tensor(img).permute(2, 0, 1).to(self.device)

                    # Route through the model's own GeneralizedRCNNTransform so
                    # the backbone sees the SAME normalization/resize it was
                    # trained with (feeding raw 0-1 pixels is a distribution
                    # mismatch).
                    images_t, _ = self.model.transform([img_t])
                    _ = self.model.backbone(images_t.tensors)

                    if self.captured_feats is None:
                        skipped += 1
                        continue

                    # Native dimensionality (FPN levels are 256-channel) — no
                    # zero-padding to 512: constant zero columns carry no
                    # information and misstate the embedding size.
                    vec = self.captured_feats[0]

                    # Centralized label inference (labels.py): YOLO txt first,
                    # then path keywords. label 1 = POTHOLE ONLY — cracks,
                    # bumps, spalls etc. are negatives.
                    lbl = infer_label_from_yolo_txt(p)
                    if lbl is None:
                        lbl = infer_label_from_path(p)
                    if lbl is None:
                        skipped += 1
                        continue

                    # Atomic append — all three lists together so an exception
                    # can never desync them.
                    features.append(vec)
                    labels.append(lbl)
                    stems.append(p.stem)
                except Exception as e:
                    print(f"Error extracting {p}: {e}")
                    skipped += 1

        handle.remove()
        if skipped > 0:
            print(f"Faster R-CNN extraction skipped {skipped} images (unreadable, no features, or no label).", flush=True)
        assert len(features) == len(labels) == len(stems), \
            "FasterRCNN extract_features: feature/label/stem lists desynced"
        return np.array(features), np.array(labels), stems

    def extract_scores(self, dataset_path: Path):
        """One pothole probability per image for the late-fusion benchmark:
        max detection score (uncensored — FasterRCNN returns all scored
        boxes). Returns (scores, labels, stems)."""
        images = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
        if not images:
            return np.zeros((0,)), np.zeros((0,)), []

        print(f"Extracting Faster R-CNN scores for {len(images)} images...", flush=True)

        run_device = self.device
        self.model.eval()

        scores, labels, stems = [], [], []
        skipped = 0
        with torch.no_grad():
            for i, p in enumerate(images):
                try:
                    if i % 100 == 0:
                        print(f"Faster R-CNN Score Progress: {i}/{len(images)}", flush=True)
                    img_raw = cv2.imread(str(p))
                    if img_raw is None:
                        skipped += 1
                        continue
                    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_t = torch.tensor(img).permute(2, 0, 1).to(run_device)
                    output = self.model([img_t])[0]
                    pred_scores = output['scores'].cpu().numpy()
                    max_score = float(pred_scores.max()) if len(pred_scores) > 0 else 0.0
                except Exception as e:
                    print(f"Error scoring {p}: {e}")
                    skipped += 1
                    continue

                lbl = infer_label_from_yolo_txt(p)
                if lbl is None:
                    lbl = infer_label_from_path(p)
                if lbl is None:
                    skipped += 1
                    continue

                scores.append(max_score)
                labels.append(lbl)
                stems.append(p.stem)

        if skipped:
            print(f"Faster R-CNN scoring skipped {skipped}/{len(images)} images.", flush=True)
        assert len(scores) == len(labels) == len(stems)
        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int64), stems

    def save(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "faster_rcnn.pth")
    
    def load(self, weights_path: Path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
