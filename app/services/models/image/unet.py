from app.services.models.base import BaseModel
from app.core.reproducibility import set_global_seed
from app.services.labels import infer_label_from_yolo_txt, infer_label_from_path
from pathlib import Path
from typing import Dict, Any
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import cv2

class PotholeSegmentationDataset(Dataset):
    # Directories that hold annotations, not inputs — must never be scanned
    # as images (Pothole-600 stores real pixel masks as PNGs in label/).
    ANNOTATION_DIRS = {"label", "labels", "masks", "mask"}

    def __init__(self, root: Path, size=256):
        self.root = root
        self.size = size
        self.imgs = sorted(p for p in
                           (list(root.rglob("*.jpg")) + list(root.rglob("*.png")))
                           if not (self.ANNOTATION_DIRS & set(p.parts)))

    def _resolve_pixel_mask(self, img_path: Path):
        """Real pixel-mask layout (Pothole-600): <split>/rgb/x.png with the
        binary mask at <split>/label/x.png. Returns the mask path or None."""
        for d in ("label", "masks", "mask"):
            cand = img_path.parent.parent / d / img_path.name
            if cand.exists():
                return cand
            cand_png = (img_path.parent.parent / d / img_path.stem).with_suffix(".png")
            if cand_png.exists():
                return cand_png
        return None

    def mask_source(self) -> str:
        """'pixel_masks' (true segmentation GT) or 'box_pseudo_masks'
        (rectangles rasterized from YOLO boxes) — recorded in metrics so a
        paper can never conflate the two IoU definitions."""
        for p in self.imgs[:20]:
            if self._resolve_pixel_mask(p) is not None:
                return "pixel_masks"
        return "box_pseudo_masks"

    def _resolve_label_path(self, img_path: Path):
        """Find the YOLO label txt for an image, or None if not found."""
        # Heuristic 1: ../labels/name.txt (YOLO standard)
        parents = list(img_path.parents)
        if len(parents) > 1 and (parents[1] / "labels").exists():
            cand = parents[1] / "labels" / img_path.with_suffix(".txt").name
            if cand.exists():
                return cand
        # Heuristic 2: same dir
        cand = img_path.with_suffix(".txt")
        if cand.exists():
            return cand
        # Heuristic 3: images/<split>/img.jpg -> labels/<split>/img.txt
        # (parts substitution, same as torchvision.py)
        if "images" in img_path.parts:
            try:
                parts = list(img_path.parts)
                parts[parts.index("images")] = "labels"
                cand = Path(*parts).with_suffix(".txt")
                if cand.exists():
                    return cand
            except (ValueError, TypeError):
                pass
        return None

    def warn_if_all_masks_empty(self):
        """Loud warning if no image has a non-empty label file — that almost
        always means label-path resolution failed, not an all-negative set."""
        if self.mask_source() == "pixel_masks":
            return  # real masks resolved — nothing to warn about
        for img_path in self.imgs:
            lbl = self._resolve_label_path(img_path)
            if lbl is not None:
                try:
                    if lbl.read_text().strip():
                        return  # at least one non-empty mask exists
                except OSError:
                    continue
        if self.imgs:
            print("=" * 70)
            print("WARNING: EVERY mask in this dataset is empty. This usually means")
            print("label-path resolution failed (no labels/ dir found), NOT that the")
            print("dataset is all-negative. Check the dataset layout.")
            print("=" * 70, flush=True)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]

        # Load Image
        image = cv2.imread(str(img_path))
        if image is None:
            # Return blank image + empty mask instead of recursing
            image = torch.zeros(3, self.size, self.size)
            mask = torch.zeros(1, self.size, self.size)
            return image, mask

        # Resize
        image = cv2.resize(image, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Make Mask. Priority: REAL pixel masks (e.g. Pothole-600's label/
        # PNGs — true segmentation GT), falling back to box-derived
        # PSEUDO-masks rasterized from YOLO boxes (IoU against those is not
        # true segmentation IoU; metrics record which source was used).
        pixel_mask = self._resolve_pixel_mask(img_path)
        if pixel_mask is not None:
            m = cv2.imread(str(pixel_mask), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
                mask = (m > 127).astype(np.float32)
                image_t = torch.tensor(image).permute(2, 0, 1)
                return image_t, torch.tensor(mask).unsqueeze(0)

        label_path = self._resolve_label_path(img_path)

        mask = np.zeros((self.size, self.size), dtype=np.float32)

        # Original Image Dimensions? We need original size to map relative coords
        # Or we can just map relative coords to 256

        if label_path is not None and label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # YOLO: class cx cy w h
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert to pixel coords on (256, 256)
                        width = self.size
                        height = self.size
                        
                        x_min = int((cx - w/2) * width)
                        y_min = int((cy - h/2) * height)
                        x_max = int((cx + w/2) * width)
                        y_max = int((cy + h/2) * height)
                        
                        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), 1.0, -1)
        
        # To Tensor
        image = torch.tensor(image).permute(2, 0, 1) # (3, H, W)
        mask = torch.tensor(mask).unsqueeze(0) # (1, H, W)
        
        return image, mask
        
    def __len__(self):
        return len(self.imgs)

class UNetWrapper(BaseModel):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.encoder = config.get("encoder", "resnet34")
        self.num_classes = config.get("num_classes", 1)
        
        self.model = smp.Unet(
            encoder_name=self.encoder, 
            encoder_weights="imagenet", 
            in_channels=3, 
            classes=self.num_classes
        )
        
        # Central device selection (cuda > mps > cpu; POTHOLE_DEVICE overrides).
        # The BatchNorm train/eval inconsistency that originally forced CPU no
        # longer reproduces on torch 2.9 — verified with parity checks in
        # tools/benchmark_device.py (U-Net: ~4.8x faster on Apple MPS).
        from app.core.device import resolve_device
        self.device = resolve_device("unet")

        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()

    def train(self, dataset_path: Path, epochs: int = 3):
        set_global_seed(42)
        print(f"Starting U-Net training on {self.device}...")
        
        # FIX: Enforce Train/Val Split logic
        train_path = dataset_path / "train"
        if not train_path.exists():
            train_path = dataset_path / "images" / "train"
        
        # Fallback to simple split if no structure
        if not train_path.exists():
             print(f"No 'train' folder found in {dataset_path}. Using Random Split (80/20).")
             full_ds = PotholeSegmentationDataset(dataset_path)
             # Keep any explicit val/valid/test folders OUT of the random
             # pool — evaluate() prefers those folders, so including them
             # here would overlap train and eval data.
             before = len(full_ds.imgs)
             full_ds.imgs = [p for p in full_ds.imgs
                             if not ({'val', 'valid', 'test'} & set(p.parts))]
             if len(full_ds.imgs) < before:
                 print(f"Excluded {before - len(full_ds.imgs)} images under val/valid/test "
                       f"folders from the random-split pool.")
             if len(full_ds) == 0:
                 print("No images found.")
                 return
             full_ds.warn_if_all_masks_empty()

             train_size = int(0.8 * len(full_ds))
             val_size = len(full_ds) - train_size
             dataset, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

             # Persist the held-out split so evaluate() can score ONLY the
             # validation samples — otherwise it would evaluate on training
             # data (leakage). Stores the full file order, the val indices,
             # and the resolved val file paths.
             self.output_dir.mkdir(parents=True, exist_ok=True)
             split_record = {
                 "files": [str(p) for p in full_ds.imgs],
                 "val_indices": [int(i) for i in val_ds.indices],
                 "val_files": [str(full_ds.imgs[i]) for i in val_ds.indices],
             }
             with open(self.output_dir / "val_indices.json", "w") as f:
                 json.dump(split_record, f)
             print(f"Persisted random-split val indices to {self.output_dir / 'val_indices.json'}")
        else:
             print(f"Training on explicit split: {train_path}")
             dataset = PotholeSegmentationDataset(train_path)
             dataset.warn_if_all_masks_empty()

        if len(dataset) == 0:
            print(f"No images found in training set.")
            return
            
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        
        self.model.train()
        
        for epoch in range(epochs):
            running_loss = 0.0
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
            print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(loader):.4f}")
        
    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        from sklearn.metrics import roc_curve, auc
        
        self.model.eval()

        # Prefer the held-out TEST split when present, then val/valid
        # (recording which split was used).
        eval_split = None
        eval_path = None
        for cand, name in [(dataset_path / "test", "test"),
                           (dataset_path / "images" / "test", "test"),
                           (dataset_path / "val", "val"),
                           (dataset_path / "images" / "val", "val"),
                           (dataset_path / "valid", "val"),
                           (dataset_path / "images" / "valid", "val")]:
            if cand.exists():
                eval_path = cand
                eval_split = name
                break

        if eval_path is not None:
            print(f"Evaluating on explicit split: {eval_path} ('{eval_split}')")
            dataset = PotholeSegmentationDataset(eval_path)
        else:
            # No explicit split folder: only evaluate on the held-out split
            # persisted by train(). NEVER fall back to the full dataset —
            # that evaluates on training data (leakage).
            split_file = self.output_dir / "val_indices.json"
            if not split_file.exists():
                raise RuntimeError("no persisted split — refusing to evaluate on training data")
            with open(split_file) as f:
                split_record = json.load(f)
            dataset = PotholeSegmentationDataset(dataset_path)
            val_files = [Path(fp) for fp in split_record.get("val_files", [])]
            dataset.imgs = [p for p in val_files if p.exists()]
            eval_split = "val"
            if len(dataset.imgs) == 0:
                raise RuntimeError(
                    f"persisted split references {len(val_files)} files but none exist "
                    f"on disk (dataset moved, or evaluate run from a different cwd than "
                    f"training) — refusing to return empty metrics.")
            if len(dataset.imgs) < len(val_files):
                print(f"WARNING: {len(val_files) - len(dataset.imgs)} persisted val files "
                      f"are missing on disk.")
            print(f"Evaluating on persisted random-split val set ({len(dataset)} samples).")

        # Exclude unreadable/corrupt images — a corrupt image is not a real
        # negative sample (same policy as FasterRCNN).
        readable = [p for p in dataset.imgs if cv2.imread(str(p)) is not None]
        skipped_corrupt = len(dataset.imgs) - len(readable)
        if skipped_corrupt:
            print(f"Excluding {skipped_corrupt} unreadable images from evaluation.")
            dataset.imgs = readable

        dataset.warn_if_all_masks_empty()
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        # NOTE: masks are box-derived pseudo-masks (rectangles rasterized from
        # YOLO boxes), so the IoU below is NOT true segmentation IoU.
        ious = []          # per-IMAGE IoU, only for images with non-empty union
        neg_total = 0      # images with empty GT mask
        neg_correct = 0    # ...whose prediction is also empty
        y_true = []
        y_scores = []

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)

                # Per-IMAGE IoU (sum over dims [1,2,3]), not per-batch.
                preds = (torch.sigmoid(outputs) > 0.5).float()
                intersection = (preds * masks).sum(dim=[1, 2, 3])
                union = preds.sum(dim=[1, 2, 3]) + masks.sum(dim=[1, 2, 3]) - intersection
                for j in range(images.shape[0]):
                    u = union[j].item()
                    if u > 0:
                        # Only images where GT or prediction is non-empty
                        # contribute to mean_iou (empty/empty is trivial).
                        ious.append(intersection[j].item() / (u + 1e-6))
                    if masks[j].sum().item() == 0:
                        neg_total += 1
                        if preds[j].sum().item() == 0:
                            neg_correct += 1

                # Collect for ROC curve
                # Binary classification: does mask contain any positive pixels?
                # This treats segmentation as "Image Classification" for Rec/Prec purposes
                batch_y_true = (masks.sum(dim=[1, 2, 3]) > 0).cpu().numpy()
                # Use max probability in prediction as score for the whole image
                batch_y_scores = torch.sigmoid(outputs).max(dim=1)[0].max(dim=1)[0].max(dim=1)[0].cpu().numpy()

                y_true.extend(batch_y_true)
                y_scores.extend(batch_y_scores)

        mean_iou = sum(ious) / len(ious) if ious else 0.0
        # Fraction of empty-GT images with an empty prediction
        negative_specificity = (neg_correct / neg_total) if neg_total > 0 else None

        # Calculate real ROC curve
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
            roc_auc = float(auc(fpr, tpr))
            if len(roc_list) > 100:
                indices = np.linspace(0, len(roc_list) - 1, 100).astype(int)
                roc_list = [roc_list[i] for i in indices]
        else:
            # ROC undefined for single-class y_true — never fabricate a diagonal
            roc_list = None
            roc_auc = None

        # Estimate CM properties from the actual Image-Level predictions
        # Note: This CM is "Image Classification" CM, not "Pixel CM".
        # Uniform >= 0.5 rule, same as YOLO / FasterRCNN.
        y_pred_binary = [1 if s >= 0.5 else 0 for s in y_scores]
        tp = sum([1 for t, p in zip(y_true, y_pred_binary) if t and p])
        tn = sum([1 for t, p in zip(y_true, y_pred_binary) if not t and not p])
        fp = sum([1 for t, p in zip(y_true, y_pred_binary) if not t and p])
        fn = sum([1 for t, p in zip(y_true, y_pred_binary) if t and not p])
        cm = [[tn, fp], [fn, tp]]

        # Recalculate P/R based on Image Classification performance (matches YOLO style)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Image-level classification accuracy — NOT IoU. Reporting IoU under
        # "accuracy" conflated two different metrics.
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

        return {
            "accuracy": float(accuracy),
            "f1": 2*p*r/(p+r) if (p+r) > 0 else 0.0,
            "precision": p,
            "recall": r,
            "mean_iou": float(mean_iou),
            # 'pixel_masks' = true segmentation IoU (e.g. Pothole-600);
            # 'box_pseudo_masks' = agreement with rasterized rectangles only.
            "mask_source": dataset.mask_source(),
            "negative_specificity": float(negative_specificity) if negative_specificity is not None else None,
            "metric_framing": "image_level_classification",
            "decision_threshold": 0.5,
            "confusion_matrix": cm,
            "roc_curve": roc_list,
            "roc_auc": roc_auc,
            "eval_split": eval_split,
            "device": str(self.device),
            "samples": len(y_true),
            "skipped_corrupt_images": skipped_corrupt
        }
        
    def predict(self, input_data: Any) -> Any:
        self.model.eval()
        with torch.no_grad():
            return self.model(input_data)

    def predict_embedding(self, img: np.ndarray) -> np.ndarray:
        """
        Extracts 512-dim feature vector for a single image (H, W, 3).
        Matches logic in tools/evaluate_fusion_unet_rf_full.py
        """
        self.model.eval()
        
        # Preprocess
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Hook
        features = []
        def hook(module, input, output):
             if isinstance(output, (list, tuple)): output = output[-1]
             # Global Average Pooling [B, C, H, W] -> [B, C]
             features.append(torch.mean(output, dim=[2, 3]).detach().cpu().numpy())
             
        handle = self.model.encoder.register_forward_hook(hook)
        
        with torch.no_grad():
            _ = self.model.encoder(img_t)
            
        handle.remove()
        
        if not features:
            # Silent zero vectors would feed garbage into the fusion models —
            # fail loudly instead (same policy as the YOLO wrapper).
            raise RuntimeError("U-Net predict_embedding: encoder hook captured no features")
        vec = features[0][0] # First batch item (resnet34 encoder = 512-dim natively)
        return vec

    def extract_features(self, dataset_path: Path):
        """Extracts encoder features (512-dim) for hybrid alignment."""
        images = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
        if not images: return None, None, []

        print(f"Extracting U-Net Features for {len(images)} images...", flush=True)
        
        self.captured_feats = None
        def hook(module, input, output):
             # output is likely a tensor [B, C, H, W]
             # Some encoders return a list of tensors (hidden states)
             if isinstance(output, (list, tuple)):
                 output = output[-1]
                 
             self.captured_feats = torch.mean(output, dim=[2, 3]).detach().cpu().numpy()

        # Hook the encoder output
        handle = self.model.encoder.register_forward_hook(hook)
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
                        print(f"U-Net Extraction Progress: {i}/{len(images)}", flush=True)
                    img_raw = cv2.imread(str(p))
                    if img_raw is None:
                        skipped += 1
                        continue
                    img = cv2.resize(img_raw, (256, 256))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)

                    _ = self.model.encoder(img_t)

                    if self.captured_feats is None:
                        skipped += 1
                        continue

                    vec = self.captured_feats[0]
                    # Standardize to 512
                    if vec.shape[0] > 512: vec = vec[:512]
                    elif vec.shape[0] < 512: vec = np.pad(vec, (0, 512 - vec.shape[0]))

                    # Centralized label inference (labels.py): YOLO txt first,
                    # then path keywords. label 1 = POTHOLE ONLY.
                    lbl = infer_label_from_yolo_txt(p)
                    if lbl is None:
                        lbl = infer_label_from_path(p)
                    if lbl is None:
                        # Unknown label — skip rather than guess.
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
            print(f"U-Net extraction skipped {skipped} images (unreadable, no features, or no label).", flush=True)
        assert len(features) == len(labels) == len(stems), \
            "UNet extract_features: feature/label/stem lists desynced"
        return np.array(features), np.array(labels), stems

    def extract_scores(self, dataset_path: Path):
        """One pothole probability per image for the late-fusion benchmark:
        max sigmoid pixel probability over the predicted mask (same image-
        level score evaluate() uses). Returns (scores, labels, stems)."""
        images = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
        if not images:
            return np.zeros((0,)), np.zeros((0,)), []

        print(f"Extracting U-Net scores for {len(images)} images...", flush=True)
        self.model.eval()

        scores, labels, stems = [], [], []
        skipped = 0
        with torch.no_grad():
            for i, p in enumerate(images):
                try:
                    if i % 100 == 0:
                        print(f"U-Net Score Progress: {i}/{len(images)}", flush=True)
                    img_raw = cv2.imread(str(p))
                    if img_raw is None:
                        skipped += 1
                        continue
                    img = cv2.resize(img_raw, (256, 256))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    img_t = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
                    out = self.model(img_t)
                    score = float(torch.sigmoid(out).max().cpu().item())
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

                scores.append(score)
                labels.append(lbl)
                stems.append(p.stem)

        if skipped:
            print(f"U-Net scoring skipped {skipped}/{len(images)} images.", flush=True)
        assert len(scores) == len(labels) == len(stems)
        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int64), stems

    def save(self):
        # Create dir if not exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "unet.pth")

    def load(self, weights_path: Path):
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
