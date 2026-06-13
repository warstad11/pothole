from app.services.models.base import BaseModel
from app.services.labels import infer_label_from_yolo_txt, infer_label_from_path
from app.core.device import device_str
from ultralytics import YOLO
from pathlib import Path
from typing import Dict, Any
import shutil
import numpy as np

class YOLOWrapper(BaseModel):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        super().__init__(model_id, config)
        self.model_name = config.get("model_name", "yolov8n.pt") # Default to nano
        # Explicit device (cuda > mps > cpu, POTHOLE_DEVICE override) passed
        # to every Ultralytics call so behavior doesn't depend on its
        # auto-detection. CPU/MPS prediction parity verified exact on
        # torch 2.9 (tools/benchmark_device.py).
        self.device = device_str("yolo")
        try:
            self.model = YOLO(self.model_name)
        except Exception as e:
            print(f"Warning: Failed to load custom YOLO model '{self.model_name}'. Error: {e}")
            print("Falling back to standard yolov8n.pt")
            self.model = YOLO("yolov8n.pt")
        self.weights_path = None  # set by train()/save()/load()

    def _current_weights(self):
        """Path of the checkpoint self.model represents, or None."""
        candidates = [self.weights_path,
                      self.output_dir / "weights" / "best.pt",
                      self.output_dir / "full_model.pt"]
        for c in candidates:
            if c and Path(c).exists():
                return Path(c)
        return None

    def train(self, dataset_path: Path, epochs: int = 10, **kwargs):
        # dataset_path must point to a data.yaml
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"YOLO requires data.yaml at {dataset_path}")

        print(f"Starting YOLO training for {epochs} epochs...")
        
        # Custom Callback
        def on_train_epoch_end(trainer):
            ep = trainer.epoch + 1
            print(f"HEARTBEAT: YOLO Epoch {ep} Done", flush=True)
            
        self.model.add_callback("on_train_epoch_end", on_train_epoch_end)

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True, # overwrite if exists
            verbose=True,
            device=kwargs.pop("device", self.device),
            **kwargs
        )
        return results

    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
        import numpy as np
        import yaml

        # Resolve evaluation split FROM data.yaml so that the Ultralytics
        # map50 and the manual loop below use the SAME images. Prefer the
        # held-out 'test' split when the dataset declares one (best.pt is
        # selected on val, so reporting on val is selection-biased).
        data_yaml = dataset_path / "data.yaml"
        eval_split = "val"
        eval_img_dir = None
        if data_yaml.exists():
            with open(data_yaml) as f:
                data_conf = yaml.safe_load(f) or {}
            for split in ("test", "val"):
                p = data_conf.get(split)
                if p:
                    cand = Path(p) if Path(p).is_absolute() else (dataset_path / p)
                    if cand.exists():
                        eval_split = split
                        eval_img_dir = cand.resolve()
                        break
        if eval_img_dir is None:
            # No data.yaml — fall back to conventional layouts (val only)
            for d in [dataset_path / "test" / "images", dataset_path / "val" / "images",
                      dataset_path / "images" / "val", dataset_path / "valid" / "images",
                      dataset_path / "images" / "valid"]:
                if d.exists():
                    eval_split = "test" if "test" in d.parts else "val"
                    eval_img_dir = d
                    break

        # Ultralytics validation for map50 on the SAME split. single_cls
        # matches the training configuration (worker trains single_cls=True).
        # val() is run on a THROWAWAY instance loaded from the saved weights:
        # validation fuses the network in-place into inference-mode tensors,
        # and any later predict() on that same object fails with "Inference
        # tensors do not track version counter". self.model stays clean for
        # the manual scoring loop below.
        weights = self._current_weights()
        val_model = YOLO(str(weights)) if weights else self.model
        metrics_obj = val_model.val(data=str(data_yaml), split=eval_split,
                                    single_cls=True, verbose=False,
                                    device=self.device)
        map50_ref = metrics_obj.box.map50
        if val_model is self.model:
            # No checkpoint on disk to isolate val() with — reload is
            # impossible, so recover the predictor by re-instantiating
            # from the pretrained name (should not happen in practice:
            # train() and load() both leave weights on disk).
            self.model = YOLO(self.model_name)

        y_true = []
        y_scores = []
        y_pred = []
        skipped_no_label = 0

        if eval_img_dir:
            val_images = sorted(list(eval_img_dir.glob("*.jpg")) + list(eval_img_dir.glob("*.png")))
            print(f" evaluating {len(val_images)} images from '{eval_split}' split manually for consistency...")

            for img_path in val_images:
                 # Ground truth via the centralized resolver. None (no label
                 # file found anywhere) means we SKIP the image — treating it
                 # as negative would silently corrupt metrics when label-path
                 # resolution fails for a layout.
                 has_pothole = infer_label_from_yolo_txt(img_path)
                 if has_pothole is None:
                     skipped_no_label += 1
                     continue

                 # Prediction - run at conf=0.001 so y_scores are UNCENSORED.
                 # (Running at a higher conf censors scores below it to 0.0,
                 # which biases the ROC curve.) The uniform image-level
                 # decision threshold of 0.5 (same for YOLO / FasterRCNN /
                 # UNet, so their P/R/F1 are comparable) is applied to the
                 # same scores; AUC remains threshold-free.
                 res = self.model(str(img_path), verbose=False, conf=0.001)

                 max_conf = 0.0
                 if len(res) > 0 and len(res[0].boxes) > 0:
                      max_conf = float(res[0].boxes.conf.max().cpu().item())

                 y_true.append(has_pothole)
                 y_scores.append(max_conf)
                 y_pred.append(1 if max_conf >= 0.5 else 0)

            if skipped_no_label:
                print(f"WARNING: skipped {skipped_no_label}/{len(val_images)} images with no "
                      f"resolvable label file.")
            if y_true and len(set(y_true)) == 1:
                print("WARNING: all evaluated images have the same ground-truth class — "
                      "check the dataset's label files / negatives.")

        # Calculate Real Metrics from Prediction Loop
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            
            # ROC
            if len(set(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_list = [[float(f), float(t)] for f, t in zip(fpr, tpr)]
                # Downsample
                if len(roc_list) > 100:
                    indices = np.linspace(0, len(roc_list)-1, 100).astype(int)
                    roc_list = [roc_list[i] for i in indices]
                roc_auc = auc(fpr, tpr)
            else:
                 # ROC is undefined when y_true has a single class — report
                 # None rather than fabricating a diagonal curve / AUC 0.5.
                 roc_list = None
                 roc_auc = None
        else:
            acc, prec, rec, f1 = 0.0, 0.0, 0.0, 0.0
            cm = [[0, 0], [0, 0]]
            roc_list = None
            roc_auc = None

        return {
            "map50": float(map50_ref), # Keep map50 as ref
            "map50_source": "ultralytics.val",
            "accuracy": float(acc), # Use Consistent Loop Metrics
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "confusion_matrix": cm,
            "roc_curve": roc_list,
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "decision_threshold": 0.5,
            # P/R/F1/accuracy here are image-level presence/absence metrics,
            # NOT box-level detection metrics.
            "metric_framing": "image_level_classification",
            "eval_split": eval_split,
            "device": self.device,
            "samples": len(y_true)
        }

    def predict(self, input_data: Any, **kwargs) -> Any:
        # Input can be path or image array
        # Pass kwargs to model (e.g. imgsz=1280, conf=0.1)
        conf = kwargs.pop('conf', 0.1)
        device = kwargs.pop('device', self.device)
        return self.model(input_data, conf=conf, device=device, **kwargs)

    def save(self):
        # Ultralytics saves automatically, but we can copy best.pt to a standard location if needed
        best_pt = self.output_dir / "weights" / "best.pt"
        if best_pt.exists():
            target = self.output_dir / "full_model.pt"
            shutil.copy(best_pt, target)
            # Reload from the checkpoint: after train() the in-memory model
            # holds inference-mode tensors that the predictor cannot reuse
            # ("Inference tensors do not track version counter"), and
            # protocol-wise evaluate() must measure the SAVED weights anyway.
            self.model = YOLO(str(target))
            self.weights_path = target

    def load(self, weights_path: Path):
        self.model = YOLO(str(weights_path))
        self.weights_path = Path(weights_path)

    def extract_features(self, dataset_path: Path):
        # Real YOLO embedding extraction: register a forward hook on the SPPF
        # layer (end of the YOLOv8 backbone, usually index 9) and global-
        # average-pool its output into a fixed-size vector per image.
        import numpy as np
        # Scan images
        images = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
        count = len(images)
        if count == 0: return np.zeros((0, 0)), np.zeros((0,)), []

        print(f"Extracting Real YOLO Features for {count} images...", flush=True)

        features = []
        labels = []
        stems = []
        skipped_no_features = 0
        skipped_no_label = 0

        # Hook for feature extraction (penultimate layer)
        # YOLOv8 backbone
        self.feature_vector = None
        def hook(module, input, output):
            # global/average pool to get fixed size vector from feature map
            # output is likely [B, C, H, W]
            if output.dim() == 4:
                self.feature_vector = output.mean(dim=[2, 3]).detach().cpu().numpy()
            else:
                self.feature_vector = output.detach().cpu().numpy()

        # Register hook on the SPPF layer (usually index 9 in backbone).
        # If this fails we MUST raise — falling back to dummy/zero features
        # would silently feed garbage into the fusion models.
        try:
             target_layer = self.model.model.model[9] # SPPF usually output of backbone
             handle = target_layer.register_forward_hook(hook)
        except Exception as e:
             raise RuntimeError(f"Could not register YOLO feature hook on backbone: {e}") from e

        self.model.eval()

        try:
            for i, p in enumerate(images):
                # Reset BEFORE each forward so a failed pass can never
                # silently reuse the previous image's features.
                self.feature_vector = None
                try:
                    if i % 100 == 0:
                        print(f"YOLO Extraction Progress: {i}/{len(images)}", flush=True)
                    _ = self.model(str(p), verbose=False)
                except Exception as e:
                    print(f"Error extracting {p}: {e}")
                    skipped_no_features += 1
                    continue

                if self.feature_vector is None:
                    # Hook never fired for this image — skip it entirely.
                    skipped_no_features += 1
                    continue

                # Native dimensionality — no truncation/zero-padding to 512.
                if self.feature_vector.ndim > 1:
                    vec = self.feature_vector[0].flatten()
                else:
                    vec = self.feature_vector.flatten()

                # Centralized label inference (labels.py): YOLO txt first,
                # then path keywords. label 1 = POTHOLE ONLY.
                lbl = infer_label_from_yolo_txt(p)
                if lbl is None:
                    lbl = infer_label_from_path(p)
                if lbl is None:
                    # Unknown label — skip rather than guess (label noise).
                    skipped_no_label += 1
                    continue

                # Atomic append: all three lists together at the end of the
                # iteration so an exception can never desync them.
                features.append(vec)
                labels.append(lbl)
                stems.append(p.stem)
        finally:
            handle.remove()

        if skipped_no_features or skipped_no_label:
            print(f"YOLO extraction skipped {skipped_no_features} images (no features captured) "
                  f"and {skipped_no_label} images (no label could be inferred).", flush=True)

        assert len(features) == len(labels) == len(stems), \
            "YOLO extract_features: feature/label/stem lists desynced"

        feat_arr = np.array(features)
        if feat_arr.shape[0] > 1 and np.allclose(feat_arr, feat_arr[0]):
            raise RuntimeError(
                "YOLO feature matrix has zero variance (all rows identical) — "
                "feature extraction is broken, refusing to return it.")

        self.feature_dim = int(feat_arr.shape[1]) if feat_arr.ndim == 2 and feat_arr.shape[0] > 0 else None
        return feat_arr, np.array(labels), stems

    def extract_scores(self, dataset_path: Path):
        """One pothole probability per image, for the late-fusion benchmark.

        Score = max detection confidence at conf=0.001 (uncensored, same as
        evaluate()'s ROC scoring). Returns (scores, labels, stems) aligned by
        stem like extract_features.
        """
        import numpy as np
        images = sorted(list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png")))
        if not images:
            return np.zeros((0,)), np.zeros((0,)), []

        print(f"Extracting YOLO scores for {len(images)} images...", flush=True)
        scores, labels, stems = [], [], []
        skipped = 0
        for i, p in enumerate(images):
            if i % 100 == 0:
                print(f"YOLO Score Progress: {i}/{len(images)}", flush=True)
            try:
                res = self.model(str(p), verbose=False, conf=0.001)
                max_conf = 0.0
                if len(res) > 0 and len(res[0].boxes) > 0:
                    max_conf = float(res[0].boxes.conf.max().cpu().item())
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

            scores.append(max_conf)
            labels.append(lbl)
            stems.append(p.stem)

        if skipped:
            print(f"YOLO scoring skipped {skipped}/{len(images)} images.", flush=True)
        assert len(scores) == len(labels) == len(stems)
        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int64), stems

    def predict_embedding(self, input_data: Any) -> np.ndarray:
        """
        Runs inference and returns the pooled feature embedding at its NATIVE
        dimensionality (consistent with extract_features — no pad/truncate).
        Raises RuntimeError on failure instead of returning zeros.
        """
        self.feature_vector = None
        def hook(module, input, output):
            if output.dim() == 4:
                self.feature_vector = output.mean(dim=[2, 3]).detach().cpu().numpy()
            else:
                self.feature_vector = output.detach().cpu().numpy()

        # Register Hook
        handle = None
        try:
             # Features come from the Backbone — SPPF is idx 9 usually.
             target = self.model.model.model[9]
             handle = target.register_forward_hook(hook)

             # Run Inference
             _ = self.model(str(input_data), verbose=False) if isinstance(input_data, Path) else self.model(input_data, verbose=False)

        except Exception as e:
            raise RuntimeError(f"YOLO embedding extraction failed: {e}") from e
        finally:
            if handle: handle.remove()

        # Process Vector
        if self.feature_vector is None:
            raise RuntimeError("YOLO embedding extraction failed: hook captured no features")

        # Single-image input: take the first batch item, native dimensionality.
        if self.feature_vector.ndim > 1:
            vec = self.feature_vector[0].flatten()
        else:
            vec = self.feature_vector.flatten()
        self.feature_dim = int(vec.shape[0])
        return vec
