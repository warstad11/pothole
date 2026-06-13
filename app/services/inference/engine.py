
from pathlib import Path
from typing import Any, List, Dict
import pandas as pd
import numpy as np
from app.core.config import SENSOR_RESAMPLE_HZ
from app.services.ingestion.alignment import DriveTimeline

class InferenceEngine:
    def __init__(self, model_path: Path = None, model_config: Dict = None):
        self.model_path = model_path
        self.config = model_config
        self.model = None
        self._load_model()
            
    def _load_model(self):
        """
        Loads the model from the specified path or config.
        """
        from app.core.config import settings
        import torch
        from app.services.models.image import YOLOWrapper, FasterRCNNWrapper, UNetWrapper
        from app.services.models.sensor import CNN1DWrapper, RandomForestWrapper
        
        # 1. Load Image Model
        # Check explicit types first
        if self.config and self.config.get("image_model") == "faster_rcnn":
            weights = None
            if self.config.get("image_model_path"):
                weights = Path(self.config["image_model_path"])
            if weights is None or not weights.exists():
                weights = settings.MODELS_DIR / "latest" / "image" / "faster_rcnn.pth"
            print(f"Loading Faster R-CNN from {weights}")
            if weights.exists():
                self.image_model = FasterRCNNWrapper("vision_inference", {"num_classes": 2})
                d = self.image_model.device
                self.image_model.model.load_state_dict(torch.load(weights, map_location=d))
                self.image_model.model.eval()
            else:
                 print(f"Faster R-CNN weights not found at {weights}")
                 self.image_model = None

        elif self.config and self.config.get("image_model") == "unet":
             weights = None
             if self.config.get("image_model_path"):
                 weights = Path(self.config["image_model_path"])
             if weights is None or not weights.exists():
                 weights = settings.MODELS_DIR / "latest" / "image" / "unet.pth"
             print(f"Loading U-Net from {weights}")
             if weights.exists():
                 self.image_model = UNetWrapper("vision_inference", {"architecture": "unet", "backbone": "resnet34"})
                 d = self.image_model.device
                 self.image_model.model.load_state_dict(torch.load(weights, map_location=d))
                 self.image_model.model.eval()
             else:
                 print(f"U-Net weights not found at {weights}")
                 self.image_model = None
                 
        else:
            # Fallback to YOLO logic (Path or Default)
            img_model_path = None
            if self.config and self.config.get("image_model"):
                mname = self.config["image_model"]
                possible_path = settings.MODELS_DIR / "image" / mname / "full_model.pt"
                if possible_path.exists():
                    img_model_path = possible_path
                elif Path(mname).exists():
                    img_model_path = Path(mname)
            
            if not img_model_path and settings.LATEST_IMAGE_MODEL.exists():
                img_model_path = settings.LATEST_IMAGE_MODEL
                
            if img_model_path and img_model_path.exists():
                print(f"Loading YOLO model from {img_model_path}")
                self.image_model = YOLOWrapper("vision_inference", {"model_name": str(img_model_path)})
            else:
                print("No suitable YOLO model found, using fallback yolov8n.pt")
                self.image_model = None # YOLOWrapper defaults to yolov8n.pt if accessed
            
        # 2. Load Sensor Model
        snr_model_path = None
        if self.config and self.config.get("sensor_model"):
            mname = self.config["sensor_model"]
            possible_path = settings.MODELS_DIR / "sensor" / mname / "full_model.pth"
            if possible_path.exists():
                snr_model_path = possible_path
            elif Path(mname).exists():
                snr_model_path = Path(mname)

        # An explicit model_path arg pointing at a sensor checkpoint takes
        # precedence over the deployment default (previously it was ignored).
        if (not snr_model_path and self.model_path
                and 'sensor' in Path(self.model_path).parts
                and Path(self.model_path).exists()):
            snr_model_path = Path(self.model_path)

        if not snr_model_path and settings.LATEST_SENSOR_MODEL.exists():
            snr_model_path = settings.LATEST_SENSOR_MODEL
        if not snr_model_path:
            # RF deploys under .joblib (the suffix routes the loader)
            rf_deploy = settings.LATEST_SENSOR_MODEL.with_suffix(".joblib")
            if rf_deploy.exists():
                snr_model_path = rf_deploy
            
        self.sensor_model = None
        if snr_model_path and snr_model_path.exists():
            print(f"Loading sensor model from {snr_model_path}")
            if snr_model_path.suffix == ".pth":
                # Only accelerometer (3 channels) is available in the sensor hardware
                self.sensor_model = CNN1DWrapper("sensor_inference", {"input_channels": 3})
                self.sensor_model.load(snr_model_path)
                self.sensor_model.model.eval()
                self.sensor_channels = 3
            else:
                self.sensor_model = RandomForestWrapper("sensor_inference", {})
                self.sensor_model.load(snr_model_path)
                self.sensor_channels = 3
        else:
            print("No suitable sensor model found, using heuristic peak detection.")

        # 3. Load Fusion Model (Feature Fusion)
        self.fusion_model = None

        # Priority: explicit config > explicit model_path arg (when it points
        # at a hybrid/fusion checkpoint) > deployed latest > legacy default.
        fusion_path = settings.MODELS_DIR / "fusion" / "yolo_cnn" / "model.pth"
        if settings.LATEST_HYBRID_MODEL.exists():
            fusion_path = settings.LATEST_HYBRID_MODEL
        if (self.model_path
                and {'hybrid', 'fusion'} & set(Path(self.model_path).parts)
                and Path(self.model_path).exists()):
            fusion_path = Path(self.model_path)

        if self.config and self.config.get("fusion_model_path"):
            path_str = self.config.get("fusion_model_path")
            if path_str == "DISABLED":
                fusion_path = None
            else:
                fusion_path = Path(path_str)

        if fusion_path and fusion_path.exists():
             import json
             # Prefer the explicit architecture config saved next to the checkpoint
             # (written by the fusion wrapper on save) over path-based guessing.
             fusion_cfg = {}
             cfg_path = fusion_path.parent / "config.json"
             if cfg_path.exists():
                 try:
                     with open(cfg_path, "r") as f:
                         saved_cfg = json.load(f)
                     for k in ("input_dim", "image_dim", "sensor_dim", "channels"):
                         if k in saved_cfg:
                             fusion_cfg[k] = saved_cfg[k]
                 except Exception as e:
                     print(f"WARNING: Failed to read fusion config.json at {cfg_path}: {e}")

             if "input_dim" not in fusion_cfg:
                 # Legacy fallback: guess architecture from the checkpoint path.
                 input_dim = 527 if "unet_rf" in str(fusion_path) else 640
                 print(f"WARNING: No config.json found next to {fusion_path}; "
                       f"guessing input_dim={input_dim} from path heuristic.")
                 fusion_cfg["input_dim"] = input_dim

             print(f"Loading Fusion Model from {fusion_path} (dim={fusion_cfg['input_dim']})")
             from app.services.models.hybrid.fusion import FeatureFusionWrapper
             self.fusion_model = FeatureFusionWrapper("fusion_inference", fusion_cfg)
             self.fusion_model.load(fusion_path)
        else:
             print("No trained Fusion Model found at models/fusion/yolo_cnn/model.pth.")

    def process_vision(self, video_path: Path, sampling_hz: float = 5.0) -> Dict[float, Any]:
        """
        Runs YOLO on video. Returns dict of {time: (score, embedding)} if fusion active, else {time: score}.
        """
        import cv2
        import torch
        
        if not video_path.exists():
            return {}
            
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        
        frame_step = int(round(fps / sampling_hz))
        if frame_step < 1: frame_step = 1
        effective_hz = fps / frame_step
        print(f"Vision sampling: requested {sampling_hz:.3f}Hz, effective {effective_hz:.3f}Hz "
              f"(fps={fps:.3f}, frame_step={frame_step})")
        
        vision_results = {}
        
        yolo = self.image_model
        if not yolo:
            from app.services.models.image import YOLOWrapper
            yolo = YOLOWrapper("vision_inference", {"model_name": "yolov8n.pt"})
        
        use_fusion = (self.fusion_model is not None)
            
        print(f"Starting YOLO inference at {sampling_hz}Hz (Fusion: {use_fusion})")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_idx = 0
        processed_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                    
                if frame_idx % frame_step == 0:
                    max_score = 0.0
                    embedding = None
                    
                    if use_fusion:
                        try:
                            # 1. Get Embedding
                            embedding = yolo.predict_embedding(frame)
                        except AttributeError: 
                            pass 

                        # 2. Get Score (for Fallback/Display)
                        # Re-running predict is slightly inefficient but safest for now as predict_embedding doesn't parse boxes.
                        # 2. Get Score (for Fallback/Display)
                        m_type = type(yolo).__name__
                        if "YOLOWrapper" in m_type:
                            results = yolo.predict(frame, imgsz=1280)
                            if results and len(results) > 0:
                                r = results[0]
                                if hasattr(r, 'boxes') and r.boxes is not None:
                                    confs = r.boxes.conf
                                    cls = r.boxes.cls
                                    pothole_indices = (cls == 0).nonzero(as_tuple=True)[0]
                                    if len(pothole_indices) > 0:
                                        max_score = float(torch.max(confs[pothole_indices]))
                        
                        elif "UNetWrapper" in m_type:
                            # Standardize Preprocessing (should match wrapper/predict_embedding)
                            img_rs = cv2.resize(frame, (256, 256))
                            img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                            t = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0).to(yolo.device)
                            with torch.no_grad():
                                out = yolo.model(t)
                                # Max Prob
                                max_score = torch.sigmoid(out).max().item()
                                
                        elif "FasterRCNNWrapper" in m_type:
                             img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                             t = torch.tensor(img_rgb).permute(2, 0, 1).to(yolo.device)
                             res = yolo.predict([t])
                             if res and len(res) > 0 and len(res[0]['scores']) > 0:
                                 max_score = float(res[0]['scores'].max())
                        
                        curr_video_time = frame_idx / fps
                        vision_results[curr_video_time] = (max_score, embedding)

                    else:
                        # Legacy Fast Path (Score Only)
                        # ... Handle standard models ...
                        model_type = type(self.image_model).__name__ if self.image_model else "YOLOWrapper"
                        
                        if model_type == "FasterRCNNWrapper":
                             # R-CNN Logic
                             img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                             img_tensor = torch.tensor(img_rgb).permute(2, 0, 1)
                             try:
                                dev = self.image_model.device
                                res = self.image_model.predict([img_tensor.to(dev)])
                                if res and len(res) > 0:
                                    out = res[0]
                                    if len(out['scores']) > 0:
                                        max_score = float(out['scores'].max())
                             except Exception as e: print(e)
                             
                        elif model_type == "UNetWrapper":
                             # U-Net Logic — resize to 256x256 to match training /
                             # predict_embedding preprocessing (see unet.py)
                             img_rs = cv2.resize(frame, (256, 256))
                             img_rgb = cv2.cvtColor(img_rs, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                             img_tensor = torch.tensor(img_rgb).permute(2, 0, 1).unsqueeze(0)
                             try:
                                dev = self.image_model.device
                                with torch.no_grad():
                                    out = self.image_model.model(img_tensor.to(dev))
                                    prob = torch.sigmoid(out)
                                    max_score = float(prob.max())
                             except Exception as e: print(e)
                        
                        else:
                             # YOLO logic
                             results = yolo.predict(frame, imgsz=1280)
                             if results and len(results) > 0:
                                 r = results[0]
                                 if hasattr(r, 'boxes') and r.boxes is not None:
                                     confs = r.boxes.conf
                                     cls = r.boxes.cls
                                     pothole_indices = (cls == 0).nonzero(as_tuple=True)[0]
                                     if len(pothole_indices) > 0:
                                         max_score = float(torch.max(confs[pothole_indices]))
                        
                        curr_video_time = frame_idx / fps
                        vision_results[curr_video_time] = max_score
                        
                    processed_count += 1
                    
                    if processed_count % 50 == 0:
                        progress = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
                        print(f"Inference Progress: {progress:.1f}%", flush=True)
                
                frames_to_skip = frame_step - 1
                for _ in range(frames_to_skip):
                    cap.grab()
                    frame_idx += 1
                frame_idx += 1
        finally:
            cap.release()
            
        return vision_results

    def process_sensor_dl(self, sensor_df: pd.DataFrame, timeline=None) -> Dict[float, Any]:
        """
        Runs DL sensor model. Returns {time: (score, embedding)} if fusion active.
        """
        import torch
        scores = {}
        if sensor_df.empty or not self.sensor_model:
            return {}
            
        col_map = {
            'accelerometer_x': 'accel_x', 'accelerometer_y': 'accel_y', 'accelerometer_z': 'accel_z',
        }
        df = sensor_df.rename(columns=col_map)
        final_cols = ['accel_x', 'accel_y', 'accel_z']
        for c in final_cols:
            if c not in df.columns: df[c] = 0.0

        data = df[final_cols].values.astype(np.float32)
        if 'seconds_elapsed' in df.columns:
            times = df['seconds_elapsed'].values
        else:
            print(f"WARNING: sensor data has no 'seconds_elapsed' timestamps; "
                  f"assuming uniform {SENSOR_RESAMPLE_HZ}Hz sampling.")
            times = np.arange(len(data)) * (1.0 / SENSOR_RESAMPLE_HZ)
        
        window_size = 100
        stride = 20
        use_fusion = (self.fusion_model is not None)
        
        num_windows = (len(data) - window_size) // stride + 1
        for i in range(num_windows):
            start = i * stride
            end = start + window_size
            window = data[start:end]
            
            # Hybrid Embedding
            embedding = None
            if use_fusion:
                 try:
                     embedding = self.sensor_model.predict_embedding(window) # numpy
                 except AttributeError: pass

            score = 0.0
            
            # Check Model Type (RF vs DL)
            model_type = type(self.sensor_model).__name__
            
            if "RandomForest" in model_type:
                 # RF Logic (expects window features or raw window handled by wrapper logic?)
                 # predict_embedding returns features. Does `proba` handle raw window?
                 # RF Wrapper .proba() is usually for features.
                 # Actually, RF wrapper `proba` calls `predict_proba`.
                 # But we passed raw data to `predict_embedding`.
                 # Let's assume we need to extract features for prediction too.
                 # Wait, `predict_embedding` returns the 15-dim features!
                 # So we can pass `embedding` (reshaped) to `proba`?
                 
                 if embedding is not None:
                     # embedding is (15,)
                     # RF expects (n_samples, n_features) -> (1, 15)
                     try:
                         probs = self.sensor_model.model.predict_proba(embedding.reshape(1, -1))
                         score = float(probs[0, 1])
                     except Exception as e:
                         # Fallback or error logging
                         score = 0.0
                 else:
                     # If fusion off, we still need score.
                     # Duplicate extraction logic?
                     # Better to just call predict_embedding regardless of use_fusion if we are RF
                     # But predict_embedding was inside "if use_fusion".
                     # Let's fix that.
                     pass 
            else:
                 # DL Logic — go through the wrapper's predict() so deployed
                 # preprocessing (per-channel z-score normalization) matches
                 # training. Input: numpy (Time, Channels); output: (1, num_classes) probs.
                 probs = self.sensor_model.predict(window)
                 score = float(probs[0, 1])

            # Re-consolidate RF logic block to ensure score is computed
            if "RandomForest" in model_type:
                if embedding is None: # If not computed above
                     try:
                         # We need features. RF Wrapper has `predict_embedding`. Use it.
                         feat_vec = self.sensor_model.predict_embedding(window)
                         probs = self.sensor_model.model.predict_proba(feat_vec.reshape(1, -1))
                         score = float(probs[0, 1])
                     except: score = 0.0
            
            # Determine v_time mapping...
            center_idx = start + window_size // 2
            t = times[center_idx]
            v_time = timeline.rel_to_video(t) if timeline else t
            if v_time >= 0:
                if use_fusion:
                    scores[v_time] = (score, embedding)
                else:
                    scores[v_time] = score
                
        return scores

    def process_drive(self, drive_path: Path) -> List[Dict]:
        """
        Runs Feature Fusion (YOLO+CNN) pipeline.
        """
        # 1. Load sensor data
        csv_path = drive_path / "sensor.csv"
        sensor_df = pd.DataFrame()
        if csv_path.exists():
            try: sensor_df = pd.read_csv(csv_path)
            except: pass
            
        # 2. Alignment
        timeline = None
        if (drive_path / "alignment.json").exists():
            timeline = DriveTimeline.load(drive_path)
        
        # 3. Sensor Inference
        sensor_results = {} 
        if self.sensor_model:
            print("Running Sensor Inference (DL)...")
            sensor_results = self.process_sensor_dl(sensor_df, timeline)
        else:
            # Heuristic Fallback
            print("Using Heuristic Sensor Peak Detection...")
            if not sensor_df.empty and 'accel_z' in sensor_df.columns:
                z = sensor_df['accel_z'].values
                if 'seconds_elapsed' in sensor_df.columns:
                    times = sensor_df['seconds_elapsed'].values
                else:
                    print(f"WARNING: sensor data has no 'seconds_elapsed' timestamps; "
                          f"assuming uniform {SENSOR_RESAMPLE_HZ}Hz sampling.")
                    times = np.arange(len(z)) * (1.0 / SENSOR_RESAMPLE_HZ)
                z_diff = np.abs(np.diff(z, prepend=z[0]))
                s_scores = np.clip(z_diff / 0.8, 0, 1.0)
                for t, s in zip(times, s_scores):
                    v_time = timeline.rel_to_video(t) if timeline else t
                    if v_time >= 0: sensor_results[v_time] = s

        # 4. Vision Inference (5Hz)
        video_path = drive_path / "video.mp4"
        if not video_path.exists():
            vids = list(drive_path.glob("*.mp4")) + list((drive_path / "Camera").glob("*.mp4"))
            if vids: video_path = vids[0]
                
        vision_results = {}
        if video_path and video_path.exists():
            print("Running Vision Inference...")
            vision_results = self.process_vision(video_path, sampling_hz=5.0)

        # 5. Hybrid Fusion
        all_times = sorted(set(sensor_results.keys()) | set(vision_results.keys()))
        preds = []
        
        use_fusion = (self.fusion_model is not None)
        
        for t in all_times:
            # Extract Sensor Data
            s_data = sensor_results.get(t, None)
            s_score = 0.0
            s_emb = None
            
            if s_data is not None:
                if isinstance(s_data, tuple):
                    s_score, s_emb = s_data
                else: 
                    s_score = float(s_data)

            # Extract Vision Data (Align)
            v_score = 0.0
            v_emb = None
            
            if vision_results:
                v_times = np.array(list(vision_results.keys()))
                diffs = np.abs(v_times - t)
                min_idx = np.argmin(diffs)
                if diffs[min_idx] < 0.2: 
                     v_data = vision_results[v_times[min_idx]]
                     if isinstance(v_data, tuple):
                         v_score, v_emb = v_data
                     else:
                         v_score = float(v_data)
            
            final_score = 0.0
            method = 'fallback'
            
            # --- FUSION LOGIC ---
            if use_fusion and v_emb is not None and s_emb is not None:
                # Validate embedding dims against the fusion head ONCE. A
                # mismatch (e.g. a checkpoint trained on differently-sized
                # embeddings) must disable fusion loudly for the whole run —
                # the old per-frame try/except silently replaced every
                # "fusion" prediction with the max() heuristic while the run
                # still claimed a fusion model was loaded.
                if not getattr(self, '_fusion_dims_checked', False):
                    got = int(np.asarray(v_emb).size + np.asarray(s_emb).size)
                    expected = int(self.fusion_model.input_dim)
                    if got != expected:
                        print(f"ERROR: fusion checkpoint expects input_dim={expected} but "
                              f"embeddings provide {got} (image {np.asarray(v_emb).size} + "
                              f"sensor {np.asarray(s_emb).size}). Disabling fusion for this "
                              f"run — retrain the fusion head against the current "
                              f"extractors. Falling back to max(v,s).")
                        use_fusion = False
                        self.fusion_model = None
                    self._fusion_dims_checked = True

            if use_fusion and self.fusion_model is not None and v_emb is not None and s_emb is not None:
                # Pass tuple. predict scales and infers. Real errors propagate —
                # no silent per-frame fallback.
                final_score = self.fusion_model.predict((v_emb, s_emb))
                method = 'feature_fusion'
            else:
                # Fallback to Late Fusion Heuristic if model/features missing
                final_score = max(s_score, v_score)

            # Keep ALL scored timestamps (including zeros) so the prediction
            # series stays uniform; thresholding happens downstream in event
            # generation, not here.
            preds.append({
                'time': t,
                'score': float(final_score),
                'v_score': float(v_score),
                's_score': float(s_score),
                'method': method
            })

        return preds
