
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from app.services.models.image.unet import UNetWrapper
import numpy as np
import cv2
import json

class SimpleImageDataset(Dataset):
    def __init__(self, root: Path, size=256):
        self.root = root
        self.size = size
        self.imgs = sorted(list(root.rglob("*.jpg")) + list(root.rglob("*.png")))

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = cv2.imread(str(img_path))
        if image is None: return self.__getitem__((idx + 1) % len(self.imgs))
        
        # U-Net needs 256x256
        image = cv2.resize(image, (self.size, self.size))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # To Tensor (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1)
        
        # Return dummy mask (not used for this specific binary eval)
        return image, torch.zeros((1, self.size, self.size))

    def __len__(self):
        return len(self.imgs)

def run_evaluation():
    # 1. Setup
    model_path = Path("runs/60/unet.pth")
    if not model_path.exists():
        print(f"Error: Model not found in runs/60")
        return

    data_root = Path("data/hybrid/Data 1 - Both")
    data_path = data_root / "images"
    json_path = data_root / "dataset_index.json"
    
    if not data_path.exists() or not json_path.exists():
        print(f"Error: Data path {data_path} or JSON {json_path} not found")
        return

    # Load Labels
    print(f"Loading labels from {json_path}...")
    with open(json_path, 'r') as f:
        label_index = json.load(f)

    print(f"Loading model from {model_path}...")
    config = {
        "encoder": "resnet34",
        "num_classes": 1
    }
    wrapper = UNetWrapper("eval_job_60", config)
    wrapper.load(model_path)
    
    # 2. Dataset
    print(f"Loading dataset from {data_path}...")
    dataset = SimpleImageDataset(data_path)
    print(f"Found {len(dataset)} images.")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # 3. Eval Loop
    wrapper.model.eval()
    device = torch.device('cpu') # Use CPU for safety
    wrapper.model.to(device)
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    print("Starting evaluation (Classification Mode: Max Pixel > 0.5 vs JSON Label)...")
    
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i % 100 == 0: print(f"Processing {i}/{len(dataset)}...", flush=True)

            # Ground Truth
            img_path = dataset.imgs[i]
            key = f"images/{img_path.name}"
            
            gt_label = 0
            if key in label_index:
                gt_label = label_index[key].get("label", 0)
            else:
                 if "pothole" in img_path.name.lower() or "bump" in img_path.name.lower():
                    gt_label = 1
            
            # Prediction
            images = images.to(device)
            outputs = wrapper.model(images) # (B, 1, H, W) logits
            probs = torch.sigmoid(outputs)
            
            # Logic: If any pixel > 0.5, we say "Pothole Detected"
            # This matches the "image classification" metric in unet.py
            max_score = probs.max().item()
            pred_label = 1 if max_score > 0.5 else 0
            
            if gt_label == 1 and pred_label == 1: tp += 1
            elif gt_label == 1 and pred_label == 0: fn += 1
            elif gt_label == 0 and pred_label == 1: fp += 1
            elif gt_label == 0 and pred_label == 0: tn += 1
            
    # 4. Metrics
    total_samples = tp + fp + tn + fn
    total_positives = tp + fn
    total_negatives = tn + fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    print("\nResults (U-Net Data 3 -> Data 1):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix: [[{tn} (TN), {fp} (FP)], [{fn} (FN), {tp} (TP)]]")
    print(f"Samples: {total_samples} (Pos: {total_positives}, Neg: {total_negatives})")

if __name__ == "__main__":
    run_evaluation()
