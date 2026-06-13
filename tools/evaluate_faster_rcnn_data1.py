
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from app.services.models.image.torchvision import FasterRCNNWrapper, PotholeDetectionDataset, collate_fn
import numpy as np
import json


def run_evaluation():
    # 1. Setup
    model_path = Path("runs/55/full_model.pt")
    if not model_path.exists():
        model_path = Path("runs/55/faster_rcnn.pth")
    if not model_path.exists():
        print(f"Error: Model not found in runs/55")
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
        "num_classes": 2,
        "backbone": "resnet50"
    }
    wrapper = FasterRCNNWrapper("eval_job_55", config)
    wrapper.load(model_path)
    
    # 2. Dataset
    print(f"Loading dataset from {data_path}...")
    dataset = PotholeDetectionDataset(data_path)
    print(f"Found {len(dataset)} images.")
    
    # Use collate_fn from torchvision wrapper logic
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    
    # 3. Eval Loop
    wrapper.model.eval()
    
    device = torch.device('cpu') 
    wrapper.model.to(device)
    
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    
    print("Starting evaluation (Classification Mode: Box Presence vs JSON Label)...")
    
    # Map dataset images to index
    # dataset.imgs contains absolute paths.
    # index uses "images/filename.jpg".
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            if i % 100 == 0: print(f"Processing {i}/{len(dataset)}...", flush=True)

            # Get Ground Truth from JSON
            img_path = dataset.imgs[i]
            key = f"images/{img_path.name}"
            
            gt_label = 0
            if key in label_index:
                gt_label = label_index[key].get("label", 0)
            else:
                # Fallback: check filename keywords
                if "pothole" in img_path.name.lower() or "bump" in img_path.name.lower():
                    gt_label = 1
                    
            # Run Inference
            images = list(image.to(device) for image in images)
            outputs = wrapper.model(images)
            
            # Since batch_size=1
            output = outputs[0]
            pred_scores = output['scores'].cpu().numpy()
            
            # Predict Positive if ANY box has score > 0.5
            pred_label = 0
            if len(pred_scores) > 0 and np.max(pred_scores) > 0.5:
                pred_label = 1
            
            # Update Metrics
            if gt_label == 1 and pred_label == 1:
                tp += 1
            elif gt_label == 1 and pred_label == 0:
                fn += 1
            elif gt_label == 0 and pred_label == 1:
                fp += 1
            elif gt_label == 0 and pred_label == 0:
                tn += 1

    # 4. Metrics
    total_samples = tp + fp + tn + fn # Should match len(dataset)
    total_positives = tp + fn
    total_negatives = tn + fp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = (tp + tn) / total_samples if total_samples > 0 else 0.0
    
    print("\nResults (Faster R-CNN Data 3 -> Data 1):")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix: [[{tn} (TN), {fp} (FP)], [{fn} (FN), {tp} (TP)]]")
    print(f"Samples: {total_samples} (Pos: {total_positives}, Neg: {total_negatives})")

if __name__ == "__main__":
    run_evaluation()
