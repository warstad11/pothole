import sys
import os
from pathlib import Path
import json
import torch

# Add project root to path
sys.path.append(os.getcwd())

from app.services.models.sensor.dl import CNN1DWrapper

def main():
    # 1. Setup paths
    model_path = Path("runs/5/model.pth") # Data 1 1D-CNN (Job 5)
    data_path = Path("data/hybrid/Data 1 - Both/sensor")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print("CLASSICAL MODULE LOADED V5 - DEBUG ACTIVE")
    print(f"Loading 1D-CNN model from {model_path}...")

    # 2. Config (Must match training config of Job 4)
    # Job 4 args: {"architecture": "1D-CNN", "layers": 3, "epochs": 20, "hidden_dim": 64}
    # We need strictly 'input_channels' and 'num_classes'. Default is 3 and 2.
    config = {
        "input_channels": 3,
        "num_classes": 2,
        "window_size": 100, # Standard
        "stride": 20       # Standard
    }

    model = CNN1DWrapper("eval_run", config)
    
    # 3. Load Weights
    # CNN1DWrapper.load expects a path, usually `metrics.json` parent? No, `load` takes weights_path directly?
    # Let's check `load` signature in dl.py: def load(self, weights_path: Path)
    # And it does `torch.load(weights_path)`
    model.load(model_path)
    
    # Need to set output_dir for metrics saving
    model.output_dir = Path("runs/eval_cnn_cross")
    model.output_dir.mkdir(parents=True, exist_ok=True)

    # 4. Evaluate using the built-in evaluate method (which uses the patched _load_data)
    print(f"Evaluating on {data_path}...")
    metrics = model.evaluate(data_path)
    
    print("\n=== Eval Results: 1D-CNN (Data 3) on Data 1 ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("==========================================")

if __name__ == "__main__":
    main()
