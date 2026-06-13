
import sys
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.getcwd())

from app.services.models.image.yolo import YOLOWrapper

def main():
    # 1. Setup paths
    model_path = Path("runs/6/full_model.pt")
    data_path = Path("data/hybrid/Data 1 - Both/images")
    
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return

    print(f"Loading YOLO model from {model_path}...")
    # Initialize Wrapper (Job ID '6', dummy config)
    wrapper = YOLOWrapper("6", {}) 
    wrapper.load(model_path)
    
    # 2. Iterate Images
    images = list(data_path. glob("*.jpg")) + list(data_path.glob("*.png"))
    print(f"Found {len(images)} images in {data_path}")
    
    y_true = []
    y_pred = []
    
    print("Evaluating...")
    for i, img_path in enumerate(images):
        if i % 100 == 0: print(f"Processing {i}/{len(images)}...", flush=True)
        
        # Ground Truth from Filename
        # E.g. "10_Pothole_3136.jpg" -> 1
        # "100_Normal_0.jpg" -> 0
        fname = img_path.name.lower()
        if any(k in fname for k in ["pothole", "positive", "damage", "cracking", "bump"]):
            gt = 1
        elif any(k in fname for k in ["normal", "plain", "negative", "undamaged", "regular", "road", "joint"]):
            gt = 0
        else:
            # Fallback or Skip? Assume Normal if unsure?
            # User data seems consistent: Pothole, Normal, Bump.
            gt = 0
            
        y_true.append(gt)
        
        # Prediction
        # YOLOWrapper.model is the Ultralytics model
        # conf=0.25 is standard
        results = wrapper.model(str(img_path), verbose=False, conf=0.25)
        
        pred = 0
        if results and len(results) > 0:
            if len(results[0].boxes) > 0:
                # If any box detected (class 0 is usually 'pothole' in this project single-class)
                # But let's check class 0
                # Assuming single class "pothole"
                pred = 1
        
        y_pred.append(pred)

    # 3. Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print("\n=== Eval Results: YOLO (Data 3) on Data 1 ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("========================================")

if __name__ == "__main__":
    main()
