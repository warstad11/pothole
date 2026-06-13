
import sys
import torch
import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

# Define the Fusion Model Class (Must match what we load in inference)
class FusionNet(torch.nn.Module):
    def __init__(self, input_dim=640, hidden_dim=128):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 2)
        )
    def forward(self, x):
        return self.fc(x)

def main():
    print("--- TRAINING FUSION MODEL FOR DEPLOYMENT ---")
    
    # 1. Load Features (YOLO + 1D-CNN)
    # Using the cached features from previous analysis
    yolo_path = Path("runs/fusion_yolo_data1/yolo_features.npz")
    cnn_path = Path("runs/fusion_rcnn_cnn_data1/features_full_checkpoint.npz")
    dataset_path = Path("data/hybrid/Data 1 - Both")
    index_path = dataset_path / "dataset_index.json"
    
    if not yolo_path.exists() or not cnn_path.exists():
        print("Error: Cached features not found. Please run evaluation scripts first.")
        return

    # Load Labels
    with open(index_path, 'r') as f:
        index = json.load(f)
    sorted_keys = sorted(index.keys())
    labels = np.array([index[k]['label'] for k in sorted_keys])
    
    # Load Data
    yd = np.load(yolo_path)
    yolo_feats = yd['feats'] # 512-dim
    
    cd = np.load(cnn_path)
    cnn_feats = cd['snr_feats'] # 128-dim
    
    # Concatenate
    X = np.concatenate([yolo_feats, cnn_feats], axis=1) # 640-dim (512+128)
    
    # CRITICAL FIX: Normalization
    # YOLO features are likely 0-1 or varying. CNN features might be different scale.
    # We MUST fit a scaler on the training data and SAVE it for inference.
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42, stratify=labels)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save Scaler
    save_dir = Path("models/fusion/yolo_cnn")
    save_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, save_dir / "scaler.pkl")
    print(f"Scaler saved to {save_dir / 'scaler.pkl'}")
    
    # Train
    model = FusionNet(input_dim=640)
    device = torch.device("cpu") # Simple MLP fits on CPU
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    X_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    
    print(f"Training on {len(X_train)} samples...")
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        out = model(X_t)
        loss = criterion(out, y_t)
        loss.backward()
        optimizer.step()
        
    # Evaluate
    model.eval()
    with torch.no_grad():
        out_test = model(torch.tensor(X_test_scaled, dtype=torch.float32))
        preds = torch.argmax(out_test, dim=1).numpy()
        
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc*100:.2f}%")
    
    if acc > 0.90:
        torch.save(model.state_dict(), save_dir / "model.pth")
        print(f"Model saved to {save_dir / 'model.pth'}")
    else:
        print("Warning: Accuracy too low. Model NOT saved.")

if __name__ == "__main__":
    main()
