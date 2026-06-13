
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

sys.path.append(".")
from app.services.models.sensor.dl import CNN1DWrapper

# Fusion Model (Input 512 + 128 = 640)
class TwoStreamNet(torch.nn.Module):
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

def train_mlp_torch(X_train, y_train, X_test, y_test, device='cpu'):
    model = TwoStreamNet(input_dim=X_train.shape[1])
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50
    best_acc = 0.0
    best_state = None
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            out_test = model(X_test_t)
            _, pred = torch.max(out_test, 1)
            acc = (pred == y_test_t).float().mean().item()
            
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
            
    if best_state: model.load_state_dict(best_state)
    return model

def main():
    print("--- EVALUATION: U-NET + 1D-CNN FUSION ---")
    
    dataset_path = Path("data/hybrid/Data 1 - Both")
    index_path = dataset_path / "dataset_index.json"
    
    with open(index_path, 'r') as f:
        index = json.load(f)
        
    # Get labels in sorted order
    sorted_keys = sorted(index.keys())
    labels = np.array([index[k]['label'] for k in sorted_keys])
    
    # 1. Load U-Net Features & Probs (From previous run)
    unet_checkpoint = Path("runs/fusion_unet_rf_data1/unet_features.npz")
    if not unet_checkpoint.exists():
        print("Error: U-Net cache not found. Please run U-Net analysis first.")
        return
        
    print("Loading U-Net features...")
    ud = np.load(unet_checkpoint)
    unet_feats = ud['feats']  # 512-dim
    unet_probs = ud['probs']
    
    # 2. Load 1D-CNN Features (From previous run)
    cnn_checkpoint = Path("runs/fusion_rcnn_cnn_data1/features_full_checkpoint.npz")
    if not cnn_checkpoint.exists():
        print("Error: 1D-CNN cache not found. Please run R-CNN analysis first.")
        return
        
    print("Loading 1D-CNN features...")
    cd = np.load(cnn_checkpoint)
    cnn_feats = cd['snr_feats'] # 128-dim
    
    # 3. Compute 1D-CNN Probabilities (Missing from cache)
    print("Computing 1D-CNN Probabilities (Recalculating)...")
    
    # We need to stage files again just for 1D-CNN?
    # Or just iterate the original paths from dataset_index?
    # Index paths are relative.
    
    cnn_probs = []
    
    cnn_weights = Path("runs/4/model.pth")
    cnn = CNN1DWrapper("cnn_eval", {"input_channels": 3, "window_size": 200, "hidden_dims": [64,128,256], "num_classes": 2, "dataset_type": "sensor"})
    cnn.load(cnn_weights)
    
    for i, k in enumerate(sorted_keys):
        if i % 100 == 0: print(f"{i}/{len(sorted_keys)}")
        
        rel_path = index[k]['sensor_path']
        fpath = dataset_path / rel_path
        
        try:
            df = pd.read_csv(fpath)
            vals = df.select_dtypes(include=[np.number]).values
            if vals.shape[1] >= 3: vals = vals[:, :3] # Ax, Ay, Az
            
            # Predict
            # cnn.predict expects numpy
            prob_vec = cnn.predict(vals) # [[p0, p1]]
            pothole_prob = prob_vec[0][1]
            cnn_probs.append(pothole_prob)
        except Exception as e:
            print(f"Error {fpath}: {e}")
            cnn_probs.append(0.0)
            
    cnn_probs = np.array(cnn_probs)
    
    # 4. Late Fusion (Avg)
    print("\n--- Late Fusion Metrics ---")
    lf_preds = ((unet_probs + cnn_probs) / 2.0 > 0.5).astype(int)
    
    print(f"Late Fusion Acc:  {accuracy_score(labels, lf_preds):.4f}")
    print(f"Late Fusion Prec: {precision_score(labels, lf_preds):.4f}")
    print(f"Late Fusion Rec:  {recall_score(labels, lf_preds):.4f}")
    
    # 5. Feature Fusion (MLP)
    print("\n--- Feature Fusion Training (Adapted) ---")
    # Concat: U-Net (512) + 1D-CNN (128) = 640
    X = np.concatenate([unet_feats, cnn_feats], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    model = train_mlp_torch(X_train, y_train, X_test, y_test)
    
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X_test, dtype=torch.float32))
        preds = torch.argmax(out, dim=1).numpy()
        
    print(f"Feature Fusion Acc:  {accuracy_score(y_test, preds):.4f}")
    print(f"Feature Fusion Prec: {precision_score(y_test, preds):.4f}")
    print(f"Feature Fusion Rec:  {recall_score(y_test, preds):.4f}")
    print(f"Confusion Matrix: {confusion_matrix(y_test, preds).tolist()}")

if __name__ == "__main__":
    main()
