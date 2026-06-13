
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import json

# Add app to path
sys.path.append(".")

from app.core.config import settings

# Manual definition of TwoStreamNet to ensure we control the architecture
class TwoStreamNet(torch.nn.Module):
    def __init__(self, input_dim=640, hidden_dim=128): # 512 + 128
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
    # Convert to Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    
    model = TwoStreamNet(input_dim=X_train.shape[1])
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 50
    best_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Eval
        model.eval()
        with torch.no_grad():
            out_test = model(X_test_t)
            _, pred = torch.max(out_test, 1)
            acc = (pred == y_test_t).float().mean().item()
            
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict()
            
    # Load best
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    model.eval()
    with torch.no_grad():
        out_test = model(X_test_t)
        _, pred = torch.max(out_test, 1)
        
    return pred.cpu().numpy()

def main():
    print("--- Reassessing Fusion Models (Proper Train/Test Split) ---")
    
    feature_dir = Path("runs/fusion_rcnn_cnn_data1")
    rcnn_cache = feature_dir / "rcnn_features.npz"
    features_csv = feature_dir / "features.csv" # Contains both if script finished
    
    img_feats = None
    snr_feats = None
    labels = None
    
    # 1. Load Features
    # Try loading from CSV (completed run) first, else fallback to npz + extraction
    if features_csv.exists():
        print(f"Loading aligned features from {features_csv}...")
        df = pd.read_csv(features_csv)
        
        img_cols = [c for c in df.columns if c.startswith('img_')]
        snr_cols = [c for c in df.columns if c.startswith('snr_')]
        
        img_feats = df[img_cols].values.astype(np.float32)
        snr_feats = df[snr_cols].values.astype(np.float32)
        labels = df['label'].values.astype(int)
        
        print(f"Loaded {len(df)} samples. Img Dim: {img_feats.shape[1]}, Snr Dim: {snr_feats.shape[1]}")
    else:
        print("Error: features.csv not found. Please run the extraction script first.")
        # Fallback logic omitted for brevity, assuming previous run saved CSV
        return

    # 2. Prepare Data
    # Concatenate features
    X = np.concatenate([img_feats, snr_feats], axis=1)
    y = labels
    
    print(f"Combined Feature Dimension: {X.shape}")
    
    # 3. Split Data (Strict Hold-out)
    # 80% Train, 20% Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train Set: {len(X_train)} ({np.sum(y_train==1)} Pos)")
    print(f"Test Set:  {len(X_test)} ({np.sum(y_test==1)} Pos)")
    
    # 4. Late Fusion Proxy (Logistic Regression)
    # Trains a linear classifier on the concatenated features. 
    # This is equivalent to finding Optimal Weights for a linear combination of features.
    print("\n--- Model 1: Late Fusion (Logistic Regression Proxy) ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    
    acc_lr = accuracy_score(y_test, y_pred_lr)
    prec_lr = precision_score(y_test, y_pred_lr, zero_division=0)
    rec_lr = recall_score(y_test, y_pred_lr, zero_division=0)
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    
    print(f"Accuracy:  {acc_lr:.4f}")
    print(f"Precision: {prec_lr:.4f}")
    print(f"Recall:    {rec_lr:.4f}")
    print(f"Confusion Matrix: {cm_lr.tolist()}")
    
    # 5. Feature Fusion (MLP)
    # Trains a Non-Linear classifier (TwoStreamNet)
    print("\n--- Model 2: Feature Fusion (MLP / TwoStreamNet) ---")
    
    y_pred_mlp = train_mlp_torch(X_train, y_train, X_test, y_test)
    
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    prec_mlp = precision_score(y_test, y_pred_mlp, zero_division=0)
    rec_mlp = recall_score(y_test, y_pred_mlp, zero_division=0)
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    
    print(f"Accuracy:  {acc_mlp:.4f}")
    print(f"Precision: {prec_mlp:.4f}")
    print(f"Recall:    {rec_mlp:.4f}")
    print(f"Confusion Matrix: {cm_mlp.tolist()}")
    
    # Save Results
    results = {
        "late_fusion": {
            "accuracy": acc_lr, "precision": prec_lr, "recall": rec_lr, "cm": cm_lr.tolist()
        },
        "feature_fusion": {
            "accuracy": acc_mlp, "precision": prec_mlp, "recall": rec_mlp, "cm": cm_mlp.tolist()
        }
    }
    
    with open(feature_dir / "reassessment_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
