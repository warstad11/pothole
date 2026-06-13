
import sqlite3
import json
import datetime
from pathlib import Path

DB_PATH = "pothole_app.db"

def queue_job(type_name, image_job_id, sensor_job_id, dataset_path):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Common Args
    args = {
        "dataset_path": str(dataset_path),
        "image_job_id": image_job_id,
        # Sensor: 1D-CNN on Data 3 (Job 4)
        "sensor_job_id": 4,
        "dataset": "Data 1 - Both",
        "image_algo": "yolov8",
        "sensor_algo": "cnn"
    }
    
    if type_name == "late_fusion":
        task_type = "hybrid_eval" # Late fusion is usually eval/validation in this pipeline
        args["fusion_type"] = "late_fusion"
        args["model"] = "late_fusion"
    elif type_name == "feature_fusion":
        task_type = "train_hybrid"
        args["fusion_type"] = "feature_fusion"
        args["model"] = "feature_fusion"
        args["epochs"] = 20
        
    args_json = json.dumps(args)
    
    cursor.execute(
        "INSERT INTO job (task_type, status, created_at, args) VALUES (?, ?, ?, ?)",
        (task_type, "PENDING", timestamp, args_json)
    )
    job_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"Queued {type_name} Job ID: {job_id} (Img: {image_job_id}, Snr: {sensor_job_id})")

if __name__ == "__main__":
    # YOLO (Data 3) = Job 6
    # RF (Data 1)   = Job 3
    # Target Data   = "data/hybrid/Data 1 - Both"
    
    target_data = "data/hybrid/Data 1 - Both"
    
    print("Queueing Late Fusion...")
    queue_job("late_fusion", 6, 3, target_data)
    
    print("Queueing Feature Fusion...")
    queue_job("feature_fusion", 6, 3, target_data)
