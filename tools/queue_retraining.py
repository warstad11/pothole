
import os
from sqlmodel import create_engine, Session, select
from app.models.db import Job, JobStatus

DB_URL = "sqlite:///pothole_app.db"

# Retraining Plan based on Audit
# 1. Faster R-CNN (Data 3) - Fix Data Leakage
# 2. Sensor Models (Data 3) - Fix Temporal Leakage
# 3. Sensor Models (Data 1) - Fix Temporal Leakage? (Data 1 is hybrid, contains sensor)
#    Actually Data 1 Sensor models were also invalid.
# 4. Hybrid Models (Data 1/3) - Blocked on Component Training? No, can queue.

RETRAIN_JOBS = [
    # Faster R-CNN
    {"task_type": "train_image", "args": {"model": "faster_rcnn", "dataset_path": "data/image/raw/Data 3 - Images"}},
    
    # Data 3 Sensor
    {"task_type": "train_sensor", "args": {"model": "cnn", "dataset_path": "data/sensor/Data 3 - Just Sensor"}},
    {"task_type": "train_sensor", "args": {"model": "random_forest", "dataset_path": "data/sensor/Data 3 - Just Sensor"}},
    
    # Data 1 Sensor (Used for Hybrid)
    # Check path: data/sensor/Data 1 - Hybrid
    # Or "data/sensor/Data 1 - Sensor"? Labels are in Data 1.
    # Previous successful runs used "data/sensor/Data 1 - Hybrid" or similar.
    # Let's verify Data 1 path first? 
    # Attempting common paths based on recent usage.
    {"task_type": "train_sensor", "args": {"model": "cnn", "dataset_path": "data/sensor/Data 1 - Hybrid (Sensor)"}},
    {"task_type": "train_sensor", "args": {"model": "random_forest", "dataset_path": "data/sensor/Data 1 - Hybrid (Sensor)"}},
]

# Note: The user mentioned 65 runs. Queuing 1 run per config is sufficient if we save the model.
# We don't need to re-run 32 random seeds unless doing a stability study.
# One valid run per dataset/model is the goal.

def queue_jobs():
    engine = create_engine(DB_URL)
    with Session(engine) as session:
        print("Queueing Retraining Jobs...")
        for j_config in RETRAIN_JOBS:
            # Check if recently queued/running to avoid duplicates?
            # Simple check
            stmt = select(Job).where(Job.task_type == j_config["task_type"]).where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
            existing = session.exec(stmt).all()
            
            # Filter by args match (naive string match or dict subset)
            dup = False
            for e in existing:
                if e.args.get('model') == j_config['args']['model'] and e.args.get('dataset_path') == j_config['args']['dataset_path']:
                    dup = True
                    break
            
            if dup:
                print(f"Skipping {j_config['args']['model']} on {j_config['args']['dataset_path']} (Already Queued/Running)")
                continue

            job = Job(
                task_type=j_config["task_type"],
                args=j_config["args"],
                status=JobStatus.QUEUED
            )
            session.add(job)
            session.commit()
            print(f"Queued Job {job.id}: {j_config['args']['model']} on {j_config['args']['dataset_path']}")

if __name__ == "__main__":
    queue_jobs()
