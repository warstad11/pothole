
import os
from sqlmodel import create_engine, Session, select, delete
from app.models.db import Job, JobStatus

DB_URL = "sqlite:///pothole_app.db"

# FULL RETRAINING MATRIX (Prioritized)
# Order: RF -> 1D-CNN -> YOLO -> RCNN -> UNet

IMAGE_DATASETS = [
    "data/image/raw/Data 3 - Images",
    "data/image/raw/RealPothole600_Converted",
    "data/image/raw/Pothole.v1-raw.yolov8"
]

SENSOR_DATASETS = [
    "data/sensor/Data 3 - Just Sensor",
    "data/sensor/Data 1"
]

def clear_queue(session):
    print("Clearing existing QUEUED jobs...")
    statement = delete(Job).where(Job.status == JobStatus.QUEUED)
    session.exec(statement)
    session.commit()
    print("Queue cleared.")

def queue_jobs():
    engine = create_engine(DB_URL)
    with Session(engine) as session:
        clear_queue(session)
        
        print("Queueing Full Retraining Matrix (Prioritized)...")
        count = 0
        
        # 1. Random Forest (Sensor)
        print(" -> Queueing Random Forest...")
        for ds in SENSOR_DATASETS:
            job = Job(
                task_type="train_sensor",
                args={"model": "random_forest", "dataset_path": ds, "epochs": 20},
                status=JobStatus.QUEUED
            )
            session.add(job)
            count += 1
            
        # 2. 1D-CNN (Sensor)
        print(" -> Queueing 1D-CNN...")
        for ds in SENSOR_DATASETS:
            job = Job(
                task_type="train_sensor",
                args={"model": "cnn", "dataset_path": ds, "epochs": 20},
                status=JobStatus.QUEUED
            )
            session.add(job)
            count += 1
            
        # 3. YOLOv8 (Image)
        print(" -> Queueing YOLOv8...")
        for ds in IMAGE_DATASETS:
            job = Job(
                task_type="train_image",
                args={"model": "yolov8", "dataset_path": ds, "epochs": 10},
                status=JobStatus.QUEUED
            )
            session.add(job)
            count += 1

        # 4. Faster R-CNN (Image)
        print(" -> Queueing Faster R-CNN...")
        for ds in IMAGE_DATASETS:
            job = Job(
                task_type="train_image",
                args={"model": "faster_rcnn", "dataset_path": ds, "epochs": 10},
                status=JobStatus.QUEUED
            )
            session.add(job)
            count += 1
            
        # 5. U-Net (Image)
        print(" -> Queueing U-Net...")
        for ds in IMAGE_DATASETS:
            job = Job(
                task_type="train_image",
                args={"model": "unet", "dataset_path": ds, "epochs": 10},
                status=JobStatus.QUEUED
            )
            session.add(job)
            count += 1
                
        session.commit()
        print(f"Successfully queued {count} jobs in priority order.")

if __name__ == "__main__":
    queue_jobs()
