
from sqlmodel import Session, create_engine
from app.models.db import Job, JobStatus

def queue_jobs():
    engine = create_engine("sqlite:///pothole_app.db")
    with Session(engine) as session:
        # Configuration
        dataset_path = "data/hybrid/Data 1 - Both"
        image_job_id = 6  # YOLO Data 3
        sensor_job_id = 2 # RF Data 3
        
        # 1. Late Fusion (Validation via Weighted Avg)
        job_base = Job(
            task_type="train_hybrid",
            status=JobStatus.QUEUED,
            args={
                "dataset_path": dataset_path,
                "image_job_id": image_job_id,
                "sensor_job_id": sensor_job_id,
                "fusion_type": "late",
                "image_algo": "yolov8",
                "sensor_algo": "random_forest"
            }
        )
        session.add(job_base)
        
        # 2. Feature Fusion (Training LR on Features)
        job_feat = Job(
            task_type="train_hybrid",
            status=JobStatus.QUEUED,
            args={
                "dataset_path": dataset_path,
                "image_job_id": image_job_id,
                "sensor_job_id": sensor_job_id,
                "fusion_type": "feature",
                "image_algo": "yolov8",
                "sensor_algo": "random_forest"
            }
        )
        session.add(job_feat)
        
        session.commit()
        print(f"Queued Hybrid Jobs: Late Fusion (ID: {job_base.id}), Feature Fusion (ID: {job_feat.id})")

if __name__ == "__main__":
    queue_jobs()
