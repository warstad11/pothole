
import json
from sqlmodel import Session, create_engine
from app.models.db import Job, JobStatus

def queue_yolo_data1():
    engine = create_engine("sqlite:///pothole_app.db")
    with Session(engine) as session:
        # Args
        dataset_path = "data/hybrid/Data 1 - Both/images" # Correct path for Data 1 images
        
        args = {
            "model": "yolov8",
            "dataset_path": dataset_path,
            "epochs": 10
        }
        
        job = Job(
            task_type="train_image",
            status=JobStatus.QUEUED,
            args=args
        )
        session.add(job)
        session.commit()
        session.refresh(job)
        print(f"Queued Job {job.id} (YOLO Data 1)")

if __name__ == "__main__":
    queue_yolo_data1()
