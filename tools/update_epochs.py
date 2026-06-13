
from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Job, JobStatus
import json

def update_epochs():
    with Session(engine) as session:
        # 1. Update QUEUED Jobs
        statement = select(Job).where(Job.status == JobStatus.QUEUED)
        queued_jobs = session.exec(statement).all()
        
        updated_count = 0
        for job in queued_jobs:
            # Check model_name inside args
            m_name = job.args.get("model_name", "")
            if m_name == "yolov8" or job.task_type == "train_image":
                # Check if it has epochs arg
                if job.args and "epochs" in job.args:
                    current_epochs = job.args["epochs"]
                    if current_epochs > 10:
                        d_name = job.args.get("dataset_name", "Unknown")
                        print(f"Updating Job #{job.id} ({d_name}) epochs from {current_epochs} to 10.")
                        new_args = job.args.copy()
                        new_args["epochs"] = 10
                        job.args = new_args
                        session.add(job)
                        updated_count += 1
        
        # 2. Handle Running Job 131
        job131 = session.get(Job, 131)
        if job131 and job131.status == JobStatus.RUNNING:
            print(f"Terminating Running Job #131 (Epochs: {job131.args.get('epochs')})...")
            job131.status = JobStatus.FAILED 
            job131.error_message = "Terminated by user request to reduce epochs."
            session.add(job131)
            
            # Clone and Re-queue with 10 epochs
            new_args = job131.args.copy()
            new_args["epochs"] = 10
            if "result" in new_args: del new_args["result"]
            if "metrics" in new_args: del new_args["metrics"]
            
            new_job = Job(
                task_type=job131.task_type,
                status=JobStatus.QUEUED,
                args=new_args
            )
            session.add(new_job)
            print(f"Re-queued Job #131 as new Job.")
        
        session.commit()
        print(f"Committed. Updated {updated_count} queued jobs.")

if __name__ == "__main__":
    update_epochs()
