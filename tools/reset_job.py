
import sys
from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Job, JobStatus
from datetime import datetime

def reset_job(job_id):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            print(f"Job {job_id} not found.")
            return
            
        print(f"Resetting Job #{job_id} to QUEUED...")
        job.status = JobStatus.QUEUED
        job.started_at = None
        job.completed_at = None
        job.error_message = None
        # job.pid = None # If PID tracking existed
        
        session.add(job)
        session.commit()
        print("Job reset complete.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        reset_job(int(sys.argv[1]))
    else:
        print("Usage: python reset_job.py <job_id>")
