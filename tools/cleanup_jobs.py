
import os
import signal
import sys
from sqlmodel import create_engine, Session, select
from app.models.db import Job, JobStatus

# Target the active DB found previously
DB_URL = "sqlite:///pothole_app.db"

RETRAIN_JOB_ID = 157

def cleanup_jobs():
    if not os.path.exists("pothole_app.db"):
        print("Error: pothole_app.db not found.")
        return

    engine = create_engine(DB_URL)
    cleaned_count = 0
    
    with Session(engine) as session:
        # Fetch active jobs
        statement = select(Job).where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
        jobs = session.exec(statement).all()
        
        print(f"Found {len(jobs)} active jobs.")
        
        for job in jobs:
            if job.id == RETRAIN_JOB_ID:
                print(f"Skipping Job {job.id} (Retraining Job) - Keeping Active.")
                continue
            
            print(f"Cancelling Job {job.id} [{job.status}]...")
            
            # Kill Process if Running
            if job.status == JobStatus.RUNNING and job.pid:
                try:
                    os.kill(job.pid, signal.SIGKILL)
                    print(f"  -> Killed PID {job.pid}")
                except ProcessLookupError:
                    print(f"  -> PID {job.pid} not found (already dead).")
                except Exception as e:
                    print(f"  -> Failed to kill PID {job.pid}: {e}")
            
            # Update DB Status
            job.status = JobStatus.CANCELED
            job.error_message = "Canceled by user cleanup request."
            session.add(job)
            cleaned_count += 1
        
        session.commit()
        print(f"Successfully canceled {cleaned_count} jobs.")

if __name__ == "__main__":
    cleanup_jobs()
