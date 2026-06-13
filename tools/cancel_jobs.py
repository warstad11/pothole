
import sys
from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Job, JobStatus

def cancel_jobs(job_ids):
    with Session(engine) as session:
        for jid in job_ids:
            job = session.get(Job, jid)
            if not job:
                print(f"Job {jid} not found.")
                continue
            
            if job.status == JobStatus.QUEUED:
                print(f"Canceling Job #{jid} (was QUEUED)...")
                job.status = JobStatus.CANCELED
                session.add(job)
            elif job.status == JobStatus.RUNNING:
                print(f"WARNING: Job #{jid} is RUNNING. Force canceling (process may still be alive)...")
                job.status = JobStatus.CANCELED
                session.add(job)
            else:
                print(f"Job #{jid} is {job.status}. Skipping cancellation.")
                
        session.commit()
        print("Batch cancellation complete.")

if __name__ == "__main__":
    ids = [150, 151, 152]
    if len(sys.argv) > 1:
        ids = [int(x) for x in sys.argv[1:]]
        
    cancel_jobs(ids)
