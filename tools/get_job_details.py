
from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Job
import sys

def get_job(job_id):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if job:
            print(f"Job #{job.id}:")
            print(f"  Task Type: {job.task_type}")
            print(f"  Status: {job.status}")
            print(f"  Args: {job.args}")
            print(f"  Created: {job.created_at}")
        else:
            print(f"Job #{job_id} not found.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        get_job(int(sys.argv[1]))
    else:
        print("Usage: python tools/get_job_details.py <job_id>")
