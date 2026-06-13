
from sqlmodel import Session, create_engine, select
from app.models.db import Job, JobStatus
from datetime import datetime

def prioritize():
    engine = create_engine("sqlite:///pothole_app.db")
    with Session(engine) as session:
        # 1. Kill Job 9 (Running -> Canceled)
        job9 = session.get(Job, 9)
        if job9 and job9.status in [JobStatus.RUNNING, JobStatus.QUEUED]:
            print("Canceling Job 9 (Holding for later reqeue)...")
            job9.status = JobStatus.CANCELED
            job9.error_message = "Held for high priority jobs 16/17"
            session.add(job9)
            
        # 2. Hold Jobs 10-15
        to_hold = [10, 11, 12, 13, 14, 15]
        for jid in to_hold:
            j = session.get(Job, jid)
            if j and j.status == JobStatus.QUEUED:
                print(f"Holding Job {jid}...")
                j.status = JobStatus.CANCELED
                j.error_message = "Held for high priority jobs 16/17"
                session.add(j)
                
        # 3. Ensure 16/17 are Queued
        for jid in [16, 17]:
            j = session.get(Job, jid)
            if j:
                print(f"Prioritizing Job {jid}...")
                j.status = JobStatus.QUEUED
                session.add(j)
                
        # 4. Re-queue Job 9 (as requested)
        # Create a new job copy of 9 so it runs later (ID > 17)
        if job9:
            new_job9 = Job(
                task_type=job9.task_type,
                status=JobStatus.QUEUED,
                args=job9.args,
                created_at=datetime.utcnow()
            )
            session.add(new_job9)
            print(f"Re-queued Job 9 as new Job (will be ID > 17)")

        session.commit()
        print("Queue reordered. Jobs 16 & 17 are now next.")

if __name__ == "__main__":
    prioritize()
