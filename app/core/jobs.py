from sqlmodel import Session, select
from typing import Dict, Any, Optional
from app.models.db import Job, JobStatus
from app.core.database import get_session, engine

class JobManager:
    def submit_job(self, task_type: str, args: Dict[str, Any] = {}) -> Job:
        with Session(engine) as session:
            job = Job(task_type=task_type, args=args, status=JobStatus.QUEUED)
            session.add(job)
            session.commit()
            session.refresh(job)
            return job

    def get_job(self, job_id: int) -> Optional[Job]:
        with Session(engine) as session:
            return session.get(Job, job_id)

    def list_jobs(self, limit: int = 200):
        with Session(engine) as session:
            statement = select(Job).order_by(Job.created_at.desc()).limit(limit)
            return session.exec(statement).all()

job_manager = JobManager()
