
import os
import shutil
import signal
import subprocess
from pathlib import Path
from sqlmodel import create_engine, Session, select, delete
from app.models.db import Job, JobStatus

# Configuration
DB_PATH = "pothole_app.db"
DB_URL = f"sqlite:///{DB_PATH}"
RUNS_DIR = Path("runs")

def kill_workers():
    print("Stopping active workers...")
    # Find python processes running worker.py
    try:
        # Check specific pattern used in run_worker.sh
        cmd = "pgrep -f 'python.*worker.py'"
        pids = subprocess.check_output(cmd, shell=True).decode().strip().split('\n')
        for pid in pids:
            if pid:
                print(f"Killing worker PID {pid}")
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
    except subprocess.CalledProcessError:
        print("No worker processes found.")

def clean_database_and_files():
    if not os.path.exists(DB_PATH):
        print("Database not found.")
        return

    engine = create_engine(DB_URL)
    with Session(engine) as session:
        # Select Jobs to DELETE
        # Keep 'real_world_inference'
        stmt = select(Job).where(Job.task_type != "real_world_inference")
        jobs_to_delete = session.exec(stmt).all()
        
        print(f"Found {len(jobs_to_delete)} jobs to delete.")
        
        ids_to_remove = []
        for job in jobs_to_delete:
            ids_to_remove.append(job.id)
            
            # Remove Runs Directory
            run_path = RUNS_DIR / str(job.id)
            if run_path.exists():
                try:
                    shutil.rmtree(run_path)
                    # print(f"Deleted {run_path}") 
                except Exception as e:
                    print(f"Failed to delete {run_path}: {e}")
            
            # Delete from DB
            session.delete(job)
            
        session.commit()
        print(f"Deleted {len(ids_to_remove)} jobs and their run artifacts.")
        
        # Verify
        remaining = session.exec(select(Job)).all()
        print(f"Remaining Jobs in DB: {len(remaining)}")
        for r in remaining:
            print(f" - ID {r.id}: {r.task_type} ({r.status})")

if __name__ == "__main__":
    confirm = input("WARNNG: This will delete ALL training jobs and results (except real_world). Type 'yes' to proceed: ")
    if confirm == "yes":
        kill_workers()
        clean_database_and_files()
    else:
        print("Aborted.")
