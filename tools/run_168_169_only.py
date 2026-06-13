
import os
import time
import subprocess
import signal
import sys
from sqlmodel import create_engine, Session, select
from app.models.db import Job, JobStatus

DB_URL = "sqlite:///pothole_app.db"
TARGET_IDS = [168, 169]

def kill_process_tree(pid_pattern):
    # Rough kill by pattern
    os.system(f"pkill -f {pid_pattern}")

def run_specific_jobs():
    print("--- Enforcing Exclusive Run for Jobs 168 & 169 ---")
    
    # 1. Kill Executing Scripts/Workers
    print("1. Killing executing scripts & workers...")
    kill_process_tree("rerun_159_160.py")
    kill_process_tree("python.*worker")
    time.sleep(2) # Give time to die

    # 2. Clean DB (Cancel others)
    print("2. Cleaning Database...")
    engine = create_engine(DB_URL)
    with Session(engine) as session:
        statement = select(Job).where(Job.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]))
        active_jobs = session.exec(statement).all()
        
        for job in active_jobs:
            if job.id not in TARGET_IDS:
                print(f"  -> Cancelling Job {job.id} [{job.task_type}]")
                job.status = JobStatus.CANCELED
                job.error_message = "Canceled to prioritize 168/169."
                session.add(job)
            else:
                print(f"  -> Preserving Job {job.id} [{job.status}]")
                # Reset to QUEUED if it was RUNNING (since we killed worker)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.QUEUED
                    job.started_at = None
                    session.add(job)
        session.commit()

    # 3. Start Transient Worker
    print("\n3. Starting Dedicated Worker...")
    with open("dedicated_worker.log", "w") as log_file:
        worker_process = subprocess.Popen(
            ["./run_worker.sh"], 
            stdout=log_file, 
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid 
        )
    worker_pid = worker_process.pid
    print(f"  -> Worker Started (PID {worker_pid})")

    # 4. Monitor
    print("\n4. Monitoring Target Jobs...")
    try:
        while True:
            all_done = True
            with Session(engine) as session:
                for jid in TARGET_IDS:
                    job = session.get(Job, jid)
                    if not job: 
                        continue
                    if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
                        all_done = False
            
            if all_done:
                print("\nJobs 168 & 169 Finished!")
                break
            
            time.sleep(5)

    except KeyboardInterrupt:
        print("\nInterrupted.")

    finally:
        print(f"\nKilling Worker PID {worker_pid}...")
        try:
            os.killpg(os.getpgid(worker_pid), signal.SIGTERM)
        except:
            pass

if __name__ == "__main__":
    run_specific_jobs()
