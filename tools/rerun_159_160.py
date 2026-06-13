
import os
import time
import subprocess
import signal
from sqlmodel import create_engine, Session, select
from app.models.db import Job, JobStatus

DB_URL = "sqlite:///pothole_app.db"

# Configurations to verify
JOBS_TO_RUN = [
    {"task_type": "train_sensor", "args": {"model": "random_forest", "dataset_path": "data/sensor/Data 3 - Just Sensor"}},
    {"task_type": "train_sensor", "args": {"model": "cnn", "dataset_path": "data/sensor/Data 3 - Just Sensor"}}
]

def run_verification():
    engine = create_engine(DB_URL)
    new_job_ids = []

    # 1. Queue Jobs
    with Session(engine) as session:
        print("Queuing Verification Jobs...")
        for j_config in JOBS_TO_RUN:
            job = Job(
                task_type=j_config["task_type"],
                args=j_config["args"],
                status=JobStatus.QUEUED
            )
            session.add(job)
            session.commit()
            session.refresh(job)
            new_job_ids.append(job.id)
            print(f"  -> Queued Job {job.id}: {j_config['args']['model']}")

    # 2. Start Worker (Transient)
    print("\nStarting Transient Worker...")
    with open("transient_worker.log", "w") as log_file:
        worker_process = subprocess.Popen(
            ["./run_worker.sh"], 
            stdout=log_file, 
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid 
        )
    worker_pid = worker_process.pid
    print(f"  -> Worker Started (PID {worker_pid})")

    # 3. Monitor
    try:
        while True:
            all_done = True
            with Session(engine) as session:
                for jid in new_job_ids:
                    job = session.get(Job, jid)
                    if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELED]:
                        all_done = False
                        # Print status if changed? (Simple poll)
            
            if all_done:
                print("\nAll verification jobs finished!")
                break
                
            time.sleep(5)
            print(".", end="", flush=True)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # 4. Kill Worker
        print(f"\nKilling Worker PID {worker_pid}...")
        try:
            os.killpg(os.getpgid(worker_pid), signal.SIGTERM)
            # worker_process.terminate()
        except:
            pass
        print("Worker terminated.")

if __name__ == "__main__":
    run_verification()
