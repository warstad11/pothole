
from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Job
from app.core.utils import get_dataset_stats
from pathlib import Path
import json

def patch_job(job_id):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            print(f"Job {job_id} not found.")
            return

        print(f"--- Patching Job #{job.id} ---")
        
        # 1. Get current result
        res = job.args.get("result", {})
        metrics = res.get("metrics", {})
        
        dataset_path = Path(res.get("dataset"))
        if not dataset_path.exists():
            print(f"Dataset path not found: {dataset_path}")
            return

        # 2. Get CORRECT stats using fixed logic
        potholes, normals = get_dataset_stats(dataset_path)
        print(f"Dataset Audit: Potholes={potholes}, Normals={normals}")
        
        # 3. Get existing performance metrics (assumed valid from Ultralytics)
        r = metrics.get("recall", 0.0)
        p = metrics.get("precision", 0.0)
        
        print(f"Model Performance: Recall={r:.4f}, Precision={p:.4f}")

        # 4. Recalculate Confusion Matrix
        # TP = Total Potholes * Recall
        tp = int(potholes * r)
        fn = potholes - tp
        
        # FP derivation from Precision
        # Precision = TP / (TP + FP)  ->  TP + FP = TP / P  ->  FP = (TP / P) - TP
        fp = 0
        if p > 0:
            fp = int((tp / p) - tp)
        
        # TN = Total Normals - FP
        tn = max(0, normals - fp)
        
        new_cm = [[tn, fp], [fn, tp]]
        print(f"New Confusion Matrix: {new_cm}")
        
        # 5. Update Job
        metrics["confusion_matrix"] = new_cm
        res["metrics"] = metrics
        
        # SQLModel/SQLAlchemy JSON mutation tracking fix:
        # Create a completely new dictionary structure to force update
        import copy
        new_args = copy.deepcopy(job.args)
        new_args["result"] = res
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(job, "args")
        
        print(f"Pre-commit check: {job.args['result']['metrics']['confusion_matrix']}")
        
        session.add(job)
        session.commit()
        session.refresh(job)
        
        # Verify persistence
        print(f"Post-refresh check: {job.args['result']['metrics']['confusion_matrix']}")
        print("Job updated successfully.")

import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            patch_job(int(arg))
    else:
        print("Please provide job IDs to patch.")
