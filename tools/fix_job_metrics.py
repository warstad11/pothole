import sys
import os
# Ensure app imports work
sys.path.append(os.getcwd())

from app.core.utils import get_dataset_stats
from app.models.db import Job
from app.core.database import engine
from sqlmodel import Session, select
from pathlib import Path
import json

def fix_metrics(job_id):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job: 
            print(f"Job {job_id} not found")
            return
        
        args = dict(job.args) # Copy
        res = args.get("result", {})
        if not res:
            print(f"Job {job_id} has no result")
            return
            
        met = res.get("metrics", {})
        ds_path_str = res.get("dataset")
        ds_path = Path(ds_path_str)
        
        if not ds_path.exists():
            # Try prepending root if relative
            if (Path.cwd() / ds_path).exists():
                ds_path = Path.cwd() / ds_path
            else:
                print(f"Dataset path {ds_path} not found")
                # Try raw
                ds_path = Path("data/image/raw") / ds_path.name
                if not ds_path.exists():
                    print(f"Cannot locate dataset for Job {job_id}")
                    return

        # Get REAL stats now
        # Note: utils.py update assumes we import the NEW code. 
        # Since I replaced the file content, this import should use the new logic.
        p, n = get_dataset_stats(ds_path)
        print(f"Job {job_id}: Dataset {ds_path} Stats: Potholes={p}, Normals={n}")
        
        # Recompute CM
        recall = met.get("recall", 0)
        precision = met.get("precision", 0)
        
        print(f"Current Metrics: P={precision}, R={recall}")
        
        tp = int(p * recall)
        fn = p - tp
        fp = 0
        if precision > 0:
            # P = TP / (TP + FP) -> P(TP+FP) = TP -> P*FP = TP - P*TP -> FP = TP(1-P)/P
            if precision < 1.0:
                 fp = int(tp * (1 - precision) / precision)
            else:
                 fp = 0
        
        tn = max(0, n - fp)
        
        cm = [[tn, fp], [fn, tp]]
        print(f"New CM: {cm}")
        
        # Update
        met["confusion_matrix"] = cm
        res["metrics"] = met
        args["result"] = res
        
        # SQLModel requires reassignment to detect change in JSON field if not using flag_modified
        job.args = args 
        session.add(job)
        session.commit()
        print(f"Job {job_id} Updated.")

if __name__ == "__main__":
    fix_metrics(105)
    fix_metrics(106)
