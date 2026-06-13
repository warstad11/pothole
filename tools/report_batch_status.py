
from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Job
import sys

def report_batch(start_id, end_id):
    with Session(engine) as session:
        statement = select(Job).where(Job.id >= start_id, Job.id <= end_id).order_by(Job.id)
        jobs = session.exec(statement).all()
        
        print(f"{'ID':<5} {'TYPE':<15} {'STATUS':<12} {'MODEL':<20} {'DATASET'}")
        print("-" * 80)
        
        for j in jobs:
            args = j.args or {}
            
            # Model Name
            model = args.get("model_name") or args.get("model") or args.get("model_type") or "Unknown"
            
            # Dataset Name
            ds_path = args.get("dataset_path") or args.get("dataset") or "Unknown"
            if "raw/" in str(ds_path):
                ds_name = str(ds_path).split("raw/")[-1]
            else:
                ds_name = str(ds_path).replace("/Users/tstading/gemini/antigravity/scratch/pothole1223/", "")
                
            print(f"{j.id:<5} {j.task_type:<15} {j.status.value:<12} {model:<20} {ds_name}")

if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 145
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 152
    report_batch(start, end)
