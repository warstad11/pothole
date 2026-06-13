
from pathlib import Path
from app.core.utils import get_dataset_stats
from app.models.db import Job
from sqlmodel import Session, select
from app.core.database import engine
import sys

# Mock settings just in case
from app.core.config import settings

def debug():
    job_id = 132
    path_str = "/Users/tstading/gemini/antigravity/scratch/pothole1223/data/image/raw/Pothole.v1-raw.yolov8"
    
    print(f"--- Debugging Job {job_id} Filtering ---")
    print(f"Path: {path_str}")
    
    # 1. Check Path existence
    p = Path(path_str)
    print(f"Exists? {p.exists()}")
    
    # 2. Check Stats
    try:
        pos, neg = get_dataset_stats(p)
        print(f"Stats: Pos={pos}, Neg={neg}, Total={pos+neg}")
    except Exception as e:
        print(f"Stats Error: {e}")

    # 3. Simulate API Logic
    print("\n--- Simulating API Logic ---")
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            print("Job not found in DB")
            return
            
        res = job.args.get("result", {})
        ds_path = res.get("dataset")
        print(f"Job Result Dataset: {ds_path}")
        
        if ds_path:
            try:
                p_stats, n_stats = get_dataset_stats(Path(ds_path))
                print(f"API Check: Total={p_stats+n_stats} (Threshold 100)")
                if (p_stats + n_stats) < 100:
                    print("RESULT: FILTERED OUT (Size < 100)")
                else:
                    print("RESULT: INCLUDED")
                    
            except Exception as e:
                print(f"API Check Error: {e}")

if __name__ == "__main__":
    debug()
