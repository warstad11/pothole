
import sys
import json
import joblib
from pathlib import Path
from app.services.models.image.yolo import YOLOWrapper
from sqlmodel import create_engine, Session, select
from app.models.db import Job

DB_URL = "sqlite:///pothole_app.db"

def recalc_yolo(job_id: int):
    engine = create_engine(DB_URL)
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            print(f"Job {job_id} not found.")
            return

        dataset_path = Path(job.args["dataset_path"])
        runs_dir = Path("runs") / str(job_id)
        
        print(f"Recalculating metrics for Job {job_id}...")
        
        # Instantiate Wrapper
        wrapper = YOLOWrapper(f"job_{job_id}", job.args)
        
        # Override output dir to point to existing run to find weights
        wrapper.output_dir = runs_dir
        # Load weights (YOLO loads from wrapper.output_dir logic or args?)
        # Wrapper init creates a NEW model object.
        # We need to tell it to load the trained weights.
        weights = runs_dir / "weights" / "best.pt"
        if not weights.exists():
            print(f"No weights found at {weights}")
            return
            
        wrapper.model = wrapper.load(weights) # Check usage: load returns model or sets it?
        # Check code: load(self, path) -> self.model = YOLO(path)
        wrapper.load(weights)
        
        # Evaluate
        metrics = wrapper.evaluate(dataset_path)
        
        print(f"New Metrics: Acc={metrics.get('accuracy')}, CM={metrics.get('confusion_matrix')}")
        
        # Update DB
        current_result = job.args.get("result", {})
        current_result["metrics"] = metrics
        
        # Update Args - Force entirely new dict structure to be safe
        new_args = dict(job.args)
        new_args["result"] = current_result
        job.args = new_args
        
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(job, "args")
        
        session.add(job)
        session.commit()
        session.refresh(job)
        print(f"Job {job_id} updated. New Acc: {job.args['result']['metrics']['accuracy']}")

if __name__ == "__main__":
    recalc_yolo(6)
    recalc_yolo(7)
    recalc_yolo(8)
