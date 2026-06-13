
from sqlmodel import Session, select, desc
from app.core.database import engine
from app.models.db import Job, JobStatus

def audit_faster_rcnn_queue():
    datasets_needed = [
        "data/image/raw/Data 3 - Images",
        "data/image/raw/Pothole.v1-raw.yolov8",
        "data/image/raw/RealPothole600_Converted"
    ]
    
    # Normalize helper (strip CWD if present)
    def normalize(p):
        return p.replace("/Users/tstading/gemini/antigravity/scratch/pothole1223/", "")

    print(f"{'ID':<5} {'STATUS':<12} {'MODEL':<20} {'DATASET'}")
    print("-" * 80)

    found_coverage = {d: False for d in datasets_needed}

    with Session(engine) as session:
        # Check QUEUED, RUNNING, COMPLETED
        # We only care about jobs that actually HAVE the 'model' key set to faster_rcnn
        # OR jobs that completed successfully/failed where we can verify they RAN faster_rcnn
        
        jobs = session.exec(select(Job).where(Job.task_type == "train_image").order_by(desc(Job.id))).all()
        
        for j in jobs:
            args = j.args or {}
            model_arg = args.get("model")
            dataset = normalize(args.get("dataset_path", ""))
            
            # Check if it IS a faster rcnn job
            is_rcnn = False
            if model_arg == "faster_rcnn":
                is_rcnn = True
            elif "faster" in str(model_arg):
                is_rcnn = True
            
            # Special check for completed jobs that might have missed the arg but ran correctly?
            # No, we established earlier that missing arg -> YOLO.
            
            if is_rcnn:
                print(f"{j.id:<5} {j.status.value:<12} {str(model_arg):<20} {dataset}")
                
                # Check coverage
                # Fuzzy match dataset
                for d in datasets_needed:
                    if d in dataset or dataset in d:
                        # Only count if Queued or Running or Completed recent?
                        # User wants them "queued".
                        if j.status in [JobStatus.QUEUED, JobStatus.RUNNING]:
                            found_coverage[d] = True
                            
    print("\nCoverage Status:")
    for d, covered in found_coverage.items():
        print(f"  {d}: {'OK (In Queue/Running)' if covered else 'MISSING'}")

if __name__ == "__main__":
    audit_faster_rcnn_queue()
