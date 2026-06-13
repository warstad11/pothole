import os
from pathlib import Path
from sqlmodel import Session
from app.core.database import engine
from app.models.db import Job, JobStatus

# Absolute Paths for Job 30 Components (Acc 0.91)
# Job 6 = YOLOv8 (Data 3)
# Job 4 = 1D-CNN (Data 3) - 3-channel
BASE_DIR = Path(os.getcwd())
IMAGE_MODEL_PATH = BASE_DIR / "runs/6/weights/best.pt"
SENSOR_MODEL_PATH = BASE_DIR / "runs/4/model.pth"
DATA_DIR = BASE_DIR / "data/iphone"

def queue_batch():
    if not IMAGE_MODEL_PATH.exists():
        print(f"ERROR: Image model not found at {IMAGE_MODEL_PATH}")
        return
    if not SENSOR_MODEL_PATH.exists():
        print(f"ERROR: Sensor model not found at {SENSOR_MODEL_PATH}")
        return

    print(f"Using Image Model: {IMAGE_MODEL_PATH}")
    print(f"Using Sensor Model: {SENSOR_MODEL_PATH}")

    with Session(engine) as session:
        # 1. Get drives
        drives = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
        print(f"Found {len(drives)} drives in {DATA_DIR}")

        count = 0
        for d in drives:
            print(f"Queueing {d.name}...")
            job = Job(
                task_type="run_inference",
                status=JobStatus.QUEUED,
                args={
                    "drive_path": str(d),
                    "model_config": {
                        # Pass Paths directly as model names to force specific load
                        "image_model": str(IMAGE_MODEL_PATH),
                        "sensor_model": str(SENSOR_MODEL_PATH)
                    },
                    "session_id": d.name
                }
            )
            session.add(job)
            count += 1
        session.commit()
        print(f"Done. Queued {count} jobs.")

if __name__ == "__main__":
    queue_batch()
