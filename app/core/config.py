import os
from pathlib import Path

# Single source of truth for the sensor sampling rate.
# ALL ingestion paths (SensorLogger, iPhone, nuScenes) resample to this rate,
# and window sizes are defined in samples at this rate
# (e.g. window_size=100 -> 1.0 s of data).
SENSOR_RESAMPLE_HZ = 100

class Settings:
    PROJECT_NAME: str = "Pothole Research Platform"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RUNS_DIR: Path = BASE_DIR / "runs"
    MODELS_DIR: Path = BASE_DIR / "models"
    REVIEWS_DIR: Path = BASE_DIR / "reviews"
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/pothole_app.db"
    
    # Latest Models (Canonical Paths)
    LATEST_DIR: Path = MODELS_DIR / "latest"
    LATEST_IMAGE_MODEL: Path = LATEST_DIR / "image" / "model.pt"
    LATEST_SENSOR_MODEL: Path = LATEST_DIR / "sensor" / "model.pth"
    LATEST_HYBRID_MODEL: Path = LATEST_DIR / "hybrid" / "model.pth"
    
    # Job Settings
    WORKER_POLL_INTERVAL: float = 1.0
    
    def __init__(self):
        # Ensure directories exist
        for path in [self.DATA_DIR, self.RUNS_DIR, self.MODELS_DIR, self.REVIEWS_DIR, 
                     self.LATEST_IMAGE_MODEL.parent, self.LATEST_SENSOR_MODEL.parent, 
                     self.LATEST_HYBRID_MODEL.parent]:
            path.mkdir(parents=True, exist_ok=True)

settings = Settings()
