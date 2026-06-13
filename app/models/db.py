from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlmodel import SQLModel, Field, JSON
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"

class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    task_type: str  # e.g., "train_image", "process_dataset"
    status: JobStatus = Field(default=JobStatus.QUEUED)
    args: Dict[str, Any] = Field(default={}, sa_type=JSON)
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    pid: Optional[int] = None  # To track process ID for re-entrancy checks

class Run(SQLModel, table=True):
    id: str = Field(primary_key=True)  # UUID or timestamp-based ID
    config: Dict[str, Any] = Field(default={}, sa_type=JSON)
    metrics: Dict[str, Any] = Field(default={}, sa_type=JSON)
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ModelRegistry(SQLModel, table=True):
    model_id: str = Field(primary_key=True)
    approach: str  # image, sensor, hybrid
    config: Dict[str, Any] = Field(default={}, sa_type=JSON)
    metrics: Dict[str, Any] = Field(default={}, sa_type=JSON)
    path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Event(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    time: float
    score: float
    v_score: float = Field(default=0.0)
    s_score: float = Field(default=0.0)
    trigger_source: Optional[str] = None # "vision", "sensor", or "both"
    video_url: Optional[str] = None
    image_url: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReviewSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: str
    reviewer_id: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ReviewLabel(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: int
    event_id: int
    is_pothole: bool
    created_at: datetime = Field(default_factory=datetime.utcnow)
