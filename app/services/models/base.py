from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

class BaseModel(ABC):
    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.config = config
        self.output_dir = Path(config.get("output_dir", f"runs/{model_id}"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def train(self, dataset_path: Path, epochs: int = 10):
        """Train the model."""
        pass

    @abstractmethod
    def evaluate(self, dataset_path: Path) -> Dict[str, float]:
        """Evaluate the model and return metrics."""
        pass

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run inference."""
        pass

    @abstractmethod
    def save(self):
        """Save model artifacts."""
        pass
    
    @abstractmethod
    def load(self, weights_path: Path):
        """Load model weights."""
        pass
