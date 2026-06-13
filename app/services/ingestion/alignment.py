import json
from pathlib import Path
from typing import Dict, Optional

class DriveTimeline:
    def __init__(self, metadata: Dict[str, float]):
        self.data = metadata
        self.start_epoch = metadata.get("start_epoch")
        self.start_video = metadata.get("start_video", 0.0)
        self.video_offset = metadata.get("video_offset", 0.0) # difference between sensor 0 and video 0

    def rel_to_epoch(self, seconds_elapsed: float) -> float:
        return self.start_epoch + seconds_elapsed

    def rel_to_video(self, seconds_elapsed: float) -> float:
        # If sensor starts at T=0, and video starts at T=video_offset
        # video_time = seconds_elapsed - video_offset
        return seconds_elapsed - self.video_offset

    def video_to_rel(self, video_time: float) -> float:
        # Inverse: seconds_elapsed = video_time + video_offset
        return video_time + self.video_offset

    def save(self, path: Path):
        with open(path / "alignment.json", "w") as f:
            json.dump(self.data, f)

    @classmethod
    def load(cls, path: Path) -> 'DriveTimeline':
        with open(path / "alignment.json", "r") as f:
            data = json.load(f)
        return cls(data)

class aligner:
    @staticmethod
    def align_drive(sensor_start_epoch: float, video_start_epoch: float) -> DriveTimeline:
        # Simple alignment assuming synchronized clocks
        # Offset is how many seconds after sensor start the video started
        video_offset = video_start_epoch - sensor_start_epoch
        return DriveTimeline({
            "start_epoch": sensor_start_epoch,
            "start_video": 0.0,
            "video_offset": video_offset
        })
