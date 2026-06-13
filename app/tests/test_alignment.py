import unittest
import shutil
from pathlib import Path
from app.services.ingestion.alignment import DriveTimeline

class TestDriveTimeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_alignment_data")
        self.test_dir.mkdir(exist_ok=True)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_save_load(self):
        data = {
            "start_epoch": 1700000000.0,
            "start_video": 0.0,
            "video_offset": 5.0
        }
        tl = DriveTimeline(data)
        tl.save(self.test_dir)
        
        loaded = DriveTimeline.load(self.test_dir)
        self.assertEqual(loaded.data['start_epoch'], 1700000000.0)
        self.assertEqual(loaded.data['video_offset'], 5.0)

    def test_rel_to_video(self):
        data = {
            "start_epoch": 1000.0, # Sensor starts at t=1000
            "start_video": 0.0,
            "video_offset": 5.0    # Video starts 5s after sensor (t=1005)
        }
        tl = DriveTimeline(data)
        
        # Point at t=1010 (10s into sensor)
        # Should be 5s into video
        rel_time = 10.0
        video_time = tl.rel_to_video(rel_time)
        self.assertEqual(video_time, 5.0)
        
    def test_video_to_rel(self):
        data = {
            "start_epoch": 1000.0,
            "start_video": 0.0,
            "video_offset": 5.0
        }
        tl = DriveTimeline(data)
        
        video_time = 10.0
        # Should be 15s into sensor
        rel_time = tl.video_to_rel(video_time)
        self.assertEqual(rel_time, 15.0)

if __name__ == '__main__':
    unittest.main()
