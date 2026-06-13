import unittest
from fastapi.testclient import TestClient
from main import app
from app.core.database import create_db_and_tables, engine
from sqlmodel import Session, SQLModel, text

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        create_db_and_tables()
        
    def tearDown(self):
        # Clean up DB
        with Session(engine) as session:
            # Drop tables roughly or just delete rows
            # For simplicity in sqlite, we can just delete file or drop tables
            # Here we just assume safe state for next test
            pass

    def test_job_submission(self):
        response = self.client.post("/api/jobs", json={"task_type": "test_job", "args": {}})
        if response.status_code != 200:
            print(f"\nResponse 422 Body: {response.json()}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertEqual(data["status"], "QUEUED")
        
        job_id = data["id"]
        response = self.client.get(f"/api/jobs/{job_id}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], job_id)

    def test_review_flow(self):
        # 1. Start session
        response = self.client.post("/api/reviews/start?run_id=test_run")
        self.assertEqual(response.status_code, 200)
        session_id = response.json()["session_id"]
        
        # 2. Get next event (might be empty/done if no events in DB)
        response = self.client.get(f"/api/reviews/{session_id}/next")
        self.assertEqual(response.status_code, 200)
        # We haven't seeded events, so likely "done" or None.
        # This checks endpoint reachability.

if __name__ == '__main__':
    unittest.main()
