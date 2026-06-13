from sqlmodel import Session, select
from app.core.database import engine
from app.models.db import Event, ReviewSession, ReviewLabel
from datetime import datetime, timezone
from typing import List, Dict, Optional

class ReviewManager:
    def create_session(self, run_id: str) -> ReviewSession:
        with Session(engine) as session:
            # Create a new review session
            review_session = ReviewSession(
                run_id=run_id,
                reviewer_id="user", # Placeholder
                created_at=datetime.utcnow(),
                status="active"
            )
            session.add(review_session)
            session.commit()
            session.refresh(review_session)
            return review_session

    def get_next_event(self, session_id: int) -> Optional[Event]:
        with Session(engine) as session:
            # Get events for the run associated with this session
            review_session = session.get(ReviewSession, session_id)
            if not review_session:
                return None
            
            # Find events that haven't been labeled in this session yet
            # This is a simplification. Ideally we check if *this* session has labeled it.
            # For now, let's just return any event from the run that doesn't have a label.
            
            # Simple query: Get all events for run
            events = session.exec(select(Event).where(Event.run_id == review_session.run_id)).all()
            
            # Get all labels for this session
            labels = session.exec(select(ReviewLabel).where(ReviewLabel.session_id == session_id)).all()
            labeled_event_ids = {l.event_id for l in labels}
            
            for event in events:
                if event.id not in labeled_event_ids:
                    return event
            
            return None

    def submit_label(self, session_id: int, event_id: int, is_pothole: bool) -> ReviewLabel:
        with Session(engine) as session:
            # Upsert logic: check for existing label for this event in this session
            statement = select(ReviewLabel).where(
                ReviewLabel.session_id == session_id,
                ReviewLabel.event_id == event_id
            )
            existing = session.exec(statement).first()
            
            if existing:
                existing.is_pothole = is_pothole
                existing.created_at = datetime.now(timezone.utc)
                session.add(existing)
                label = existing
            else:
                label = ReviewLabel(
                    session_id=session_id,
                    event_id=event_id,
                    is_pothole=is_pothole,
                    created_at=datetime.now(timezone.utc)
                )
                session.add(label)
                
            session.commit()
            session.refresh(label)
            return label

    def get_stats(self, session_id: int) -> Dict[str, float]:
        with Session(engine) as session:
            labels = session.exec(select(ReviewLabel).where(ReviewLabel.session_id == session_id)).all()
            if not labels:
                return {"precision": 0.0, "progress": 0.0, "total_labeled": 0}
            
            tp = sum(1 for l in labels if l.is_pothole)
            total = len(labels)
            precision = tp / total if total > 0 else 0.0
            
            return {
                "precision": precision,
                "total_labeled": total,
                "true_positives": tp,
                "false_positives": total - tp
            }
