"""
api/routers/feedback.py — Submit thumbs up/down feedback on Q&A responses.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from api.dependencies import get_current_user
from core.database import insert_feedback, get_engine, qa_logs_table

router = APIRouter(prefix="/feedback", tags=["feedback"])


class FeedbackRequest(BaseModel):
    qa_log_id: int
    rating: int        # 1 = thumbs up, -1 = thumbs down
    comment: str = ""


@router.post("/")
def submit_feedback(req: FeedbackRequest, current_user: dict = Depends(get_current_user)):
    """Submit feedback rating for a specific Q&A record."""
    if req.rating not in (1, -1):
        raise HTTPException(status_code=400, detail="Rating must be 1 (thumbs up) or -1 (thumbs down)")

    # Verify qa_log exists
    engine = get_engine()
    with engine.connect() as conn:
        exists = conn.execute(
            select(qa_logs_table.c.id).where(qa_logs_table.c.id == req.qa_log_id)
        ).fetchone()
    if not exists:
        raise HTTPException(status_code=404, detail=f"QA log {req.qa_log_id} not found")

    feedback_id = insert_feedback(
        qa_log_id=req.qa_log_id,
        user_id=current_user["id"],
        rating=req.rating,
        comment=req.comment,
    )
    return {"message": "Feedback recorded", "id": feedback_id}
