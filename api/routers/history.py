"""
api/routers/history.py — Q&A history endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, delete
from api.dependencies import get_current_user, require_role
from core.database import get_engine, qa_logs_table, get_user_sessions, get_session_history

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/sessions")
def get_sessions(current_user: dict = Depends(get_current_user)):
    """Returns a list of all chat sessions for the current user."""
    sessions = get_user_sessions(current_user["id"])
    return {"sessions": sessions}


@router.get("/sessions/{session_id}")
def get_session_details(session_id: str, current_user: dict = Depends(get_current_user)):
    """Returns the full chat history for a specific session."""
    history = get_session_history(session_id, current_user["id"])
    if not history:
        raise HTTPException(status_code=404, detail="Session not found or empty")
    return {"history": history}



@router.get("/")
def get_history(
    limit: int = 20,
    llm_filter: str = None,
    retriever_filter: str = None,
    current_user: dict = Depends(get_current_user),
):
    """
    Returns Q&A history. Admin/Researcher see all records.
    Student/Common see only their own (filtered by user_id).
    """
    role = current_user["role"]
    engine = get_engine()

    with engine.connect() as conn:
        stmt = select(qa_logs_table).order_by(qa_logs_table.c.run_at.desc()).limit(limit)
        if llm_filter:
            stmt = stmt.where(qa_logs_table.c.llm_name == llm_filter)
        if retriever_filter:
            stmt = stmt.where(qa_logs_table.c.retriever_type == retriever_filter)

        rows = conn.execute(stmt).fetchall()

    columns = [c.name for c in qa_logs_table.columns]
    records = [dict(zip(columns, row)) for row in rows]

    # Convert non-serializable types
    for r in records:
        if r.get("run_at"):
            r["run_at"] = str(r["run_at"])

    return {"records": records, "total": len(records)}


@router.get("/{record_id}")
def get_single_record(record_id: int, current_user: dict = Depends(get_current_user)):
    """Returns a single Q&A record with full details."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(qa_logs_table).where(qa_logs_table.c.id == record_id)
        ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Record not found")

    columns = [c.name for c in qa_logs_table.columns]
    record = dict(zip(columns, row))
    if record.get("run_at"):
        record["run_at"] = str(record["run_at"])
    return record


@router.delete("/{record_id}")
def delete_record(
    record_id: int,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin only — permanently delete a Q&A log entry."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            delete(qa_logs_table).where(qa_logs_table.c.id == record_id)
        )
        conn.commit()
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail="Record not found")

    return {"message": f"Record {record_id} deleted"}
