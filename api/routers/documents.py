"""
api/routers/documents.py — Document listing and category endpoints.
"""
from fastapi import APIRouter, Depends
from sqlalchemy import select, distinct
from api.dependencies import get_current_user
from core.database import get_engine, documents_table, chunks_table

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("/")
def get_documents(current_user: dict = Depends(get_current_user)):
    """Returns all ingested documents with metadata."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                documents_table.c.id,
                documents_table.c.filename,
                documents_table.c.total_pages,
                documents_table.c.uploaded_at,
            ).order_by(documents_table.c.filename)
        ).fetchall()

    return {
        "documents": [
            {
                "id": r.id,
                "filename": r.filename,
                "total_pages": r.total_pages,
                "uploaded_at": str(r.uploaded_at) if r.uploaded_at else None,
            }
            for r in rows
        ],
        "total": len(rows),
    }


@router.get("/categories")
def get_categories(current_user: dict = Depends(get_current_user)):
    """
    Returns unique categories derived from chunk filenames.
    Uses the parent folder name extracted from the filename path.
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(distinct(chunks_table.c.filename))
            .order_by(chunks_table.c.filename)
        ).fetchall()

    # Extract folder/category from filenames like "Publications/file.pdf"
    categories = set()
    for (fname,) in rows:
        if fname and "/" in fname:
            categories.add(fname.split("/")[0])
        elif fname:
            categories.add("General")

    return {"categories": sorted(categories)}
