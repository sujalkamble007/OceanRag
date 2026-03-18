"""
api/routers/eval_results.py — API endpoint for research evaluation results.
"""

from fastapi import APIRouter, Depends, Query
from api.dependencies import get_current_user
from core.database import get_eval_results, get_all_experiments

router = APIRouter(prefix="/eval", tags=["evaluation"])


@router.get("/results")
def list_eval_results(
    phase: str = Query(None, description="Filter by phase: A, B, C, D, FULL, QUICK"),
    limit: int = Query(200, ge=1, le=1000),
    current_user: dict = Depends(get_current_user),
):
    """Get per-question evaluation results, optionally filtered by phase."""
    results = get_eval_results(phase=phase, limit=limit)
    # Convert datetime objects to strings for JSON serialization
    for r in results:
        if r.get("created_at"):
            r["created_at"] = str(r["created_at"])
    return {"results": results, "count": len(results)}


@router.get("/experiments")
def list_experiments(
    current_user: dict = Depends(get_current_user),
):
    """Get aggregated experiment results (one row per config)."""
    results = get_all_experiments()
    return {"results": results, "count": len(results)}


@router.get("/summary")
def eval_summary(
    current_user: dict = Depends(get_current_user),
):
    """
    Returns summary statistics grouped by phase.
    Useful for the frontend dashboard.
    """
    import pandas as pd

    results = get_all_experiments()
    if not results:
        return {"phases": {}, "total_experiments": 0}

    df = pd.DataFrame(results)

    summary = {"total_experiments": len(df), "phases": {}}

    group_col = "phase" if "phase" in df.columns else "chunk_strategy"
    if group_col in df.columns:
        for name, group in df.groupby(group_col):
            if name is None:
                name = "legacy"
            metrics = {}
            for col in ["precision_at_k", "recall_at_k", "mrr", "hit_rate",
                         "faithfulness", "rouge_l", "bleu", "bertscore",
                         "latency_seconds", "cost_per_query"]:
                if col in group.columns:
                    val = group[col].mean()
                    metrics[col] = round(float(val), 4) if val == val else 0.0  # NaN check
            metrics["count"] = len(group)
            summary["phases"][str(name)] = metrics

    return summary
