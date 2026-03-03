"""
api/routers/experiments.py — Experiment results, leaderboard, and best config endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException
from api.dependencies import get_current_user, require_role
from core.database import get_all_experiments, get_best_config

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("/")
def get_experiments(
    chunk_strategy: str = None,
    embedding_model: str = None,
    retriever_type: str = None,
    llm_name: str = None,
    min_faithfulness: float = None,
    current_user: dict = Depends(require_role("admin", "researcher", "student")),
):
    """
    Returns experiment results with optional filters.
    Common users cannot access this endpoint (enforced by require_role).
    """
    experiments = get_all_experiments()

    # Apply optional filters
    if chunk_strategy:
        experiments = [e for e in experiments if e.get("chunk_strategy") == chunk_strategy]
    if embedding_model:
        experiments = [e for e in experiments if e.get("embedding_model") == embedding_model]
    if retriever_type:
        experiments = [e for e in experiments if e.get("retriever_type") == retriever_type]
    if llm_name:
        experiments = [e for e in experiments if e.get("llm_name") == llm_name]
    if min_faithfulness is not None:
        experiments = [e for e in experiments if (e.get("faithfulness") or 0) >= min_faithfulness]

    # Serialize datetime
    for e in experiments:
        if e.get("run_at"):
            e["run_at"] = str(e["run_at"])

    return {"experiments": experiments, "total": len(experiments)}


@router.get("/leaderboard")
def get_leaderboard(
    top_n: int = 10,
    current_user: dict = Depends(get_current_user),
):
    """
    Returns top_n experiments sorted by composite score.
    composite = (precision_at_k + recall_at_k + faithfulness + answer_relevancy) / 4
    Also adds rouge_l to differentiate experiments with identical scores.
    """
    experiments = get_all_experiments()

    def composite(exp):
        return (
            (exp.get("precision_at_k") or 0) +
            (exp.get("recall_at_k") or 0) +
            (exp.get("faithfulness") or 0) +
            (exp.get("answer_relevancy") or 0) +
            (exp.get("rouge_l") or 0)
        ) / 5

    ranked = sorted(experiments, key=composite, reverse=True)[:top_n]
    for i, e in enumerate(ranked):
        e["rank"] = i + 1
        e["composite_score"] = round(composite(e), 4)
        if e.get("run_at"):
            e["run_at"] = str(e["run_at"])

    return {"leaderboard": ranked, "total": len(ranked)}


@router.get("/best")
def get_best(current_user: dict = Depends(get_current_user)):
    """Returns the single best experiment configuration."""
    best = get_best_config()
    if not best:
        raise HTTPException(status_code=404, detail="No experiments found yet. Run Phase 4 first.")
    if best.get("run_at"):
        best["run_at"] = str(best["run_at"])
    return best
