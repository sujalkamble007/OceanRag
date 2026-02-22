"""
retrieval_logger.py — Phase 2: Log retrieval runs to PostgreSQL and CSV.
"""

import os
import csv
import json
from sqlalchemy import select, func
import database
from database import retrieval_logs_table


def log_retrieval(retrieval_output: dict, embedding_model_name: str,
                  chunk_strategy: str) -> int:
    """
    Log a single retrieval run to PostgreSQL.
    Takes output from any retriever (similarity/mmr/hybrid).
    Returns the new log id.
    """
    log_data = {
        "query_text": retrieval_output["query"],
        "retriever_type": retrieval_output["retriever_type"],
        "embedding_model": embedding_model_name,
        "chunk_strategy": chunk_strategy,
        "top_k": retrieval_output["top_k"],
        "results": json.dumps(retrieval_output["results"]),
        "latency_seconds": retrieval_output["latency_seconds"],
    }
    log_id = database.insert_retrieval_log(log_data)
    print(f"📝 Logged {retrieval_output['retriever_type']} retrieval run to PostgreSQL (id: {log_id})")
    return log_id


def log_all_retrievers(all_results: dict, embedding_model_name: str,
                       chunk_strategy: str) -> None:
    """
    Log all 3 retriever outputs from run_all_retrievers() to PostgreSQL.
    """
    for rtype in ("similarity", "mmr", "hybrid"):
        log_retrieval(all_results[rtype], embedding_model_name, chunk_strategy)
    print("📝 Logged 3 retrieval runs to PostgreSQL")


def save_results_to_csv(all_results: dict, output_dir: str) -> None:
    """
    Flatten all results into rows and append to retrieval_results.csv.
    Creates file with header if it doesn't exist.
    """
    filepath = os.path.join(output_dir, "retrieval_results.csv")
    file_exists = os.path.exists(filepath)

    fieldnames = [
        "query", "retriever_type", "rank", "score",
        "filename", "page_number", "chunk_id",
        "latency_seconds", "preview",
    ]

    rows = []
    for rtype in ("similarity", "mmr", "hybrid"):
        data = all_results[rtype]
        for r in data["results"]:
            rows.append({
                "query": data["query"],
                "retriever_type": rtype,
                "rank": r["rank"],
                "score": r["score"],
                "filename": r["filename"],
                "page_number": r["page_number"],
                "chunk_id": r["chunk_id"],
                "latency_seconds": data["latency_seconds"],
                "preview": r.get("page_content", "")[:100],
            })

    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"📊 Results saved → {filepath}")


def get_retrieval_summary() -> None:
    """
    Query PostgreSQL retrieval_logs and print a summary table.
    """
    engine = database.get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                retrieval_logs_table.c.retriever_type,
                func.count().label("runs"),
                func.avg(retrieval_logs_table.c.latency_seconds).label("avg_latency"),
            ).group_by(retrieval_logs_table.c.retriever_type)
        ).fetchall()

    if not rows:
        print("  No retrieval logs yet.")
        return

    print(f"\n  {'Retriever Type':<18} | {'Runs':>5} | {'Avg Latency':>12}")
    print(f"  {'-'*18}-+-{'-'*5}-+-{'-'*12}")
    for row in rows:
        rtype, runs, avg_lat = row
        print(f"  {rtype:<18} | {runs:>5} | {avg_lat:>10.4f}s")
    print()
