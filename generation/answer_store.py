"""
generation/answer_store.py — Save Q&A records and model comparisons to PostgreSQL.
"""

import json
from core import database
from core.config import DEFAULT_CHUNK_CONFIG, DEFAULT_EMBEDDING_CONFIG


def save_qa(query: str, retrieval_output: dict, generation_result: dict,
            prompt: dict, sources: list, session_id: str = None, user_id: int = None) -> int:
    """Save a complete Q&A record to PostgreSQL qa_logs. Returns new record id."""
    qa_data = {
        "session_id": session_id,
        "user_id": user_id,
        "query_text": query,
        "retriever_type": retrieval_output.get("retriever_type", ""),
        "embedding_model": DEFAULT_EMBEDDING_CONFIG["name"],
        "chunk_strategy": DEFAULT_CHUNK_CONFIG["name"],
        "top_k": retrieval_output.get("top_k", 5),
        "llm_name": generation_result.get("llm_name", ""),
        "llm_model_id": generation_result.get("model_id", ""),
        "context_chunks": json.dumps(retrieval_output.get("results", [])),
        "prompt_text": prompt.get("user", ""),
        "answer_text": generation_result.get("answer", ""),
        "sources": json.dumps(sources),
        "input_tokens": generation_result.get("input_tokens", 0),
        "output_tokens": generation_result.get("output_tokens", 0),
        "latency_seconds": generation_result.get("latency_seconds", 0),
        "cost_usd": generation_result.get("cost_usd", 0.0),
    }
    record_id = database.insert_qa_log(qa_data)
    print(f"💾 Q&A saved to PostgreSQL (id: {record_id})")
    return record_id


def save_comparison(query: str, retriever_type: str, top_k: int,
                    generation_results: dict) -> int:
    """Save a multi-model comparison record. Returns new record id."""
    serializable = {
        llm_key: {
            "llm_name": result.get("llm_name", ""),
            "answer": result.get("answer", "")[:500],
            "latency_seconds": result.get("latency_seconds", 0),
            "cost_usd": result.get("cost_usd", 0.0),
        }
        for llm_key, result in generation_results.items()
    }

    comparison_data = {
        "query_text": query,
        "retriever_type": retriever_type,
        "top_k": top_k,
        "results": json.dumps(serializable),
    }
    record_id = database.insert_model_comparison(comparison_data)
    print(f"💾 Comparison saved to PostgreSQL (id: {record_id})")
    return record_id


def print_qa_history(limit: int = 5):
    """Print recent Q&A history from PostgreSQL."""
    rows = database.get_qa_history(limit)
    if not rows:
        print("  No Q&A history yet.")
        return

    print(f"\n── Recent Q&A History (last {limit}) {'─' * 30}")
    for i, row in enumerate(rows, 1):
        llm = row.get("llm_name", "unknown")
        run_at = row.get("run_at", "")[:16]
        query = row.get("query_text", "")[:60]
        answer = row.get("answer_text", "")[:80]
        latency = row.get("latency_seconds", 0)
        cost = row.get("cost_usd", 0)
        cost_str = "FREE" if cost == 0 else f"${cost:.4f}"

        print(f"  #{i} | {llm} | {run_at} | {cost_str} | {latency}s")
        print(f"     Q: {query}...")
        print(f"     A: {answer}...")
        print()
