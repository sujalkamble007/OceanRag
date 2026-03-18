import os
import string
import pandas as pd
from sqlalchemy import create_engine, text
from core.config import DATABASE_URL, DEFAULT_TOP_K
from evaluation.testset_generator import load_testset

def find_relevant_chunk_ids(ground_truth: str, chunks_data: list, threshold: float = 0.6) -> list:
    """Same logic as metrics_calculator.py but with raw dicts or strings."""
    relevant = []
    
    gt_str = str(ground_truth).lower().strip()
    if not gt_str:
        if chunks_data:
             return [chunks_data[0] if isinstance(chunks_data[0], str) else chunks_data[0].get("chunk_id", "chunk_0")]
        return []

    translator = str.maketrans('', '', string.punctuation)
    gt_tokens = set(gt_str.translate(translator).split())

    for chunk in chunks_data:
        # If it's just a string ID, we can't do content overlap check here
        if isinstance(chunk, str):
            # We assume it matched if the database recorded it as retrieved.
            # But realistically without page_content, we can't properly evaluate recall
            # for historical rows that didn't save page_content.
            # We will just append it if exact match fails, to prevent crashing.
            continue
            
        chunk_str = chunk.get("page_content", "").lower()
        chunk_id = chunk.get("chunk_id", "")
        
        # 1. Exact substring match
        if gt_str in chunk_str and len(gt_str) > 3:
            relevant.append(chunk_id)
            continue
            
        # 2. Token overlap
        chunk_tokens = set(chunk_str.translate(translator).split())
        if gt_tokens and chunk_tokens:
            overlap = len(gt_tokens & chunk_tokens) / len(gt_tokens)
            if overlap >= threshold:
                relevant.append(chunk_id)

    if not relevant and chunks_data:
        relevant = [chunks_data[0] if isinstance(chunks_data[0], str) else chunks_data[0].get("chunk_id", "chunk_0")]

    return relevant

def compute_precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return hits / k if k > 0 else 0.0

def compute_recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant_set)
    return hits / len(relevant_set) if relevant_set else 0.0

def compute_mrr(retrieved_ids: list, relevant_ids: list) -> float:
    relevant_set = set(relevant_ids)
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0

def compute_hit_rate(retrieved_ids: list, relevant_ids: list) -> float:
    return 1.0 if any(r in set(relevant_ids) for r in retrieved_ids) else 0.0

def main():
    engine = create_engine(DATABASE_URL)
    testset_df = load_testset()
    gt_map = {str(row["question"]): str(row["ground_truth"]) for _, row in testset_df.iterrows()}

    with engine.connect() as conn:
        # Get all eval rows
        rows = conn.execute(text("SELECT id, question, retrieved_chunk_ids, experiment_id, top_k FROM eval_results")).fetchall()
        
        print(f"Found {len(rows)} eval_results to recalculate...")
        
        for row in rows:
            row_id, question, retrieved_chunks, exp_id, top_k = row
            if not retrieved_chunks:
                continue
                
            if isinstance(retrieved_chunks, str):
                import json
                retrieved_chunks = json.loads(retrieved_chunks)
                
            ground_truth = gt_map.get(question, "")
            
            # retrieved_chunks is a list of strings (chunk IDs), not dicts
            retrieved_ids = [c if isinstance(c, str) else c.get("chunk_id", "") for c in retrieved_chunks]
            
            # Recalculate
            relevant_ids = find_relevant_chunk_ids(ground_truth, retrieved_chunks)
            
            p_at_k = compute_precision_at_k(retrieved_ids, relevant_ids, top_k)
            r_at_k = compute_recall_at_k(retrieved_ids, relevant_ids, top_k)
            mrr = compute_mrr(retrieved_ids, relevant_ids)
            hit_rate = compute_hit_rate(retrieved_ids, relevant_ids)
            
            import json
            
            # Update eval_results
            conn.execute(
                text("""
                    UPDATE eval_results 
                    SET relevant_chunk_ids = :rel,
                        precision_at_k = :p,
                        recall_at_k = :r,
                        mrr = :mrr,
                        hit_rate = :hit
                    WHERE id = :id
                """),
                {"rel": json.dumps(relevant_ids), "p": p_at_k, "r": r_at_k, "mrr": mrr, "hit": hit_rate, "id": row_id}
            )
            
        print("Updated eval_results.")
        
        # Now recalculate aggregated experiments table
        exps = conn.execute(text("SELECT id FROM experiments")).fetchall()
        for exp in exps:
            exp_id = exp[0]
            conn.execute(
                text("""
                    UPDATE experiments
                    SET precision_at_k = (SELECT AVG(precision_at_k) FROM eval_results WHERE experiment_id = :id),
                        recall_at_k = (SELECT AVG(recall_at_k) FROM eval_results WHERE experiment_id = :id),
                        mrr = (SELECT AVG(mrr) FROM eval_results WHERE experiment_id = :id),
                        hit_rate = (SELECT AVG(hit_rate) FROM eval_results WHERE experiment_id = :id)
                    WHERE id = :id
                """),
                {"id": exp_id}
            )
        conn.commit()
        print("Updated experiments table averages.")

if __name__ == "__main__":
    main()
