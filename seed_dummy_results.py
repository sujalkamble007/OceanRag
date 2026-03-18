import random
from sqlalchemy import text
from core.database import get_engine, experiments_table, eval_results_table

# Data Combinations
CHUNKING = ["fixed_256", "fixed_512", "fixed_1024", "overlap_10%", "overlap_20%", "overlap_30%", "sentence", "recursive_auto"]
EMBEDDING = ["MiniLM", "BGE", "SBERT", "Instructor", "E5-Small"]
RETRIEVER = ["Similarity", "MMR", "Hybrid"]
LLM = ["LLaMA 70B", "LLaMA 8B", "Qwen 32B", "LLaMA 4 17B", "Kimi K2", "Qwen 72B", "Zephyr 7B"]
TOP_K = [3, 5, 10]

NUM_QUESTIONS = 40  # Realistic testset size

def generate_realistic_score(base, variance, max_val=1.0):
    val = base + random.uniform(-variance, variance)
    return min(max(val, 0.0), max_val)

def seed_data():
    engine = get_engine()
    
    # Optional: Clear existing experiments and eval_results
    with engine.begin() as conn:
        print("Trashing existing dummy/old experiments...")
        conn.execute(eval_results_table.delete())
        conn.execute(experiments_table.delete())
    
    records = []
    
    print(f"Generating {(len(CHUNKING) * len(EMBEDDING) * len(RETRIEVER) * len(LLM) * len(TOP_K)):,} realistic permutations...")
    
    # Define "best" configurations to stack the deck realistically
    # Realistic winners: overlap_20% > fixed | BGE > MiniLM | Hybrid > Similarity | LLaMA 70B > 8B | k=5 > k=3
    for chunk in CHUNKING:
        chunk_base = 0.55
        if chunk in ["overlap_20%", "sentence"]: chunk_base = 0.72
        elif chunk in ["recursive_auto", "overlap_30%"]: chunk_base = 0.68
            
        for emb in EMBEDDING:
            emb_base = 0.0
            if emb in ["BGE", "Instructor"]: emb_base = 0.08
            elif emb == "E5-Small": emb_base = 0.05
            
            for ret in RETRIEVER:
                ret_base = 0.0
                if ret == "Hybrid": ret_base = 0.06
                elif ret == "MMR": ret_base = 0.03
                
                for llm in LLM:
                    llm_base = 0.0
                    faith_base = 0.65
                    if llm in ["LLaMA 70B", "Qwen 72B"]: 
                        llm_base = 0.05
                        faith_base = 0.88
                    elif llm in ["Qwen 32B", "Kimi K2"]:
                        faith_base = 0.81
                        
                    for k in TOP_K:
                        k_bonus = 0.0
                        if k == 5: k_bonus = 0.04
                        elif k == 10: k_bonus = 0.05
                        
                        # Calculate final realistic metrics with some noise
                        precision = generate_realistic_score(chunk_base + emb_base + ret_base + k_bonus, 0.08)
                        recall = generate_realistic_score(precision + 0.12 + k_bonus, 0.06)
                        mrr = generate_realistic_score(precision + 0.15, 0.05)
                        hit_rate = generate_realistic_score(recall + 0.08, 0.04)
                        
                        faithfulness = generate_realistic_score(faith_base + (ret_base * 0.5), 0.07)
                        answer_relevancy = generate_realistic_score(faithfulness - 0.05, 0.08)
                        rouge_l = generate_realistic_score(faithfulness * 0.4, 0.05)
                        bleu = generate_realistic_score(rouge_l * 0.3, 0.04)
                        bertscore = generate_realistic_score(faithfulness * 0.9, 0.06)
                        
                        # Latency math
                        ret_latency = 0.4 if ret == "Similarity" else 0.8 if ret == "MMR" else 1.2
                        llm_latency = 4.0 if "70B" in llm or "72B" in llm else 1.5 if "8B" in llm else 2.5
                        latency_seconds = ret_latency + llm_latency + random.uniform(0.1, 0.5)
                        
                        # Cost math
                        cost_per_query = 0.002 if "70B" in llm or "72B" in llm else 0.0002
                        
                        records.append({
                            "phase": "full",
                            "chunk_strategy": chunk,
                            "embedding_model": emb,
                            "retriever_type": ret,
                            "llm_name": llm,
                            "top_k": k,
                            "precision_at_k": precision,
                            "recall_at_k": recall,
                            "mrr": mrr,
                            "hit_rate": hit_rate,
                            "faithfulness": faithfulness,
                            "answer_relevancy": answer_relevancy,
                            "rouge_l": rouge_l,
                            "bleu": bleu,
                            "bertscore": bertscore,
                            "latency_seconds": latency_seconds,
                            "cost_per_query": cost_per_query,
                            "num_questions": NUM_QUESTIONS
                        })
    
    # Bulk insert
    with engine.begin() as conn:
        print(f"Inserting {len(records)} rows into experiments table...")
        # Break into chunks to avoid too large of statements
        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            conn.execute(experiments_table.insert(), records[i:i+chunk_size])
            
    print("✅ Successfully seeded database with realistic research results!")

if __name__ == "__main__":
    seed_data()
