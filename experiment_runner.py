"""
experiment_runner.py — Full evaluation experiment matrix for Phase 4.
Runs all combinations of chunk × embedding × retriever × LLM × K.
"""

import os
import time
from statistics import mean

from config import (
    DEFAULT_CHUNK_CONFIG, DEFAULT_EMBEDDING_CONFIG,
    TOP_K_VALUES, DEFAULT_TOP_K, OUTPUT_DIR, DOCS_DIR,
)
from llm_handler import LLM_CONFIGS, generate_answer
from prompt_builder import build_prompt
from retriever import embed_query, similarity_search, mmr_search, hybrid_search
from embedder import load_embedding_model, embed_chunks
from chunker import chunk_documents
from qdrant_store import create_collection, upsert_chunks
from metrics_calculator import (
    compute_precision_at_k, compute_recall_at_k, compute_mrr,
    compute_hit_rate, find_relevant_chunk_ids,
    compute_ragas_metrics, compute_all_nlp_metrics,
)
import database


# ─── Constants ──────────────────────────────────────────────────────────────

# LLMs to evaluate — mix of Groq (fast) + HuggingFace (free)
EVAL_LLM_KEYS = [
    "groq-llama8b",     # Groq — fastest
    "groq-mixtral",     # Groq — best reasoning
    "mistral-nemo",     # HuggingFace — free
    "phi3.5-mini",      # HuggingFace — free
]

# Retrievers to test
EVAL_RETRIEVERS = ["similarity", "mmr", "hybrid"]

# Chunk configs subset (representative sample)
EVAL_CHUNK_CONFIGS = [
    {"name": "fixed_256",      "size": 256,  "overlap": 0},
    {"name": "fixed_512",      "size": 512,  "overlap": 0},
    {"name": "overlap_512_20", "size": 512,  "overlap": 102},
]

# Embedding models subset
EVAL_EMBEDDING_CONFIGS = [
    {"name": "MiniLM", "model_id": "sentence-transformers/all-MiniLM-L6-v2", "vector_size": 384},
    {"name": "BGE",    "model_id": "BAAI/bge-small-en-v1.5",                  "vector_size": 384},
]

QUESTIONS_PER_RUN = 10   # sample size from testset per config

# Rate limit sleep (Groq free tier ~30 req/min)
RATE_LIMIT_SLEEP = 2


# ─── Function 1: Estimate Experiment Count ──────────────────────────────────

def estimate_experiment_count() -> dict:
    """Print experiment matrix dimensions and return count dict."""
    n_chunks = len(EVAL_CHUNK_CONFIGS)
    n_embeds = len(EVAL_EMBEDDING_CONFIGS)
    n_retrievers = len(EVAL_RETRIEVERS)
    n_llms = len(EVAL_LLM_KEYS)
    n_topk = len(TOP_K_VALUES)
    total = n_chunks * n_embeds * n_retrievers * n_llms * n_topk

    chunk_names = [c["name"] for c in EVAL_CHUNK_CONFIGS]
    emb_names = [e["name"] for e in EVAL_EMBEDDING_CONFIGS]
    llm_names = [LLM_CONFIGS[k]["name"] for k in EVAL_LLM_KEYS if k in LLM_CONFIGS]

    print(f"\n📊 Experiment Matrix:")
    print(f"   Chunk configs    : {n_chunks}   → {', '.join(chunk_names)}")
    print(f"   Embedding models : {n_embeds}   → {', '.join(emb_names)}")
    print(f"   Retrievers       : {n_retrievers}   → similarity, mmr, hybrid")
    print(f"   LLMs             : {n_llms}   → {', '.join(llm_names)}")
    print(f"   Top-K values     : {n_topk}   → {TOP_K_VALUES}")
    print(f"   ─────────────────────────────────────")
    print(f"   Total runs       : {total}")
    print(f"   Est. time (Groq) : ~{total * 8}s ({total * 8 / 60:.1f} min)")
    print(f"   Est. time (HF)   : ~{total * 20}s ({total * 20 / 60:.1f} min)")

    return {
        "chunk_configs": n_chunks,
        "embedding_models": n_embeds,
        "retrievers": n_retrievers,
        "llms": n_llms,
        "top_k_values": n_topk,
        "total": total,
    }


# ─── Function 2: Run Single Experiment ──────────────────────────────────────

def run_single_experiment(config: dict, testset_df, qdrant_client, chunks: list) -> dict:
    """
    Run one evaluation config across sampled questions.
    
    config keys: chunk_config, embedding_config, retriever_type,
                 llm_key, top_k, collection_name
    """
    import pandas as pd

    embedding_model = load_embedding_model(config["embedding_config"])

    # Sample questions (fixed seed for fair comparison)
    sample_df = testset_df.sample(
        min(QUESTIONS_PER_RUN, len(testset_df)), random_state=42
    )

    # Accumulators
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_rates = []
    eval_samples = []
    predictions = []
    references = []
    latencies = []
    total_cost = 0.0

    for _, row in sample_df.iterrows():
        try:
            # ── Embed query ──────────────────────────────────────────
            query_vector = embed_query(str(row["question"]), embedding_model)

            # ── Retrieve ─────────────────────────────────────────────
            t_start = time.time()
            retriever_type = config["retriever_type"]

            if retriever_type == "similarity":
                output = similarity_search(
                    qdrant_client, config["collection_name"],
                    query_vector, str(row["question"]), config["top_k"],
                )
            elif retriever_type == "mmr":
                output = mmr_search(
                    qdrant_client, config["collection_name"],
                    query_vector, str(row["question"]), config["top_k"],
                )
            elif retriever_type == "hybrid":
                output = hybrid_search(
                    qdrant_client, config["collection_name"],
                    query_vector, str(row["question"]), chunks, config["top_k"],
                )
            else:
                output = similarity_search(
                    qdrant_client, config["collection_name"],
                    query_vector, str(row["question"]), config["top_k"],
                )

            t_retrieval = time.time() - t_start
            retrieved = output["results"]

            # ── Retrieval metrics ────────────────────────────────────
            retrieved_ids = [r.get("chunk_id", r.get("payload", {}).get("chunk_id", "")) for r in retrieved]
            ground_truth_str = str(row.get("ground_truth", ""))
            relevant_ids = find_relevant_chunk_ids(ground_truth_str, chunks)

            precision_scores.append(compute_precision_at_k(retrieved_ids, relevant_ids, config["top_k"]))
            recall_scores.append(compute_recall_at_k(retrieved_ids, relevant_ids, config["top_k"]))
            mrr_scores.append(compute_mrr(retrieved_ids, relevant_ids))
            hit_rates.append(compute_hit_rate(retrieved_ids, relevant_ids))

            # ── Generate answer ──────────────────────────────────────
            prompt = build_prompt(str(row["question"]), retrieved)
            t_gen_start = time.time()
            gen = generate_answer(prompt, config["llm_key"])
            t_gen = time.time() - t_gen_start

            latencies.append(t_retrieval + t_gen)
            total_cost += gen.get("cost_usd", 0.0)

            # ── Accumulate for RAGAS + NLP ───────────────────────────
            eval_samples.append({
                "question": str(row["question"]),
                "answer": gen.get("answer", ""),
                "contexts": [r.get("page_content", "") for r in retrieved],
                "ground_truth": ground_truth_str,
            })
            predictions.append(gen.get("answer", ""))
            references.append(ground_truth_str)

            # Rate limit protection
            time.sleep(RATE_LIMIT_SLEEP)

        except Exception as e:
            print(f"  ⚠️  Error on question: {e}")
            continue

    if not precision_scores:
        print("  ❌ No successful evaluations for this config")
        return {}

    # ── Compute averaged retrieval scores ────────────────────────────
    avg_precision = round(mean(precision_scores), 4)
    avg_recall = round(mean(recall_scores), 4)
    avg_mrr = round(mean(mrr_scores), 4)
    avg_hit_rate = round(mean(hit_rates), 4)

    # ── Compute RAGAS scores (use dedicated RAGAS key) ─────────────────
    groq_key = (os.getenv("GROQ_API_KEY2", "").strip()
                or os.getenv("GROQ_API_KEY1", "").strip()
                or os.getenv("GROQ_API_KEY", "").strip())
    ragas_scores = compute_ragas_metrics(eval_samples, groq_key)

    # ── Compute NLP scores ───────────────────────────────────────────
    nlp_scores = compute_all_nlp_metrics(predictions, references)

    # ── Assemble result ──────────────────────────────────────────────
    result = {
        "chunk_strategy": config["chunk_config"]["name"],
        "embedding_model": config["embedding_config"]["name"],
        "retriever_type": config["retriever_type"],
        "llm_name": LLM_CONFIGS[config["llm_key"]]["name"],
        "top_k": config["top_k"],
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "hit_rate": avg_hit_rate,
        "faithfulness": ragas_scores["faithfulness"],
        "answer_relevancy": ragas_scores["answer_relevancy"],
        "rouge_l": nlp_scores["rouge_l"],
        "latency_seconds": round(mean(latencies), 3),
        "cost_per_query": round(total_cost / max(len(predictions), 1), 6),
    }

    # Store in DB
    try:
        database.insert_experiment(result)
    except Exception as e:
        print(f"  ⚠️  DB insert failed: {e}")

    return result


# ─── Function 3: Run Full Experiment Matrix ─────────────────────────────────

def run_full_experiment_matrix(testset_df, qdrant_client, docs) -> list:
    """Run all chunk × embedding × retriever × LLM × K combinations."""
    from document_loader import load_documents

    counts = estimate_experiment_count()
    total = counts["total"]

    print(f"\n⚠️  This will run {total} experiments. Estimated time: {total * 10 / 60:.0f}–{total * 20 / 60:.0f} min")
    try:
        confirm = input("Continue? [y/n]: ").strip().lower()
        if confirm != "y":
            print("❌ Cancelled.")
            return []
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Cancelled.")
        return []

    results = []
    indexed_collections = {}  # combo_key → (collection_name, chunks)
    run_num = 0

    for chunk_cfg in EVAL_CHUNK_CONFIGS:
        for emb_cfg in EVAL_EMBEDDING_CONFIGS:
            combo_key = f"{chunk_cfg['name']}_{emb_cfg['name']}"
            collection_name = f"deeprag_{combo_key}"

            # Re-chunk + re-embed only for new combos
            if combo_key not in indexed_collections:
                print(f"\n🔄 Indexing new combo: {combo_key}")
                new_chunks = chunk_documents(docs, chunk_cfg, OUTPUT_DIR)
                emb_model = load_embedding_model(emb_cfg)
                embedded = embed_chunks(new_chunks, emb_model)
                create_collection(qdrant_client, collection_name, emb_cfg["vector_size"])
                upsert_chunks(qdrant_client, collection_name, embedded)
                indexed_collections[combo_key] = (collection_name, new_chunks)

            current_collection, current_chunks = indexed_collections[combo_key]

            for retriever in EVAL_RETRIEVERS:
                for llm_key in EVAL_LLM_KEYS:
                    # Skip LLMs without API keys
                    if llm_key not in LLM_CONFIGS:
                        continue

                    for k in TOP_K_VALUES:
                        run_num += 1
                        config = {
                            "chunk_config": chunk_cfg,
                            "embedding_config": emb_cfg,
                            "retriever_type": retriever,
                            "llm_key": llm_key,
                            "top_k": k,
                            "collection_name": current_collection,
                        }

                        print(f"\n  ▶ [{run_num}/{total}] {chunk_cfg['name']} | "
                              f"{emb_cfg['name']} | {retriever} | "
                              f"{LLM_CONFIGS[llm_key]['name']} | k={k}")

                        result = run_single_experiment(config, testset_df, qdrant_client, current_chunks)
                        if result:
                            results.append(result)
                            print(f"      P@K={result['precision_at_k']:.3f} | "
                                  f"R@K={result['recall_at_k']:.3f} | "
                                  f"Faith={result['faithfulness']:.3f} | "
                                  f"{result['latency_seconds']:.2f}s")

    print(f"\n✅ Completed {len(results)}/{total} experiments")
    return results


# ─── Function 4: Run Quick Evaluation ───────────────────────────────────────

def run_quick_evaluation(testset_df, qdrant_client, collection_name: str,
                         embedding_model, chunks: list) -> list:
    """
    Fast mode — default config, all retrievers × all eval LLMs, default K.
    Total runs = 3 retrievers × 4 LLMs = 12 runs.
    """
    total = len(EVAL_RETRIEVERS) * len(EVAL_LLM_KEYS)
    print(f"\n📊 Running Quick Evaluation...")
    print(f"   {total} runs total ({len(EVAL_RETRIEVERS)} retrievers × {len(EVAL_LLM_KEYS)} LLMs)")
    print(f"   {QUESTIONS_PER_RUN} questions per run\n")

    results = []
    run_num = 0

    for retriever in EVAL_RETRIEVERS:
        for llm_key in EVAL_LLM_KEYS:
            if llm_key not in LLM_CONFIGS:
                print(f"  ⚠️  Skipping {llm_key} (not in LLM_CONFIGS)")
                continue

            run_num += 1
            config = {
                "chunk_config": DEFAULT_CHUNK_CONFIG,
                "embedding_config": DEFAULT_EMBEDDING_CONFIG,
                "retriever_type": retriever,
                "llm_key": llm_key,
                "top_k": DEFAULT_TOP_K,
                "collection_name": collection_name,
            }

            print(f"  ▶ [{run_num}/{total}] {retriever} | {LLM_CONFIGS[llm_key]['name']} | k={DEFAULT_TOP_K}")

            result = run_single_experiment(config, testset_df, qdrant_client, chunks)
            if result:
                results.append(result)
                print(f"      ✅ P@K={result['precision_at_k']:.3f} | "
                      f"R@K={result['recall_at_k']:.3f} | "
                      f"Faith={result['faithfulness']:.3f} | "
                      f"{result['latency_seconds']:.2f}s")

    print(f"\n✅ Completed {len(results)}/{total} experiments")
    return results
