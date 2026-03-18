"""
experiment_runner.py — Full evaluation experiment matrix for research.
Runs all combinations of chunk × embedding × retriever × LLM × K.
Saves per-question results to eval_results + aggregated to experiments.
"""

import os
import time
from statistics import mean

from core.config import (
    CHUNK_CONFIGS, EMBEDDING_CONFIGS,
    TOP_K_VALUES, DEFAULT_TOP_K, DEFAULT_CHUNK_CONFIG,
    DEFAULT_EMBEDDING_CONFIG, OUTPUT_DIR, DOCS_DIR,
)
from generation.llm_handler import LLM_CONFIGS, generate_answer
from generation.prompt_builder import build_prompt
from retrieval.retriever import embed_query, similarity_search, mmr_search, hybrid_search
from core.embedder import load_embedding_model, embed_chunks
from core.chunker import chunk_documents
from core.qdrant_store import create_collection, upsert_chunks
from evaluation.metrics_calculator import (
    compute_precision_at_k, compute_recall_at_k, compute_mrr,
    compute_hit_rate, find_relevant_chunk_ids,
    compute_ragas_metrics, compute_all_nlp_metrics,
    compute_rouge_l, compute_bleu,
)
from core import database


# ─── Constants ──────────────────────────────────────────────────────────────

# Use ALL configs from config.py
EVAL_CHUNK_CONFIGS = CHUNK_CONFIGS
EVAL_EMBEDDING_CONFIGS = EMBEDDING_CONFIGS

# LLMs to evaluate — all available
EVAL_LLM_KEYS = [k for k in LLM_CONFIGS.keys()]

# Retrievers to test
EVAL_RETRIEVERS = ["similarity", "mmr", "hybrid"]

QUESTIONS_PER_RUN = 20    # questions per config for research validity
RATE_LIMIT_SLEEP = 0.3


# ─── Function 1: Estimate Experiment Count ──────────────────────────────────

def estimate_experiment_count(
    chunk_configs=None, emb_configs=None, retrievers=None, llm_keys=None, topk_values=None
) -> dict:
    """Print experiment matrix dimensions and return count dict."""
    chunks = chunk_configs or EVAL_CHUNK_CONFIGS
    embeds = emb_configs or EVAL_EMBEDDING_CONFIGS
    rets = retrievers or EVAL_RETRIEVERS
    llms = llm_keys or EVAL_LLM_KEYS
    topks = topk_values or TOP_K_VALUES

    n_chunks = len(chunks)
    n_embeds = len(embeds)
    n_retrievers = len(rets)
    n_llms = len(llms)
    n_topk = len(topks)
    total = n_chunks * n_embeds * n_retrievers * n_llms * n_topk

    chunk_names = [c["name"] for c in chunks]
    emb_names = [e["name"] for e in embeds]
    llm_names = [LLM_CONFIGS[k]["name"] for k in llms if k in LLM_CONFIGS]

    print(f"\n📊 Experiment Matrix:")
    print(f"   Chunk configs    : {n_chunks}   → {', '.join(chunk_names)}")
    print(f"   Embedding models : {n_embeds}   → {', '.join(emb_names)}")
    print(f"   Retrievers       : {n_retrievers}   → {', '.join(rets)}")
    print(f"   LLMs             : {n_llms}   → {', '.join(llm_names)}")
    print(f"   Top-K values     : {n_topk}   → {topks}")
    print(f"   ─────────────────────────────────────")
    print(f"   Total runs       : {total}")
    print(f"   Questions/run    : {QUESTIONS_PER_RUN}")
    print(f"   Est. time (Groq) : ~{total * 8 * QUESTIONS_PER_RUN / 3600:.1f} hrs")

    return {
        "chunk_configs": n_chunks,
        "embedding_models": n_embeds,
        "retrievers": n_retrievers,
        "llms": n_llms,
        "top_k_values": n_topk,
        "total": total,
    }


# ─── Function 2: Run Single Experiment ──────────────────────────────────────

def run_single_experiment(config: dict, testset_df, qdrant_client, chunks: list,
                          use_ragas: bool = True, embedding_model=None,
                          phase: str = "", save_per_question: bool = True) -> dict:
    """
    Run one evaluation config across sampled questions.
    Saves per-question results to eval_results table.

    config keys: chunk_config, embedding_config, retriever_type,
                 llm_key, top_k, collection_name
    """
    import pandas as pd

    if embedding_model is None:
        embedding_model = load_embedding_model(config["embedding_config"])

    # Sample questions
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
    per_question_results = []

    llm_name = LLM_CONFIGS[config["llm_key"]]["name"]
    chunk_name = config["chunk_config"]["name"]
    embed_name = config["embedding_config"]["name"]

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

            p_at_k = compute_precision_at_k(retrieved_ids, relevant_ids, config["top_k"])
            r_at_k = compute_recall_at_k(retrieved_ids, relevant_ids, config["top_k"])
            q_mrr = compute_mrr(retrieved_ids, relevant_ids)
            q_hit = compute_hit_rate(retrieved_ids, relevant_ids)

            precision_scores.append(p_at_k)
            recall_scores.append(r_at_k)
            mrr_scores.append(q_mrr)
            hit_rates.append(q_hit)

            # ── Generate answer ──────────────────────────────────────
            prompt = build_prompt(str(row["question"]), retrieved)
            t_gen_start = time.time()
            gen = generate_answer(prompt, config["llm_key"])
            t_gen = time.time() - t_gen_start

            latencies.append(t_retrieval + t_gen)
            total_cost += gen.get("cost_usd", 0.0)

            answer = gen.get("answer", "")
            q_rouge = compute_rouge_l(answer, ground_truth_str)
            q_bleu = compute_bleu(answer, ground_truth_str)

            # ── Accumulate for RAGAS + NLP ───────────────────────────
            eval_samples.append({
                "question": str(row["question"]),
                "answer": answer,
                "contexts": [r.get("page_content", "") for r in retrieved],
                "ground_truth": ground_truth_str,
            })
            predictions.append(answer)
            references.append(ground_truth_str)

            # ── Per-question result ──────────────────────────────────
            if save_per_question:
                per_question_results.append({
                    "phase": phase,
                    "chunk_strategy": chunk_name,
                    "embedding_model": embed_name,
                    "retriever_type": retriever_type,
                    "llm_name": llm_name,
                    "top_k": config["top_k"],
                    "question": str(row["question"]),
                    "ground_truth": ground_truth_str,
                    "generated_answer": answer,
                    "retrieved_chunk_ids": retrieved_ids,
                    "relevant_chunk_ids": relevant_ids,
                    "precision_at_k": p_at_k,
                    "recall_at_k": r_at_k,
                    "mrr": q_mrr,
                    "hit_rate": q_hit,
                    "rouge_l": q_rouge,
                    "bleu": q_bleu,
                    "bertscore": 0.0,  # computed in batch below
                    "faithfulness": 0.0,  # computed in batch below
                    "retrieval_latency_ms": round(t_retrieval * 1000, 2),
                    "generation_latency_ms": round(t_gen * 1000, 2),
                    "cost_usd": gen.get("cost_usd", 0.0),
                })

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

    # ── Compute RAGAS scores ─────────────────────────────────────────
    if use_ragas:
        groq_key = (os.getenv("GROQ_API_KEY2", "").strip()
                    or os.getenv("GROQ_API_KEY1", "").strip()
                    or os.getenv("GROQ_API_KEY", "").strip())
        ragas_scores = compute_ragas_metrics(eval_samples, groq_key)
    else:
        ragas_scores = {"faithfulness": 0.0, "answer_relevancy": 0.0}

    # ── Compute NLP scores ───────────────────────────────────────────
    nlp_scores = compute_all_nlp_metrics(predictions, references)

    # ── Save per-question results to DB ──────────────────────────────
    if save_per_question and per_question_results:
        # Update batch bertscore and faithfulness
        for pqr in per_question_results:
            pqr["bertscore"] = nlp_scores.get("bertscore", 0.0)
            pqr["faithfulness"] = ragas_scores.get("faithfulness", 0.0)
        try:
            database.insert_eval_results_batch(per_question_results)
        except Exception as e:
            print(f"  ⚠️  Failed to save per-question results: {e}")

    # ── Assemble aggregated result ───────────────────────────────────
    result = {
        "phase": phase,
        "chunk_strategy": chunk_name,
        "embedding_model": embed_name,
        "retriever_type": config["retriever_type"],
        "llm_name": llm_name,
        "top_k": config["top_k"],
        "precision_at_k": avg_precision,
        "recall_at_k": avg_recall,
        "mrr": avg_mrr,
        "hit_rate": avg_hit_rate,
        "faithfulness": ragas_scores.get("faithfulness", 0.0),
        "answer_relevancy": ragas_scores.get("answer_relevancy", 0.0),
        "rouge_l": nlp_scores.get("rouge_l", 0.0),
        "bleu": nlp_scores.get("bleu", 0.0),
        "bertscore": nlp_scores.get("bertscore", 0.0),
        "latency_seconds": round(mean(latencies), 3),
        "cost_per_query": round(total_cost / max(len(predictions), 1), 6),
        "num_questions": len(predictions),
    }

    # Store aggregated in DB
    try:
        exp_id = database.insert_experiment(result)
        # Update per-question results with experiment_id
        if save_per_question and per_question_results:
            for pqr in per_question_results:
                pqr["experiment_id"] = exp_id
    except Exception as e:
        print(f"  ⚠️  DB insert failed: {e}")

    return result

# ─── Function 3: Run Phase A — Chunking Evaluation ─────────────────────────

def run_phase_a(testset_df, qdrant_client, docs, use_ragas=True) -> list:
    """
    Phase A: Vary chunking strategy only.
    Fixed: MiniLM embedding, similarity retriever, groq-llama8b, k=5.
    """
    fix_embed = EMBEDDING_CONFIGS[0]  # MiniLM
    fix_retriever = "similarity"
    fix_llm = "groq-llama8b"
    fix_k = 5

    total = len(EVAL_CHUNK_CONFIGS)
    print(f"\n{'='*60}")
    print(f"  PHASE A — Chunking Strategy Evaluation ({total} configs)")
    print(f"  Fixed: {fix_embed['name']} | {fix_retriever} | {LLM_CONFIGS[fix_llm]['name']} | k={fix_k}")
    print(f"{'='*60}")

    results = []
    indexed = {}

    for i, chunk_cfg in enumerate(EVAL_CHUNK_CONFIGS, 1):
        combo_key = f"{chunk_cfg['name']}_{fix_embed['name']}"
        collection_name = f"deeprag_{combo_key}"

        if combo_key not in indexed:
            print(f"\n🔄 [{i}/{total}] Indexing: {combo_key}")
            new_chunks = chunk_documents(docs, chunk_cfg, OUTPUT_DIR)
            emb_model = load_embedding_model(fix_embed)
            embedded = embed_chunks(new_chunks, emb_model)
            create_collection(qdrant_client, collection_name, fix_embed["vector_size"])
            upsert_chunks(qdrant_client, collection_name, embedded)
            indexed[combo_key] = (collection_name, new_chunks, emb_model)

        coll, chunks, emb_model = indexed[combo_key]
        config = {
            "chunk_config": chunk_cfg,
            "embedding_config": fix_embed,
            "retriever_type": fix_retriever,
            "llm_key": fix_llm,
            "top_k": fix_k,
            "collection_name": coll,
        }

        print(f"  ▶ [{i}/{total}] {chunk_cfg['name']}")
        result = run_single_experiment(config, testset_df, qdrant_client, chunks,
                                        use_ragas=use_ragas, embedding_model=emb_model,
                                        phase="A")
        if result:
            results.append(result)
            print(f"      P@K={result['precision_at_k']:.3f} | R@K={result['recall_at_k']:.3f}")

    print(f"\n✅ Phase A complete: {len(results)}/{total} experiments")
    return results


# ─── Function 4: Run Phase B — Embedding Evaluation ────────────────────────

def run_phase_b(testset_df, qdrant_client, docs, chunk_config=None, use_ragas=True) -> list:
    """
    Phase B: Vary embedding model only.
    Fixed: Given chunk config (or default), similarity retriever, groq-llama8b, k=5.
    """
    fix_chunk = chunk_config or DEFAULT_CHUNK_CONFIG
    fix_retriever = "similarity"
    fix_llm = "groq-llama8b"
    fix_k = 5

    total = len(EVAL_EMBEDDING_CONFIGS)
    print(f"\n{'='*60}")
    print(f"  PHASE B — Embedding Model Evaluation ({total} configs)")
    print(f"  Fixed: {fix_chunk['name']} | {fix_retriever} | {LLM_CONFIGS[fix_llm]['name']} | k={fix_k}")
    print(f"{'='*60}")

    results = []
    indexed = {}

    for i, emb_cfg in enumerate(EVAL_EMBEDDING_CONFIGS, 1):
        combo_key = f"{fix_chunk['name']}_{emb_cfg['name']}"
        collection_name = f"deeprag_{combo_key}"

        if combo_key not in indexed:
            print(f"\n🔄 [{i}/{total}] Indexing: {combo_key}")
            new_chunks = chunk_documents(docs, fix_chunk, OUTPUT_DIR)
            emb_model = load_embedding_model(emb_cfg)
            embedded = embed_chunks(new_chunks, emb_model)
            create_collection(qdrant_client, collection_name, emb_cfg["vector_size"])
            upsert_chunks(qdrant_client, collection_name, embedded)
            indexed[combo_key] = (collection_name, new_chunks, emb_model)

        coll, chunks, emb_model = indexed[combo_key]
        config = {
            "chunk_config": fix_chunk,
            "embedding_config": emb_cfg,
            "retriever_type": fix_retriever,
            "llm_key": fix_llm,
            "top_k": fix_k,
            "collection_name": coll,
        }

        print(f"  ▶ [{i}/{total}] {emb_cfg['name']}")
        result = run_single_experiment(config, testset_df, qdrant_client, chunks,
                                        use_ragas=use_ragas, embedding_model=emb_model,
                                        phase="B")
        if result:
            results.append(result)
            print(f"      P@K={result['precision_at_k']:.3f} | R@K={result['recall_at_k']:.3f}")

    print(f"\n✅ Phase B complete: {len(results)}/{total} experiments")
    return results


# ─── Function 5: Run Phase C — Retriever Evaluation ────────────────────────

def run_phase_c(testset_df, qdrant_client, docs,
                chunk_config=None, embed_config=None, use_ragas=True) -> list:
    """
    Phase C: Vary retriever type × top_k.
    Fixed: Given chunk + embedding configs, groq-llama8b.
    """
    fix_chunk = chunk_config or DEFAULT_CHUNK_CONFIG
    fix_embed = embed_config or DEFAULT_EMBEDDING_CONFIG
    fix_llm = "groq-llama8b"

    total = len(EVAL_RETRIEVERS) * len(TOP_K_VALUES)
    print(f"\n{'='*60}")
    print(f"  PHASE C — Retriever Evaluation ({total} configs)")
    print(f"  Fixed: {fix_chunk['name']} | {fix_embed['name']} | {LLM_CONFIGS[fix_llm]['name']}")
    print(f"{'='*60}")

    # Index once
    combo_key = f"{fix_chunk['name']}_{fix_embed['name']}"
    collection_name = f"deeprag_{combo_key}"
    print(f"\n🔄 Indexing: {combo_key}")
    new_chunks = chunk_documents(docs, fix_chunk, OUTPUT_DIR)
    emb_model = load_embedding_model(fix_embed)
    embedded = embed_chunks(new_chunks, emb_model)
    create_collection(qdrant_client, collection_name, fix_embed["vector_size"])
    upsert_chunks(qdrant_client, collection_name, embedded)

    results = []
    run_num = 0

    for retriever in EVAL_RETRIEVERS:
        for k in TOP_K_VALUES:
            run_num += 1
            config = {
                "chunk_config": fix_chunk,
                "embedding_config": fix_embed,
                "retriever_type": retriever,
                "llm_key": fix_llm,
                "top_k": k,
                "collection_name": collection_name,
            }

            print(f"  ▶ [{run_num}/{total}] {retriever} | k={k}")
            result = run_single_experiment(config, testset_df, qdrant_client, new_chunks,
                                            use_ragas=use_ragas, embedding_model=emb_model,
                                            phase="C")
            if result:
                results.append(result)
                print(f"      P@K={result['precision_at_k']:.3f} | MRR={result['mrr']:.3f}")

    print(f"\n✅ Phase C complete: {len(results)}/{total} experiments")
    return results


# ─── Function 6: Run Phase D — LLM Comparison ──────────────────────────────

def run_phase_d(testset_df, qdrant_client, docs,
                chunk_config=None, embed_config=None,
                retriever_type="similarity", top_k=5, use_ragas=True) -> list:
    """
    Phase D: Vary LLM only.
    Fixed: Everything else from best configs.
    """
    fix_chunk = chunk_config or DEFAULT_CHUNK_CONFIG
    fix_embed = embed_config or DEFAULT_EMBEDDING_CONFIG

    available_llms = [k for k in EVAL_LLM_KEYS if k in LLM_CONFIGS]
    total = len(available_llms)
    print(f"\n{'='*60}")
    print(f"  PHASE D — LLM Comparison ({total} models)")
    print(f"  Fixed: {fix_chunk['name']} | {fix_embed['name']} | {retriever_type} | k={top_k}")
    print(f"{'='*60}")

    # Index once
    combo_key = f"{fix_chunk['name']}_{fix_embed['name']}"
    collection_name = f"deeprag_{combo_key}"
    print(f"\n🔄 Indexing: {combo_key}")
    new_chunks = chunk_documents(docs, fix_chunk, OUTPUT_DIR)
    emb_model = load_embedding_model(fix_embed)
    embedded = embed_chunks(new_chunks, emb_model)
    create_collection(qdrant_client, collection_name, fix_embed["vector_size"])
    upsert_chunks(qdrant_client, collection_name, embedded)

    results = []

    for i, llm_key in enumerate(available_llms, 1):
        config = {
            "chunk_config": fix_chunk,
            "embedding_config": fix_embed,
            "retriever_type": retriever_type,
            "llm_key": llm_key,
            "top_k": top_k,
            "collection_name": collection_name,
        }

        print(f"  ▶ [{i}/{total}] {LLM_CONFIGS[llm_key]['name']}")
        result = run_single_experiment(config, testset_df, qdrant_client, new_chunks,
                                        use_ragas=use_ragas, embedding_model=emb_model,
                                        phase="D")
        if result:
            results.append(result)
            print(f"      Faith={result['faithfulness']:.3f} | ROUGE-L={result['rouge_l']:.3f} | "
                  f"Latency={result['latency_seconds']:.2f}s")

    print(f"\n✅ Phase D complete: {len(results)}/{total} experiments")
    return results


# ─── Function 7: Run Full Experiment Matrix ─────────────────────────────────

def run_full_experiment_matrix(testset_df, qdrant_client, docs, use_ragas=True) -> list:
    """Run all chunk × embedding × retriever × LLM × K combinations."""
    counts = estimate_experiment_count()
    total = counts["total"]

    print(f"\n⚠️  This will run {total} experiments.")
    try:
        confirm = input("Continue? [y/n]: ").strip().lower()
        if confirm != "y":
            print("❌ Cancelled.")
            return []
    except (EOFError, KeyboardInterrupt):
        print("\n❌ Cancelled.")
        return []

    results = []
    indexed_collections = {}
    run_num = 0

    for chunk_cfg in EVAL_CHUNK_CONFIGS:
        for emb_cfg in EVAL_EMBEDDING_CONFIGS:
            combo_key = f"{chunk_cfg['name']}_{emb_cfg['name']}"
            collection_name = f"deeprag_{combo_key}"

            if combo_key not in indexed_collections:
                print(f"\n🔄 Indexing new combo: {combo_key}")
                new_chunks = chunk_documents(docs, chunk_cfg, OUTPUT_DIR)
                emb_model = load_embedding_model(emb_cfg)
                embedded = embed_chunks(new_chunks, emb_model)
                create_collection(qdrant_client, collection_name, emb_cfg["vector_size"])
                upsert_chunks(qdrant_client, collection_name, embedded)
                indexed_collections[combo_key] = (collection_name, new_chunks, emb_model)

            current_collection, current_chunks, current_emb = indexed_collections[combo_key]

            for retriever in EVAL_RETRIEVERS:
                for llm_key in EVAL_LLM_KEYS:
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

                        result = run_single_experiment(
                            config, testset_df, qdrant_client, current_chunks,
                            use_ragas=use_ragas, embedding_model=current_emb,
                            phase="FULL",
                        )
                        if result:
                            results.append(result)
                            print(f"      P@K={result['precision_at_k']:.3f} | "
                                  f"R@K={result['recall_at_k']:.3f}")

    print(f"\n✅ Completed {len(results)}/{total} experiments")
    return results


# ─── Function 8: Run Quick Evaluation ───────────────────────────────────────

def run_quick_evaluation(testset_df, qdrant_client, collection_name: str,
                         embedding_model, chunks: list,
                         use_ragas: bool = True) -> list:
    """
    Fast mode — default config, all retrievers × all eval LLMs, default K.
    Total runs = 3 retrievers × N LLMs.
    """
    available_llms = [k for k in EVAL_LLM_KEYS if k in LLM_CONFIGS]
    total = len(EVAL_RETRIEVERS) * len(available_llms)
    ragas_label = "with RAGAS" if use_ragas else "NO RAGAS (fast mode)"
    print(f"\n📊 Running Quick Evaluation [{ragas_label}]...")
    print(f"   {total} runs total ({len(EVAL_RETRIEVERS)} retrievers × {len(available_llms)} LLMs)")
    print(f"   {QUESTIONS_PER_RUN} questions per run")

    results = []
    run_num = 0

    for retriever in EVAL_RETRIEVERS:
        for llm_key in available_llms:
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

            result = run_single_experiment(config, testset_df, qdrant_client, chunks,
                                            use_ragas=use_ragas,
                                            embedding_model=embedding_model,
                                            phase="QUICK")
            if result:
                results.append(result)
                faith_str = f"Faith={result['faithfulness']:.3f} | " if use_ragas else ""
                print(f"      ✅ P@K={result['precision_at_k']:.3f} | "
                      f"R@K={result['recall_at_k']:.3f} | "
                      f"{faith_str}"
                      f"{result['latency_seconds']:.2f}s")

    print(f"\n✅ Completed {len(results)}/{total} experiments")
    return results
