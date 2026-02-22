"""
run_evaluation.py — Phase 4 entry point.
Runs testset generation, experiment evaluation, export, and leaderboard.
"""

from pathlib import Path

from config import (
    QDRANT_COLLECTION_NAME, DEFAULT_CHUNK_CONFIG,
    DEFAULT_EMBEDDING_CONFIG, DEFAULT_TOP_K, OUTPUT_DIR, DOCS_DIR,
)
import database
from document_loader import load_documents
from chunker import chunk_documents
from embedder import load_embedding_model
from qdrant_store import get_qdrant_client, get_collection_info
from testset_generator import generate_testset, load_testset, validate_testset
from experiment_runner import (
    run_single_experiment, run_quick_evaluation,
    run_full_experiment_matrix, EVAL_LLM_KEYS,
)
from results_exporter import (
    export_to_csv, print_leaderboard,
    print_metric_comparison_table, generate_chart_data,
)
from llm_handler import LLM_CONFIGS


def main():
    """Phase 4 pipeline: init → testset → evaluate → export → leaderboard."""

    print()
    print("=" * 60)
    print("  PHASE 4: EVALUATION MODULE")
    print("=" * 60)
    print()

    # ─── Step 1: Initialize from Phase 1+2+3 ────────────────────────────
    print("=" * 55)
    print("  STEP 1: INITIALIZE FROM PHASE 1 + 2 + 3")
    print("=" * 55)

    database.init_db(reset=False)

    client = get_qdrant_client()
    info = get_collection_info(client, QDRANT_COLLECTION_NAME)

    if info["vectors_count"] == 0:
        print("❌ No vectors found in Qdrant. Run Phase 1 first (python main.py).")
        return None

    embedding_model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)

    print("\n📄 Loading documents for BM25 index...")
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs, DEFAULT_CHUNK_CONFIG, OUTPUT_DIR)

    print("✅ Phase 1+2+3 initialized")

    # ─── Step 2: Database ready ─────────────────────────────────────────
    print()
    print("=" * 55)
    print("  STEP 2: DATABASE READY")
    print("=" * 55)
    print("  ✅ experiments table available")
    print("  ✅ qa_logs table available")

    # ─── Step 3: Testset ────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  STEP 3: QA TESTSET")
    print("=" * 55)

    if Path("output/testset.csv").exists():
        df = load_testset()
    else:
        print("🔄 Generating testset using Groq (llama-3.1-8b-instant)...")
        df = generate_testset(docs, embedding_model)

    df = validate_testset(df)
    print(f"📋 Using {len(df)} questions for evaluation\n")

    # ─── Step 4: Mode selection ─────────────────────────────────────────
    print("═" * 50)
    print("  DeepRAG Evaluation Mode")
    print("═" * 50)
    print("  1. Quick Eval  (3 retrievers × 4 LLMs, default config) ~15 min")
    print("  2. Full Matrix (all combinations)                      ~2–4 hrs")
    print("  3. Single Run  (fastest sanity check)                  ~3 min")

    try:
        mode_input = input("Choose [1/2/3, default=1]: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Exiting.")
        return None

    mode = mode_input if mode_input in ("1", "2", "3") else "1"

    results = []

    if mode == "1":
        # Quick evaluation
        results = run_quick_evaluation(
            df, client, QDRANT_COLLECTION_NAME,
            embedding_model, chunks,
        )

    elif mode == "2":
        # Full matrix
        results = run_full_experiment_matrix(df, client, docs)

    elif mode == "3":
        # Single run — fastest sanity check
        first_available = None
        for k in EVAL_LLM_KEYS:
            if k in LLM_CONFIGS:
                first_available = k
                break
        if not first_available:
            first_available = list(LLM_CONFIGS.keys())[0]

        config = {
            "chunk_config": DEFAULT_CHUNK_CONFIG,
            "embedding_config": DEFAULT_EMBEDDING_CONFIG,
            "retriever_type": "mmr",
            "llm_key": first_available,
            "top_k": DEFAULT_TOP_K,
            "collection_name": QDRANT_COLLECTION_NAME,
        }
        print(f"\n  ▶ Single run: mmr | {LLM_CONFIGS[first_available]['name']} | k={DEFAULT_TOP_K}")
        result = run_single_experiment(config, df, client, chunks)
        if result:
            results = [result]

    if not results:
        print("❌ No experiment results. Check API keys and configuration.")
        return None

    # ─── Step 5: Export + Display ───────────────────────────────────────
    print()
    print("=" * 55)
    print("  STEP 5: RESULTS")
    print("=" * 55)

    export_to_csv(results)
    print_leaderboard(results)
    print_metric_comparison_table(results)
    generate_chart_data(results)
    print("\n📁 All outputs saved to output/")

    # ─── Step 6: Show best config ───────────────────────────────────────
    print()
    best = database.get_best_config()
    if best:
        print("═" * 50)
        print("  🏆 RECOMMENDED CONFIGURATION")
        print("═" * 50)
        print(f"  Chunking    : {best.get('chunk_strategy', 'N/A')}")
        print(f"  Embedding   : {best.get('embedding_model', 'N/A')}")
        print(f"  Retriever   : {best.get('retriever_type', 'N/A')}")
        print(f"  LLM         : {best.get('llm_name', 'N/A')}")
        print(f"  Top-K       : {best.get('top_k', 'N/A')}")
        print(f"  ──────────────────────────────────────")
        print(f"  Precision@K : {best.get('precision_at_k', 'N/A')}")
        print(f"  Recall@K    : {best.get('recall_at_k', 'N/A')}")
        print(f"  Faithfulness: {best.get('faithfulness', 'N/A')}")
        print(f"  Latency     : {best.get('latency_seconds', 'N/A')}s")
        print(f"  Cost/Query  : FREE (Groq + HuggingFace)")
        print()

    print("🎉 Phase 4 Complete!")
    return results


if __name__ == "__main__":
    main()
