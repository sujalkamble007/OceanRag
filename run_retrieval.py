"""
run_retrieval.py — Phase 2 entry point.

Runs all 3 retrieval strategies, compares across Top-K values,
logs everything to PostgreSQL, and offers an interactive query mode.
"""

from config import (
    QDRANT_COLLECTION_NAME, DEFAULT_CHUNK_CONFIG,
    DEFAULT_EMBEDDING_CONFIG, TOP_K_VALUES, OUTPUT_DIR, DOCS_DIR,
)
import database
from document_loader import load_documents
from chunker import chunk_documents
from embedder import load_embedding_model
from qdrant_store import get_qdrant_client, get_collection_info
from retriever import (
    embed_query, similarity_search, mmr_search, hybrid_search,
    run_all_retrievers, print_retrieval_results,
)
from retrieval_logger import (
    log_retrieval, log_all_retrievers, save_results_to_csv,
    get_retrieval_summary,
)


def main():
    """Phase 2 pipeline: retrieve, compare, log, interact."""

    print()
    print("=" * 60)
    print("  PHASE 2: RETRIEVAL ENGINE")
    print("=" * 60)
    print()

    # ─── Step 1: Initialize from Phase 1 ────────────────────────────────
    print("=" * 55)
    print("  STEP 1: INITIALIZE FROM PHASE 1")
    print("=" * 55)

    # Ensure tables exist (creates retrieval_logs if missing)
    database.init_db(reset=False)

    # Connect to Qdrant
    client = get_qdrant_client()
    info = get_collection_info(client, QDRANT_COLLECTION_NAME)

    if info["vectors_count"] == 0:
        print("❌ No vectors found in Qdrant. Run Phase 1 first (python main.py).")
        return None

    # Load embedding model
    embedding_model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)

    # Reload chunks for BM25 (hybrid search needs original Document objects)
    print("\n📄 Loading documents for BM25 index...")
    docs = load_documents(DOCS_DIR)
    chunks = chunk_documents(docs, DEFAULT_CHUNK_CONFIG, OUTPUT_DIR)
    print()

    # ─── Step 2: Verify retrieval_logs table ────────────────────────────
    print("=" * 55)
    print("  STEP 2: DATABASE READY")
    print("=" * 55)
    print("  ✅ retrieval_logs table available")
    print()

    # ─── Step 3: Test all 3 strategies on one query ─────────────────────
    print("=" * 55)
    print("  STEP 3: TEST RETRIEVAL — ALL 3 STRATEGIES")
    print("=" * 55)

    test_query = "What are the environmental obligations under UNCLOS?"
    all_results = run_all_retrievers(
        client, QDRANT_COLLECTION_NAME,
        test_query, embedding_model, chunks, k=5,
    )
    print_retrieval_results(all_results)
    log_all_retrievers(all_results, DEFAULT_EMBEDDING_CONFIG["name"],
                       DEFAULT_CHUNK_CONFIG["name"])
    save_results_to_csv(all_results, OUTPUT_DIR)

    # ─── Step 4: Top-K latency comparison ───────────────────────────────
    print("=" * 55)
    print("  STEP 4: TOP-K LATENCY COMPARISON")
    print("=" * 55)

    latency_table = []
    for k in TOP_K_VALUES:
        r = run_all_retrievers(
            client, QDRANT_COLLECTION_NAME,
            test_query, embedding_model, chunks, k=k,
        )
        log_all_retrievers(r, DEFAULT_EMBEDDING_CONFIG["name"],
                           DEFAULT_CHUNK_CONFIG["name"])
        latency_table.append({
            "k": k,
            "similarity": r["similarity"]["latency_seconds"],
            "mmr": r["mmr"]["latency_seconds"],
            "hybrid": r["hybrid"]["latency_seconds"],
        })

    print(f"\n  {'K':<5} | {'Similarity':>11} | {'MMR':>8} | {'Hybrid':>8}")
    print(f"  {'-'*5}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}")
    for row in latency_table:
        print(f"  {row['k']:<5} | {row['similarity']:>9.4f}s | {row['mmr']:>6.4f}s | {row['hybrid']:>6.4f}s")
    print()

    # Retrieval summary from DB
    print("=" * 55)
    print("  RETRIEVAL LOG SUMMARY")
    print("=" * 55)
    get_retrieval_summary()

    # ─── Step 5: Interactive mode ───────────────────────────────────────
    print("=" * 55)
    print("  INTERACTIVE MODE")
    print("=" * 55)
    print("  Type a query and select a retrieval strategy.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            query = input("🔍 Enter a query (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting interactive mode.")
            break

        if not query:
            print("  ⚠️  Empty query, please try again.")
            continue
        if query.lower() == "quit":
            print("👋 Goodbye!")
            break

        # Choose retriever
        print("  1 = Similarity  2 = MMR  3 = Hybrid  4 = All")
        try:
            choice = input("  Strategy [1/2/3/4]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting interactive mode.")
            break

        if choice not in ("1", "2", "3", "4"):
            choice = "4"

        # Choose k
        try:
            k_input = input("  Top-K [3/5/10]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting interactive mode.")
            break

        try:
            k = int(k_input)
            if k not in (3, 5, 10):
                k = 5
        except ValueError:
            k = 5

        query_vector = embed_query(query, embedding_model)

        if choice == "4":
            results = run_all_retrievers(
                client, QDRANT_COLLECTION_NAME,
                query, embedding_model, chunks, k=k,
            )
            print_retrieval_results(results)
            log_all_retrievers(results, DEFAULT_EMBEDDING_CONFIG["name"],
                               DEFAULT_CHUNK_CONFIG["name"])
            save_results_to_csv(results, OUTPUT_DIR)
        else:
            retriever_map = {
                "1": ("similarity", lambda: similarity_search(
                    client, QDRANT_COLLECTION_NAME, query_vector, query, k)),
                "2": ("mmr", lambda: mmr_search(
                    client, QDRANT_COLLECTION_NAME, query_vector, query, k)),
                "3": ("hybrid", lambda: hybrid_search(
                    client, QDRANT_COLLECTION_NAME, query_vector, query, chunks, k)),
            }
            rtype, run_fn = retriever_map[choice]
            result = run_fn()

            # Print single retriever result
            print(f"\n── {rtype.upper()} SEARCH (latency: {result['latency_seconds']}s) ──")
            for r in result["results"]:
                print(f"  [{r['rank']}] Score: {r['score']:.4f} | {r['filename']} | Page {r['page_number']}")
                preview = r.get("page_content", "")[:80]
                if preview:
                    print(f"      Preview: {preview}...")

            log_retrieval(result, DEFAULT_EMBEDDING_CONFIG["name"],
                          DEFAULT_CHUNK_CONFIG["name"])
            print()

    print()
    print("🎉 Phase 2 Complete!")
    return (client, QDRANT_COLLECTION_NAME, embedding_model, chunks)


if __name__ == "__main__":
    main()
