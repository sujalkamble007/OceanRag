"""
run_generation.py — Phase 3 entry point.
Runs RAG pipeline: retrieve → generate → store → compare → interactive mode.
"""

from config import (
    QDRANT_COLLECTION_NAME, DEFAULT_CHUNK_CONFIG,
    DEFAULT_EMBEDDING_CONFIG, OUTPUT_DIR, DOCS_DIR,
)
import database
from document_loader import load_documents
from chunker import chunk_documents
from embedder import load_embedding_model
from qdrant_store import get_qdrant_client, get_collection_info
from llm_handler import get_available_llms, LLM_CONFIGS
from generation_pipeline import (
    run_rag_query, run_multimodel_comparison, print_rag_result,
)
from answer_store import print_qa_history


def main():
    """Phase 3 pipeline: init → test RAG → compare models → interactive."""

    print()
    print("=" * 60)
    print("  PHASE 3: LLM GENERATION")
    print("=" * 60)
    print()

    # ─── Step 1: Initialize from Phase 1 ────────────────────────────────
    print("=" * 55)
    print("  STEP 1: INITIALIZE FROM PHASE 1 + 2")
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

    # ─── Step 2: Init new DB tables ─────────────────────────────────────
    print()
    print("=" * 55)
    print("  STEP 2: DATABASE READY")
    print("=" * 55)
    print("  ✅ qa_logs table available")
    print("  ✅ model_comparisons table available")

    # ─── Step 3: Check available LLMs ───────────────────────────────────
    print()
    print("=" * 55)
    print("  STEP 3: CHECK AVAILABLE LLMs")
    print("=" * 55)

    available = get_available_llms()
    if not available:
        print("❌ No LLMs available. Add HF_API_TOKEN to .env file.")
        print("   Get free token: huggingface.co/settings/tokens")
        return None

    first_llm = available[0]["key"]

    # ─── Step 4: Test RAG query ─────────────────────────────────────────
    print("=" * 55)
    print("  STEP 4: TEST RAG QUERY")
    print("=" * 55)

    result = run_rag_query(
        query="What are the environmental obligations under UNCLOS?",
        qdrant_client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding_model=embedding_model,
        chunks=chunks,
        retriever_type="mmr",
        llm_key=first_llm,
        top_k=5,
    )
    print_rag_result(result)

    # ─── Step 5: Multimodel comparison (if 2+ LLMs) ────────────────────
    if len(available) >= 2:
        print("=" * 55)
        print("  STEP 5: MULTIMODEL COMPARISON")
        print("=" * 55)

        # Use first 3 LLMs to keep it fast
        compare_keys = [m["key"] for m in available[:3]]
        run_multimodel_comparison(
            query="What are ISA regulations for deep-sea mining contractors?",
            qdrant_client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding_model=embedding_model,
            chunks=chunks,
            retriever_type="similarity",
            top_k=5,
            llm_keys=compare_keys,
        )
    else:
        print("\n  ℹ️  Skipping multimodel comparison (only 1 LLM available)")

    # ─── Step 6: Show history ───────────────────────────────────────────
    print()
    print("=" * 55)
    print("  STEP 6: Q&A HISTORY")
    print("=" * 55)
    print_qa_history(limit=3)

    # ─── Step 7: Interactive mode ───────────────────────────────────────
    print("=" * 55)
    print("  INTERACTIVE MODE — DeepRAG")
    print("=" * 55)
    print("  Commands: 'quit' | 'history' | 'compare' | or type a query")
    print()

    while True:
        try:
            query = input("🔍 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        if not query:
            continue

        if query.lower() == "quit":
            print("👋 Goodbye!")
            break

        if query.lower() == "history":
            print_qa_history()
            continue

        if query.lower() == "compare":
            try:
                cmp_query = input("  Query for comparison: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n👋 Exiting.")
                break

            if cmp_query:
                compare_keys = [m["key"] for m in available[:3]]
                run_multimodel_comparison(
                    query=cmp_query,
                    qdrant_client=client,
                    collection_name=QDRANT_COLLECTION_NAME,
                    embedding_model=embedding_model,
                    chunks=chunks,
                    retriever_type="mmr",
                    top_k=5,
                    llm_keys=compare_keys,
                )
            continue

        # Regular query — choose retriever, LLM, top_k
        print("  1=similarity  2=mmr  3=hybrid [default=2]: ", end="")
        try:
            ret_choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break
        retriever_map = {"1": "similarity", "2": "mmr", "3": "hybrid"}
        retriever_type = retriever_map.get(ret_choice, "mmr")

        # LLM selection
        print("  Available LLMs:")
        for idx, m in enumerate(available, 1):
            cost_label = "FREE" if m.get("input_cost_per_1k", 0) == 0 else "PAID"
            print(f"    {idx}. {m['name']} ({cost_label})")
        print(f"  LLM [default=1]: ", end="")
        try:
            llm_choice = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        try:
            llm_idx = int(llm_choice) - 1
            if llm_idx < 0 or llm_idx >= len(available):
                llm_idx = 0
        except (ValueError, IndexError):
            llm_idx = 0
        llm_key = available[llm_idx]["key"]

        # Top-K
        print("  Top-K [3/5/10, default=5]: ", end="")
        try:
            k_input = input().strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Exiting.")
            break

        try:
            k = int(k_input)
            if k not in (3, 5, 10):
                k = 5
        except ValueError:
            k = 5

        # Run
        result = run_rag_query(
            query=query,
            qdrant_client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding_model=embedding_model,
            chunks=chunks,
            retriever_type=retriever_type,
            llm_key=llm_key,
            top_k=k,
        )
        print_rag_result(result)

    print()
    print("🎉 Phase 3 Complete!")
    return (client, QDRANT_COLLECTION_NAME, embedding_model, chunks)


if __name__ == "__main__":
    main()
