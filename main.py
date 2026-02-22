"""
main.py — Entry point orchestrating the full Phase 1 pipeline.
"""

from tqdm import tqdm

from config import (
    DOCS_DIR, OUTPUT_DIR, QDRANT_COLLECTION_NAME,
    DEFAULT_CHUNK_CONFIG, DEFAULT_EMBEDDING_CONFIG,
    CHUNK_CONFIGS,
)
import database
from document_loader import load_documents, summarize_documents
from chunker import chunk_documents, compare_chunk_strategies
from embedder import load_embedding_model, embed_chunks
from qdrant_store import (
    get_qdrant_client, create_collection, upsert_chunks,
    search_similar, get_collection_info,
)


def run_phase1(rebuild=False):
    """
    Orchestrates the full Phase 1 pipeline:
      1. Database init
      2. Load documents
      3. Chunk documents
      4. Connect to Qdrant
      5. Embed chunks (if needed)
      6. Store in Qdrant (if needed)
      7. Store metadata in PostgreSQL (if needed)
      8. Test retrieval
      9. Print summary

    Args:
        rebuild: If True, re-embeds and re-uploads even if Qdrant has vectors.

    Returns:
        Tuple of (qdrant_client, collection_name, embedding_model).
    """

    # ─── Step 1: Database Init ──────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 1: DATABASE INIT")
    print("=" * 55)
    database.init_db(reset=rebuild)
    print()

    # ─── Step 2: Load Documents ─────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 2: LOADING DOCUMENTS")
    print("=" * 55)
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("❌ No documents loaded. Place PDF files in the 'docs/' directory.")
        return None

    file_pages = summarize_documents(docs)

    # Insert each unique document into PostgreSQL and build filename → id map
    doc_id_map = {}
    for fname, page_count in file_pages.items():
        # Derive filepath from the first page of this file
        filepath = ""
        for d in docs:
            if d.metadata.get("filename") == fname:
                filepath = d.metadata.get("filepath", "")
                break
        doc_id = database.insert_document(fname, filepath, page_count)
        doc_id_map[fname] = doc_id

    # ─── Step 3: Chunk Documents ────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 3: CHUNKING")
    print("=" * 55)
    chunks = chunk_documents(docs, DEFAULT_CHUNK_CONFIG, OUTPUT_DIR)
    print()

    # ─── Step 4: Connect to Qdrant ──────────────────────────────────────────
    print("=" * 55)
    print("  STEP 4: CONNECTING TO QDRANT CLOUD")
    print("=" * 55)
    client = get_qdrant_client()
    collection_info = get_collection_info(client, QDRANT_COLLECTION_NAME)
    print()

    # Decide whether to embed + upload
    need_embedding = rebuild or collection_info["vectors_count"] == 0

    if not need_embedding:
        print("ℹ️  Using existing Qdrant collection (skip embedding/upload).")
        print()
        # Still need the embedding model for test queries
        embedding_model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)
    else:
        # ─── Step 5 & 6: Embed + Upload ────────────────────────────────────
        print("=" * 55)
        print("  STEP 5 & 6: EMBEDDING + QDRANT UPLOAD")
        print("=" * 55)
        embedding_model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)
        embedded_chunks = embed_chunks(chunks, embedding_model)

        vector_size = DEFAULT_EMBEDDING_CONFIG["vector_size"]
        create_collection(client, QDRANT_COLLECTION_NAME, vector_size)
        point_ids = upsert_chunks(client, QDRANT_COLLECTION_NAME, embedded_chunks)
        print()

        # ─── Step 7: Save metadata to PostgreSQL ──────────────────────────
        print("=" * 55)
        print("  STEP 7: SAVING METADATA TO POSTGRESQL")
        print("=" * 55)
        chunk_records = []
        for i, chunk in enumerate(chunks):
            meta = chunk.metadata
            fname = meta.get("filename", "")
            chunk_records.append({
                "chunk_id": meta["chunk_id"],
                "document_id": doc_id_map.get(fname),
                "filename": fname,
                "page_number": meta.get("page_number"),
                "chunk_strategy": meta.get("chunk_strategy"),
                "chunk_size": meta.get("chunk_size"),
                "chunk_overlap": meta.get("chunk_overlap"),
                "char_count": meta.get("char_count"),
                "content_preview": chunk.page_content[:200],
                "qdrant_point_id": point_ids[i],
                "embedding_model": DEFAULT_EMBEDDING_CONFIG["name"],
            })
        database.insert_chunks_batch(chunk_records)
        print(f"✅ Saved {len(chunks)} chunk records to PostgreSQL.")
        print()

    # ─── Step 8: Test Retrieval ─────────────────────────────────────────────
    print("=" * 55)
    print("  STEP 8: TEST RETRIEVAL")
    print("=" * 55)
    test_queries = [
        "What are the environmental obligations under UNCLOS?",
        "ISA regulations for deep-sea mining",
        "Environmental Impact Assessment requirements",
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        query_vector = embedding_model.embed_query(query)
        search_similar(client, QDRANT_COLLECTION_NAME, query_vector, k=3)

    print()

    # ─── Step 9: Summary ────────────────────────────────────────────────────
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    chunk_stats = database.get_chunk_stats()
    final_info = get_collection_info(client, QDRANT_COLLECTION_NAME)

    print(f"  Total documents : {len(doc_id_map)}")
    print(f"  Total chunks    : {chunk_stats['total_chunks']}")
    print(f"  Qdrant vectors  : {final_info['vectors_count']}")
    print(f"  PostgreSQL rows : {chunk_stats['total_chunks']}")
    print(f"  Embedding model : {DEFAULT_EMBEDDING_CONFIG['name']} ({DEFAULT_EMBEDDING_CONFIG['model_id']})")
    print(f"  Chunk strategy  : {DEFAULT_CHUNK_CONFIG['name']}")
    print()
    print("🎉 Phase 1 Complete! Pipeline is working end-to-end.")

    return (client, QDRANT_COLLECTION_NAME, embedding_model)


def compare_all_configs():
    """
    Loads documents and runs compare_chunk_strategies() with all CHUNK_CONFIGS.
    Call manually for experiment matrix comparison.
    """
    docs = load_documents(DOCS_DIR)
    if not docs:
        print("❌ No documents found.")
        return
    summarize_documents(docs)
    compare_chunk_strategies(docs, CHUNK_CONFIGS, OUTPUT_DIR)


if __name__ == "__main__":
    run_phase1(rebuild=False)
