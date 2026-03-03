# core/__init__.py
# Exposes Phase 1 components cleanly

from core.config import (
    DOCS_DIR, OUTPUT_DIR, QDRANT_COLLECTION_NAME,
    CHUNK_CONFIGS, EMBEDDING_CONFIGS,
    DEFAULT_CHUNK_CONFIG, DEFAULT_EMBEDDING_CONFIG,
    TOP_K_VALUES, DEFAULT_TOP_K
)
from core.database import (
    init_db, insert_document, insert_chunk,
    insert_experiment, get_chunk_stats,
    insert_retrieval_log, insert_qa_log,
    insert_model_comparison, get_qa_history,
    get_all_experiments, get_best_config
)
from core.document_loader import load_documents, summarize_documents
from core.chunker import chunk_documents, compare_chunk_strategies
from core.embedder import load_embedding_model, embed_chunks, embed_query
from core.qdrant_store import (
    get_qdrant_client, create_collection,
    upsert_chunks, search_similar,
    get_collection_info, delete_collection
)
