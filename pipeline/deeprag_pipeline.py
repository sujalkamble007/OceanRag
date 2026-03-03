# pipeline/deeprag_pipeline.py

import os
from pathlib import Path
from core import (
    init_db, load_documents, summarize_documents,
    chunk_documents, load_embedding_model, embed_chunks,
    get_qdrant_client, create_collection, upsert_chunks,
    get_collection_info, insert_document, insert_chunk,
    DEFAULT_CHUNK_CONFIG, DEFAULT_EMBEDDING_CONFIG,
    QDRANT_COLLECTION_NAME, DOCS_DIR, OUTPUT_DIR,
    DEFAULT_TOP_K
)
from retrieval import (
    similarity_search, mmr_search, hybrid_search,
    run_all_retrievers, print_retrieval_results,
    log_all_retrievers, save_results_to_csv
)
from generation import (
    get_available_llms, run_rag_query, stream_rag_query,
    run_multimodel_comparison, print_rag_result,
    print_qa_history
)
from evaluation import (
    generate_testset, load_testset, validate_testset,
    run_quick_evaluation, run_full_experiment_matrix,
    export_to_csv, print_leaderboard,
    print_metric_comparison_table, generate_chart_data
)
from core.config import validate_config
from core.database import get_best_config


class DeepRAGPipeline:
    """
    Master pipeline class.
    Holds all state across phases.
    All run_phaseX.py files instantiate this class.
    """

    def __init__(self):
        self.qdrant_client     = None
        self.collection_name   = QDRANT_COLLECTION_NAME
        self.embedding_model   = None
        self.chunks            = []
        self.docs              = []
        self.available_llms    = []
        self.is_phase1_ready   = False
        self.is_phase2_ready   = False
        self.is_phase3_ready   = False

    # ────────────────────────────────────────────────────
    # PHASE 1
    # ────────────────────────────────────────────────────
    def run_phase1(self, rebuild: bool = False):
        """
        Load docs → chunk → embed → store in Qdrant + PostgreSQL.
        Sets: self.qdrant_client, self.embedding_model, self.chunks
        """
        print("\n" + "="*55)
        print("  PHASE 1: DOCUMENT PIPELINE")
        print("="*55)

        # Validate config first
        validate_config()

        # Init DB
        init_db()
        print("✅ PostgreSQL tables ready\n")

        # Connect to Qdrant right away to check if we need to rebuild
        self.qdrant_client = get_qdrant_client()
        info = get_collection_info(self.qdrant_client, self.collection_name)
        vector_count = info.get("vectors_count", 0)

        if vector_count > 0 and not rebuild:
            print(f"⚡ Qdrant has {vector_count} vectors. Skipping re-index.")
            print("   Pass rebuild=True to force re-index.\n")
            self.embedding_model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)
            
            print(f"⚡ Chunks will be loaded on demand for hybrid search.\n")
            # Don't preload 37K rows — load lazily on first hybrid search call
            self.chunks = []  # Populated lazily
                
        else:
            # Need to build from scratch
            self.docs = load_documents(DOCS_DIR)
            file_page_map = summarize_documents(self.docs)

            # Insert document records to PostgreSQL
            doc_id_map = {}
            for fname, page_count in file_page_map.items():
                filepath = next(
                    (d.metadata["filepath"] for d in self.docs
                     if d.metadata["filename"] == fname), ""
                )
                doc_id = insert_document(fname, filepath, page_count)
                doc_id_map[fname] = doc_id

            # Chunk
            self.chunks = chunk_documents(
                self.docs, DEFAULT_CHUNK_CONFIG, OUTPUT_DIR
            )

            # Embed
            self.embedding_model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)
            embedded = embed_chunks(self.chunks, self.embedding_model)

            # Store in Qdrant
            create_collection(
                self.qdrant_client, self.collection_name,
                DEFAULT_EMBEDDING_CONFIG["vector_size"]
            )
            point_ids = upsert_chunks(
                self.qdrant_client, self.collection_name, embedded
            )

            # Store metadata in PostgreSQL
            for chunk, point_id in zip(self.chunks, point_ids):
                m = chunk.metadata
                insert_chunk({
                    "chunk_id":       m["chunk_id"],
                    "document_id":    doc_id_map.get(m.get("filename",""), None),
                    "filename":       m.get("filename",""),
                    "page_number":    m.get("page_number", 0),
                    "chunk_strategy": m.get("chunk_strategy",""),
                    "chunk_size":     m.get("chunk_size", 0),
                    "chunk_overlap":  m.get("chunk_overlap", 0),
                    "char_count":     m.get("char_count", 0),
                    "content_preview": chunk.page_content[:200],
                    "qdrant_point_id": str(point_id),
                    "embedding_model": DEFAULT_EMBEDDING_CONFIG["name"]
                })
            print(f"✅ Saved {len(self.chunks)} chunks to PostgreSQL\n")

        self.is_phase1_ready = True
        print("🎉 Phase 1 Complete\n")
        return self

    # ────────────────────────────────────────────────────
    # PHASE 2
    # ────────────────────────────────────────────────────
    def run_phase2(self, test_queries: list = None):
        """
        Test all 3 retrievers on sample queries.
        Requires phase1 to be run first.
        """
        if not self.is_phase1_ready:
            self.run_phase1()

        print("\n" + "="*55)
        print("  PHASE 2: RETRIEVAL ENGINE")
        print("="*55)

        if test_queries is None:
            test_queries = [
                "What are the environmental obligations under UNCLOS?",
                "ISA regulations for deep-sea mining",
                "Environmental Impact Assessment requirements",
            ]

        for query in test_queries:
            all_results  = run_all_retrievers(
                self.qdrant_client, self.collection_name,
                query, self.embedding_model, self.chunks, k=DEFAULT_TOP_K
            )
            print_retrieval_results(all_results)
            log_all_retrievers(
                all_results,
                DEFAULT_EMBEDDING_CONFIG["name"],
                DEFAULT_CHUNK_CONFIG["name"]
            )
            save_results_to_csv(all_results, OUTPUT_DIR)

        self.is_phase2_ready = True
        print("🎉 Phase 2 Complete\n")
        return self

    # ────────────────────────────────────────────────────
    # PHASE 3
    # ────────────────────────────────────────────────────
    def run_phase3(self):
        """
        Interactive RAG query mode with LLM generation.
        Requires phase1 to be run first.
        """
        if not self.is_phase1_ready:
            self.run_phase1()

        print("\n" + "="*55)
        print("  PHASE 3: LLM GENERATION")
        print("="*55)

        self.available_llms = get_available_llms()
        if not self.available_llms:
            raise EnvironmentError(
                "No LLMs available. Set GROQ_API_KEY or HF_API_TOKEN in .env"
            )

        # Test query
        test_result = run_rag_query(
            query="What are the environmental obligations under UNCLOS?",
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            chunks=self.chunks,
            retriever_type="mmr",
            llm_key=self.available_llms[0]["key"],
            top_k=DEFAULT_TOP_K
        )
        print_rag_result(test_result)

        # Multimodel comparison if 2+ LLMs
        if len(self.available_llms) >= 2:
            run_multimodel_comparison(
                query="What are ISA regulations for deep-sea mining contractors?",
                qdrant_client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                chunks=self.chunks,
                llm_keys=[m["key"] for m in self.available_llms]
            )

        self.is_phase3_ready = True
        print("🎉 Phase 3 Complete\n")
        return self

    # ────────────────────────────────────────────────────
    # PHASE 4
    # ────────────────────────────────────────────────────
    def run_phase4(self, mode: str = "quick", use_ragas: bool = True):
        """
        Phase 4: Evaluation Module.
        
        Args:
            mode: 'quick' | 'full' | 'single'
            use_ragas: Set False to skip RAGAS LLM-based evaluation (much faster).
                       Retrieval metrics (P@K, R@K, MRR) still run either way.
        Requires phase1 to be run first.
        """
        if not self.is_phase1_ready:
            self.run_phase1()

        print("\n" + "="*55)
        print("  PHASE 4: EVALUATION MODULE")
        print("="*55)

        # Load or generate testset
        testset_path = Path(OUTPUT_DIR) / "testset.csv"
        if testset_path.exists():
            df = load_testset()
        else:
            print("🔄 Generating testset via Groq...")
            df = generate_testset(self.docs, self.embedding_model)
        df = validate_testset(df)

        # Run evaluation
        if mode == "quick":
            results = run_quick_evaluation(
                df, self.qdrant_client, self.collection_name,
                self.embedding_model, self.chunks,
                use_ragas=use_ragas
            )
        elif mode == "full":
            results = run_full_experiment_matrix(
                df, self.qdrant_client, self.chunks
            )
        elif mode == "single":
            from evaluation.experiment_runner import run_single_experiment
            from core.config import DEFAULT_CHUNK_CONFIG, DEFAULT_EMBEDDING_CONFIG, DEFAULT_TOP_K
            config = {
                "chunk_config":     DEFAULT_CHUNK_CONFIG,
                "embedding_config": DEFAULT_EMBEDDING_CONFIG,
                "retriever_type":   "mmr",
                "llm_key":          self.available_llms[0]["key"] if self.available_llms else "default",
                "top_k":            DEFAULT_TOP_K,
                "collection_name":  self.collection_name
            }
            results = [run_single_experiment(
                config, df, self.qdrant_client, self.chunks
            )]

        # Export + Display
        export_to_csv(results)
        print_leaderboard(results)
        print_metric_comparison_table(results)
        generate_chart_data(results)

        # Best config
        best = get_best_config()
        if best:
            print("\n" + "="*44)
            print("  🏆 RECOMMENDED CONFIGURATION")
            print("="*44)
            print(f"  Chunking   : {best['chunk_strategy']}")
            print(f"  Embedding  : {best['embedding_model']}")
            print(f"  Retriever  : {best['retriever_type']}")
            print(f"  LLM        : {best['llm_name']}")
            print(f"  Top-K      : {best['top_k']}")
            print(f"  Precision  : {best['precision_at_k']}")
            print(f"  Recall     : {best['recall_at_k']}")
            print(f"  Faithfulness: {best['faithfulness']}")
            print(f"  Latency    : {best['latency_seconds']}s")
            print(f"  Cost       : FREE (Groq + HuggingFace)")

        print("\n🎉 Phase 4 Complete\n")
        return results

    # ────────────────────────────────────────────────────
    # INTERACTIVE MODE (used by run_phase3.py)
    # ────────────────────────────────────────────────────
    def interactive_query(self):
        """
        Loop for interactive question answering.
        Called after Phase 3 setup.
        """
        if not self.is_phase1_ready:
            self.run_phase1()

        self.available_llms = get_available_llms()

        while True:
            print("\n" + "="*44)
            print("  DeepRAG — Interactive Mode")
            print("="*44)
            print("  Commands: 'quit' | 'history' | 'compare'")
            print("  Or type a question\n")
            user_input = input("> ").strip()

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "history":
                print_qa_history(limit=5)
                continue
            if user_input.lower() == "compare":
                query = input("Query for comparison: ").strip()
                run_multimodel_comparison(
                    query=query,
                    qdrant_client=self.qdrant_client,
                    collection_name=self.collection_name,
                    embedding_model=self.embedding_model,
                    chunks=self.chunks,
                    llm_keys=[m["key"] for m in self.available_llms]
                )
                continue

            # Normal query
            print("Retriever: 1=similarity  2=mmr  3=hybrid [default=2]: ", end="")
            r_choice = input().strip() or "2"
            retriever_map = {"1": "similarity", "2": "mmr", "3": "hybrid"}
            retriever = retriever_map.get(r_choice, "mmr")

            print("LLMs available:")
            for i, m in enumerate(self.available_llms, 1):
                print(f"  {i}. {m['name']} ({m['provider']})")
            print(f"Choose [1-{len(self.available_llms)}, default=1]: ", end="")
            l_choice = input().strip() or "1"
            try:
                llm_key = self.available_llms[int(l_choice)-1]["key"]
            except (ValueError, IndexError):
                llm_key = self.available_llms[0]["key"]

            print("Top-K [3/5/10, default=5]: ", end="")
            k_choice = input().strip() or "5"
            k = int(k_choice) if k_choice in ["3","5","10"] else 5

            result = run_rag_query(
                query=user_input,
                qdrant_client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding_model=self.embedding_model,
                chunks=self.chunks,
                retriever_type=retriever,
                llm_key=llm_key,
                top_k=k
            )
            print_rag_result(result)

    # ────────────────────────────────────────────────────
    # PHASE 5 — API Method
    # ────────────────────────────────────────────────────
    def run_rag_query(
        self,
        query: str,
        retriever_type: str = "similarity",
        llm_key: str = "groq-llama8b",
        top_k: int = 5,
        user_id: int = None,
    ) -> dict:
        """
        Thin wrapper around generation.run_rag_query for the FastAPI backend.
        Returns a structured dict with: answer, sources, chunks, latency, cost, record_id.
        """
        if not self.is_phase1_ready:
            raise RuntimeError("Pipeline not ready. run_phase1() must complete first.")

        result = run_rag_query(
            query=query,
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            chunks=self.chunks,
            retriever_type=retriever_type,
            llm_key=llm_key,
            top_k=top_k,
        )

        # Build structured retrieved_chunks list for API response
        raw_chunks = result.get("retrieved_chunks", [])
        chunk_list = []
        for i, c in enumerate(raw_chunks):
            meta = c.metadata if hasattr(c, "metadata") else {}
            chunk_list.append({
                "rank": i + 1,
                "score": round(meta.get("score", 0.0), 4),
                "filename": meta.get("filename", ""),
                "page_number": meta.get("page_number", 0),
                "preview": (c.page_content[:250] if hasattr(c, "page_content") else str(c))[:250],
            })

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "retrieved_chunks": chunk_list,
            "llm_name": result.get("llm_name", llm_key),
            "latency_seconds": result.get("latency_seconds", 0.0),
            "cost_usd": result.get("cost_usd", 0.0),
            "record_id": result.get("record_id", 0),
        }

    # ────────────────────────────────────────────────────
    # PHASE 5b — Streaming API Method
    # ────────────────────────────────────────────────────
    def stream_rag_query_api(
        self,
        query: str,
        retriever_type: str = "similarity",
        llm_key: str = "groq-llama8b",
        top_k: int = 5,
        user_id: int = None,
    ):
        """Streaming wrapper for the FastAPI streaming endpoint.
        Yields SSE event dicts from generation.stream_rag_query."""
        if not self.is_phase1_ready:
            raise RuntimeError("Pipeline not ready. run_phase1() must complete first.")

        yield from stream_rag_query(
            query=query,
            qdrant_client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding_model=self.embedding_model,
            chunks=self.chunks,
            retriever_type=retriever_type,
            llm_key=llm_key,
            top_k=top_k,
        )

