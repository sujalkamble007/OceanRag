import os
from pipeline.deeprag_pipeline import DeepRAGPipeline
from evaluation.testset_generator import load_testset
from evaluation.experiment_runner import run_phase_a
from core.database import init_db
from core.document_loader import load_documents
from core.config import DOCS_DIR

def main():
    init_db()
    testset_df = load_testset()
    pipeline = DeepRAGPipeline()
    
    print("📄 Loading a small sample of documents...")
    all_docs = load_documents(DOCS_DIR)
    
    # Keep only first 2 documents (maybe ~50-100 pages, fast to embed)
    docs_to_keep = []
    current_docs = set()
    for d in all_docs:
        fn = d.metadata.get("filename", "")
        current_docs.add(fn)
        if len(current_docs) <= 2:
            docs_to_keep.append(d)
            
    print(f"✅ Fast Mode: Kept {len(docs_to_keep)} pages from {len(current_docs)} documents.")
    
    run_phase_a(testset_df, pipeline.qdrant_client, docs_to_keep, use_ragas=False)

if __name__ == "__main__":
    main()
