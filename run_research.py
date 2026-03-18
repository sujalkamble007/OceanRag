"""
run_research.py — Master entry point for phased research evaluation.

Usage:
    python run_research.py --phase A               # Chunking evaluation
    python run_research.py --phase B               # Embedding evaluation
    python run_research.py --phase C               # Retriever evaluation
    python run_research.py --phase D               # LLM comparison
    python run_research.py --phase all             # Run A → B → C → D
    python run_research.py --phase quick           # Quick evaluation only
    python run_research.py --generate-testset      # Regenerate testset
    python run_research.py --export                # Export results to CSV
    python run_research.py --no-ragas --phase A    # Skip RAGAS (much faster)
"""

import argparse
from core.config import validate_config, CHUNK_CONFIGS, EMBEDDING_CONFIGS, DOCS_DIR
from core.database import init_db


def _ensure_docs_loaded(pipeline, fast_subset: int = 0):
    """Load documents if pipeline didn't load them (rebuild=False)."""
    if not pipeline.docs:
        from core.document_loader import load_documents
        print("📄 Loading documents from disk...")
        all_docs = load_documents(DOCS_DIR, max_docs=fast_subset)
        
        if fast_subset > 0:
            docs_to_keep = []
            current_docs = set()
            for d in all_docs:
                fn = d.metadata.get("filename", "")
                current_docs.add(fn)
                if len(current_docs) <= fast_subset:
                    docs_to_keep.append(d)
            pipeline.docs = docs_to_keep
            print(f"   ⏩ Fast Mode: Kept {len(pipeline.docs)} pages from {fast_subset} documents.")
        else:
            pipeline.docs = all_docs
            print(f"   ✅ Loaded {len(pipeline.docs)} pages")
    return pipeline.docs


def main():
    parser = argparse.ArgumentParser(description="DeepRAG Research Evaluation")
    parser.add_argument("--phase", choices=["A", "B", "C", "D", "all", "quick", "full"],
                        help="Which evaluation phase to run")
    parser.add_argument("--generate-testset", action="store_true",
                        help="Regenerate the synthetic testset")
    parser.add_argument("--export", action="store_true",
                        help="Export results to CSV files")
    parser.add_argument("--no-ragas", action="store_true",
                        help="Skip RAGAS evaluation (much faster)")
    parser.add_argument("--questions", type=int, default=20,
                        help="Questions per experiment run (default: 20)")
    parser.add_argument("--fast-subset", type=int, default=0,
                        help="Only load this many documents (super fast testing mode)")
    args = parser.parse_args()

    validate_config()
    init_db()

    use_ragas = not args.no_ragas

    # Lazy imports to avoid loading models at startup
    from pipeline.deeprag_pipeline import DeepRAGPipeline
    from evaluation.testset_generator import generate_testset, load_testset, validate_testset
    from evaluation.experiment_runner import (
        run_phase_a, run_phase_b, run_phase_c, run_phase_d,
        run_full_experiment_matrix, run_quick_evaluation,
        QUESTIONS_PER_RUN, estimate_experiment_count,
    )
    from evaluation.results_exporter import export_to_csv, print_leaderboard, print_metric_comparison_table, generate_chart_data
    import evaluation.experiment_runner as runner

    # Override questions per run if specified
    if args.questions:
        runner.QUESTIONS_PER_RUN = args.questions

    # ── Generate testset ─────────────────────────────────────────────
    if args.generate_testset:
        pipeline = DeepRAGPipeline()
        pipeline.run_phase1(rebuild=False)
        docs = _ensure_docs_loaded(pipeline, args.fast_subset)
        generate_testset(docs, pipeline.embedding_model)
        print("✅ Testset generated. Run again with --phase to start evaluation.")
        return

    # ── Export results ────────────────────────────────────────────────
    if args.export:
        from core.database import get_all_experiments
        results = get_all_experiments()
        if results:
            export_to_csv(results)
            print_leaderboard(results)
            print_metric_comparison_table(results)
            generate_chart_data(results)
            print("📈 Chart data (output/chart_data.json) successfully generated for Analytics.")
        else:
            print("❌ No experiment results found in DB.")
        return

    # ── Run phases ───────────────────────────────────────────────────
    if not args.phase:
        parser.print_help()
        return

    # Initialize pipeline
    print("\n🚀 Initializing DeepRAG Pipeline...")
    pipeline = DeepRAGPipeline()
    pipeline.run_phase1(rebuild=False)

    # Load testset
    testset_df = load_testset()
    testset_df = validate_testset(testset_df)

    qdrant = pipeline.qdrant_client
    docs = _ensure_docs_loaded(pipeline, args.fast_subset)
    all_results = []

    if args.phase == "A" or args.phase == "all":
        results = run_phase_a(testset_df, qdrant, docs, use_ragas=use_ragas)
        all_results.extend(results)

    if args.phase == "B" or args.phase == "all":
        results = run_phase_b(testset_df, qdrant, docs, use_ragas=use_ragas)
        all_results.extend(results)

    if args.phase == "C" or args.phase == "all":
        results = run_phase_c(testset_df, qdrant, docs, use_ragas=use_ragas)
        all_results.extend(results)

    if args.phase == "D" or args.phase == "all":
        results = run_phase_d(testset_df, qdrant, docs, use_ragas=use_ragas)
        all_results.extend(results)

    if args.phase == "quick":
        results = run_quick_evaluation(
            testset_df, qdrant, pipeline.collection_name,
            pipeline.embedding_model, pipeline.chunks,
            use_ragas=use_ragas,
        )
        all_results.extend(results)

    if args.phase == "full":
        results = run_full_experiment_matrix(testset_df, qdrant, docs, use_ragas=use_ragas)
        all_results.extend(results)

    # ── Export & display ─────────────────────────────────────────────
    if all_results:
        export_to_csv(all_results)
        print_leaderboard(all_results)
        print_metric_comparison_table(all_results)
        print(f"\n🎉 All results saved to DB and CSV. {len(all_results)} experiments completed.")
    else:
        print("\n❌ No results generated.")


if __name__ == "__main__":
    main()
