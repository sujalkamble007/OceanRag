import sys
from pipeline.deeprag_pipeline import DeepRAGPipeline

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════╗
║        DeepRAG — Full Pipeline       ║
║  Phase 1 → 2 → 3 → 4 (Interactive)  ║
╚══════════════════════════════════════╝
""")
    pipeline = DeepRAGPipeline()

    # Phase 1: Build index
    pipeline.run_phase1(rebuild=False)

    # Phase 2: Test retrieval
    pipeline.run_phase2()

    # Phase 3: Test generation
    pipeline.run_phase3()

    # Phase 4: Evaluate
    eval_mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    pipeline.run_phase4(mode=eval_mode)

    # Interactive
    print("\n🚀 All phases complete. Starting interactive mode...\n")
    pipeline.interactive_query()
