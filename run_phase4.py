import sys
from pipeline.deeprag_pipeline import DeepRAGPipeline

if __name__ == "__main__":
    args = sys.argv[1:]

    # Parse mode: quick / full / single  (default: quick)
    mode = "quick"
    for a in args:
        if a in ("quick", "full", "single"):
            mode = a

    # Parse --no-ragas flag
    use_ragas = "--no-ragas" not in args

    # Usage:
    #   python run_phase4.py                    → quick mode, RAGAS ON
    #   python run_phase4.py --no-ragas         → quick mode, RAGAS OFF (fast)
    #   python run_phase4.py full               → full mode, RAGAS ON
    #   python run_phase4.py full --no-ragas    → full mode, RAGAS OFF

    pipeline = DeepRAGPipeline()
    pipeline.run_phase1(rebuild=False)
    pipeline.run_phase4(mode=mode, use_ragas=use_ragas)
