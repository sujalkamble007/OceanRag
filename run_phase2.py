from pipeline.deeprag_pipeline import DeepRAGPipeline

if __name__ == "__main__":
    pipeline = DeepRAGPipeline()
    pipeline.run_phase1(rebuild=False)
    pipeline.run_phase2()
