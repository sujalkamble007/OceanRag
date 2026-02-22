"""
metrics_calculator.py — All retrieval + generation + NLP metrics for Phase 4.
"""

import os
from statistics import mean

import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION A — Retrieval Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_precision_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """Precision@K = |retrieved[:k] ∩ relevant| / k"""
    if k == 0:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return round(hits / k, 4)


def compute_recall_at_k(retrieved_ids: list, relevant_ids: list, k: int) -> float:
    """Recall@K = |retrieved[:k] ∩ relevant| / |relevant|"""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return round(hits / max(len(relevant_ids), 1), 4)


def compute_mrr(retrieved_ids: list, relevant_ids: list) -> float:
    """Mean Reciprocal Rank — 1/(rank of first relevant result)."""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return round(1.0 / (i + 1), 4)
    return 0.0


def compute_hit_rate(retrieved_ids: list, relevant_ids: list) -> float:
    """1.0 if any retrieved doc is relevant, else 0.0."""
    return 1.0 if any(r in set(relevant_ids) for r in retrieved_ids) else 0.0


def find_relevant_chunk_ids(ground_truth: str, chunks: list, threshold: float = 0.6) -> list:
    """
    Finds which chunks are ground-truth relevant by token overlap.
    Falls back to first chunk if none meet threshold.
    """
    relevant = []
    gt_tokens = set(str(ground_truth).lower().split())

    if not gt_tokens:
        if chunks:
            return [chunks[0].metadata.get("chunk_id", "chunk_0")]
        return []

    for chunk in chunks:
        chunk_tokens = set(chunk.page_content.lower().split())
        overlap = len(gt_tokens & chunk_tokens) / len(gt_tokens)
        if overlap >= threshold:
            relevant.append(chunk.metadata.get("chunk_id", ""))

    # Fallback: if nothing matched, use first chunk
    if not relevant and chunks:
        relevant = [chunks[0].metadata.get("chunk_id", "chunk_0")]

    return relevant


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION B — Generation Metrics via RAGAS
# ═══════════════════════════════════════════════════════════════════════════


def compute_ragas_metrics(eval_samples: list, groq_api_key: str) -> dict:
    """
    Compute RAGAS metrics (faithfulness, answer_relevancy, context_precision, context_recall).
    Uses HuggingFace as primary LLM (FREE). Groq as fallback.
    Compatible with RAGAS v0.4.x.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        )
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import EvaluationDataset, SingleTurnSample
        from langchain_huggingface import HuggingFaceEmbeddings
        from config import DEFAULT_EMBEDDING_CONFIG

        # ── Set up LLM for RAGAS (Groq — 8B model handles eval prompts) ──
        ragas_llm = None
        ragas_groq_key = (os.getenv("GROQ_API_KEY2", "").strip()
                          or os.getenv("GROQ_API_KEY1", "").strip()
                          or groq_api_key)

        if ragas_groq_key:
            from langchain_groq import ChatGroq
            ragas_llm = LangchainLLMWrapper(ChatGroq(
                api_key=ragas_groq_key,
                model_name="llama-3.1-8b-instant",
                temperature=0.0,
                max_tokens=2048,
            ))
            print("  🤖 RAGAS eval using Groq (llama-3.1-8b)")

        # ── Embeddings for RAGAS ─────────────────────────────────────────
        ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
            model_name=DEFAULT_EMBEDDING_CONFIG["model_id"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        ))

        # ── Build EvaluationDataset ──────────────────────────────────────
        samples = []
        for s in eval_samples:
            samples.append(SingleTurnSample(
                user_input=s["question"],
                response=s.get("answer", ""),
                retrieved_contexts=s.get("contexts", []),
                reference=str(s.get("ground_truth", "")),
            ))
        dataset = EvaluationDataset(samples=samples)

        # ── Run evaluation ───────────────────────────────────────────────
        scores = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            raise_exceptions=False,
        )
        return {
            "faithfulness": round(float(scores["faithfulness"]), 4),
            "answer_relevancy": round(float(scores["answer_relevancy"]), 4),
            "context_precision": round(float(scores["context_precision"]), 4),
            "context_recall": round(float(scores["context_recall"]), 4),
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}. Using 0.0 scores.")
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  SECTION C — NLP Metrics
# ═══════════════════════════════════════════════════════════════════════════


def compute_rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F-measure between prediction and reference."""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return round(scorer.score(reference, prediction)["rougeL"].fmeasure, 4)


def compute_bleu(prediction: str, reference: str) -> float:
    """Sentence-level BLEU with smoothing."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothie = SmoothingFunction().method1
    ref_tokens = reference.split()
    pred_tokens = prediction.split()
    if not ref_tokens or not pred_tokens:
        return 0.0
    return round(
        sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie), 4
    )


def compute_bertscore_batch(predictions: list, references: list) -> float:
    """
    BERTScore F1 averaged across all pairs.
    Uses distilbert-base-uncased for 8GB RAM compatibility.
    Always call in batch to avoid reloading model.
    """
    if not predictions or not references:
        return 0.0
    try:
        from bert_score import score as bert_score_fn
        _, _, F1 = bert_score_fn(
            predictions, references,
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
        )
        return round(F1.mean().item(), 4)
    except Exception as e:
        print(f"  ⚠️  BERTScore failed: {e}. Returning 0.0.")
        return 0.0


def compute_all_nlp_metrics(predictions: list, references: list) -> dict:
    """Compute averaged ROUGE-L, BLEU, and BERTScore across all pairs."""
    if not predictions or not references:
        return {"rouge_l": 0.0, "bleu": 0.0, "bertscore": 0.0}

    rouge_scores = [compute_rouge_l(p, r) for p, r in zip(predictions, references)]
    bleu_scores = [compute_bleu(p, r) for p, r in zip(predictions, references)]
    bert_f1 = compute_bertscore_batch(predictions, references)

    return {
        "rouge_l": round(mean(rouge_scores), 4),
        "bleu": round(mean(bleu_scores), 4),
        "bertscore": bert_f1,
    }
