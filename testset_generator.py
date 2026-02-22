"""
testset_generator.py — Generates synthetic QA test set from documents.

Custom implementation that bypasses RAGAS's fragile multi-stage transform
pipeline (HeadlinesExtractor crashes with LLMDidNotFinishException).
Uses direct Groq API calls — simple, reliable, and fast.

Output: CSV with 'question' and 'ground_truth' columns (same format RAGAS uses).
"""

import os
import json
import random
import time
from pathlib import Path
import pandas as pd

from config import DEFAULT_EMBEDDING_CONFIG

TESTSET_PATH = "./output/testset.csv"
TESTSET_SIZE = 40
MAX_DOCS_FOR_SAMPLING = 80  # sample diverse chunks


# ─── Function 1: Generate Testset ───────────────────────────────────────────

def generate_testset(docs: list, embedding_model) -> pd.DataFrame:
    """
    Generate synthetic QA test set from documents using direct LLM calls.
    Samples diverse chunks and asks the LLM to generate Q&A pairs.
    """
    from groq import Groq

    # ── API key (all 3 Groq keys as fallback) ────────────────────────────
    groq_key = (os.getenv("GROQ_API_KEY1", "").strip()
                or os.getenv("GROQ_API_KEY", "").strip()
                or os.getenv("GROQ_API_KEY2", "").strip())

    if not groq_key:
        raise ValueError("No Groq API key found. Set GROQ_API_KEY in .env")

    client = Groq(api_key=groq_key)

    # ── Sample diverse chunks ────────────────────────────────────────────
    random.seed(42)
    sample_size = min(MAX_DOCS_FOR_SAMPLING, len(docs))
    sampled = random.sample(docs, sample_size)
    print(f"   📑 Sampled {sample_size} pages from {len(docs)} total")

    # ── Build text chunks (merge small pages, truncate large ones) ───────
    chunks = []
    for doc in sampled:
        text = doc.page_content.strip()
        if len(text) > 200:  # skip near-empty pages
            chunks.append(text[:2000])  # cap at 2000 chars per chunk

    random.shuffle(chunks)
    chunks = chunks[:TESTSET_SIZE + 10]  # a few extras in case some fail
    print(f"📝 Generating {TESTSET_SIZE} QA pairs from {len(chunks)} chunks...")

    # ── Generate Q&A pairs ───────────────────────────────────────────────
    qa_pairs = []
    failed = 0

    for i, chunk in enumerate(chunks):
        if len(qa_pairs) >= TESTSET_SIZE:
            break

        prompt = f"""Based on the following text passage, generate exactly ONE question-answer pair.

The question should be specific, factual, and answerable from the passage.
The answer should be concise and directly supported by the text.

PASSAGE:
{chunk}

Respond in EXACTLY this JSON format (no extra text):
{{"question": "your question here", "ground_truth": "your answer here"}}"""

        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You generate factual QA pairs from text. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=512,
            )

            raw = resp.choices[0].message.content.strip()

            # Parse JSON (handle markdown code blocks)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            pair = json.loads(raw)
            if pair.get("question") and pair.get("ground_truth"):
                qa_pairs.append(pair)
                if (i + 1) % 10 == 0 or len(qa_pairs) == TESTSET_SIZE:
                    print(f"   ✅ {len(qa_pairs)}/{TESTSET_SIZE} pairs generated")

        except Exception as e:
            failed += 1
            if failed <= 3:  # only log first few
                print(f"   ⚠️  Chunk {i+1} failed: {str(e)[:60]}")

        # Small delay to respect rate limits
        time.sleep(0.5)

    # ── Save ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(qa_pairs)
    Path("./output").mkdir(exist_ok=True)
    df.to_csv(TESTSET_PATH, index=False)

    print(f"✅ Generated {len(df)} QA pairs ({failed} failures)")
    print(f"   Saved → {TESTSET_PATH}")

    return df


# ─── Function 2: Load Testset ───────────────────────────────────────────────

def load_testset() -> pd.DataFrame:
    """Load existing testset from CSV."""
    if Path(TESTSET_PATH).exists():
        df = pd.read_csv(TESTSET_PATH)
        print(f"📋 Loaded testset: {len(df)} questions from {TESTSET_PATH}")
        return df
    raise FileNotFoundError(
        f"No testset found at {TESTSET_PATH}. Run generate_testset() first."
    )


# ─── Function 3: Validate Testset ───────────────────────────────────────────

def validate_testset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop empty/duplicate rows and return cleaned testset."""
    orig = len(df)

    # Normalize column names from RAGAS format
    rename_map = {}
    if "user_input" in df.columns and "question" not in df.columns:
        rename_map["user_input"] = "question"
    if "reference" in df.columns and "ground_truth" not in df.columns:
        rename_map["reference"] = "ground_truth"
    if rename_map:
        df = df.rename(columns=rename_map)

    # Drop rows with empty question or ground_truth
    df = df.dropna(subset=["question"])
    if "ground_truth" in df.columns:
        df = df.dropna(subset=["ground_truth"])

    # Drop duplicate questions
    df = df.drop_duplicates(subset=["question"])
    df = df.reset_index(drop=True)

    dropped = orig - len(df)
    if dropped > 0:
        print(f"   Dropped {dropped} invalid/duplicate rows")
    print(f"✅ Testset validated: {len(df)} usable questions")
    return df
