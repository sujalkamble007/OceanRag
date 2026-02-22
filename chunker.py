"""
chunker.py — Text splitting strategies using LangChain RecursiveCharacterTextSplitter.
"""

import os
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(docs: list, chunk_config: dict, output_dir: str) -> list:
    """
    Splits documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        docs: List of LangChain Document objects.
        chunk_config: Dict with keys 'name', 'size', 'overlap'.
        output_dir: Directory to save chunk summary CSV.

    Returns:
        List of chunk Document objects with enriched metadata.
    """
    strategy = chunk_config["name"]
    chunk_size = chunk_config["size"]
    chunk_overlap = chunk_config["overlap"]

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_documents(docs)

    # Enrich metadata for each chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{strategy}_chunk_{i}"
        chunk.metadata["chunk_strategy"] = strategy
        chunk.metadata["chunk_size"] = chunk_size
        chunk.metadata["chunk_overlap"] = chunk_overlap
        chunk.metadata["char_count"] = len(chunk.page_content)

    _save_chunk_summary(chunks, strategy, output_dir)

    print(f"✅ Created {len(chunks)} chunks.")
    print(f"   Strategy: {strategy} | Size: {chunk_size} | Overlap: {chunk_overlap}")
    print(f"📊 Chunk summary saved → {output_dir}/chunks_{strategy}.csv")

    return chunks


def _save_chunk_summary(chunks: list, strategy: str, output_dir: str):
    """Saves a CSV summary of all chunks for the given strategy."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    records = []
    for chunk in chunks:
        records.append({
            "chunk_id": chunk.metadata.get("chunk_id"),
            "filename": chunk.metadata.get("filename"),
            "page_number": chunk.metadata.get("page_number"),
            "char_count": chunk.metadata.get("char_count"),
            "preview": chunk.page_content[:100].replace("\n", " "),
        })

    df = pd.DataFrame(records)
    filepath = os.path.join(output_dir, f"chunks_{strategy}.csv")
    df.to_csv(filepath, index=False)


def compare_chunk_strategies(docs: list, configs: list, output_dir: str) -> dict:
    """
    Runs chunk_documents() for every config and prints a comparison table.

    Args:
        docs: List of LangChain Document objects.
        configs: List of chunk config dicts.
        output_dir: Directory to save chunk summary CSVs.

    Returns:
        Dict mapping strategy name → list of chunk Documents.
    """
    results = {}

    print("\n📊 Chunk Strategy Comparison:")
    print(f"{'Strategy':<20} {'Total Chunks':>14} {'Avg Chars':>10}")
    print("-" * 46)

    for config in configs:
        chunks = chunk_documents(docs, config, output_dir)
        avg_chars = sum(c.metadata["char_count"] for c in chunks) / max(len(chunks), 1)
        results[config["name"]] = chunks
        print(f"{config['name']:<20} {len(chunks):>14} {avg_chars:>10.1f}")

    print()
    return results
