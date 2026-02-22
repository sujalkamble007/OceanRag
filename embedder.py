"""
embedder.py — HuggingFace embedding model loading and chunk embedding.
"""

from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(embedding_config: dict):
    """
    Loads a HuggingFace embedding model.

    Args:
        embedding_config: Dict with keys 'name', 'model_id', 'vector_size'.

    Returns:
        HuggingFaceEmbeddings object.
    """
    model_name = embedding_config["name"]
    model_id = embedding_config["model_id"]

    print(f"⏳ Loading {model_name} model ({model_id})...")
    print("   (First run will download the model — this may take a minute)")

    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"✅ Embedding model '{model_name}' loaded.")
    return embeddings


def embed_chunks(chunks: list, embedding_model, batch_size: int = 500) -> list:
    """
    Embeds a list of LangChain Document chunks using the given embedding model.
    Uses batch embedding for significantly faster processing.

    Args:
        chunks: List of LangChain Document objects.
        embedding_model: HuggingFaceEmbeddings object.
        batch_size: Number of chunks to embed at once (default 500).

    Returns:
        List of dicts, each containing:
            - "vector": the embedding (list of floats)
            - "chunk_id": from chunk metadata
            - "page_content": chunk text
            - "metadata": full chunk metadata dict
    """
    embedded = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(chunks), batch_size), desc="Embedding chunks", total=total_batches):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        vectors = embedding_model.embed_documents(texts)

        for chunk, vector in zip(batch, vectors):
            embedded.append({
                "vector": vector,
                "chunk_id": chunk.metadata["chunk_id"],
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            })

    print(f"✅ Embedded {len(embedded)} chunks.")
    return embedded
