"""
retriever.py — Phase 2: All 3 retrieval strategies for OceanRAG.

Strategies:
  1. similarity_search  — Pure vector cosine similarity via Qdrant
  2. mmr_search          — Max Marginal Relevance (diversity-aware)
  3. hybrid_search       — BM25 keyword + vector score fusion
"""

import time
import numpy as np
from qdrant_client import QdrantClient
from langchain_community.retrievers import BM25Retriever


# ─── Helper ─────────────────────────────────────────────────────────────────

def _cosine_similarity(vec_a: list, vec_b: list) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


# ─── Function 1: Embed Query ────────────────────────────────────────────────

def embed_query(query_text: str, embedding_model) -> list:
    """Convert a text query into a vector embedding."""
    vector = embedding_model.embed_query(query_text)
    print(f"🔢 Query embedded — vector size: {len(vector)}")
    return vector


# ─── Function 2: Similarity Search ──────────────────────────────────────────

def similarity_search(client: QdrantClient, collection_name: str,
                      query_vector: list, query_text: str, k: int = 5) -> dict:
    """
    Pure vector similarity search — finds the k most similar vectors in Qdrant.
    Uses query_points() API (qdrant-client v1.12+).
    """
    start = time.time()

    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=k,
    )

    end = time.time()

    results = []
    for i, hit in enumerate(response.points):
        results.append({
            "rank": i + 1,
            "score": float(hit.score),
            "chunk_id": hit.payload.get("chunk_id", ""),
            "filename": hit.payload.get("filename", ""),
            "page_number": hit.payload.get("page_number", 0),
            "chunk_strategy": hit.payload.get("chunk_strategy", ""),
            "page_content": hit.payload.get("content_preview", ""),
            "retriever_type": "similarity",
        })

    return {
        "retriever_type": "similarity",
        "query": query_text,
        "top_k": k,
        "latency_seconds": round(end - start, 4),
        "results": results,
    }


# ─── Function 3: MMR Search ─────────────────────────────────────────────────

def mmr_search(client: QdrantClient, collection_name: str,
               query_vector: list, query_text: str,
               k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.5) -> dict:
    """
    Max Marginal Relevance — retrieves diverse results by penalizing
    chunks too similar to already-selected chunks.
    """
    start = time.time()

    # 1) Fetch fetch_k candidates from Qdrant
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=fetch_k,
        with_payload=True,
    )

    if not response.points:
        return {
            "retriever_type": "mmr",
            "query": query_text,
            "top_k": k,
            "latency_seconds": round(time.time() - start, 4),
            "results": [],
        }

    # 2) Retrieve vectors for candidates
    candidate_ids = [p.id for p in response.points]
    points_with_vectors = client.retrieve(
        collection_name=collection_name,
        ids=candidate_ids,
        with_vectors=True,
        with_payload=True,
    )

    # Build lookup: point_id -> (payload, vector, original_score)
    score_map = {p.id: p.score for p in response.points}
    candidates = []
    for p in points_with_vectors:
        vec = p.vector
        # Handle dict-style vectors (named vectors)
        if isinstance(vec, dict):
            vec = list(vec.values())[0]
        candidates.append({
            "id": p.id,
            "payload": p.payload,
            "vector": vec,
            "score": score_map.get(p.id, 0.0),
        })

    # 3) MMR selection algorithm
    selected = []
    remaining = list(candidates)

    for _ in range(min(k, len(remaining))):
        best_idx = -1
        best_mmr = float("-inf")

        for idx, cand in enumerate(remaining):
            sim_to_query = _cosine_similarity(cand["vector"], query_vector)

            if selected:
                max_sim_to_selected = max(
                    _cosine_similarity(cand["vector"], s["vector"])
                    for s in selected
                )
            else:
                max_sim_to_selected = 0.0

            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx >= 0:
            selected.append(remaining.pop(best_idx))

    end = time.time()

    # 4) Format results
    results = []
    for i, s in enumerate(selected):
        results.append({
            "rank": i + 1,
            "score": float(s["score"]),
            "chunk_id": s["payload"].get("chunk_id", ""),
            "filename": s["payload"].get("filename", ""),
            "page_number": s["payload"].get("page_number", 0),
            "chunk_strategy": s["payload"].get("chunk_strategy", ""),
            "page_content": s["payload"].get("content_preview", ""),
            "retriever_type": "mmr",
        })

    return {
        "retriever_type": "mmr",
        "query": query_text,
        "top_k": k,
        "latency_seconds": round(end - start, 4),
        "results": results,
    }


# ─── Function 4: Hybrid Search ──────────────────────────────────────────────

def hybrid_search(client: QdrantClient, collection_name: str,
                  query_vector: list, query_text: str,
                  chunks: list, k: int = 5) -> dict:
    """
    Hybrid search — BM25 keyword (sparse) + Qdrant vector (dense).
    Scores normalized to [0, 1] and combined with equal weighting (0.5 each).
    """
    start = time.time()

    # 1) Vector search — get top 2*k results
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=2 * k,
    )

    vector_scores = {}
    max_vscore = 0.0
    for hit in response.points:
        cid = hit.payload.get("chunk_id", "")
        vector_scores[cid] = float(hit.score)
        if hit.score > max_vscore:
            max_vscore = float(hit.score)

    # Normalize vector scores to [0, 1]
    if max_vscore > 0:
        vector_scores = {cid: s / max_vscore for cid, s in vector_scores.items()}

    # 2) BM25 search — keyword search over chunks
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2 * k
    bm25_docs = bm25_retriever.invoke(query_text)

    bm25_scores = {}
    total_bm25 = len(bm25_docs)
    for rank, doc in enumerate(bm25_docs):
        cid = doc.metadata.get("chunk_id", "")
        # Higher rank → higher score, normalized to [0, 1]
        bm25_scores[cid] = (total_bm25 - rank) / total_bm25 if total_bm25 > 0 else 0

    # 3) Combine scores
    all_chunk_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
    combined = {}
    for cid in all_chunk_ids:
        vs = vector_scores.get(cid, 0.0)
        bs = bm25_scores.get(cid, 0.0)
        combined[cid] = 0.5 * vs + 0.5 * bs

    # 4) Sort and take top k
    top_ids = sorted(combined.keys(), key=lambda c: combined[c], reverse=True)[:k]

    # 5) Build results — use payload from Qdrant where available, fall back to chunk metadata
    # Build a payload lookup from vector results
    payload_map = {}
    for hit in response.points:
        cid = hit.payload.get("chunk_id", "")
        payload_map[cid] = hit.payload

    # Also build lookup from BM25 docs
    bm25_meta_map = {}
    for doc in bm25_docs:
        cid = doc.metadata.get("chunk_id", "")
        bm25_meta_map[cid] = {
            "filename": doc.metadata.get("filename", ""),
            "page_number": doc.metadata.get("page_number", 0),
            "chunk_strategy": doc.metadata.get("chunk_strategy", ""),
            "content_preview": doc.page_content[:200],
        }

    end = time.time()

    results = []
    for i, cid in enumerate(top_ids):
        # Prefer Qdrant payload, fall back to BM25 metadata
        payload = payload_map.get(cid, bm25_meta_map.get(cid, {}))
        results.append({
            "rank": i + 1,
            "score": round(combined[cid], 4),
            "vector_score": round(vector_scores.get(cid, 0.0), 4),
            "bm25_score": round(bm25_scores.get(cid, 0.0), 4),
            "chunk_id": cid,
            "filename": payload.get("filename", ""),
            "page_number": payload.get("page_number", 0),
            "chunk_strategy": payload.get("chunk_strategy", ""),
            "page_content": payload.get("content_preview", ""),
            "retriever_type": "hybrid",
        })

    return {
        "retriever_type": "hybrid",
        "query": query_text,
        "top_k": k,
        "latency_seconds": round(end - start, 4),
        "results": results,
    }


# ─── Function 5: Run All Retrievers ─────────────────────────────────────────

def run_all_retrievers(client: QdrantClient, collection_name: str,
                       query_text: str, embedding_model,
                       chunks: list, k: int = 5) -> dict:
    """Run all 3 retrievers on the same query and return results for comparison."""
    query_vector = embed_query(query_text, embedding_model)

    sim_results = similarity_search(client, collection_name, query_vector, query_text, k)
    mmr_results = mmr_search(client, collection_name, query_vector, query_text, k)
    hyb_results = hybrid_search(client, collection_name, query_vector, query_text, chunks, k)

    return {
        "query": query_text,
        "k": k,
        "similarity": sim_results,
        "mmr": mmr_results,
        "hybrid": hyb_results,
    }


# ─── Function 6: Print Retrieval Results ────────────────────────────────────

def print_retrieval_results(results: dict):
    """Print formatted comparison of all 3 retrievers."""
    print()
    print("═" * 60)
    print(f"  RETRIEVAL RESULTS")
    print(f"  Query: \"{results['query']}\"")
    print(f"  Top-K: {results['k']}")
    print("═" * 60)

    for rtype in ("similarity", "mmr", "hybrid"):
        data = results[rtype]
        label = rtype.upper() + " SEARCH"
        latency = data["latency_seconds"]

        print(f"\n── {label} (latency: {latency}s) {'─' * (40 - len(label))}")

        for r in data["results"]:
            score_str = f"Score: {r['score']:.4f}"
            if rtype == "hybrid":
                score_str += f" (vec: {r.get('vector_score', 0):.2f} | bm25: {r.get('bm25_score', 0):.2f})"
            print(f"  [{r['rank']}] {score_str} | {r['filename']} | Page {r['page_number']}")
            preview = r.get("page_content", "")[:80]
            if preview:
                print(f"      Preview: {preview}...")

    print()
