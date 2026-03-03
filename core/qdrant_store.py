"""
ingestion/qdrant_store.py — Qdrant Cloud operations: collection management, upsert, search.
"""

import uuid
import time
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter
)
from core.config import QDRANT_URL, QDRANT_API_KEY


def get_qdrant_client() -> QdrantClient:
    """Creates and returns an authenticated Qdrant Cloud client."""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)
    try:
        client.get_collections()
        print("✅ Connected to Qdrant Cloud")
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant Cloud: {e}")
        raise
    return client


def create_collection(client: QdrantClient, collection_name: str, vector_size: int):
    """Creates a Qdrant collection if it doesn't already exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name in collections:
        print(f"📦 Collection '{collection_name}' already exists, skipping creation.")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✅ Created collection '{collection_name}' (vector_size={vector_size}).")


def upsert_chunks(client: QdrantClient, collection_name: str, embedded_chunks: list) -> list:
    """
    Upserts embedded chunks into Qdrant in small batches with retry logic.
    Returns list of point UUID strings.
    """
    BATCH_SIZE = 20
    MAX_RETRIES = 3
    point_ids = []
    points = []

    for item in embedded_chunks:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, item["chunk_id"]))
        point_ids.append(point_id)
        payload = {
            "chunk_id": item["chunk_id"],
            "filename": item["metadata"].get("filename", ""),
            "page_number": item["metadata"].get("page_number", 0),
            "chunk_strategy": item["metadata"].get("chunk_strategy", ""),
            "chunk_size": item["metadata"].get("chunk_size", 0),
            "char_count": item["metadata"].get("char_count", 0),
            "content_preview": item["page_content"][:200],
        }
        points.append(PointStruct(id=point_id, vector=item["vector"], payload=payload))

    total_batches = (len(points) + BATCH_SIZE - 1) // BATCH_SIZE
    for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Uploading to Qdrant", total=total_batches):
        batch = points[i : i + BATCH_SIZE]
        for attempt in range(MAX_RETRIES):
            try:
                client.upsert(collection_name=collection_name, points=batch)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"\n⚠️  Retry {attempt+1}/{MAX_RETRIES} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    print(f"\n❌ Failed after {MAX_RETRIES} retries: {e}")
                    raise

    print(f"✅ Upserted {len(points)} vectors to Qdrant Cloud.")
    return point_ids


def search_similar(client: QdrantClient, collection_name: str, query_vector: list, k: int = 5) -> list:
    """Searches for the most similar vectors in the collection."""
    response = client.query_points(collection_name=collection_name, query=query_vector, limit=k)
    formatted = []
    for i, hit in enumerate(response.points, 1):
        result = {
            "score": hit.score,
            "payload": hit.payload,
            "page_content": hit.payload.get("content_preview", ""),
        }
        formatted.append(result)
        print(f"  [{i}] Score: {hit.score:.4f} | "
              f"{hit.payload.get('filename', 'N/A')} | "
              f"Page {hit.payload.get('page_number', '?')}")
        preview = hit.payload.get("content_preview", "")[:80]
        print(f"      Preview: {preview}...")
    return formatted


def get_collection_info(client: QdrantClient, collection_name: str) -> dict:
    """Returns collection stats: points count, status."""
    try:
        info = client.get_collection(collection_name)
        stats = {"vectors_count": info.points_count, "status": str(info.status)}
        print(f"📦 Collection '{collection_name}' — {stats['vectors_count']} vectors ({stats['status']})")
        return stats
    except Exception:
        print(f"📦 Collection '{collection_name}' — does not exist yet")
        return {"vectors_count": 0, "status": "not_found"}


def delete_collection(client: QdrantClient, collection_name: str):
    """Deletes the entire collection after user confirmation."""
    confirm = input(f"⚠️  Delete collection '{collection_name}'? (yes/no): ").strip().lower()
    if confirm == "yes":
        client.delete_collection(collection_name)
        print(f"🗑️  Collection '{collection_name}' deleted.")
    else:
        print("❌ Deletion cancelled.")
