"""
resume_step7.py — ONLY does Step 7: batch insert chunk metadata to PostgreSQL.
Reads from the saved CSV — no PDF loading, no chunking, no embedding.
Uses a SINGLE database connection for ALL inserts.
"""

import csv
import uuid
from config import DEFAULT_EMBEDDING_CONFIG
from database import get_engine, init_db, documents_table, chunks_table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy import select

CSV_PATH = "./output/chunks_fixed_512.csv"
CHUNK_STRATEGY = "fixed_512"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 0


def run_step7():
    # Ensure tables exist (no reset)
    init_db(reset=False)

    # Read all chunk data from CSV
    print(f"📄 Reading chunks from {CSV_PATH}...")
    with open(CSV_PATH, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    print(f"  ✅ Read {len(rows)} chunks from CSV")

    engine = get_engine()
    with engine.connect() as conn:
        # 1) Insert documents (unique filenames from CSV)
        filenames = sorted(set(r["filename"] for r in rows))
        print(f"\n📝 Inserting {len(filenames)} documents...")
        for fname in filenames:
            stmt = pg_insert(documents_table).values(
                filename=fname, filepath="", total_pages=0
            ).on_conflict_do_nothing(index_elements=["filename"])
            conn.execute(stmt)
        conn.commit()

        # Fetch doc ID mapping
        doc_rows = conn.execute(
            select(documents_table.c.id, documents_table.c.filename)
        ).fetchall()
        doc_id_map = {r.filename: r.id for r in doc_rows}
        print(f"  ✅ {len(doc_id_map)} documents ready")

        # 2) Batch insert chunks
        print(f"\n📦 Inserting {len(rows)} chunks in batches of 500...")
        BATCH = 500
        total = len(rows)

        for i in range(0, total, BATCH):
            batch = rows[i : i + BATCH]
            values = []
            for r in batch:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, r["chunk_id"]))
                values.append({
                    "chunk_id": r["chunk_id"],
                    "document_id": doc_id_map.get(r["filename"]),
                    "filename": r["filename"],
                    "page_number": int(r["page_number"]),
                    "chunk_strategy": CHUNK_STRATEGY,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "char_count": int(r["char_count"]),
                    "content_preview": r["preview"][:200],
                    "qdrant_point_id": point_id,
                    "embedding_model": DEFAULT_EMBEDDING_CONFIG["name"],
                })
            stmt = pg_insert(chunks_table).values(values).on_conflict_do_nothing(
                index_elements=["chunk_id"]
            )
            conn.execute(stmt)
            conn.commit()
            done = min(i + BATCH, total)
            print(f"  Progress: {done}/{total} ({100*done//total}%)", end="\r")

        print(f"\n  ✅ {total} chunks saved to PostgreSQL!")

    print("\n🎉 Step 7 complete!")


if __name__ == "__main__":
    run_step7()
