"""
database.py — PostgreSQL connection + table setup using SQLAlchemy.
Uses a single shared engine for all operations (critical for remote DBs like Neon).
"""

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text,
    Float, DateTime, ForeignKey, select, JSON
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD


# ─── SQLAlchemy Metadata ────────────────────────────────────────────────────
metadata = MetaData()

documents_table = Table(
    "documents", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("filename", String(255), nullable=False, unique=True),
    Column("filepath", Text),
    Column("total_pages", Integer),
    Column("uploaded_at", DateTime, server_default=func.now()),
)

chunks_table = Table(
    "chunks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chunk_id", String(255), unique=True, nullable=False),
    Column("document_id", Integer, ForeignKey("documents.id")),
    Column("filename", String(255)),
    Column("page_number", Integer),
    Column("chunk_strategy", String(100)),
    Column("chunk_size", Integer),
    Column("chunk_overlap", Integer),
    Column("char_count", Integer),
    Column("content_preview", Text),
    Column("qdrant_point_id", String(255)),
    Column("embedding_model", String(100)),
    Column("created_at", DateTime, server_default=func.now()),
)

experiments_table = Table(
    "experiments", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("chunk_strategy", String(100)),
    Column("embedding_model", String(100)),
    Column("retriever_type", String(100)),
    Column("llm_name", String(100)),
    Column("top_k", Integer),
    Column("precision_at_k", Float),
    Column("recall_at_k", Float),
    Column("mrr", Float),
    Column("hit_rate", Float),
    Column("faithfulness", Float),
    Column("answer_relevancy", Float),
    Column("rouge_l", Float),
    Column("latency_seconds", Float),
    Column("cost_per_query", Float),
    Column("run_at", DateTime, server_default=func.now()),
)

retrieval_logs_table = Table(
    "retrieval_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_text", Text, nullable=False),
    Column("retriever_type", String(100)),
    Column("embedding_model", String(100)),
    Column("chunk_strategy", String(100)),
    Column("top_k", Integer),
    Column("results", JSON),
    Column("latency_seconds", Float),
    Column("run_at", DateTime, server_default=func.now()),
)


# ─── Shared Engine (singleton) ──────────────────────────────────────────────
_engine = None


def get_engine():
    """Creates and returns a shared SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is not None:
        return _engine

    connection_string = (
        f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    # Use SSL for remote hosts (e.g. Neon), skip for localhost
    if POSTGRES_HOST not in ("localhost", "127.0.0.1"):
        connection_string += "?sslmode=require"
    _engine = create_engine(connection_string, echo=False, pool_pre_ping=True)
    return _engine


def init_db(reset=False):
    """Creates all tables if they don't exist. If reset=True, drops and recreates."""
    engine = get_engine()
    if reset:
        metadata.drop_all(engine)
        print("🗑️  Dropped existing tables")
    metadata.create_all(engine)
    print("✅ PostgreSQL tables ready (documents, chunks, experiments, retrieval_logs)")
    return engine


def insert_document(filename: str, filepath: str, total_pages: int) -> int:
    """
    Inserts a document record and returns its id.
    Uses ON CONFLICT to handle re-runs safely — returns existing id if already present.
    """
    engine = get_engine()
    with engine.connect() as conn:
        # Try to insert, on conflict (duplicate filename) do nothing
        stmt = pg_insert(documents_table).values(
            filename=filename,
            filepath=filepath,
            total_pages=total_pages,
        ).on_conflict_do_nothing(index_elements=["filename"])
        conn.execute(stmt)
        conn.commit()

        # Fetch the id (whether just inserted or already existing)
        result = conn.execute(
            select(documents_table.c.id).where(documents_table.c.filename == filename)
        ).scalar()
        return result


def insert_chunk(chunk_data: dict) -> None:
    """
    Inserts a record into the chunks table.
    Uses ON CONFLICT DO NOTHING to handle re-runs safely.
    """
    engine = get_engine()
    with engine.connect() as conn:
        stmt = pg_insert(chunks_table).values(
            chunk_id=chunk_data["chunk_id"],
            document_id=chunk_data.get("document_id"),
            filename=chunk_data.get("filename"),
            page_number=chunk_data.get("page_number"),
            chunk_strategy=chunk_data.get("chunk_strategy"),
            chunk_size=chunk_data.get("chunk_size"),
            chunk_overlap=chunk_data.get("chunk_overlap"),
            char_count=chunk_data.get("char_count"),
            content_preview=chunk_data.get("content_preview"),
            qdrant_point_id=chunk_data.get("qdrant_point_id"),
            embedding_model=chunk_data.get("embedding_model"),
        ).on_conflict_do_nothing(index_elements=["chunk_id"])
        conn.execute(stmt)
        conn.commit()


def insert_chunks_batch(chunk_data_list: list, batch_size: int = 500) -> None:
    """
    Inserts many chunk records in batches using a single connection.
    Uses ON CONFLICT DO NOTHING to handle re-runs safely.
    Much faster than insert_chunk() for large datasets.
    """
    engine = get_engine()
    with engine.connect() as conn:
        for i in range(0, len(chunk_data_list), batch_size):
            batch = chunk_data_list[i : i + batch_size]
            stmt = pg_insert(chunks_table).values(batch).on_conflict_do_nothing(
                index_elements=["chunk_id"]
            )
            conn.execute(stmt)
            conn.commit()


def get_chunk_stats() -> dict:
    """Returns summary: total chunks, chunks per strategy, chunks per document."""
    engine = get_engine()
    with engine.connect() as conn:
        # Total chunks
        total = conn.execute(
            chunks_table.select().with_only_columns(func.count())
        ).scalar()

        # Chunks per strategy
        strategy_rows = conn.execute(
            select(chunks_table.c.chunk_strategy, func.count().label("count"))
            .group_by(chunks_table.c.chunk_strategy)
        ).fetchall()
        per_strategy = {row[0]: row[1] for row in strategy_rows}

        # Chunks per document (filename)
        doc_rows = conn.execute(
            select(chunks_table.c.filename, func.count().label("count"))
            .group_by(chunks_table.c.filename)
        ).fetchall()
        per_document = {row[0]: row[1] for row in doc_rows}

    return {
        "total_chunks": total,
        "per_strategy": per_strategy,
        "per_document": per_document,
    }


def insert_experiment(experiment_data: dict) -> None:
    """Inserts a row into the experiments table (used in Phase 4)."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(
            experiments_table.insert().values(**experiment_data)
        )
        conn.commit()


def insert_retrieval_log(log_data: dict) -> int:
    """
    Insert a retrieval run record into retrieval_logs.
    log_data keys: query_text, retriever_type, embedding_model,
                   chunk_strategy, top_k, results (list of dicts), latency_seconds
    Returns new log id.
    """
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            retrieval_logs_table.insert().values(**log_data).returning(retrieval_logs_table.c.id)
        )
        conn.commit()
        return result.scalar()
