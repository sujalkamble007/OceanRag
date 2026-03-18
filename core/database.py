"""
core/database.py — PostgreSQL connection + table setup using SQLAlchemy.
Uses a single shared engine for all operations (critical for remote DBs like Neon).
"""

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text,
    Float, DateTime, ForeignKey, select, JSON
)
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from core.config import DATABASE_URL


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
    Column("phase", String(10)),
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
    Column("bleu", Float),
    Column("bertscore", Float),
    Column("latency_seconds", Float),
    Column("cost_per_query", Float),
    Column("num_questions", Integer),
    Column("run_at", DateTime, server_default=func.now()),
)

eval_results_table = Table(
    "eval_results", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", Integer),
    Column("phase", String(10)),
    Column("chunk_strategy", String(100)),
    Column("embedding_model", String(100)),
    Column("retriever_type", String(100)),
    Column("llm_name", String(100)),
    Column("top_k", Integer),
    Column("question", Text),
    Column("ground_truth", Text),
    Column("generated_answer", Text),
    Column("retrieved_chunk_ids", JSON),
    Column("relevant_chunk_ids", JSON),
    Column("precision_at_k", Float),
    Column("recall_at_k", Float),
    Column("mrr", Float),
    Column("hit_rate", Float),
    Column("rouge_l", Float),
    Column("bleu", Float),
    Column("bertscore", Float),
    Column("faithfulness", Float),
    Column("retrieval_latency_ms", Float),
    Column("generation_latency_ms", Float),
    Column("cost_usd", Float),
    Column("created_at", DateTime, server_default=func.now()),
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

qa_logs_table = Table(
    "qa_logs", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("session_id", String(100)),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("query_text", Text, nullable=False),
    Column("retriever_type", String(100)),
    Column("embedding_model", String(100)),
    Column("chunk_strategy", String(100)),
    Column("top_k", Integer),
    Column("llm_name", String(100)),
    Column("llm_model_id", String(200)),
    Column("context_chunks", JSON),
    Column("prompt_text", Text),
    Column("answer_text", Text),
    Column("sources", JSON),
    Column("input_tokens", Integer),
    Column("output_tokens", Integer),
    Column("latency_seconds", Float),
    Column("cost_usd", Float),
    Column("run_at", DateTime, server_default=func.now()),
)

model_comparisons_table = Table(
    "model_comparisons", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("query_text", Text, nullable=False),
    Column("retriever_type", String(100)),
    Column("top_k", Integer),
    Column("results", JSON),
    Column("run_at", DateTime, server_default=func.now()),
)


# ─── Shared Engine (singleton) ──────────────────────────────────────────────
_engine = None


def get_engine():
    """Creates and returns a shared SQLAlchemy engine (singleton)."""
    global _engine
    if _engine is not None:
        return _engine

    if not DATABASE_URL:
        raise EnvironmentError("DATABASE_URL is not set.")
    
    # Neon Serverless Postgres drops idle connections after ~60s.
    # pool_pre_ping checks the connection before use.
    # pool_recycle=55 ensures we recycle before Neon drops them.
    # pool_timeout=10 means requests fail fast (not hang) if no connection available.
    # connect_args sets psycopg2-level socket timeout so auth never hangs forever.
    _engine = create_engine(
        DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
        pool_recycle=55,          # Recycle before Neon's ~60s idle timeout
        pool_timeout=10,          # Wait max 10s for a connection from pool
        pool_use_lifo=True,       # Use LIFO to reduce total open connections
        connect_args={"connect_timeout": 15},  # psycopg2 socket-level timeout
    )
    return _engine


def init_db(reset=False):
    """Creates all tables if they don't exist. If reset=True, drops and recreates."""
    engine = get_engine()
    if reset:
        metadata.drop_all(engine)
        print("🗑️  Dropped existing tables")
    metadata.create_all(engine)
    print("✅ PostgreSQL tables ready (documents, chunks, experiments, retrieval_logs, qa_logs, model_comparisons)")
    return engine


def insert_document(filename: str, filepath: str, total_pages: int) -> int:
    """
    Inserts a document record and returns its id.
    Uses ON CONFLICT to handle re-runs safely — returns existing id if already present.
    """
    engine = get_engine()
    with engine.connect() as conn:
        stmt = pg_insert(documents_table).values(
            filename=filename,
            filepath=filepath,
            total_pages=total_pages,
        ).on_conflict_do_nothing(index_elements=["filename"])
        conn.execute(stmt)
        conn.commit()

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
        total = conn.execute(
            chunks_table.select().with_only_columns(func.count())
        ).scalar()

        strategy_rows = conn.execute(
            select(chunks_table.c.chunk_strategy, func.count().label("count"))
            .group_by(chunks_table.c.chunk_strategy)
        ).fetchall()
        per_strategy = {row[0]: row[1] for row in strategy_rows}

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


def insert_experiment(experiment_data: dict) -> int:
    """Inserts a row into the experiments table. Returns new experiment id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            experiments_table.insert().values(**experiment_data).returning(experiments_table.c.id)
        )
        conn.commit()
        return result.scalar()


def insert_eval_result(result_data: dict) -> int:
    """Insert a single per-question eval result. Returns new id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            eval_results_table.insert().values(**result_data).returning(eval_results_table.c.id)
        )
        conn.commit()
        return result.scalar()


def insert_eval_results_batch(results: list, batch_size: int = 100) -> None:
    """Batch insert eval results for efficiency."""
    engine = get_engine()
    with engine.connect() as conn:
        for i in range(0, len(results), batch_size):
            batch = results[i : i + batch_size]
            conn.execute(eval_results_table.insert(), batch)
            conn.commit()


def get_eval_results(phase: str = None, limit: int = 500) -> list:
    """Get eval results, optionally filtered by phase."""
    engine = get_engine()
    with engine.connect() as conn:
        query = select(eval_results_table).order_by(eval_results_table.c.created_at.desc())
        if phase:
            query = query.where(eval_results_table.c.phase == phase)
        query = query.limit(limit)
        rows = conn.execute(query).fetchall()

    columns = [c.name for c in eval_results_table.columns]
    return [dict(zip(columns, row)) for row in rows]


def insert_retrieval_log(log_data: dict) -> int:
    """Insert a retrieval run record into retrieval_logs. Returns new log id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            retrieval_logs_table.insert().values(**log_data).returning(retrieval_logs_table.c.id)
        )
        conn.commit()
        return result.scalar()


def insert_qa_log(qa_data: dict) -> int:
    """Insert a Q&A record into qa_logs. Returns new record id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            qa_logs_table.insert().values(**qa_data).returning(qa_logs_table.c.id)
        )
        conn.commit()
        return result.scalar()


def insert_model_comparison(comparison_data: dict) -> int:
    """Insert a multi-model comparison record. Returns new record id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            model_comparisons_table.insert().values(**comparison_data).returning(
                model_comparisons_table.c.id
            )
        )
        conn.commit()
        return result.scalar()


def get_qa_history(limit: int = 20) -> list:
    """Returns last `limit` rows from qa_logs ordered by run_at DESC."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                qa_logs_table.c.id,
                qa_logs_table.c.query_text,
                qa_logs_table.c.llm_name,
                qa_logs_table.c.answer_text,
                qa_logs_table.c.latency_seconds,
                qa_logs_table.c.cost_usd,
                qa_logs_table.c.run_at,
            )
            .order_by(qa_logs_table.c.run_at.desc())
            .limit(limit)
        ).fetchall()

    return [
        {
            "id": r.id,
            "query_text": r.query_text,
            "llm_name": r.llm_name,
            "answer_text": r.answer_text,
            "latency_seconds": r.latency_seconds,
            "cost_usd": r.cost_usd,
            "run_at": str(r.run_at) if r.run_at else "",
        }
        for r in rows
    ]


def get_all_experiments() -> list:
    """Returns all rows from experiments table as list of dicts, ordered by run_at DESC."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(experiments_table).order_by(experiments_table.c.run_at.desc())
        ).fetchall()

    columns = [c.name for c in experiments_table.columns]
    return [dict(zip(columns, row)) for row in rows]


def get_best_config() -> dict:
    """
    Returns single experiment row with highest composite score.
    composite = (precision_at_k + recall_at_k + faithfulness + answer_relevancy) / 4
    """
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(experiments_table).order_by(experiments_table.c.run_at.desc())
        ).fetchall()

    if not rows:
        return {}

    columns = [c.name for c in experiments_table.columns]
    experiments = [dict(zip(columns, row)) for row in rows]

    def composite(exp):
        return (
            (exp.get("precision_at_k") or 0) +
            (exp.get("recall_at_k") or 0) +
            (exp.get("faithfulness") or 0) +
            (exp.get("answer_relevancy") or 0)
        ) / 4

    return max(experiments, key=composite)


# ─── Phase 5 Tables ─────────────────────────────────────────────────────────

users_table = Table(
    "users", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String(100), unique=True, nullable=False),
    Column("email", String(200), unique=True, nullable=False),
    Column("hashed_password", Text, nullable=False),
    Column("role", String(50), server_default="student"),
    Column("is_active", String(10), server_default="true"),
    Column("created_at", DateTime, server_default=func.now()),
)

feedback_table = Table(
    "feedback", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("qa_log_id", Integer, ForeignKey("qa_logs.id")),
    Column("user_id", Integer, ForeignKey("users.id")),
    Column("rating", Integer),        # 1=thumbs up, -1=thumbs down
    Column("comment", Text),
    Column("created_at", DateTime, server_default=func.now()),
)


# ─── Phase 5 DB Functions ────────────────────────────────────────────────────

def create_user(username: str, email: str, hashed_password: str, role: str = "student") -> dict:
    """Insert a new user. Returns the created user dict."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            users_table.insert().values(
                username=username,
                email=email,
                hashed_password=hashed_password,
                role=role,
                is_active="true",
            ).returning(*users_table.c)
        )
        conn.commit()
        row = result.fetchone()
        return dict(zip([c.name for c in users_table.columns], row))


def get_user_by_email(email: str) -> dict | None:
    """Fetch a user by email. Returns dict or None."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(users_table).where(users_table.c.email == email)
        ).fetchone()
    if not row:
        return None
    return dict(zip([c.name for c in users_table.columns], row))


def get_user_by_username(username: str) -> dict | None:
    """Fetch a user by username. Returns dict or None."""
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(users_table).where(users_table.c.username == username)
        ).fetchone()
    if not row:
        return None
    return dict(zip([c.name for c in users_table.columns], row))


def insert_feedback(qa_log_id: int, user_id: int, rating: int, comment: str = "") -> int:
    """Insert a feedback record. Returns the new feedback id."""
    engine = get_engine()
    with engine.connect() as conn:
        result = conn.execute(
            feedback_table.insert().values(
                qa_log_id=qa_log_id,
                user_id=user_id,
                rating=rating,
                comment=comment,
            ).returning(feedback_table.c.id)
        )
        conn.commit()
        return result.scalar()


def get_feedback_stats() -> dict:
    """Returns thumbs-up/down counts and average rating per LLM."""
    engine = get_engine()
    with engine.connect() as conn:
        total = conn.execute(
            select(func.count()).select_from(feedback_table)
        ).scalar()
        avg_rating = conn.execute(
            select(func.avg(feedback_table.c.rating)).select_from(feedback_table)
        ).scalar()
        thumbs_up = conn.execute(
            select(func.count()).select_from(feedback_table)
            .where(feedback_table.c.rating == 1)
        ).scalar()
        thumbs_down = conn.execute(
            select(func.count()).select_from(feedback_table)
            .where(feedback_table.c.rating == -1)
        ).scalar()

    return {
        "total_feedback": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "avg_rating": round(float(avg_rating), 3) if avg_rating else 0.0,
    }


def get_dashboard_stats() -> dict:
    """Returns aggregated stats for the dashboard."""
    engine = get_engine()
    with engine.connect() as conn:
        total_docs = conn.execute(select(func.count()).select_from(documents_table)).scalar()
        total_chunks = conn.execute(select(func.count()).select_from(chunks_table)).scalar()
        total_queries = conn.execute(select(func.count()).select_from(qa_logs_table)).scalar()
        total_experiments = conn.execute(select(func.count()).select_from(experiments_table)).scalar()

        avg_faith_row = conn.execute(
            select(func.avg(experiments_table.c.faithfulness)).select_from(experiments_table)
        ).scalar()
        avg_latency_row = conn.execute(
            select(func.avg(qa_logs_table.c.latency_seconds)).select_from(qa_logs_table)
        ).scalar()

        # Most used LLM
        llm_row = conn.execute(
            select(qa_logs_table.c.llm_name, func.count().label("cnt"))
            .group_by(qa_logs_table.c.llm_name)
            .order_by(func.count().desc())
            .limit(1)
        ).fetchone()

        # Most used retriever
        ret_row = conn.execute(
            select(qa_logs_table.c.retriever_type, func.count().label("cnt"))
            .group_by(qa_logs_table.c.retriever_type)
            .order_by(func.count().desc())
            .limit(1)
        ).fetchone()

    return {
        "total_documents": total_docs or 0,
        "total_chunks": total_chunks or 0,
        "total_queries": total_queries or 0,
        "total_experiments": total_experiments or 0,
        "avg_faithfulness": round(float(avg_faith_row), 3) if avg_faith_row else 0.0,
        "avg_latency": round(float(avg_latency_row), 3) if avg_latency_row else 0.0,
        "most_used_llm": llm_row[0] if llm_row else "N/A",
        "most_used_retriever": ret_row[0] if ret_row else "N/A",
    }


def get_user_sessions(user_id: int) -> list:
    """Returns a list of unique session dicts for a user, ordered by most recent."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                qa_logs_table.c.session_id,
                qa_logs_table.c.query_text,
                qa_logs_table.c.run_at
            )
            .where(qa_logs_table.c.user_id == user_id)
            .where(qa_logs_table.c.session_id.isnot(None))
            .order_by(qa_logs_table.c.run_at.desc())
        ).fetchall()
        
    sessions = []
    seen = set()
    for r in rows:
        if r.session_id not in seen:
            seen.add(r.session_id)
            sessions.append({
                "session_id": r.session_id,
                "title": r.query_text[:50] + "..." if len(r.query_text) > 50 else r.query_text,
                "last_active": str(r.run_at) if r.run_at else ""
            })
    return sessions


def get_session_history(session_id: str, user_id: int) -> list:
    """Returns all Q&A pairs for a specific session."""
    import json
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            select(
                qa_logs_table.c.query_text,
                qa_logs_table.c.answer_text,
                qa_logs_table.c.sources,
                qa_logs_table.c.run_at
            )
            .where(qa_logs_table.c.session_id == session_id)
            .where(qa_logs_table.c.user_id == user_id)
            .order_by(qa_logs_table.c.run_at.asc())
        ).fetchall()
    
    history = []
    for r in rows:
        history.append({
            "query_text": r.query_text,
            "answer_text": r.answer_text,
            "sources": json.loads(r.sources) if r.sources else [],
            "run_at": str(r.run_at) if r.run_at else ""
        })
    return history
