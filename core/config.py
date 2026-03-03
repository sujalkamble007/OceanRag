# core/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────
DOCS_DIR    = os.getenv("DOCS_DIR",   "./docs")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "./output")

# Auto-create output folder
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# ── Qdrant ───────────────────────────────────────────────
QDRANT_URL             = os.getenv("QDRANT_URL")
QDRANT_API_KEY         = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "deeprag_chunks")

# ── PostgreSQL ───────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    pg_host = os.getenv("POSTGRES_HOST")
    pg_port = os.getenv("POSTGRES_PORT", "5432")
    pg_db   = os.getenv("POSTGRES_DB")
    pg_user = os.getenv("POSTGRES_USER")
    pg_pwd  = os.getenv("POSTGRES_PASSWORD")
    if pg_host and pg_db and pg_user and pg_pwd:
        DATABASE_URL = f"postgresql://{pg_user}:{pg_pwd}@{pg_host}:{pg_port}/{pg_db}"
        if pg_host not in ("localhost", "127.0.0.1"):
            DATABASE_URL += "?sslmode=require&connect_timeout=10"

# ── LLM Keys ─────────────────────────────────────────────
GROQ_API_KEY  = os.getenv("GROQ_API_KEY",  "")
HF_API_TOKEN  = os.getenv("HF_API_TOKEN",  "")

# ── Chunking Configs ─────────────────────────────────────
CHUNK_CONFIGS = [
    {"name": "fixed_256",      "size": 256,  "overlap": 0},
    {"name": "fixed_512",      "size": 512,  "overlap": 0},
    {"name": "fixed_1024",     "size": 1024, "overlap": 0},
    {"name": "overlap_512_10", "size": 512,  "overlap": 51},
    {"name": "overlap_512_20", "size": 512,  "overlap": 102},
    {"name": "overlap_512_30", "size": 512,  "overlap": 153},
]

# ── Embedding Configs ────────────────────────────────────
EMBEDDING_CONFIGS = [
    {"name": "MiniLM", "model_id": "sentence-transformers/all-MiniLM-L6-v2", "vector_size": 384},
    {"name": "BGE",    "model_id": "BAAI/bge-small-en-v1.5",                  "vector_size": 384},
    {"name": "SBERT",  "model_id": "sentence-transformers/all-mpnet-base-v2", "vector_size": 768},
]

# ── Defaults ─────────────────────────────────────────────
DEFAULT_CHUNK_CONFIG     = CHUNK_CONFIGS[1]      # fixed_512
DEFAULT_EMBEDDING_CONFIG = EMBEDDING_CONFIGS[0]  # MiniLM

# ── Retrieval ─────────────────────────────────────────────
TOP_K_VALUES = [3, 5, 10]
DEFAULT_TOP_K = 5

# ── Validation ───────────────────────────────────────────
def validate_config():
    """Call at startup to catch missing env vars early."""
    errors = []
    if not QDRANT_URL:      errors.append("QDRANT_URL not set in .env")
    if not QDRANT_API_KEY:  errors.append("QDRANT_API_KEY not set in .env")
    if not DATABASE_URL:    errors.append("DATABASE_URL not set in .env")
    if not GROQ_API_KEY and not HF_API_TOKEN:
        errors.append("Neither GROQ_API_KEY nor HF_API_TOKEN set in .env")
    if errors:
        print("\n❌ Configuration errors:")
        for e in errors:
            print(f"   • {e}")
        raise EnvironmentError("Fix .env file before running.")
    print("✅ Configuration validated\n")
