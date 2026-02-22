"""
config.py — All settings loaded from .env environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── Qdrant Cloud Settings ──────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "https://0b8ad691-381f-4af2-88a6-69811726b9ad.us-east-1-1.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.fmS_earKPd2v1gYxqk-5U-y8vr_2Mnm9Na7L_gpAm8M")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "OceanRag")

# ─── PostgreSQL Settings ────────────────────────────────────────────────────
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "ep-spring-morning-ainz0ze4-pooler.c-4.us-east-1.aws.neon.tech")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "OceanRag")
POSTGRES_USER = os.getenv("POSTGRES_USER", "neondb_owner")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "npg_Y5VFLJgUX9yR")

# ─── Directory Paths ────────────────────────────────────────────────────────
DOCS_DIR = "./docs"
OUTPUT_DIR = "./output"

# ─── Chunking Configurations ────────────────────────────────────────────────
CHUNK_CONFIGS = [
    {"name": "fixed_256",      "size": 256,  "overlap": 0},
    {"name": "fixed_512",      "size": 512,  "overlap": 0},
    {"name": "fixed_1024",     "size": 1024, "overlap": 0},
    {"name": "overlap_512_10", "size": 512,  "overlap": 51},
    {"name": "overlap_512_20", "size": 512,  "overlap": 102},
    {"name": "overlap_512_30", "size": 512,  "overlap": 153},
]

# ─── Embedding Configurations ───────────────────────────────────────────────
EMBEDDING_CONFIGS = [
    {"name": "MiniLM", "model_id": "sentence-transformers/all-MiniLM-L6-v2",  "vector_size": 384},
    {"name": "BGE",    "model_id": "BAAI/bge-small-en-v1.5",                   "vector_size": 384},
    {"name": "SBERT",  "model_id": "sentence-transformers/all-mpnet-base-v2",  "vector_size": 768},
]

# ─── Default Selections ─────────────────────────────────────────────────────
DEFAULT_CHUNK_CONFIG     = CHUNK_CONFIGS[1]       # fixed_512
DEFAULT_EMBEDDING_CONFIG = EMBEDDING_CONFIGS[0]   # MiniLM
TOP_K_VALUES = [3, 5, 10]
DEFAULT_TOP_K = 5
