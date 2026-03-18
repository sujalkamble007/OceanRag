"""
api/main.py — FastAPI application entry point.

Startup: initialize DeepRAGPipeline once (Phase 1 with existing index).
All routers registered here. Pipeline shared via get_pipeline() dependency.
"""

import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from sqlalchemy import text
from pipeline.deeprag_pipeline import DeepRAGPipeline
from core.database import init_db, get_engine

from api.auth.router import router as auth_router
from api.routers.query import router as query_router
from api.routers.history import router as history_router
from api.routers.experiments import router as experiments_router
from api.routers.dashboard import router as dashboard_router
from api.routers.feedback import router as feedback_router
from api.routers.documents import router as documents_router
from api.routers.eval_results import router as eval_router

# ── Global pipeline singleton ───────────────────────────────────────────────
_pipeline: DeepRAGPipeline = None


def get_pipeline() -> DeepRAGPipeline:
    """FastAPI dependency — returns the shared pipeline instance."""
    return _pipeline


# ── Keep-alive task for Neon free tier ──────────────────────────────────────
async def _neon_keep_alive():
    """Ping Neon every 4 min to prevent free-tier compute suspension."""
    while True:
        await asyncio.sleep(240)
        try:
            engine = get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("💓 Neon keep-alive ping OK")
        except Exception as e:
            print(f"⚠️  Neon keep-alive ping failed: {e}")


# ── Lifespan: startup + shutdown ────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    print("\n🚀 Starting DeepRAG API...")
    print("   Initializing database tables...")
    init_db()
    print("   Loading DeepRAG pipeline (Phase 1)...")
    _pipeline = DeepRAGPipeline()
    _pipeline.run_phase1(rebuild=False)

    # Start background keep-alive for Neon
    keep_alive_task = asyncio.create_task(_neon_keep_alive())
    print("💓 Neon keep-alive started (every 4 min)")
    print("✅ DeepRAG API ready — visit http://localhost:8000/docs\n")
    yield
    keep_alive_task.cancel()
    print("👋 Shutting down DeepRAG API")


# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DeepRAG API",
    description=(
        "🌊 OceanRAG — RAG system for Deep-Sea Governance & UNCLOS documents.\n\n"
        "**Authentication:** All endpoints (except /auth/*) require a Bearer JWT token.\n\n"
        "**Flow:** Register → Login → Copy `access_token` → Click 'Authorize' above → Use endpoints."
    ),
    version="5.0.0",
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
        # FRONTEND_URL, 
        # "http://localhost:3000",
        # "http://localhost:5173",
        # "http://localhost:5174",
        # "http://localhost:5175"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(query_router)
app.include_router(history_router)
app.include_router(experiments_router)
app.include_router(dashboard_router)
app.include_router(feedback_router)
app.include_router(documents_router)
app.include_router(eval_router)


# ── Root endpoints ───────────────────────────────────────────────────────────
@app.get("/", tags=["root"])
def root():
    return {
        "message": "🌊 DeepRAG API is running",
        "docs": "http://localhost:8000/docs",
        "version": "5.0.0",
        "endpoints": {
            "auth": "/auth/register | /auth/login | /auth/me",
            "query": "POST /query/",
            "history": "GET /history/",
            "experiments": "GET /experiments/leaderboard",
            "dashboard": "GET /dashboard/charts | /dashboard/stats",
        },
    }


@app.get("/health", tags=["root"])
def health():
    return {
        "status": "healthy",
        "pipeline_ready": _pipeline is not None and _pipeline.is_phase1_ready,
    }
