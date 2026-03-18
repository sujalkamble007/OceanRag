<div align="center">
  <h1>🌊 OceanRAG</h1>
  <p><strong>Production-grade Deep-Sea Governance Research Assistant powered by a full-stack RAG pipeline.</strong></p>
  <p><em>Ingests 7,000+ pages of dense PDF legal text, retrieves context in under a second, and generates grounded answers with full citation trails — with a beautiful multi-user web interface built on top.</em></p>

  <p>
    <img src="https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python" alt="Python" />
    <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi" alt="FastAPI" />
    <img src="https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react" alt="React" />
    <img src="https://img.shields.io/badge/Qdrant-Vector_DB-purple?style=for-the-badge" alt="Qdrant" />
    <img src="https://img.shields.io/badge/Neon-PostgreSQL-blue?style=for-the-badge&logo=postgresql" alt="Neon Postgres" />
  </p>
</div>

<br/>

<div align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-api-reference">API Reference</a> •
  <a href="#-user-roles">User Roles</a>
</div>

---

## 🌐 Overview

**OceanRAG** is a production-ready research assistant for deep-sea mining governance and UNCLOS regulations. It combines a sophisticated RAG (Retrieval-Augmented Generation) backend with a modern single-page web application.

Every query goes through a 4-phase pipeline:
1. **Document Ingestion** — parses, chunks, embeds, and stores 100+ PDFs (~37,000 chunks)
2. **Hybrid Retrieval** — fuses dense vector search and BM25 keyword search for precision recall
3. **LLM Generation** — injects retrieved context into a grounded prompt and streams the response
4. **Evaluation Matrix** — benchmarks every configuration combination to find statistically optimal hyper-parameters

The system features full user authentication, role-based access control, **Conversational Memory** (chat history tied to sessions), real-time streaming, analytics dashboards, and an automated research evaluation framework.

---

## ✨ Features

### 🤖 Core RAG Pipeline
| Feature | Detail |
|---|---|
| Document Ingestion | LangChain PDF loader with timeout protection |
| Chunking Strategies | 6+ strategies: Fixed, Sentence, Recursive, Semantic |
| Embedding Models | `all-MiniLM-L6-v2`, `bge-small-en-v1.5`, `paraphrase-MiniLM-L6-v2` |
| Retrieval Modes | `Similarity`, `MMR (Max Marginal Relevance)`, `Hybrid (BM25 + Vector)` |
| LLM Support | Groq: Llama 3 (8B/70B) • HuggingFace: Qwen 2.5 72B, Zephyr 7B |
| Response Streaming | Real-time token-by-token Server-Sent Events (SSE) |

### 💬 Conversational Memory
- Each chat session gets a persistent `session_id` (UUID)
- Previous QA turns are injected into the LLM prompt automatically (last 3 turns)
- Left sidebar slider lists all past sessions with dates and titles
- Click any past session to restore full conversation history and continue chatting
- New Chat button generates a fresh `session_id`

### 👥 Multi-User Authentication
- JWT-based secure authentication
- Role-based access control with 4 roles:
  - `common_user` — Basic chat access (Similarity retrieval, top-K capped at 3)
  - `student` — Advanced retrieval, top-K up to 5
  - `researcher` — All models, hybrid retrieval, Eval Matrix access
  - `admin` — Full access, user management, all features

### 📊 Analytics & Research Evaluation
- **Dashboard** — Q&A log viewer with filters, latency and cost tracking
- **Research Results** — Automated 4-phase evaluation matrix:
  - Phase A: Chunking strategy comparison
  - Phase B: Embedding model comparison
  - Phase C: Retriever × Top-K sweep
  - Phase D: LLM head-to-head
- Leaderboard with composite scores across all tested configurations

### 🎨 Frontend Interface
- **Landing Page** — animated ocean-themed hero, feature highlights, CTA
- **Top Navbar** — Logo, Chat Engine tab, Analytics & Research Results links, profile + logout
- **Chat Interface** — animated collapsible session history sidebar, settings panel (LLM, retriever, top-K)
- **Real-time Streaming** — token-by-token rendering with typing indicator
- **Source Citations** — documents consulted shown beneath each answer
- **Thumbs Up/Down Feedback** — per-answer rating stored in DB
- Responsive dark design built with TailwindCSS + Framer Motion

---

## 🏗️ Architecture

The system is built as three decoupled layers — a React SPA, a FastAPI backend, and a cloud-native data layer.

```mermaid
graph TD
    classDef main fill:#f9f9f9,stroke:#333,stroke-width:2px;
    classDef db fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef ext fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef ui fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef auth fill:#fce4ec,stroke:#c62828,stroke-width:2px;

    subgraph "Phase 0: Frontend & Auth"
        U[User Browser] --> LAND(Landing Page)
        LAND --> LOGIN(Login / Register)
        LOGIN --> JWT[JWT Token]
        JWT --> CHAT(Chat Interface)
        CHAT --> SIDEBAR(Session History Sidebar)
        CHAT --> PARAMS(Parameters Panel)
    end

    subgraph "Phase 1: Document Ingestion & Indexing"
        A[PDF Documents] --> B(LangChain PDFLoader)
        B --> C{Chunking Strategies\nFixed · Sentence · Recursive}
        C --> D[HuggingFace Embeddings\nMiniLM · BGE · SBERT]
        D --> E[(Qdrant Cloud\n37K+ Vectors)]
        C --> F[(Neon Postgres\nChunk Metadata)]
    end

    subgraph "Phase 2: Hybrid Retrieval"
        G[User Query + session_id] --> H[Embed Query]
        H -. Vector Search .-> E
        G -. BM25 Keyword Search .-> F
        E --> I{Retrieval Fusion\nSimilarity · MMR · Hybrid}
        F --> I
    end

    subgraph "Phase 3: LLM Generation"
        I --> J[Prompt Builder\n+ Chat History Injection]
        J --> K[LLM Router Interface]
        K --> L1[Groq — Llama 3 8B/70B]
        K --> L2[HuggingFace — Qwen 2.5/Zephyr]
        L1 --> M[Streaming Answer\nSSE Tokens]
        L2 --> M
        M --> N[(Neon Postgres\nqa_logs + session_id)]
    end

    subgraph "Phase 4: Conversational Memory"
        N -. get_session_history .-> J
        SIDEBAR -. GET /history/sessions .-> N
        CHAT -. POST /query/stream .-> G
    end

    subgraph "Phase 5: Research Evaluation"
        O[RAGAS + NLP Metrics\nROUGE · BLEU · BERTScore\nPrecision@K · MRR] -. Validates .-> J
        O -. Validates .-> I
        P[Testset Generator\n50 Synthetic QAs] --> O
        Q[run_research.py\nPhase A/B/C/D] --> O
        O --> R[(eval_results table)]
    end

    class A,G,M main;
    class E,F,N,R db;
    class L1,L2,D,P ext;
    class U,CHAT,SIDEBAR,PARAMS ui;
    class JWT,LOGIN auth;
```


### Request Lifecycle (Chat Query)

```
Browser → POST /query/stream (query, session_id, llm_key, retriever_type, top_k)
         ↓
    JWT Auth Middleware validates Bearer token
         ↓
    DeepRAGPipeline.stream_rag_query_api()
         ↓
    ┌── Retrieval Engine ───────────────────────────────────┐
    │  embed(query) → Qdrant nearest-neighbor search        │
    │  (Similarity / MMR / Hybrid-BM25 depending on role)  │
    └───────────────────────────────────────────────────────┘
         ↓  top-K chunks
    ┌── Session History ─────────────────────────────────────┐
    │  get_session_history(session_id) → last 3 QA turns     │
    │  format as "User: ...\nDeepRAG: ..." block             │
    └────────────────────────────────────────────────────────┘
         ↓  chat_history_str
    ┌── Prompt Builder ──────────────────────────────────────┐
    │  SYSTEM_PROMPT + retrieved_chunks + chat_history       │
    │  + "Answer only from the context. Cite sources."       │
    └────────────────────────────────────────────────────────┘
         ↓  assembled prompt
    LLM Router → Groq / HuggingFace API
         ↓  token stream
    SSE events: { event: "token", data: { token: "..." } }
                { event: "done",  data: { sources, latency, record_id } }
         ↓
    _save_qa_background() → INSERT INTO qa_logs (session_id, user_id, ...)
```

---

## 📂 Project Structure

```
OceanRAG/
│
├── api/                            # FastAPI Application Layer
│   ├── main.py                     # App init, CORS, routers, pipeline singleton
│   ├── dependencies.py             # get_current_user, get_pipeline injectors
│   ├── auth/                       # JWT token logic
│   ├── middleware/                 # Rate limiting, logging middleware
│   └── routers/
│       ├── auth.py                 # POST /auth/login, /auth/register
│       ├── query.py                # POST /query, /query/stream (SSE)
│       ├── history.py              # GET /history/sessions, /sessions/{id}
│       ├── feedback.py             # POST /feedback/ (thumbs up/down)
│       └── eval_results.py         # GET /eval-results/ leaderboard
│
├── core/                           # Shared domain utilities
│   ├── config.py                   # CHUNK_CONFIGS, EMBEDDING_CONFIGS, LLM keys
│   ├── database.py                 # SQLAlchemy tables + all CRUD functions
│   │                               #   ↳ qa_logs (session_id, user_id)
│   │                               #   ↳ get_user_sessions / get_session_history
│   ├── document_loader.py          # PDF loader with timeout protection
│   └── chunker.py                  # Fixed, Sentence, Recursive, Semantic chunkers
│
├── retrieval/
│   └── retriever.py                # similarity_search, mmr_search, hybrid_search
│
├── generation/
│   ├── generation_pipeline.py      # run_rag_query / stream_rag_query
│   │                               #   ↳ fetches chat history → injects into prompt
│   ├── prompt_builder.py           # SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
│   │                               #   ↳ build_prompt(query, chunks, chat_history)
│   ├── llm_handler.py              # Groq + HuggingFace API routers
│   └── answer_store.py             # save_qa(session_id, user_id, ...)
│
├── pipeline/
│   └── deeprag_pipeline.py         # DeepRAGPipeline master class
│       ├── run_phase1()            # Ingest docs → embed → Qdrant + Postgres
│       ├── run_phase2()            # Test all retrievers
│       ├── run_rag_query()         # Sync API wrapper (accepts session_id)
│       └── stream_rag_query_api()  # Streaming API wrapper (accepts session_id)
│
├── evaluation/
│   ├── experiment_runner.py        # run_phase_a/b/c/d, run_single_experiment
│   ├── testset_generator.py        # Groq-powered synthetic QA generation
│   ├── metrics_calculator.py       # P@K, R@K, MRR, ROUGE-L, BLEU, BERTScore, RAGAS
│   └── results_exporter.py         # CSV export, leaderboard, chart JSON
│
├── frontend/                       # React 18 + Vite + TailwindCSS + Framer Motion
│   └── src/
│       ├── pages/
│       │   ├── Landing.jsx         # Animated hero page
│       │   ├── Login.jsx           # Auth pages
│       │   ├── Register.jsx
│       │   ├── Chat.jsx            # Chat + sliding session sidebar + streaming
│       │   ├── Dashboard.jsx       # Q&A log analytics
│       │   └── EvalResults.jsx     # Research leaderboard UI
│       ├── components/layout/
│       │   ├── Navbar.jsx          # Top bar: Logo · Chat · Analytics · Research · Profile
│       │   └── MainLayout.jsx      # Full-width layout (Navbar + Outlet)
│       ├── api/client.js           # Axios base client with JWT interceptor
│       └── store/authStore.js      # Zustand auth state (token, user, role)
│
├── docs/Publications/              # Input PDF research papers (not committed)
├── output/                         # CSV exports, evaluation matrix results
│
├── run_phase1.py                   # CLI: Ingest documents
├── run_research.py                 # CLI: Run phased evaluation (A/B/C/D or all)
├── migrate_qa_logs.py              # One-time DB migration: add session_id, user_id
├── requirements.txt
├── pyproject.toml
└── .env                            # API keys and DB connection strings
```

---



## 🚀 Quick Start

### Prerequisites
| Service | Description | Free Tier |
|---|---|---|
| [Groq](https://console.groq.com/keys) | LLM inference API | ✅ Yes |
| [Qdrant Cloud](https://cloud.qdrant.io/) | Vector database | ✅ Yes |
| [Neon](https://neon.tech/) | Serverless Postgres | ✅ Yes |
| Python 3.13+ | Runtime | — |
| Node.js 18+ | Frontend build | — |

### 1. Clone & Setup Backend
```bash
git clone https://github.com/sujalkamble007/OceanRAG.git
cd OceanRAG

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate           # Windows

pip install -r requirements.txt
pip install -e .                   # Install oceanrag as a local package
```

### 2. Environment Variables
Create a `.env` file in the project root:
```env
# ── Vector Database ──────────────────────────────────
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=OceanRag

# ── Relational Database (Neon Postgres) ──────────────
DATABASE_URL=postgresql://user:password@ep-your-db.region.neon.tech/neondb?sslmode=require

# ── LLMs ─────────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key
HF_API_TOKEN=your_huggingface_token   # Optional

# ── Auth ─────────────────────────────────────────────
SECRET_KEY=a-long-random-secret-key
```

### 3. Ingest Documents
Place your PDF files in `docs/Publications/`, then run the ingestion pipeline:
```bash
python run_phase1.py
```
This will chunk, embed, and index all documents into Qdrant and Postgres. *(One-time setup — skip if Qdrant already has vectors.)*

### 4. Start the Backend (FastAPI)
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
API will be available at `http://localhost:8000/docs`

### 5. Start the Frontend (React)
```bash
cd frontend
npm install
npm run dev
```
App will be available at `http://localhost:5173`

---

## 📂 Project Structure

```
OceanRAG/
├── api/                    # FastAPI Application
│   ├── main.py             # App entry point, routers, pipeline injection
│   └── routers/
│       ├── auth.py         # JWT login/register endpoints
│       ├── query.py        # /query and /query/stream endpoints
│       ├── history.py      # /history/sessions chat memory endpoints
│       ├── feedback.py     # Thumbs up/down rating endpoints
│       └── eval_results.py # Evaluation result viewer endpoints
│
├── core/                   # Shared utilities
│   ├── database.py         # SQLAlchemy models, CRUD, session history
│   ├── config.py           # Chunk, embedding, LLM config constants
│   ├── document_loader.py  # PDF parsing with timeout protection
│   └── chunker.py          # Chunking strategies
│
├── retrieval/              # Retrieval engine
│   └── retriever.py        # Similarity, MMR, Hybrid search
│
├── generation/             # Generation module
│   ├── generation_pipeline.py  # run_rag_query, stream_rag_query with chat history
│   ├── prompt_builder.py       # System + user prompt templates with history injection
│   ├── llm_handler.py          # Groq / HuggingFace router
│   └── answer_store.py         # qa_logs persistence with session_id
│
├── pipeline/
│   └── deeprag_pipeline.py     # Master orchestrator class for the API
│
├── evaluation/             # Research Evaluation Framework
│   ├── experiment_runner.py
│   ├── testset_generator.py
│   ├── metrics_calculator.py
│   └── results_exporter.py
│
├── frontend/               # React + Vite SPA
│   └── src/
│       ├── pages/
│       │   ├── Landing.jsx     # Animated landing page
│       │   ├── Login.jsx       # Auth pages
│       │   ├── Register.jsx
│       │   ├── Chat.jsx        # Main chat interface + session sidebar
│       │   ├── Dashboard.jsx   # Q&A analytics
│       │   └── EvalResults.jsx # Research leaderboard
│       ├── components/layout/
│       │   ├── Navbar.jsx      # Top navbar with nav links + profile
│       │   └── MainLayout.jsx  # Full-width layout wrapper
│       ├── api/client.js       # Axios API client
│       └── store/authStore.js  # Zustand auth state
│
├── run_phase1.py           # Document ingestion runner
├── run_research.py         # Phased evaluation entry point
└── migrate_qa_logs.py      # DB migration: adds session_id + user_id to qa_logs
```

---

## 📡 API Reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/register` | ❌ | Create a new user account |
| `POST` | `/auth/login` | ❌ | Login and receive a JWT token |
| `POST` | `/query` | ✅ | Run a full RAG query (sync) |
| `POST` | `/query/stream` | ✅ | Run a RAG query with SSE streaming |
| `GET` | `/history/sessions` | ✅ | List all past chat sessions for the current user |
| `GET` | `/history/sessions/{session_id}` | ✅ | Get full chat history for a session |
| `POST` | `/feedback/` | ✅ | Submit thumbs up/down rating for a Q&A record |
| `GET` | `/eval-results/` | ✅ | Get research evaluation leaderboard |
| `GET` | `/docs` | ❌ | Swagger interactive API documentation |

### Example: Start a Chat Session
```bash
# 1. Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "you@example.com", "password": "yourpassword"}'

# 2. Query (with session_id for Conversational Memory)
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the environmental obligations under UNCLOS Article 145?",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "llm_key": "groq-llama8b",
    "retriever_type": "mmr",
    "top_k": 5
  }'
```

---

## 👤 User Roles

| Role | Chat | MMR/Hybrid | All LLMs | Eval Matrix | Top-K |
|------|------|------------|----------|-------------|-------|
| `common_user` | ✅ | ❌ | ❌ | ❌ | 1–3 |
| `student` | ✅ | ✅ | ❌ | ❌ | 1–5 |
| `researcher` | ✅ | ✅ | ✅ | ✅ | 1–10 |
| `admin` | ✅ | ✅ | ✅ | ✅ | 1–10 |

---

## 🔬 Running the Research Evaluation

```bash
# Generate a 50-question synthetic testset
python run_research.py --generate-testset

# Run Phase A (Chunking strategy comparison)
python run_research.py --phase A

# Run all 4 phases sequentially (A → B → C → D)
python run_research.py --phase all

# Export results to CSV
python run_research.py --export
```

Results are saved to `output/` as CSVs and stored in the `eval_results` Postgres table, viewable in the **Research Results** page of the frontend.

---

## 🧠 How Conversational Memory Works

1. When a user opens the Chat page, a new `session_id` (UUID) is generated in the browser.
2. Every query sent to `/query/stream` includes this `session_id`.
3. The backend saves each QA turn to the `qa_logs` table with the `session_id`.
4. On the **next query in the same session**, the backend calls `get_session_history()` to fetch the last 3 turns, formats them as a `User: / DeepRAG:` block, and injects them into the LLM prompt.
5. The frontend sidebar fetches `/history/sessions` to list all past chats.
6. Clicking a session restores the full history by calling `/history/sessions/{session_id}`.

---

## 🎓 Key Design Decisions

- **Session memory via DB, not in-memory** — ensures history survives page refreshes and server restarts
- **Background thread for DB saves** — the `_save_qa_background` function runs in a `ThreadPoolExecutor` so it never blocks the streaming response
- **Hybrid retrieval default** — provides best-of-both-worlds for legal/technical text where exact term matching matters
- **Role-gated compute** — heavier models (70B) and retrieval methods are reserved for trusted roles to control API costs
- **Framer Motion sidebar** — the session sidebar slides with an `AnimatePresence` width transition, matching a native app feel

---

<div align="center">
  <p><em>"The best way to understand something is to build it."</em></p>
  <p>Built with care by <strong>Sujal Kamble</strong></p>
  <p>OceanRAG — find exactly what matters in the deepest of oceans. 🌊</p>
</div>
