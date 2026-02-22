# 🌊 OceanRAG — Deep-Sea Governance Research Assistant

A **Retrieval-Augmented Generation (RAG)** pipeline for semantic search over ocean governance research papers. OceanRAG ingests PDFs, chunks text, generates vector embeddings, and stores them in Qdrant Cloud — enabling natural language queries across thousands of pages of deep-sea mining regulations, UNCLOS documentation, and environmental impact assessments.

---

## ✨ What It Does

### Phase 1 — Document Ingestion & Indexing
1. **Ingests** 104 ocean governance PDFs (7,333 pages) using LangChain
2. **Chunks** documents into 37,013 overlapping text segments
3. **Embeds** each chunk into a 384-dimensional vector using MiniLM
4. **Stores** vectors in Qdrant Cloud for fast similarity search
5. **Tracks** metadata (document → chunk → vector mapping) in PostgreSQL

### Phase 2 — Retrieval Engine
6. **Similarity Search** — Pure vector cosine similarity via Qdrant
7. **MMR Search** — Max Marginal Relevance for diverse, non-redundant results
8. **Hybrid Search** — BM25 keyword + vector score fusion (best of both worlds)
9. **Logs** every retrieval run to PostgreSQL with latency metrics
10. **Interactive Mode** — Query the system live from your terminal

```
📄 PDFs → 🔪 Chunks → 🧠 Embeddings → 🔍 Qdrant Cloud → 💬 Ranked Results
                                         ↘ 🗄️ PostgreSQL (metadata + logs)
```

---

## 🏗️ Project Structure

```
OceanRAG/
├── config.py              # Environment & configuration
├── database.py            # PostgreSQL — tables, CRUD, batch inserts, retrieval logs
├── document_loader.py     # PDF loading with timeout protection
├── chunker.py             # Text splitting (RecursiveCharacterTextSplitter)
├── embedder.py            # HuggingFace embeddings (batch processing)
├── qdrant_store.py        # Qdrant Cloud — upsert, search, collection mgmt
├── main.py                # Phase 1 entry point (9-step pipeline)
├── retriever.py           # Phase 2 — 3 retrieval strategies (similarity, MMR, hybrid)
├── retrieval_logger.py    # Phase 2 — PostgreSQL logging + CSV export
├── run_retrieval.py       # Phase 2 entry point (5-step pipeline + interactive mode)
├── .env                   # API keys & DB credentials (not committed)
├── docs/Publications/     # Source PDFs
└── output/
    ├── chunks_fixed_512.csv       # Chunk summaries
    └── retrieval_results.csv      # Retrieval comparison results
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.13 | Core runtime |
| **PDF Parsing** | LangChain + PyPDF | Extract text from research papers |
| **Text Splitting** | RecursiveCharacterTextSplitter | Chunk documents into segments |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vector embeddings |
| **Vector DB** | Qdrant Cloud (v1.16) | Similarity search (Cosine distance) |
| **Metadata DB** | PostgreSQL (Neon) | Document, chunk, and retrieval metadata |
| **ORM** | SQLAlchemy | Database operations & connection pooling |
| **Keyword Search** | rank-bm25 | BM25 scoring for hybrid search |
| **Client Library** | qdrant-client v1.17 | Python SDK for Qdrant Cloud |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/OceanRAG.git
cd OceanRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install rank-bm25
```

### 2. Configure Environment

Create a `.env` file with your credentials:

```env
# Qdrant Cloud
QDRANT_URL=https://your-cluster.cloud.qdrant.io
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=OceanRag

# PostgreSQL (Neon)
POSTGRES_HOST=your-neon-endpoint.neon.tech
POSTGRES_PORT=5432
POSTGRES_DB=neondb
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
```

### 3. Add Documents

```bash
mkdir -p docs/Publications
# Copy your ocean governance PDFs into docs/Publications/
```

### 4. Run Phase 1 — Ingest & Index

```bash
source .venv/bin/activate
python main.py
```

| Step | What It Does | Time |
|------|-------------|------|
| 1 | Initialize PostgreSQL tables | ~2s |
| 2 | Load PDFs + extract text | ~2 min |
| 3 | Chunk text (fixed_512 strategy) | ~10s |
| 4 | Connect to Qdrant Cloud | ~1s |
| 5 | Generate embeddings (MiniLM) | ~4.5 min |
| 6 | Upload vectors to Qdrant | ~46 min |
| 7 | Save metadata to PostgreSQL | ~3 min |
| 8 | Test retrieval queries | ~5s |
| 9 | Print summary | instant |

> **Note:** Steps 5-6 are skipped on re-runs if Qdrant already has vectors.

### 5. Run Phase 2 — Retrieval Engine

```bash
python run_retrieval.py
```

| Step | What It Does |
|------|-------------|
| 1 | Initialize from Phase 1 (reuse existing index) |
| 2 | Ensure retrieval_logs table exists |
| 3 | Test all 3 strategies on a sample query |
| 4 | Compare latency across Top-K values (3, 5, 10) |
| 5 | Interactive mode — query the system live |

---

## 🔍 Retrieval Strategies

### 1. Similarity Search
Pure vector cosine similarity — finds the k most similar vectors in Qdrant.
- **Best for:** General semantic queries
- **Speed:** Fastest

### 2. MMR Search (Max Marginal Relevance)
Fetches candidates, then re-ranks to maximize diversity. Penalizes chunks that are too similar to already-selected results.
- **Best for:** Getting diverse perspectives from different sources
- **Speed:** Medium (requires fetching vectors for re-ranking)

### 3. Hybrid Search
Combines BM25 keyword scoring (sparse) + Qdrant vector search (dense). Both scores are normalized to [0, 1] and averaged with equal weighting.
- **Best for:** Queries with specific technical terms + semantic meaning
- **Speed:** Slowest (runs BM25 over all chunks)

---

## 📊 Example Output

```
════════════════════════════════════════════════════════════
  RETRIEVAL RESULTS
  Query: "What are the environmental obligations under UNCLOS?"
  Top-K: 5
════════════════════════════════════════════════════════════

── SIMILARITY SEARCH (latency: 7.46s) ─────────────────────
  [1] Score: 0.7262 | Publications-30.pdf | Page 75
  [2] Score: 0.7257 | Publications-33.pdf | Page 186
  [3] Score: 0.7252 | Publications-51.pdf | Page 22

── MMR SEARCH (latency: 1.66s) ─────────────────────────────
  [1] Score: 0.7262 | Publications-30.pdf | Page 75
  [2] Score: 0.7257 | Publications-33.pdf | Page 186
  [3] Score: 0.6852 | Publications-84.pdf | Page 27    ← more diverse

── HYBRID SEARCH (latency: 1.27s) ──────────────────────────
  [1] Score: 0.9996 | Publications-33.pdf | Page 186   ← keyword + vector match
  [2] Score: 0.5000 | Publications-30.pdf | Page 75
  [3] Score: 0.4993 | Publications-51.pdf | Page 22
```

### Top-K Latency Comparison

```
K     | Similarity |      MMR |   Hybrid
------+------------+---------+---------
3     |    1.23s   |  3.40s  |  1.23s
5     |    1.04s   |  1.56s  |  1.75s
10    |    5.38s   |  6.82s  |  3.62s
```

---

## 📊 Pipeline Statistics

| Metric | Value |
|--------|-------|
| Documents ingested | 104 PDFs |
| Pages extracted | 7,333 |
| Chunks created | 37,013 |
| Chunk strategy | fixed_512 (512 chars, no overlap) |
| Embedding model | all-MiniLM-L6-v2 (384 dimensions) |
| Vector similarity | Cosine |
| Qdrant vectors | 37,013 |
| PostgreSQL tables | documents, chunks, experiments, retrieval_logs |

---

## ⚙️ Configuration

### Chunking Strategies (`config.py`)

| Strategy | Size | Overlap |
|----------|------|---------|
| `fixed_256` | 256 | 0 |
| **`fixed_512`** (default) | 512 | 0 |
| `fixed_1024` | 1024 | 0 |
| `overlap_512_10` | 512 | 51 (10%) |
| `overlap_512_20` | 512 | 102 (20%) |
| `overlap_512_30` | 512 | 153 (30%) |

### Embedding Models

| Model | ID | Dimensions |
|-------|----|------------|
| **MiniLM** (default) | all-MiniLM-L6-v2 | 384 |
| BGE | bge-small-en-v1.5 | 384 |
| SBERT | all-mpnet-base-v2 | 768 |

---

## 🗄️ Database Schema

### `documents`
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Auto-increment ID |
| filename | VARCHAR (UNIQUE) | PDF filename |
| filepath | VARCHAR | Full path |
| total_pages | INTEGER | Page count |

### `chunks`
| Column | Type | Description |
|--------|------|-------------|
| chunk_id | VARCHAR PK | Deterministic chunk ID |
| document_id | FK → documents | Parent document |
| filename | VARCHAR | Source PDF |
| page_number | INTEGER | Page in source |
| chunk_strategy | VARCHAR | e.g., fixed_512 |
| qdrant_point_id | VARCHAR | UUID in Qdrant |
| embedding_model | VARCHAR | Model used |

### `retrieval_logs`
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Auto-increment ID |
| query_text | TEXT | The search query |
| retriever_type | VARCHAR | similarity / mmr / hybrid |
| embedding_model | VARCHAR | Model used |
| top_k | INTEGER | Number of results |
| results | JSONB | Full ranked results |
| latency_seconds | FLOAT | Query time |
| run_at | TIMESTAMP | When the query ran |

---

## 🗺️ Roadmap

- [x] **Phase 1** — Document ingestion & indexing pipeline
- [x] **Phase 2** — Retrieval engine (3 strategies + logging)
- [ ] **Phase 3** — LLM integration & answer generation
- [ ] **Phase 4** — Experiment tracking & evaluation

---

## 📄 License

MIT
