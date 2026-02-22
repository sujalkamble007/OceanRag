# 🌊 OceanRAG — Deep-Sea Governance Research Assistant

A **Retrieval-Augmented Generation (RAG)** pipeline for semantic search over ocean governance research papers. OceanRAG ingests PDFs, chunks text, generates vector embeddings, and stores them in a cloud vector database — enabling natural language queries across thousands of pages of deep-sea mining regulations, UNCLOS documentation, and environmental impact assessments.

---

## ✨ What It Does

1. **Ingests** 104 ocean governance PDFs (7,333 pages) using LangChain
2. **Chunks** documents into 37,013 overlapping text segments
3. **Embeds** each chunk into a 384-dimensional vector using MiniLM
4. **Stores** vectors in Qdrant Cloud for fast similarity search
5. **Tracks** metadata (document → chunk → vector mapping) in PostgreSQL
6. **Retrieves** the most relevant passages for any natural language query

```
📄 PDFs → 🔪 Chunks → 🧠 Embeddings → 🔍 Qdrant Cloud → 💬 Answers
                                         ↘ 🗄️ PostgreSQL (metadata)
```

---

## 🏗️ Architecture

```
OceanRAG/
├── config.py              # Environment & configuration
├── database.py            # PostgreSQL (Neon) — tables, CRUD, batch inserts
├── document_loader.py     # PDF loading with timeout protection
├── chunker.py             # Text splitting (RecursiveCharacterTextSplitter)
├── embedder.py            # HuggingFace embeddings (batch processing)
├── qdrant_store.py        # Qdrant Cloud — upsert, search, collection mgmt
├── main.py                # Phase 1 orchestration pipeline (9 steps)
├── .env                   # API keys & DB credentials (not committed)
├── docs/Publications/     # Source PDFs
└── output/                # Chunk summaries (CSV)
```

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.13 | Core runtime |
| **PDF Parsing** | LangChain + PyPDF | Extract text from research papers |
| **Text Splitting** | RecursiveCharacterTextSplitter | Chunk documents into segments |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vector embeddings |
| **Vector DB** | Qdrant Cloud | Similarity search (Cosine distance) |
| **Metadata DB** | PostgreSQL (Neon) | Document & chunk metadata tracking |
| **ORM** | SQLAlchemy | Database operations & connection pooling |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/OceanRAG.git
cd OceanRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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

Place your PDF files in the `docs/` directory:

```bash
mkdir -p docs/Publications
# Copy your PDFs into docs/Publications/
```

### 4. Run Phase 1 Pipeline

```bash
python main.py
```

This runs the full 9-step pipeline:

| Step | What It Does | Time |
|------|-------------|------|
| 1 | Initialize PostgreSQL tables | ~2s |
| 2 | Load PDFs + extract text | ~3 min |
| 3 | Chunk text (fixed_512 strategy) | ~10s |
| 4 | Connect to Qdrant Cloud | ~1s |
| 5 | Generate embeddings (MiniLM) | ~4.5 min |
| 6 | Upload vectors to Qdrant | ~46 min |
| 7 | Save metadata to PostgreSQL | ~3 min |
| 8 | Test retrieval queries | ~5s |
| 9 | Print summary | instant |

> **Note:** Steps 5-6 are skipped on subsequent runs if Qdrant already has vectors (idempotent design).

---

## 🔍 Example Queries & Results

```python
from qdrant_store import get_qdrant_client, search_similar
from embedder import load_embedding_model
from config import QDRANT_COLLECTION_NAME, DEFAULT_EMBEDDING_CONFIG

client = get_qdrant_client()
model = load_embedding_model(DEFAULT_EMBEDDING_CONFIG)
vec = model.embed_query("What are the environmental obligations under UNCLOS?")
search_similar(client, QDRANT_COLLECTION_NAME, vec, k=3)
```

**Output:**
```
🔍 Query: 'What are the environmental obligations under UNCLOS?'
  [1] Score: 0.7262 | Publications-30.pdf | Page 75
  [2] Score: 0.7257 | Publications-33.pdf | Page 186
  [3] Score: 0.7252 | Publications-51.pdf | Page 22

🔍 Query: 'ISA regulations for deep-sea mining'
  [1] Score: 0.8224 | Publications-56.pdf | Page 25

🔍 Query: 'Environmental Impact Assessment requirements'
  [1] Score: 0.8495 | Publications-67.pdf | Page 163
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
| PostgreSQL records | 37,013 chunks + 104 documents |

---

## ⚙️ Configuration Options

### Chunking Strategies

Defined in `config.py` — switch by changing `DEFAULT_CHUNK_CONFIG`:

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

## 📁 Database Schema

### `documents` table
| Column | Type | Description |
|--------|------|-------------|
| id | SERIAL PK | Auto-increment ID |
| filename | VARCHAR (UNIQUE) | PDF filename |
| filepath | VARCHAR | Full path to file |
| total_pages | INTEGER | Page count |

### `chunks` table
| Column | Type | Description |
|--------|------|-------------|
| chunk_id | VARCHAR PK | Deterministic chunk identifier |
| document_id | FK → documents | Parent document |
| filename | VARCHAR | Source PDF |
| page_number | INTEGER | Page in source PDF |
| chunk_strategy | VARCHAR | e.g., fixed_512 |
| chunk_size / overlap | INTEGER | Chunking params |
| char_count | INTEGER | Character count |
| content_preview | TEXT | First 200 chars |
| qdrant_point_id | VARCHAR | UUID in Qdrant |
| embedding_model | VARCHAR | Model used |

---

## 🗺️ Roadmap

- [x] **Phase 1** — Document ingestion & indexing pipeline
- [ ] **Phase 2** — RAG query pipeline (LLM integration)
- [ ] **Phase 3** — Web UI for interactive search
- [ ] **Phase 4** — Experiment tracking & evaluation

---

## 📄 License

MIT
