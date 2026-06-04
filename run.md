# ⚖️ Legal RAG — Run Guide

> Run every command from the **project root**. One terminal per step unless noted.

---

## Prerequisites

- Docker Desktop running
- Python 3.10+ with a virtual environment set up
- BGE model weights downloaded into `data/models/`
  - `data/models/bge-m3/` — from `BAAI/bge-m3` on Hugging Face
  - `data/models/bge-reranker-v2-m3/` — from `BAAI/bge-reranker-v2-m3` on Hugging Face
- A Gemini API key → [aistudio.google.com](https://aistudio.google.com) (free)
- Your PDF dataset in a `dataset/` folder at the project root

---

## Setup (Once)

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in your config
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

---

## Step 1 — Start Docker Services

```bash
docker compose up -d
```

Starts Qdrant (port 6333), embedding server (port 8081), reranker server (port 8082).

Verify all three are healthy:
```bash
curl http://localhost:6333/healthz   # → {"title":"qdrant - ..."}
curl http://localhost:8081/health    # → {"status":"ok","mode":"embed"}
curl http://localhost:8082/health    # → {"status":"ok","mode":"rerank"}
```

---

## Step 2 — Preprocess PDFs → SQLite

```bash
python run_preprocess.py
```

Reads all PDFs from `dataset/`, extracts text, metadata, and paragraph chunks, writes to `data/db/legal.db`.

⏱ ~400 documents might take time. Fully idempotent — safe to re-run. 

---

## Step 3 — Index into Qdrant

```bash
python run_indexing.py
```

Reads SQLite rows, generates BGE-M3 dense + sparse embeddings in batches of 8, upserts into Qdrant collection `legal_chunks`.

⏱ 400 documents time depending on hardware.

---

## Step 4 — Start Retrieval Service

```bash
python run_retreival.py
```

Starts the search API on `http://localhost:8000`. Keep this terminal running.

Verify:
```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

Test a search:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "Section 34 arbitration award", "top_k": 10, "rerank_top_n": 3}'
```

---

## Step 5 — Launch the Chat UI

Open a **new terminal tab**:

```bash
.venv/bin/streamlit run services/agent/src/app.py
```

Opens at **http://localhost:8501** in your browser.

- Type a legal question in the chat input
- Click **🔄 New Chat** in the sidebar to reset the conversation
- Expand **📄 Sources** under each answer to see retrieved chunks
- Debug logs are saved to `data/debug_logs/` after every LLM call

---

## Wipe & Start Fresh

```bash
# Stop Docker
docker compose down

# Delete all data (keeps folder structure)
rm -f data/db/legal.db data/db/legal.db-shm data/db/legal.db-wal
rm -rf data/qdrant/collections data/qdrant/aliases data/qdrant/*.json
rm -f data/debug_logs/*.json

# Then re-run Steps 1 → 5
```

---

## Debug Logs

Each LLM call writes a timestamped JSON file to `data/debug_logs/`:

```json
{
  "timestamp": "2025-06-04 09:00:00",
  "query": "What is the test for patent illegality?",
  "history": [...],
  "retrieved_context": [...],
  "prompts": { "system": "...", "user": "..." },
  "response": "...",
  "token_usage": { "input": 1842, "output": 312, "total": 2154 },
  "estimated_cost_usd": 0.00025
}
```
