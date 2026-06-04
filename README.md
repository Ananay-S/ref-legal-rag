# Legal RAG

A local, offline Retrieval-Augmented Generation (RAG) pipeline for Indian Supreme Court judgements. Processes PDFs end-to-end — extraction → chunking → hybrid vector search → LLM-powered chat.

## What It Does

1. **Preprocessing** — Extracts text and metadata from PDFs (case name, parties, date, jurisdiction, paragraph chunks) and stores them in SQLite.
2. **Indexing** — Generates dense (1024-dim) and sparse (BM25-style) embeddings via `BGE-M3` and upserts them into Qdrant as hybrid vectors.
3. **Retrieval** — A lightweight HTTP search server accepts a query, runs a hybrid dense+sparse search, reranks results with `BGE-Reranker-v2-m3`, and returns the top chunks.
4. **Agent + Chat UI** — A Streamlit app sends the query and retrieved context to Gemini (via `google-genai` SDK), displays the answer, and logs every LLM call with token usage and estimated cost.

## Architecture

```
PDFs  →  preprocessing  →  SQLite
                              ↓
                          indexing  →  Qdrant (dense + sparse)
                                           ↓
query  →  retrieval service  ←  Qdrant search + rerank
               ↓
         agent (prompt builder + Gemini API)
               ↓
         Streamlit chat UI
```

## Stack

| Component | Choice |
|---|---|
| Embeddings + Reranker | `BAAI/bge-m3` + `BAAI/bge-reranker-v2-m3` (Docker CPU) |
| Vector DB | Qdrant (Docker) |
| LLM | Gemini 1.5 Flash via `google-genai` |
| UI | Streamlit |
| Storage | SQLite + local JSON fallback |
| Config | `.env` file |

## Project Structure

```
├── docker-compose.yml          # Qdrant + model servers
├── requirements.txt
├── run_preprocess.py           # Step 2 entrypoint
├── run_indexing.py             # Step 3 entrypoint
├── run_retreival.py            # Step 4 entrypoint
├── phase2_common.py            # Shared utils (embeddings, similarity, chunking)
├── phase2_models.py            # Embedder + Reranker wrappers
├── phase2_store.py             # Qdrant / local vector store abstraction
│
├── services/
│   ├── preprocessing/src/      # PDF → SQLite pipeline
│   ├── indexing/src/           # SQLite → Qdrant pipeline
│   ├── retrieval/src/          # Search HTTP API server
│   ├── model_server/src/       # BGE embed + rerank HTTP servers
│   └── agent/
│       ├── system_prompt.md    # System prompt loaded at runtime
│       └── src/
│           ├── agent.py        # Retrieval → prompt → Gemini → log
│           ├── prompt_builder.py  # Context dedup, group, sort
│           ├── logger.py       # Debug log writer
│           └── app.py          # Streamlit chat UI
│
└── data/                       # Created at runtime, not tracked in git
    ├── db/                     # legal.db (SQLite)
    ├── qdrant/                 # Qdrant persistence
    ├── models/                 # BGE model weights (download separately)
    ├── preprocessed/           # Intermediate JSON outputs
    └── debug_logs/             # Per-call LLM logs (JSON)
```

## Requirements

- macOS or Linux
- Python 3.10+
- Docker Desktop running
- BGE model weights in `data/models/` (see [run.md](run.md))
- A Gemini API key — get one free at [aistudio.google.com](https://aistudio.google.com)

## Dataset

This pipeline is designed for Indian Supreme Court judgement PDFs.  
Place your PDFs in a `dataset/` folder at the project root (not tracked in git).

> A curated dataset of ~400 Supreme Court judgements (2025) can be obtained from [Indian Kanoon](https://indiankanoon.org) or similar legal databases.
>https://www.kaggle.com/datasets/adarshsingh0903/legal-dataset-sc-judgments-india-19502024?resource=download&select=supreme_court_judgments

## Quick Start

See **[run.md](run.md)** for the full step-by-step guide.
