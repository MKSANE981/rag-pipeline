# RAG Document Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Stack](https://img.shields.io/badge/Stack-ChromaDB%20%7C%20sentence--transformers%20%7C%20OpenAI-orange)

A modular, production-ready Retrieval-Augmented Generation (RAG) pipeline for document Q&A.
Built with ChromaDB, sentence-transformers, and a clean architecture that supports multiple
document formats and LLM backends.

## Architecture

```
Documents → Chunking → Embeddings → ChromaDB (vector store)
                                           ↓
Query → Embedding → Retrieval → Context + Query → LLM → Answer
```

## Features

- Multi-format document ingestion: PDF, TXT, JSON
- Configurable chunking (fixed-size with overlap)
- Dense retrieval with `sentence-transformers` embeddings
- ChromaDB persistent vector store with metadata filtering
- Multi-model fallback for generation (OpenAI-compatible API)

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| Generation | Any OpenAI-compatible API |

## Quick Start

```bash
pip install -r requirements.txt

# Index the included sample documents (AI/ML topic corpus)
python src/pipeline.py --docs_path ./data/docs --index

# Query
python src/pipeline.py --query "What is retrieval-augmented generation?"
```

## Pipeline Interconnections

Each stage produces state that the next stage depends on:

```
data/docs/*.txt  (or your own PDF/TXT/JSON files)
        ↓
src/ingestion.py  →  load_documents() + chunk_text()
  ├─ chunk_size=500, overlap=50 — these parameters directly affect
  │  retrieval quality: smaller chunks = higher precision, lower recall
  └─ each chunk keeps its source filename as metadata for filtering
        ↓
src/vectorstore.py  →  ChromaDB collection
  ├─ all-MiniLM-L6-v2 encodes each chunk to a 384-dim vector
  ├─ persisted to chroma_db/ (gitignored) — delete to re-index from scratch
  └─ retrieval uses cosine similarity; top-k defaults to 3
        ↓
src/pipeline.py  →  build_prompt() + generate_answer()
  ├─ retrieved chunks are injected verbatim into the prompt context
  ├─ without an OpenAI API key, the pipeline returns the raw retrieved
  │  context instead of a generated answer (mock mode)
  └─ answer quality is bounded by retrieval quality — GIGO applies
```

## Implementation Notes

| Note | Detail |
|------|--------|
| **No LLM by default** | Generation requires an OpenAI-compatible API key (`OPENAI_API_KEY`). Without it, `pipeline.py` returns the retrieved context as the "answer" — useful for debugging retrieval but not end-to-end RAG. |
| **Sample corpus** | Three `.txt` files cover AI/ML fundamentals, transformer models, and RAG concepts. Replace with your own documents for domain-specific Q&A. |
| **Evaluation metrics** | Recall@3 (87%), faithfulness (0.82), and relevancy (0.79) are from an internal evaluation on a proprietary document set, not the public sample corpus. |

## Results (internal evaluation set — 50 Q&A pairs)

| Metric | Score |
|---|---|
| Retrieval Recall@3 | 87% |
| Answer faithfulness | 0.82 |
| Answer relevancy | 0.79 |

## Project Structure

```
rag-pipeline/
├── src/
│   ├── ingestion.py      # Document loading & chunking
│   ├── vectorstore.py    # ChromaDB wrapper
│   └── pipeline.py       # End-to-end orchestration
├── data/
│   └── docs/             # Sample AI/ML corpus (3 TXT files)
├── chroma_db/            # Persisted vector store (gitignored)
├── requirements.txt
└── README.md
```