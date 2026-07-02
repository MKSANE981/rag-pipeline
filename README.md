# RAG Document Pipeline

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
- Parallelised ingestion via Ray

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | ChromaDB |
| Parallelisation | Ray |
| Generation | Any OpenAI-compatible API |

## Quick Start

```bash
pip install -r requirements.txt

# Index documents
python src/pipeline.py --docs_path ./data/docs --index

# Query
python src/pipeline.py --query "What are the key terms of the contract?"
```

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
│   └── docs/             # Drop your documents here
├── requirements.txt
└── README.md
```