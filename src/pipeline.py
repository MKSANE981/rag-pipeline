"""End-to-end RAG pipeline — ingestion, retrieval, prompt building."""

import argparse
from ingestion import process_documents
from vectorstore import VectorStore


def build_context(chunks: list, max_words: int = 1500) -> str:
    parts, total = [], 0
    for chunk in chunks:
        n = len(chunk["text"].split())
        if total + n > max_words:
            break
        parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        total += n
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    return (
        "Use the context below to answer the question accurately and concisely.\n"
        "If the answer is not in the context, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def ask(query: str, store: VectorStore, n_results: int = 5) -> dict:
    chunks = store.query(query, n_results=n_results)
    context = build_context(chunks)
    prompt = build_prompt(query, context)
    sources = sorted({c["source"] for c in chunks})
    return {"prompt": prompt, "context": context, "sources": sources, "chunks": chunks}


def main():
    parser = argparse.ArgumentParser(description="RAG Document Pipeline")
    parser.add_argument("--docs_path", default="./data/docs")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--index", action="store_true", help="(Re)index documents")
    parser.add_argument("--reset", action="store_true", help="Reset the vector store")
    parser.add_argument("--n_results", type=int, default=5)
    args = parser.parse_args()

    store = VectorStore()

    if args.reset:
        store.reset()

    if args.index or store.count() == 0:
        print(f"Indexing documents from {args.docs_path} ...")
        records = process_documents(args.docs_path)
        store.add_documents(records)
        print(f"Vector store now contains {store.count()} chunks.")

    if args.query:
        result = ask(args.query, store, args.n_results)
        print("\n=== Sources ===")
        print(", ".join(result["sources"]))
        print("\n=== Context (first 400 chars) ===")
        print(result["context"][:400], "...\n")
        print("=== Prompt ready for LLM ===")
        print(result["prompt"][:500], "...")


if __name__ == "__main__":
    main()