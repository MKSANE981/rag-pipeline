"""
End-to-end RAG pipeline — ties ingestion, retrieval and prompt building together.

The design is deliberately LLM-agnostic: this module builds a ready-to-use
prompt string but does not call any LLM. That way you can plug in OpenAI,
Mistral, a local llama.cpp server, or anything else without changing the
retrieval logic.

Typical flow:
    1. Index documents once:   python pipeline.py --index --docs_path ./data/docs
    2. Ask questions:          python pipeline.py --query "What is X?"
    3. Feed the returned prompt to your LLM of choice.
"""

import argparse
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from ingestion import process_documents
from vectorstore import VectorStore


def build_context(chunks: list, max_words: int = 1500) -> str:
    """
    Concatenate retrieved chunks into a single context block.

    We cap the total word count to avoid blowing past the LLM's context
    window. Chunks are already ranked by relevance (closest first), so
    we just stop adding once we hit the limit.

    Args:
        chunks: Output of VectorStore.query().
        max_words: Approximate word budget for the context block.

    Returns:
        Formatted context string with source labels.
    """
    parts, total = [], 0
    for chunk in chunks:
        n = len(chunk["text"].split())
        if total + n > max_words:
            break
        parts.append(f"[{chunk['source']}]\n{chunk['text']}")
        total += n
    return "\n\n---\n\n".join(parts)


def build_prompt(query: str, context: str) -> str:
    """
    Assemble the final prompt string to send to an LLM.

    The instruction to say "I don't know" when the answer isn't in context
    is important — without it, most models will hallucinate an answer.

    Args:
        query: The user's question.
        context: The context block built by build_context().

    Returns:
        A ready-to-use prompt string.
    """
    return (
        "Use the context below to answer the question accurately and concisely.\n"
        "If the answer is not in the context, say so - do not make things up.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def ask(query: str, store: VectorStore, n_results: int = 5) -> dict:
    """
    Run a full retrieval-augmented query against the vector store.

    This is the main entry point for question answering. It returns a dict
    rather than just the prompt so callers can inspect the retrieved chunks,
    check sources, or log retrieval quality independently.

    Args:
        query: The user's question.
        store: An initialized VectorStore instance.
        n_results: Number of chunks to retrieve before context truncation.

    Returns:
        Dict with keys: prompt, context, sources (list), chunks (list).
    """
    chunks = store.query(query, n_results=n_results)
    context = build_context(chunks)
    prompt = build_prompt(query, context)
    sources = sorted({c["source"] for c in chunks})
    return {"prompt": prompt, "context": context, "sources": sources, "chunks": chunks}


def main():
    parser = argparse.ArgumentParser(description="RAG Document Pipeline")
    parser.add_argument("--docs_path", default="./data/docs",
                        help="Folder containing documents to index")
    parser.add_argument("--query", type=str, default=None,
                        help="Question to ask against the indexed documents")
    parser.add_argument("--index", action="store_true",
                        help="(Re)index documents before querying")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe the vector store before indexing")
    parser.add_argument("--n_results", type=int, default=5,
                        help="Number of chunks to retrieve per query")
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
        print("\n=== Sources used ===")
        print(", ".join(result["sources"]))
        print("\n=== Context snippet ===")
        print(result["context"][:400], "...\n")
        print("=== Prompt (feed this to your LLM) ===")
        print(result["prompt"][:500], "...")


if __name__ == "__main__":
    main()