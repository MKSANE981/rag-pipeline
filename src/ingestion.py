"""
Document ingestion module — load, parse and chunk raw files.

Supports .txt, .pdf (via pdfplumber) and .json. The chunking strategy
uses a sliding window over words (not characters) so chunk boundaries
always fall on word edges. Overlap avoids losing context at chunk seams.
"""

from pathlib import Path
from typing import List, Dict, Any


def load_document(path: str) -> str:
    """
    Load a single document and return its full text as a string.

    We dispatch by file extension. PDFs are parsed page by page with
    pdfplumber, which handles multi-column layouts better than PyPDF2.
    JSON files are flattened to key: value lines so they're readable
    as plain text by the embedding model.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        The extracted text content.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            # Some pages may return None (e.g. image-only pages) — we skip those
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    elif suffix == ".json":
        import json
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return "\n".join(f"{k}: {v}" for k, v in data.items())
        return "\n".join(str(item) for item in data)

    else:
        raise ValueError(f"Unsupported format: {suffix}. Add a parser for it here.")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Split a long text into overlapping word-level chunks.

    Why word-level and not character-level? Because embedding models
    tokenize by subwords, and splitting mid-word produces garbage tokens.
    Splitting on whitespace is a safe approximation.

    The overlap parameter controls how many words are repeated between
    consecutive chunks. A value of 64 (out of 512) is ~12.5% — enough
    to preserve sentence context across boundaries without inflating the
    index too much.

    Args:
        text: The full document text.
        chunk_size: Number of words per chunk.
        overlap: Number of words repeated from the previous chunk.

    Returns:
        List of text chunks (strings).
    """
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def process_documents(
    docs_path: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> List[Dict[str, Any]]:
    """
    Walk a directory, load all supported files and return chunked records.

    Each record carries the chunk text, its source filename and its
    position index. The source and chunk_id fields are stored as metadata
    in ChromaDB, so we can filter by document or trace back answers to
    specific files.

    Unsupported files are silently skipped (with a warning) so you can
    drop a mixed folder of docs without pre-filtering.

    Args:
        docs_path: Path to a folder containing documents.
        chunk_size: Words per chunk (passed to chunk_text).
        overlap: Overlap words between chunks (passed to chunk_text).

    Returns:
        List of dicts with keys: text, source, chunk_id.
    """
    supported = {".txt", ".pdf", ".json"}
    records = []

    for path in Path(docs_path).rglob("*"):
        if path.suffix.lower() not in supported:
            continue
        try:
            text = load_document(str(path))
            for i, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
                records.append({
                    "text": chunk,
                    "source": path.name,
                    "chunk_id": i,
                })
        except Exception as e:
            print(f"[warn] Skipping {path.name}: {e}")

    print(f"Processed {len(records)} chunks from {docs_path}")
    return records