"""Document loading, parsing and chunking."""

from pathlib import Path
from typing import List, Dict, Any


def load_document(path: str) -> str:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        import pdfplumber
        with pdfplumber.open(path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif suffix == ".json":
        import json
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            return "\n".join(f"{k}: {v}" for k, v in data.items())
        return "\n".join(str(item) for item in data)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Split text into overlapping word-level chunks."""
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
    """Load and chunk all documents in a directory."""
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