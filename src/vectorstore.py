"""
Vector store module — embedding and semantic search via ChromaDB.

We use sentence-transformers (all-MiniLM-L6-v2 by default) because it
strikes a good balance between speed and quality for English text.
ChromaDB handles persistence, HNSW indexing and cosine similarity search
out of the box, which keeps this module simple.

If you need to swap in a different embedding model (e.g. multilingual),
just change model_name in the constructor — everything else stays the same.
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional


class VectorStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    Handles embedding, indexing and retrieval in one place so the rest
    of the pipeline doesn't need to know about ChromaDB internals.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_path: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        """
        Initialize the vector store.

        Args:
            collection_name: Name of the ChromaDB collection. You can
                have multiple collections (one per corpus) in the same
                persist_path.
            persist_path: Directory where ChromaDB writes its data.
                Gets created automatically if it doesn't exist.
            model_name: HuggingFace sentence-transformers model to use
                for embedding. all-MiniLM-L6-v2 is fast (~80ms/batch)
                and good enough for most English corpora.
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            # cosine similarity works better than L2 for sentence embeddings
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, records: List[Dict[str, Any]], batch_size: int = 100) -> None:
        """
        Embed and index a list of chunked document records.

        We batch inserts to avoid memory spikes when indexing large corpora.
        ChromaDB deduplicates by ID, so re-indexing the same file won't
        create duplicate chunks — it will just overwrite them.

        Args:
            records: Output of ingestion.process_documents().
            batch_size: Number of chunks to embed and insert at once.
        """
        if not records:
            return
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.collection.add(
                documents=[r["text"] for r in batch],
                metadatas=[{"source": r["source"], "chunk_id": r["chunk_id"]} for r in batch],
                # IDs must be unique strings — we combine source + chunk index
                ids=[f"{r['source']}_chunk_{r['chunk_id']}" for r in batch],
            )
        print(f"Indexed {len(records)} chunks into '{self.collection.name}'.")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the top-k most semantically similar chunks.

        The query is embedded with the same model used during indexing,
        then ChromaDB performs an approximate nearest-neighbor search
        (HNSW) over the stored embeddings.

        Args:
            query_text: The user's question or search string.
            n_results: Number of chunks to return.
            source_filter: If provided, restrict results to chunks from
                this specific source file (exact match on the 'source'
                metadata field).

        Returns:
            List of dicts with keys: text, source, chunk_id, distance.
            Lower distance = more similar (cosine distance in [0, 2]).
        """
        where = {"source": source_filter} if source_filter else None
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )
        return [
            {
                "text": doc,
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def count(self) -> int:
        """Return the total number of indexed chunks."""
        return self.collection.count()

    def reset(self) -> None:
        """
        Delete and recreate the collection from scratch.

        Use this when you want to re-index everything with a different
        chunking strategy or a new document set, without leftover data.
        """
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embed_fn,
        )
        print("Collection reset.")