"""ChromaDB vector store with sentence-transformers embeddings."""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional


class VectorStore:
    def __init__(
        self,
        collection_name: str = "documents",
        persist_path: str = "./chroma_db",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.client = chromadb.PersistentClient(path=persist_path)
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, records: List[Dict[str, Any]], batch_size: int = 100) -> None:
        if not records:
            return
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            self.collection.add(
                documents=[r["text"] for r in batch],
                metadatas=[{"source": r["source"], "chunk_id": r["chunk_id"]} for r in batch],
                ids=[f"{r['source']}_chunk_{r['chunk_id']}" for r in batch],
            )
        print(f"Indexed {len(records)} chunks into '{self.collection.name}'.")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
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
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            embedding_function=self.embed_fn,
        )
        print("Collection reset.")