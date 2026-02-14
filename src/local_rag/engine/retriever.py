from __future__ import annotations

import hashlib
from dataclasses import dataclass

from local_rag.engine.chunker import Chunk
from local_rag.engine.embedder import Embedder
from local_rag.store.chroma import ChromaStore


@dataclass
class SearchResult:
    text: str
    source: str
    score: float
    metadata: dict


class Retriever:
    """Orchestrates embedding and searching against the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store: ChromaStore,
        batch_size: int = 64,
    ):
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_embedding = self._embedder.embed([query])
        raw = self._store.query(
            query_embeddings=query_embedding,
            top_k=top_k,
            where=filters,
        )

        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i]
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue
            results.append(SearchResult(
                text=raw["documents"][0][i],
                source=raw["metadatas"][0][i].get("source", ""),
                score=similarity,
                metadata=raw["metadatas"][0][i],
            ))

        return results

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into the store in batches."""
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start:start + self._batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed(texts)
            ids = [self._chunk_id(c) for c in batch]
            metadatas = [c.metadata for c in batch]

            self._store.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        source = chunk.metadata.get("source", "")
        content = f"{source}::{chunk.index}::{chunk.text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
