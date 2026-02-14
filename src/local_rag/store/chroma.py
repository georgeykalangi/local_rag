"""ChromaDB vector store wrapper for local RAG."""

from __future__ import annotations

import sys
from typing import Any

# Python 3.14 breaks pydantic v1's type inference (used internally by chromadb).
# Patch _set_default_and_type before importing chromadb to avoid ConfigError.
if sys.version_info >= (3, 14):
    import pydantic.v1.fields as _pf
    from pydantic.v1.fields import Undefined as _Undefined
    import pydantic.v1.errors as _errors

    def _patched_set_default_and_type(self: _pf.ModelField) -> None:
        if self.default_factory is not None:
            if self.type_ is _Undefined:
                raise _errors.ConfigError(
                    f"you need to set the type of field {self.name!r} "
                    "when using default_factory"
                )
            return

        default_value = self.get_default()

        if default_value is not None and self.type_ is _Undefined:
            self.type_ = default_value.__class__
            self.outer_type_ = self.type_
            self.annotation = self.type_

        if self.type_ is _Undefined:
            self.type_ = str
            self.outer_type_ = str
            self.allow_none = True

        if self.required is False and default_value is None:
            self.allow_none = True

    _pf.ModelField._set_default_and_type = _patched_set_default_and_type

import chromadb


class ChromaStore:
    """Wrapper around ChromaDB for vector storage and retrieval."""

    def __init__(self, persist_dir: str, collection_name: str = "default"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Add documents with embeddings to the collection."""
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        """Insert or update documents by ID."""
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict[str, Any]:
        """Query the collection by embedding similarity."""
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
    ) -> None:
        """Delete documents by IDs or metadata filter."""
        kwargs: dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        self._collection.delete(**kwargs)

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def list_collections(self) -> list[str]:
        """Return names of all collections in the client."""
        return [c.name for c in self._client.list_collections()]
