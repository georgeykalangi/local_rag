from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    """Base class for embedding providers."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return vectors."""
        ...


class LocalEmbedder(Embedder):
    """Embedding provider using sentence-transformers locally."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]


class OpenAIEmbedder(Embedder):
    """Embedding provider using the OpenAI API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self._client = OpenAI()
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]


class VoyageEmbedder(Embedder):
    """Embedding provider using the Voyage AI API."""

    def __init__(self, model: str = "voyage-3"):
        import voyageai

        self._client = voyageai.Client()
        self._model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        result = self._client.embed(texts, model=self._model)
        return result.embeddings


def create_embedder(provider: str, model: str) -> Embedder:
    """Factory function to create the appropriate embedder."""
    if provider == "local":
        return LocalEmbedder(model=model)
    elif provider == "openai":
        return OpenAIEmbedder(model=model)
    elif provider == "voyage":
        return VoyageEmbedder(model=model)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
