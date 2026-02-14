from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VALID_PROVIDERS = ("local", "openai", "voyage")


@dataclass
class ChunkingConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100


@dataclass
class EmbeddingsConfig:
    provider: str = "local"
    local_model: str = "all-MiniLM-L6-v2"
    openai_model: str = "text-embedding-3-small"
    voyage_model: str = "voyage-3"
    batch_size: int = 64


@dataclass
class IndexingConfig:
    supported_extensions: list[str] = field(
        default_factory=lambda: [".pdf", ".md", ".txt", ".py", ".js", ".ts", ".docx", ".rst"]
    )
    max_file_size_mb: int = 10
    ignore_file: str = ".ragignore"
    recursive: bool = True


@dataclass
class RetrievalConfig:
    default_top_k: int = 5
    similarity_threshold: float = 0.3


@dataclass
class StorageConfig:
    chroma_persist_dir: str = "~/.local_rag/chroma_db"
    collection_name: str = "default"


@dataclass
class RAGConfig:
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embeddings: EmbeddingsConfig = field(default_factory=EmbeddingsConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def load_config(path: str | None = None) -> RAGConfig:
    """Load RAG config from a YAML file. Falls back to LOCAL_RAG_CONFIG env var."""
    if path is None:
        path = os.environ.get("LOCAL_RAG_CONFIG")
    if path is None:
        raise FileNotFoundError("No config path provided and LOCAL_RAG_CONFIG not set")

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    config = RAGConfig(
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        embeddings=EmbeddingsConfig(**raw.get("embeddings", {})),
        indexing=IndexingConfig(**raw.get("indexing", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        storage=StorageConfig(**raw.get("storage", {})),
    )

    # Validate provider
    if config.embeddings.provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Invalid embedding provider '{config.embeddings.provider}'. "
            f"Must be one of: {VALID_PROVIDERS}"
        )

    # Expand ~ in paths
    config.storage.chroma_persist_dir = str(
        Path(config.storage.chroma_persist_dir).expanduser()
    )

    return config
