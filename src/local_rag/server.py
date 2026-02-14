"""MCP server for local RAG system."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from local_rag.config import RAGConfig, load_config
from local_rag.engine.chunker import chunk_text
from local_rag.engine.embedder import create_embedder
from local_rag.engine.loader import discover_files, load_document
from local_rag.engine.retriever import Retriever
from local_rag.store.chroma import ChromaStore

# Initialize MCP server
mcp = FastMCP("local-rag")

# Global state
_config: RAGConfig | None = None
_retriever: Retriever | None = None
_store: ChromaStore | None = None


def _ensure_initialized() -> tuple[RAGConfig, Retriever, ChromaStore]:
    """Ensure the RAG system is initialized, loading config if needed."""
    global _config, _retriever, _store

    if _config is None:
        # Try to load from env var or use defaults
        config_path = os.environ.get("LOCAL_RAG_CONFIG")
        if config_path:
            _config = load_config(config_path)
        else:
            # Use defaults
            _config = RAGConfig()

    if _store is None:
        _store = ChromaStore(
            persist_dir=_config.storage.chroma_persist_dir,
            collection_name=_config.storage.collection_name,
        )

    if _retriever is None:
        # Select the model based on the provider
        if _config.embeddings.provider == "local":
            model = _config.embeddings.local_model
        elif _config.embeddings.provider == "openai":
            model = _config.embeddings.openai_model
        elif _config.embeddings.provider == "voyage":
            model = _config.embeddings.voyage_model
        else:
            model = _config.embeddings.local_model

        embedder = create_embedder(
            provider=_config.embeddings.provider,
            model=model,
        )
        _retriever = Retriever(
            embedder=embedder,
            store=_store,
            batch_size=_config.embeddings.batch_size,
        )

    return _config, _retriever, _store


# Core functions (can be tested directly)


def search(
    query: str,
    top_k: int = 5,
    similarity_threshold: float = 0.0,
) -> dict:
    """Search indexed documents using semantic similarity."""
    _, retriever, _ = _ensure_initialized()

    results = retriever.search(
        query=query,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )

    return {
        "query": query,
        "num_results": len(results),
        "results": [
            {
                "text": r.text,
                "source": r.source,
                "score": round(r.score, 4),
                "metadata": r.metadata,
            }
            for r in results
        ],
    }


@mcp.tool()
def rag_search(
    query: Annotated[str, Field(description="Search query text")],
    top_k: Annotated[int, Field(description="Number of results to return", ge=1, le=50)] = 5,
    similarity_threshold: Annotated[
        float, Field(description="Minimum similarity score (0.0-1.0)", ge=0.0, le=1.0)
    ] = 0.0,
) -> dict:
    """
    Search indexed documents using semantic similarity.

    Returns relevant text chunks with their sources and similarity scores.
    Higher scores indicate better matches.
    """
    return search(query, top_k, similarity_threshold)


def index_file(file_path: str) -> dict:
    """Index a single file into the RAG system."""
    config, retriever, _ = _ensure_initialized()

    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    if not path.is_file():
        return {"success": False, "error": f"Not a file: {file_path}"}

    try:
        # Load document
        doc = load_document(path)

        # Chunk the document
        chunks = chunk_text(
            text=doc.text,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
            min_chunk_size=config.chunking.min_chunk_size,
            metadata={
                "source": doc.source,
                "file_type": doc.file_type,
                **doc.metadata,
            },
        )

        if not chunks:
            return {
                "success": False,
                "error": "Document produced no chunks (possibly empty)",
            }

        # Index chunks
        retriever.index_chunks(chunks)

        return {
            "success": True,
            "file": str(path),
            "chunks_indexed": len(chunks),
            "file_type": doc.file_type,
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to index file: {str(e)}"}


@mcp.tool()
def rag_index_file(
    file_path: Annotated[str, Field(description="Absolute path to the file to index")],
) -> dict:
    """
    Index a single file into the RAG system.

    Supports: PDF, DOCX, Markdown, text files, and common code files.
    The file will be loaded, chunked, embedded, and stored for future searches.
    """
    return index_file(file_path)


def index_directory(directory_path: str, recursive: bool = True) -> dict:
    """Index all supported files in a directory."""
    config, retriever, _ = _ensure_initialized()

    dir_path = Path(directory_path).expanduser().resolve()
    if not dir_path.exists():
        return {"success": False, "error": f"Directory not found: {directory_path}"}

    if not dir_path.is_dir():
        return {"success": False, "error": f"Not a directory: {directory_path}"}

    try:
        # Discover files
        files = discover_files(
            folder=str(dir_path),
            extensions=config.indexing.supported_extensions,
            recursive=recursive,
            max_size_mb=config.indexing.max_file_size_mb,
            ignore_file=config.indexing.ignore_file,
        )

        if not files:
            return {
                "success": True,
                "message": "No files found matching criteria",
                "files_indexed": 0,
                "total_chunks": 0,
            }

        indexed_files = 0
        total_chunks = 0
        errors = []

        for file_path in files:
            try:
                doc = load_document(file_path)
                chunks = chunk_text(
                    text=doc.text,
                    chunk_size=config.chunking.chunk_size,
                    chunk_overlap=config.chunking.chunk_overlap,
                    min_chunk_size=config.chunking.min_chunk_size,
                    metadata={
                        "source": doc.source,
                        "file_type": doc.file_type,
                        **doc.metadata,
                    },
                )

                if chunks:
                    retriever.index_chunks(chunks)
                    indexed_files += 1
                    total_chunks += len(chunks)

            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})

        result = {
            "success": True,
            "directory": str(dir_path),
            "files_discovered": len(files),
            "files_indexed": indexed_files,
            "total_chunks": total_chunks,
        }

        if errors:
            result["errors"] = errors[:10]  # Limit error list
            if len(errors) > 10:
                result["additional_errors"] = len(errors) - 10

        return result

    except Exception as e:
        return {"success": False, "error": f"Failed to index directory: {str(e)}"}


@mcp.tool()
def rag_index_directory(
    directory_path: Annotated[str, Field(description="Absolute path to the directory to index")],
    recursive: Annotated[bool, Field(description="Recursively index subdirectories")] = True,
) -> dict:
    """
    Index all supported files in a directory.

    Discovers and indexes all files matching supported extensions (PDF, DOCX, MD, TXT, code files).
    Respects .ragignore file if present. Returns summary of indexing results.
    """
    return index_directory(directory_path, recursive)


def list_collections() -> dict:
    """List all available ChromaDB collections."""
    try:
        _, _, store = _ensure_initialized()
        collection_names = store.list_collections()

        return {
            "collections": collection_names,
            "total_collections": len(collection_names),
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to list collections: {str(e)}"}


@mcp.tool()
def rag_list_collections() -> dict:
    """
    List all available ChromaDB collections.

    Returns collection names and their document counts.
    """
    return list_collections()


def get_stats() -> dict:
    """Get statistics about the current RAG collection."""
    try:
        config, _, store = _ensure_initialized()

        count = store.count()

        return {
            "collection_name": config.storage.collection_name,
            "document_count": count,
            "config": {
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "embedding_provider": config.embeddings.provider,
                "top_k_default": config.retrieval.default_top_k,
                "similarity_threshold": config.retrieval.similarity_threshold,
            },
        }

    except Exception as e:
        return {"success": False, "error": f"Failed to get stats: {str(e)}"}


@mcp.tool()
def rag_get_stats() -> dict:
    """
    Get statistics about the current RAG collection.

    Returns document count, collection name, and configuration details.
    """
    return get_stats()


def configure(
    collection_name: str | None = None,
    embedding_provider: str | None = None,
) -> dict:
    """Reconfigure the RAG system."""
    global _config, _retriever, _store

    try:
        config, _, _ = _ensure_initialized()

        changes = []

        if collection_name and collection_name != config.storage.collection_name:
            config.storage.collection_name = collection_name
            _store = None  # Force re-initialization
            _retriever = None
            changes.append(f"collection_name -> {collection_name}")

        if embedding_provider and embedding_provider != config.embeddings.provider:
            if embedding_provider not in ("local", "openai", "voyage"):
                return {
                    "success": False,
                    "error": f"Invalid provider: {embedding_provider}. Must be: local, openai, or voyage",
                }
            config.embeddings.provider = embedding_provider
            _retriever = None  # Force re-initialization
            changes.append(f"embedding_provider -> {embedding_provider}")

        if changes:
            # Re-initialize with new config
            _ensure_initialized()
            return {
                "success": True,
                "message": "Configuration updated",
                "changes": changes,
                "note": "If you changed the embedding provider, you should re-index your documents",
            }
        else:
            return {
                "success": True,
                "message": "No changes made",
            }

    except Exception as e:
        return {"success": False, "error": f"Failed to configure: {str(e)}"}


@mcp.tool()
def rag_configure(
    collection_name: Annotated[
        str | None, Field(description="Name of the collection to use")
    ] = None,
    embedding_provider: Annotated[
        str | None, Field(description="Embedding provider: local, openai, or voyage")
    ] = None,
) -> dict:
    """
    Reconfigure the RAG system.

    Allows switching collections or embedding providers at runtime.
    Note: Changing providers requires re-indexing documents.
    """
    return configure(collection_name, embedding_provider)
