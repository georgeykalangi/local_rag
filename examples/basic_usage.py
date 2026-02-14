#!/usr/bin/env python3
"""
Example: Basic usage of the local RAG system.

This script demonstrates how to use the RAG system programmatically
to index documents and perform semantic search.
"""

from pathlib import Path

from local_rag.config import RAGConfig
from local_rag.engine.chunker import chunk_text
from local_rag.engine.embedder import create_embedder
from local_rag.engine.loader import discover_files, load_document
from local_rag.engine.retriever import Retriever
from local_rag.store.chroma import ChromaStore


def main():
    # Initialize configuration
    config = RAGConfig()
    print(f"Using embedding provider: {config.embeddings.provider}")
    print(f"Chunk size: {config.chunking.chunk_size}")
    print()

    # Initialize vector store
    store = ChromaStore(
        persist_dir=config.storage.chroma_persist_dir,
        collection_name="example_collection",
    )
    print(f"ChromaDB initialized")

    # Initialize embedder
    if config.embeddings.provider == "local":
        model = config.embeddings.local_model
    elif config.embeddings.provider == "openai":
        model = config.embeddings.openai_model
    else:
        model = config.embeddings.voyage_model

    embedder = create_embedder(provider=config.embeddings.provider, model=model)
    print(f"Embedder initialized: {config.embeddings.provider}/{model}")

    # Initialize retriever
    retriever = Retriever(
        embedder=embedder,
        store=store,
        batch_size=config.embeddings.batch_size,
    )
    print("Retriever initialized\n")

    # Create sample documents
    print("Creating sample documents...")
    sample_dir = Path("example_data")
    sample_dir.mkdir(exist_ok=True)

    (sample_dir / "python_guide.md").write_text("""
# Python Programming Guide

Python is a versatile language used for web dev, data science, and automation.

## Key Features
- Easy to learn and read
- Extensive standard library
- Large ecosystem of packages
- Great for beginners and experts
""")

    print(f"Created sample documents in {sample_dir}/\n")

    # Index documents
    print("Indexing documents...")
    files = discover_files(
        folder=str(sample_dir),
        extensions=[".md", ".txt"],
        recursive=True,
        max_size_mb=10,
    )

    for file_path in files:
        doc = load_document(file_path)
        chunks = chunk_text(
            text=doc.text,
            chunk_size=500,
            chunk_overlap=50,
            min_chunk_size=50,
            metadata={"source": doc.source},
        )
        retriever.index_chunks(chunks)
        print(f"  Indexed {file_path.name}: {len(chunks)} chunks")

    # Search
    print("\nSearching...")
    results = retriever.search(query="Python programming features", top_k=3)
    print(f"Found {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. Score: {r.score:.3f}")
        print(f"   {r.text[:100]}...")


if __name__ == "__main__":
    main()
