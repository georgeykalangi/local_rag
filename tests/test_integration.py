"""Integration tests for the complete RAG pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from local_rag.config import RAGConfig
from local_rag.engine.chunker import chunk_text
from local_rag.engine.embedder import create_embedder
from local_rag.engine.loader import discover_files, load_document
from local_rag.engine.retriever import Retriever
from local_rag.server import (
    configure,
    get_stats,
    index_directory,
    index_file,
    search,
)
from local_rag.store.chroma import ChromaStore


@pytest.fixture
def temp_docs_dir(tmp_path):
    """Create a temporary directory with sample documents."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    # Create various document types
    (docs_dir / "python_tutorial.md").write_text("""
# Python Programming Tutorial

## Introduction
Python is a high-level programming language known for its simplicity and readability.

## Variables and Data Types
In Python, you can create variables without declaring their type:
- Strings: `name = "Alice"`
- Integers: `age = 25`
- Floats: `price = 19.99`

## Functions
Functions in Python are defined using the `def` keyword:
```python
def greet(name):
    return f"Hello, {name}!"
```

## Conclusion
Python is widely used for web development, data science, and automation.
""")

    (docs_dir / "algorithms.txt").write_text("""
Common Sorting Algorithms

1. Bubble Sort
   Time Complexity: O(n²)
   Space Complexity: O(1)
   Description: Repeatedly steps through the list, compares adjacent elements and swaps them if needed.

2. Quick Sort
   Time Complexity: O(n log n) average, O(n²) worst
   Space Complexity: O(log n)
   Description: Divide-and-conquer algorithm that picks a pivot and partitions the array.

3. Merge Sort
   Time Complexity: O(n log n)
   Space Complexity: O(n)
   Description: Divides array into halves, sorts them and merges them back together.
""")

    (docs_dir / "machine_learning.md").write_text("""
# Machine Learning Basics

## Supervised Learning
Supervised learning uses labeled data to train models. Common algorithms:
- Linear Regression: Predicts continuous values
- Logistic Regression: Predicts binary outcomes
- Decision Trees: Uses tree-like model of decisions
- Neural Networks: Mimics human brain structure

## Unsupervised Learning
Works with unlabeled data to find patterns:
- K-Means Clustering: Groups similar data points
- Principal Component Analysis (PCA): Reduces dimensionality
- Autoencoders: Neural networks for feature learning

## Key Concepts
- Training Data: Dataset used to train the model
- Test Data: Dataset used to evaluate the model
- Overfitting: Model performs well on training but poorly on new data
- Underfitting: Model is too simple to capture patterns
""")

    (docs_dir / "api_docs.py").write_text("""
\"\"\"
API Documentation for User Management

This module provides functions for managing user accounts.
\"\"\"

def create_user(username: str, email: str, password: str) -> dict:
    \"\"\"
    Create a new user account.

    Args:
        username: Unique username for the account
        email: User's email address
        password: Account password (will be hashed)

    Returns:
        Dictionary containing user_id and confirmation message
    \"\"\"
    pass

def get_user(user_id: int) -> dict:
    \"\"\"
    Retrieve user information by ID.

    Args:
        user_id: The unique identifier for the user

    Returns:
        Dictionary with user details
    \"\"\"
    pass

def update_user(user_id: int, **kwargs) -> bool:
    \"\"\"
    Update user information.

    Args:
        user_id: The user to update
        **kwargs: Fields to update (username, email, etc.)

    Returns:
        True if update succeeded
    \"\"\"
    pass
""")

    return docs_dir


def test_full_pipeline_with_loader_and_retriever(tmp_path):
    """Test the complete RAG pipeline from loading to searching."""
    # Create test documents
    test_file = tmp_path / "test_doc.txt"
    test_file.write_text("""
    The cat sat on the mat.
    The dog ran in the park.
    Birds fly in the sky.
    Fish swim in the ocean.
    """ * 10)  # Make it long enough to chunk

    # Initialize components
    config = RAGConfig()
    store = ChromaStore(
        persist_dir=str(tmp_path / "chroma_test"),
        collection_name="test_pipeline",
    )
    embedder = create_embedder(provider="local", model="all-MiniLM-L6-v2")
    retriever = Retriever(embedder=embedder, store=store, batch_size=32)

    # Load and process document
    doc = load_document(test_file)
    assert doc.text
    assert doc.source == str(test_file)

    # Chunk the document
    chunks = chunk_text(
        text=doc.text,
        chunk_size=100,
        chunk_overlap=20,
        min_chunk_size=50,
        metadata={"source": doc.source},
    )
    assert len(chunks) > 0

    # Index chunks
    retriever.index_chunks(chunks)

    # Search
    results = retriever.search(query="cat on mat", top_k=3)
    assert len(results) > 0
    assert "cat" in results[0].text.lower() or "mat" in results[0].text.lower()


def test_mcp_server_full_workflow(temp_docs_dir):
    """Test the complete MCP server workflow."""
    # Configure to use a test collection
    config_result = configure(collection_name="integration_test")
    assert config_result["success"] is True

    # Get initial stats
    stats = get_stats()
    assert stats["collection_name"] == "integration_test"
    initial_count = stats["document_count"]

    # Index the entire directory
    index_result = index_directory(str(temp_docs_dir), recursive=True)
    assert index_result["success"] is True
    assert index_result["files_indexed"] == 4
    assert index_result["total_chunks"] > 0

    # Verify stats updated
    stats = get_stats()
    assert stats["document_count"] > initial_count

    # Search for Python-related content
    python_results = search(query="Python programming language", top_k=3)
    assert python_results["num_results"] > 0
    # Should find content from python_tutorial.md
    assert any("python" in r["text"].lower() for r in python_results["results"])

    # Search for algorithm content
    algo_results = search(query="sorting algorithms time complexity", top_k=3)
    assert algo_results["num_results"] > 0
    # Should find content from algorithms.txt
    assert any(
        "sort" in r["text"].lower() or "complexity" in r["text"].lower()
        for r in algo_results["results"]
    )

    # Search for ML content
    ml_results = search(query="machine learning supervised", top_k=3)
    assert ml_results["num_results"] > 0
    assert any(
        "learning" in r["text"].lower() or "supervised" in r["text"].lower()
        for r in ml_results["results"]
    )

    # Test similarity threshold
    filtered_results = search(
        query="Python programming",
        top_k=5,
        similarity_threshold=0.5,
    )
    # All results should have score >= 0.5
    assert all(r["score"] >= 0.5 for r in filtered_results["results"])


def test_mcp_server_file_indexing(tmp_path):
    """Test indexing individual files through MCP server."""
    # Create test files
    file1 = tmp_path / "doc1.txt"
    file1.write_text("Artificial intelligence is transforming technology. " * 20)

    file2 = tmp_path / "doc2.md"
    file2.write_text("# Blockchain Technology\n\nBlockchain is a distributed ledger. " * 20)

    # Configure test collection
    configure(collection_name="file_test")

    # Index files individually
    result1 = index_file(str(file1))
    assert result1["success"] is True
    assert result1["chunks_indexed"] > 0

    result2 = index_file(str(file2))
    assert result2["success"] is True
    assert result2["chunks_indexed"] > 0

    # Search for content from each file
    ai_results = search(query="artificial intelligence", top_k=2)
    assert ai_results["num_results"] > 0

    blockchain_results = search(query="blockchain distributed ledger", top_k=2)
    assert blockchain_results["num_results"] > 0


def test_discover_and_index_workflow(temp_docs_dir):
    """Test discovering files and indexing them."""
    # Discover files
    files = discover_files(
        folder=str(temp_docs_dir),
        extensions=[".md", ".txt", ".py"],
        recursive=True,
        max_size_mb=10,
    )
    assert len(files) == 4

    # Load and verify each file
    for file_path in files:
        doc = load_document(file_path)
        assert doc.text
        assert doc.source == str(file_path)
        assert doc.file_type in [".md", ".txt", ".py"]


def test_collection_isolation():
    """Test that different collections are isolated."""
    # Create two collections with different data
    configure(collection_name="collection_a")
    index_file_result = index_file("/dev/null")  # Will fail but that's ok for setup

    # Add to collection A
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Collection A specific content about dogs. " * 20)
        f.flush()
        file_a = f.name

    configure(collection_name="collection_a")
    index_file(file_a)
    results_a = search(query="dogs", top_k=3)

    # Switch to collection B and add different content
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Collection B specific content about cats. " * 20)
        f.flush()
        file_b = f.name

    configure(collection_name="collection_b")
    index_file(file_b)
    results_b = search(query="cats", top_k=3)

    # Verify isolation
    assert results_b["num_results"] > 0

    # Search for dogs in collection B should not find the collection A content
    dogs_in_b = search(query="dogs", top_k=3)
    # Collection B shouldn't have dog content
    if dogs_in_b["num_results"] > 0:
        # If we get results, they should have lower scores than cats
        cats_in_b = search(query="cats", top_k=3)
        if cats_in_b["num_results"] > 0:
            assert cats_in_b["results"][0]["score"] > dogs_in_b["results"][0]["score"]

    # Clean up
    Path(file_a).unlink()
    Path(file_b).unlink()


def test_error_handling():
    """Test error handling in MCP server."""
    # Test non-existent file
    result = index_file("/nonexistent/path/to/file.txt")
    assert result["success"] is False
    assert "not found" in result["error"].lower()

    # Test non-existent directory
    result = index_directory("/nonexistent/directory")
    assert result["success"] is False
    assert "not found" in result["error"].lower()

    # Test invalid provider
    result = configure(embedding_provider="invalid_provider")
    assert result["success"] is False
    assert "invalid" in result["error"].lower()
