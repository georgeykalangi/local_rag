"""Tests for the MCP server."""

from __future__ import annotations

import pytest

from local_rag.server import (
    configure,
    get_stats,
    index_file,
    list_collections,
    search,
)


def test_get_stats_initializes_system():
    """Test that get_stats initializes the system."""
    result = get_stats()
    assert "collection_name" in result
    assert "document_count" in result
    assert "config" in result


def test_list_collections():
    """Test listing collections."""
    result = list_collections()
    assert "collections" in result
    assert "total_collections" in result


def test_search_returns_correct_format():
    """Test that search returns the correct format."""
    result = search(query="test query", top_k=5)
    assert result["query"] == "test query"
    assert "num_results" in result
    assert "results" in result
    assert isinstance(result["results"], list)


def test_index_file_not_found():
    """Test indexing a non-existent file."""
    result = index_file("/nonexistent/file.txt")
    assert result["success"] is False
    assert "not found" in result["error"].lower()


def test_index_file_success(tmp_path):
    """Test successfully indexing a file."""
    # Create a test file
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test document with some content for indexing.")

    result = index_file(str(test_file))
    assert result["success"] is True
    assert result["chunks_indexed"] >= 1
    assert result["file_type"] == ".txt"


def test_search_after_indexing(tmp_path):
    """Test searching after indexing a document."""
    # Create and index a test file
    test_file = tmp_path / "search_test.txt"
    content = "The quick brown fox jumps over the lazy dog. " * 10
    test_file.write_text(content)

    index_result = index_file(str(test_file))
    assert index_result["success"] is True

    # Search for content
    search_result = search(query="quick brown fox", top_k=3)
    assert search_result["num_results"] > 0
    assert "fox" in search_result["results"][0]["text"].lower()


def test_configure_invalid_provider():
    """Test configuring with invalid provider."""
    result = configure(embedding_provider="invalid")
    assert result["success"] is False
    assert "invalid" in result["error"].lower()


def test_configure_collection():
    """Test changing collection name."""
    result = configure(collection_name="test_collection")
    assert result["success"] is True
    assert "collection_name -> test_collection" in result["changes"]

    # Verify the change
    stats = get_stats()
    assert stats["collection_name"] == "test_collection"

    # Change back to default
    configure(collection_name="default")
