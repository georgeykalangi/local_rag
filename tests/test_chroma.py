import tempfile
import pytest
from local_rag.store.chroma import ChromaStore


@pytest.fixture
def store(tmp_path):
    return ChromaStore(persist_dir=str(tmp_path / "chroma"), collection_name="test")


def test_store_initializes_collection(store):
    assert store.count() == 0


def test_add_and_count(store):
    store.add(
        ids=["doc1", "doc2"],
        documents=["First document", "Second document"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )
    assert store.count() == 2


def test_query_returns_results(store):
    store.add(
        ids=["doc1", "doc2", "doc3"],
        documents=["Python is great", "Java is verbose", "Rust is fast"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.9, 0.1, 0.0]],
        metadatas=[{"lang": "python"}, {"lang": "java"}, {"lang": "rust"}],
    )
    results = store.query(query_embeddings=[[1.0, 0.0, 0.0]], top_k=2)
    assert len(results["ids"][0]) == 2
    assert "doc1" in results["ids"][0]


def test_query_with_filter(store):
    store.add(
        ids=["doc1", "doc2"],
        documents=["Python code", "Java code"],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
        metadatas=[{"lang": "python"}, {"lang": "java"}],
    )
    results = store.query(
        query_embeddings=[[0.5, 0.5]], top_k=10,
        where={"lang": "python"},
    )
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "doc1"


def test_delete_by_ids(store):
    store.add(
        ids=["doc1", "doc2"],
        documents=["First", "Second"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"type": "text"}, {"type": "text"}],
    )
    store.delete(ids=["doc1"])
    assert store.count() == 1


def test_delete_by_filter(store):
    store.add(
        ids=["doc1", "doc2"],
        documents=["First", "Second"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )
    store.delete(where={"source": "a.txt"})
    assert store.count() == 1


def test_list_collections(store):
    collections = store.list_collections()
    assert "test" in collections


def test_upsert_overwrites(store):
    store.add(
        ids=["doc1"],
        documents=["Original"],
        embeddings=[[0.1, 0.2]],
        metadatas=[{"v": "1"}],
    )
    store.upsert(
        ids=["doc1"],
        documents=["Updated"],
        embeddings=[[0.3, 0.4]],
        metadatas=[{"v": "2"}],
    )
    assert store.count() == 1
    results = store.query(query_embeddings=[[0.3, 0.4]], top_k=1)
    assert results["documents"][0][0] == "Updated"
