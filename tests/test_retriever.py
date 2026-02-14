import hashlib
import pytest
from unittest.mock import MagicMock, patch
from local_rag.engine.retriever import Retriever, SearchResult


@pytest.fixture
def mock_embedder():
    embedder = MagicMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    return embedder


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.query.return_value = {
        "ids": [["id1", "id2"]],
        "documents": [["First chunk text", "Second chunk text"]],
        "metadatas": [[
            {"source": "a.txt", "chunk_index": 0},
            {"source": "b.txt", "chunk_index": 1},
        ]],
        "distances": [[0.1, 0.5]],
    }
    return store


def test_search_returns_results(mock_embedder, mock_store):
    retriever = Retriever(embedder=mock_embedder, store=mock_store)
    results = retriever.search("test query", top_k=2)
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].text == "First chunk text"
    assert results[0].source == "a.txt"


def test_search_filters_by_threshold(mock_embedder, mock_store):
    retriever = Retriever(embedder=mock_embedder, store=mock_store)
    results = retriever.search("test query", top_k=10, similarity_threshold=0.8)
    assert len(results) == 1
    assert results[0].source == "a.txt"


def test_search_passes_filters_to_store(mock_embedder, mock_store):
    retriever = Retriever(embedder=mock_embedder, store=mock_store)
    retriever.search("query", top_k=5, filters={"source": "a.txt"})
    mock_store.query.assert_called_once()
    call_kwargs = mock_store.query.call_args[1]
    assert call_kwargs["where"] == {"source": "a.txt"}


def test_search_calls_embedder_with_query(mock_embedder, mock_store):
    retriever = Retriever(embedder=mock_embedder, store=mock_store)
    retriever.search("my question")
    mock_embedder.embed.assert_called_once_with(["my question"])


def test_index_documents(mock_embedder, mock_store):
    mock_embedder.embed.return_value = [[0.1, 0.2], [0.3, 0.4]]
    retriever = Retriever(embedder=mock_embedder, store=mock_store)

    from local_rag.engine.chunker import Chunk
    chunks = [
        Chunk(text="chunk one", index=0, metadata={"source": "a.txt"}),
        Chunk(text="chunk two", index=1, metadata={"source": "a.txt"}),
    ]
    retriever.index_chunks(chunks)
    mock_store.upsert.assert_called_once()
    call_kwargs = mock_store.upsert.call_args[1]
    assert len(call_kwargs["ids"]) == 2
    assert len(call_kwargs["embeddings"]) == 2


def test_index_batches_large_input(mock_embedder, mock_store):
    mock_embedder.embed.side_effect = lambda texts: [[0.1] * 3] * len(texts)
    retriever = Retriever(embedder=mock_embedder, store=mock_store, batch_size=2)

    from local_rag.engine.chunker import Chunk
    chunks = [Chunk(text=f"chunk {i}", index=i, metadata={}) for i in range(5)]
    retriever.index_chunks(chunks)
    assert mock_embedder.embed.call_count == 3
