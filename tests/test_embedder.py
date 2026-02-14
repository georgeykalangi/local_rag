import pytest
from unittest.mock import patch, MagicMock
from local_rag.engine.embedder import (
    create_embedder,
    LocalEmbedder,
)


def test_create_embedder_local():
    embedder = create_embedder(provider="local", model="all-MiniLM-L6-v2")
    assert isinstance(embedder, LocalEmbedder)


def test_create_embedder_openai():
    mock_openai_mod = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        from local_rag.engine.embedder import OpenAIEmbedder

        embedder = create_embedder(provider="openai", model="text-embedding-3-small")
        assert isinstance(embedder, OpenAIEmbedder)


def test_create_embedder_voyage():
    mock_voyage_mod = MagicMock()
    with patch.dict("sys.modules", {"voyageai": mock_voyage_mod}):
        from local_rag.engine.embedder import VoyageEmbedder

        embedder = create_embedder(provider="voyage", model="voyage-3")
        assert isinstance(embedder, VoyageEmbedder)


def test_create_embedder_invalid():
    with pytest.raises(ValueError, match="Unknown"):
        create_embedder(provider="bad", model="x")


def test_local_embedder_returns_vectors():
    embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
    vectors = embedder.embed(["Hello world", "Test sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert all(isinstance(v, float) for v in vectors[0])


def test_local_embedder_single_text():
    embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
    vectors = embedder.embed(["Single text"])
    assert len(vectors) == 1


def test_local_embedder_empty_list():
    embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
    vectors = embedder.embed([])
    assert vectors == []


def test_openai_embedder_calls_api():
    mock_openai_mod = MagicMock()
    with patch.dict("sys.modules", {"openai": mock_openai_mod}):
        from local_rag.engine.embedder import OpenAIEmbedder

        embedder = OpenAIEmbedder(model="text-embedding-3-small")
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3]),
            MagicMock(embedding=[0.4, 0.5, 0.6]),
        ]
        with patch.object(embedder, "_client") as mock_client:
            mock_client.embeddings.create.return_value = mock_response
            vectors = embedder.embed(["text1", "text2"])
        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2, 0.3]


def test_voyage_embedder_calls_api():
    mock_voyage_mod = MagicMock()
    with patch.dict("sys.modules", {"voyageai": mock_voyage_mod}):
        from local_rag.engine.embedder import VoyageEmbedder

        embedder = VoyageEmbedder(model="voyage-3")
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
        with patch.object(embedder, "_client") as mock_client:
            mock_client.embed.return_value = mock_result
            vectors = embedder.embed(["text1", "text2"])
        assert len(vectors) == 2
        assert vectors[0] == [0.1, 0.2]
