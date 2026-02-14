import os
import tempfile
import yaml
import pytest
from local_rag.config import load_config, RAGConfig


def _write_config(tmp_path, overrides=None):
    """Helper to write a config.yaml with optional overrides."""
    base = {
        "chunking": {"chunk_size": 500, "chunk_overlap": 100, "min_chunk_size": 50},
        "embeddings": {
            "provider": "local",
            "local_model": "all-MiniLM-L6-v2",
            "openai_model": "text-embedding-3-small",
            "voyage_model": "voyage-3",
            "batch_size": 32,
        },
        "indexing": {
            "supported_extensions": [".txt", ".md"],
            "max_file_size_mb": 5,
            "ignore_file": ".ragignore",
            "recursive": True,
        },
        "retrieval": {"default_top_k": 3, "similarity_threshold": 0.4},
        "storage": {
            "chroma_persist_dir": str(tmp_path / "chroma"),
            "collection_name": "test",
        },
    }
    if overrides:
        for section, values in overrides.items():
            base.setdefault(section, {}).update(values)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(base))
    return str(config_path)


def test_load_config_returns_rag_config(tmp_path):
    path = _write_config(tmp_path)
    config = load_config(path)
    assert isinstance(config, RAGConfig)
    assert config.chunking.chunk_size == 500
    assert config.chunking.chunk_overlap == 100
    assert config.embeddings.provider == "local"
    assert config.retrieval.default_top_k == 3


def test_load_config_expands_home_dir(tmp_path):
    path = _write_config(
        tmp_path, {"storage": {"chroma_persist_dir": "~/test_chroma"}}
    )
    config = load_config(path)
    assert "~" not in config.storage.chroma_persist_dir
    assert os.path.expanduser("~") in config.storage.chroma_persist_dir


def test_load_config_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/config.yaml")


def test_load_config_from_env_var(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    monkeypatch.setenv("LOCAL_RAG_CONFIG", path)
    config = load_config()
    assert config.chunking.chunk_size == 500


def test_load_config_validates_provider(tmp_path):
    path = _write_config(tmp_path, {"embeddings": {"provider": "invalid"}})
    with pytest.raises(ValueError):
        load_config(path)
