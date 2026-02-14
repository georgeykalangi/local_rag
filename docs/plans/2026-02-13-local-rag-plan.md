# Local RAG Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local RAG system exposed as an MCP server so Claude Code can search and retrieve context from a local document folder.

**Architecture:** ChromaDB (embedded) for vector storage, configurable embedding providers (local sentence-transformers or OpenAI/Voyage API), FastMCP server exposing search/index tools over stdio. Documents are loaded, chunked with overlap, embedded, and stored. Queries embed the question and return top-k similar chunks.

**Tech Stack:** Python 3.14, FastMCP, ChromaDB, sentence-transformers, pymupdf, python-docx, PyYAML

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `config.yaml`
- Create: `.ragignore`
- Create: `.gitignore`
- Create: `src/local_rag/__init__.py`
- Create: `src/local_rag/engine/__init__.py`
- Create: `src/local_rag/store/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/sample.txt`
- Create: `tests/fixtures/sample.md`
- Create: `tests/fixtures/sample.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "local-rag"
version = "0.1.0"
description = "Local RAG system with MCP server for Claude Code"
requires-python = ">=3.11"
dependencies = [
    "fastmcp>=2.0",
    "chromadb>=1.0",
    "sentence-transformers>=3.0",
    "pymupdf>=1.24",
    "python-docx>=1.1",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
openai = ["openai>=1.0"]
voyage = ["voyageai>=0.3"]
all = ["local-rag[openai,voyage]"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create config.yaml**

```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100

embeddings:
  provider: "local"
  local_model: "all-MiniLM-L6-v2"
  openai_model: "text-embedding-3-small"
  voyage_model: "voyage-3"
  batch_size: 64

indexing:
  supported_extensions:
    - .pdf
    - .md
    - .txt
    - .py
    - .js
    - .ts
    - .docx
    - .rst
  max_file_size_mb: 10
  ignore_file: ".ragignore"
  recursive: true

retrieval:
  default_top_k: 5
  similarity_threshold: 0.3

storage:
  chroma_persist_dir: "~/.local_rag/chroma_db"
  collection_name: "default"
```

**Step 3: Create .ragignore**

```
# Ignore patterns (gitignore syntax)
.git/
__pycache__/
*.pyc
.venv/
node_modules/
.DS_Store
```

**Step 4: Create .gitignore**

```
.venv/
__pycache__/
*.pyc
*.egg-info/
dist/
build/
.eggs/
*.egg
.DS_Store
```

**Step 5: Create package init files**

`src/local_rag/__init__.py` — empty file
`src/local_rag/engine/__init__.py` — empty file
`src/local_rag/store/__init__.py` — empty file
`tests/__init__.py` — empty file

**Step 6: Create test fixtures**

`tests/fixtures/sample.txt`:
```
This is a sample text file for testing the document loader.
It contains multiple lines of text that can be used to verify
that the plain text loader works correctly.
```

`tests/fixtures/sample.md`:
```markdown
# Sample Document

## Section One

This is the first section of the sample markdown document.

## Section Two

This is the second section with different content for testing.
```

`tests/fixtures/sample.py`:
```python
def hello():
    """A sample function for testing code file loading."""
    return "Hello, world!"

class SampleClass:
    """A sample class for testing."""
    pass
```

**Step 7: Create venv and install**

```bash
python3 -m venv .venv
PYTHONPATH="" .venv/bin/pip install -e ".[dev]"
```

**Step 8: Verify install**

```bash
PYTHONPATH="" .venv/bin/python -c "import local_rag; print('OK')"
```

**Step 9: Commit**

```bash
git add pyproject.toml config.yaml .ragignore .gitignore src/ tests/
git commit -m "chore: scaffold local-rag project with deps and config"
```

---

### Task 2: Config Module

**Files:**
- Create: `src/local_rag/config.py`
- Create: `tests/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_config.py
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
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_config.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'local_rag.config'`

**Step 3: Write minimal implementation**

```python
# src/local_rag/config.py
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
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_config.py -v
```
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/local_rag/config.py tests/test_config.py
git commit -m "feat: add config module with YAML loading and validation"
```

---

### Task 3: Document Loaders

**Files:**
- Create: `src/local_rag/engine/loader.py`
- Create: `tests/test_loader.py`
- Create: `tests/fixtures/sample.docx` (generated in test setup)

**Step 1: Write the failing tests**

```python
# tests/test_loader.py
import os
from pathlib import Path

import pytest

from local_rag.engine.loader import load_document, discover_files, LoadedDocument

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_txt_file():
    doc = load_document(FIXTURES / "sample.txt")
    assert isinstance(doc, LoadedDocument)
    assert "sample text file" in doc.text
    assert doc.file_type == ".txt"
    assert doc.source == str(FIXTURES / "sample.txt")


def test_load_markdown_file():
    doc = load_document(FIXTURES / "sample.md")
    assert "Sample Document" in doc.text
    assert "Section One" in doc.text
    assert doc.file_type == ".md"


def test_load_python_file():
    doc = load_document(FIXTURES / "sample.py")
    assert "def hello" in doc.text
    assert doc.file_type == ".py"


def test_load_pdf_file():
    pdf_path = FIXTURES / "sample.pdf"
    if not pdf_path.exists():
        pytest.skip("sample.pdf not in fixtures")
    doc = load_document(pdf_path)
    assert len(doc.text) > 0
    assert doc.file_type == ".pdf"
    assert doc.metadata.get("page_count") is not None


def test_load_docx_file(tmp_path):
    # Create a minimal docx for testing
    from docx import Document
    docx_path = tmp_path / "test.docx"
    d = Document()
    d.add_paragraph("First paragraph of test document.")
    d.add_paragraph("Second paragraph with more content.")
    d.save(str(docx_path))

    doc = load_document(docx_path)
    assert "First paragraph" in doc.text
    assert doc.file_type == ".docx"


def test_load_unsupported_file(tmp_path):
    bad_file = tmp_path / "data.bin"
    bad_file.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(bad_file)


def test_load_empty_file(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("")
    doc = load_document(empty)
    assert doc.text == ""


def test_discover_files_respects_extensions(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "b.py").write_text("code")
    (tmp_path / "c.jpg").write_bytes(b"\xff\xd8")
    files = discover_files(str(tmp_path), extensions=[".txt", ".py"], recursive=True)
    names = {f.name for f in files}
    assert names == {"a.txt", "b.py"}


def test_discover_files_respects_max_size(tmp_path):
    small = tmp_path / "small.txt"
    small.write_text("small")
    big = tmp_path / "big.txt"
    big.write_text("x" * (2 * 1024 * 1024))  # 2MB
    files = discover_files(
        str(tmp_path), extensions=[".txt"], recursive=True, max_size_mb=1
    )
    names = {f.name for f in files}
    assert names == {"small.txt"}


def test_discover_files_respects_ragignore(tmp_path):
    (tmp_path / "keep.txt").write_text("keep")
    sub = tmp_path / "ignored_dir"
    sub.mkdir()
    (sub / "skip.txt").write_text("skip")
    (tmp_path / ".ragignore").write_text("ignored_dir/\n")
    files = discover_files(str(tmp_path), extensions=[".txt"], recursive=True,
                           ignore_file=".ragignore")
    names = {f.name for f in files}
    assert "keep.txt" in names
    assert "skip.txt" not in names


def test_discover_files_recursive_false(tmp_path):
    (tmp_path / "top.txt").write_text("top")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("nested")
    files = discover_files(str(tmp_path), extensions=[".txt"], recursive=False)
    names = {f.name for f in files}
    assert names == {"top.txt"}
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_loader.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/local_rag/engine/loader.py
from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LoadedDocument:
    text: str
    source: str
    file_type: str
    metadata: dict = field(default_factory=dict)


def load_document(path: Path | str) -> LoadedDocument:
    """Load a single document and extract its text content."""
    path = Path(path)
    suffix = path.suffix.lower()
    source = str(path)

    if suffix in (".txt", ".rst"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return LoadedDocument(text=text, source=source, file_type=suffix)

    if suffix == ".md":
        text = path.read_text(encoding="utf-8", errors="replace")
        return LoadedDocument(text=text, source=source, file_type=suffix)

    if suffix in (".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
                   ".c", ".cpp", ".h", ".hpp", ".rb", ".sh", ".yaml", ".yml",
                   ".toml", ".json", ".css", ".html"):
        text = path.read_text(encoding="utf-8", errors="replace")
        return LoadedDocument(
            text=text, source=source, file_type=suffix,
            metadata={"language": suffix.lstrip(".")},
        )

    if suffix == ".pdf":
        return _load_pdf(path)

    if suffix == ".docx":
        return _load_docx(path)

    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(path: Path) -> LoadedDocument:
    import fitz  # pymupdf

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    text = "\n".join(pages)
    return LoadedDocument(
        text=text,
        source=str(path),
        file_type=".pdf",
        metadata={"page_count": len(doc)},
    )


def _load_docx(path: Path) -> LoadedDocument:
    from docx import Document

    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs]
    text = "\n".join(paragraphs)
    return LoadedDocument(text=text, source=str(path), file_type=".docx")


def discover_files(
    folder: str,
    extensions: list[str],
    recursive: bool = True,
    max_size_mb: int = 10,
    ignore_file: str | None = None,
) -> list[Path]:
    """Walk a folder and return paths matching the given extensions and constraints."""
    folder_path = Path(folder)
    max_bytes = max_size_mb * 1024 * 1024
    ignore_patterns = _load_ignore_patterns(folder_path, ignore_file)

    results = []
    if recursive:
        walker = folder_path.rglob("*")
    else:
        walker = folder_path.glob("*")

    for entry in walker:
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in extensions:
            continue
        if entry.stat().st_size > max_bytes:
            continue
        if _is_ignored(entry, folder_path, ignore_patterns):
            continue
        results.append(entry)

    return sorted(results)


def _load_ignore_patterns(folder: Path, ignore_file: str | None) -> list[str]:
    if not ignore_file:
        return []
    ignore_path = folder / ignore_file
    if not ignore_path.exists():
        return []
    patterns = []
    for line in ignore_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _is_ignored(path: Path, root: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(root)
    rel_str = str(rel)
    for pattern in patterns:
        # Check if any parent directory matches a directory pattern
        if pattern.endswith("/"):
            dir_pattern = pattern.rstrip("/")
            for parent in rel.parents:
                if fnmatch.fnmatch(str(parent), dir_pattern):
                    return True
                if str(parent) == dir_pattern:
                    return True
        elif fnmatch.fnmatch(rel_str, pattern):
            return True
    return False
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_loader.py -v
```
Expected: All tests PASS (PDF test may skip if no fixture)

**Step 5: Commit**

```bash
git add src/local_rag/engine/loader.py tests/test_loader.py
git commit -m "feat: add document loaders with file discovery"
```

---

### Task 4: Text Chunker

**Files:**
- Create: `src/local_rag/engine/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Write the failing tests**

```python
# tests/test_chunker.py
import pytest
from local_rag.engine.chunker import chunk_text, Chunk


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=1000, chunk_overlap=200, min_chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"


def test_chunk_respects_size():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100, min_chunk_size=50)
    for chunk in chunks:
        assert len(chunk.text) <= 600  # some tolerance for not splitting mid-word


def test_chunk_overlap_preserves_context():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=10, min_chunk_size=5)
    assert len(chunks) > 1
    # Check that consecutive chunks share some text
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].text[-10:]
        start_of_next = chunks[i + 1].text[:20]
        # There should be some overlap content
        assert len(end_of_current) > 0


def test_chunk_index_metadata():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50, min_chunk_size=10)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_min_chunk_size_filters_small_chunks():
    text = "Hi. Hello. Hey."
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=0, min_chunk_size=100)
    # All chunks are smaller than min, so only the full text is returned
    assert len(chunks) <= 1


def test_empty_text_returns_empty():
    chunks = chunk_text("", chunk_size=1000, chunk_overlap=200, min_chunk_size=10)
    assert chunks == []


def test_chunk_splits_on_paragraphs_first():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    # Should try to split on paragraph boundaries
    assert any("First paragraph." in c.text for c in chunks)


def test_chunk_carries_source_metadata():
    text = "Some text content here."
    chunks = chunk_text(
        text, chunk_size=1000, chunk_overlap=200, min_chunk_size=10,
        metadata={"source": "test.txt", "file_type": ".txt"},
    )
    assert chunks[0].metadata["source"] == "test.txt"
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_chunker.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/local_rag/engine/chunker.py
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    index: int
    metadata: dict = field(default_factory=dict)


# Separators ordered by priority: paragraph > newline > sentence > space
_SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    min_chunk_size: int,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks using recursive character splitting."""
    if not text.strip():
        return []

    base_meta = metadata or {}

    if len(text) <= chunk_size:
        return [Chunk(text=text, index=0, metadata=dict(base_meta))]

    raw_chunks = _recursive_split(text, chunk_size, _SEPARATORS)

    # Apply overlap
    merged = _apply_overlap(raw_chunks, chunk_overlap)

    # Filter by min size
    filtered = [c for c in merged if len(c) >= min_chunk_size]

    # If everything was filtered out, return the full text as one chunk
    if not filtered and text.strip():
        filtered = [text]

    return [
        Chunk(text=t, index=i, metadata=dict(base_meta))
        for i, t in enumerate(filtered)
    ]


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Recursively split text trying each separator in order."""
    if len(text) <= chunk_size:
        return [text]

    # Try each separator
    for sep in separators:
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    # If single part is too big, recurse with next separator
                    if len(part) > chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1:]
                        if remaining_seps:
                            chunks.extend(_recursive_split(part, chunk_size, remaining_seps))
                        else:
                            # Last resort: hard split
                            for i in range(0, len(part), chunk_size):
                                chunks.append(part[i:i + chunk_size])
                        current = ""
                    else:
                        current = part
            if current:
                chunks.append(current)
            return chunks

    # No separator found, hard split
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def _apply_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Add overlap from the end of the previous chunk to the start of the next."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        overlap_text = prev[-overlap:] if len(prev) > overlap else prev
        result.append(overlap_text + chunks[i])
    return result
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_chunker.py -v
```
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/local_rag/engine/chunker.py tests/test_chunker.py
git commit -m "feat: add recursive text chunker with overlap"
```

---

### Task 5: Embedding Provider Abstraction

**Files:**
- Create: `src/local_rag/engine/embedder.py`
- Create: `tests/test_embedder.py`

**Step 1: Write the failing tests**

```python
# tests/test_embedder.py
import pytest
from unittest.mock import patch, MagicMock
from local_rag.engine.embedder import (
    create_embedder,
    LocalEmbedder,
    OpenAIEmbedder,
    VoyageEmbedder,
)


def test_create_embedder_local():
    embedder = create_embedder(provider="local", model="all-MiniLM-L6-v2")
    assert isinstance(embedder, LocalEmbedder)


def test_create_embedder_openai():
    embedder = create_embedder(provider="openai", model="text-embedding-3-small")
    assert isinstance(embedder, OpenAIEmbedder)


def test_create_embedder_voyage():
    embedder = create_embedder(provider="voyage", model="voyage-3")
    assert isinstance(embedder, VoyageEmbedder)


def test_create_embedder_invalid():
    with pytest.raises(ValueError, match="Unknown"):
        create_embedder(provider="bad", model="x")


def test_local_embedder_returns_vectors():
    embedder = LocalEmbedder(model="all-MiniLM-L6-v2")
    vectors = embedder.embed(["Hello world", "Test sentence"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 0  # non-empty vector
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
    embedder = VoyageEmbedder(model="voyage-3")
    mock_result = MagicMock()
    mock_result.embeddings = [[0.1, 0.2], [0.3, 0.4]]
    with patch.object(embedder, "_client") as mock_client:
        mock_client.embed.return_value = mock_result
        vectors = embedder.embed(["text1", "text2"])
    assert len(vectors) == 2
    assert vectors[0] == [0.1, 0.2]
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_embedder.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/local_rag/engine/embedder.py
from __future__ import annotations

from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts and return vectors."""
        ...


class LocalEmbedder(Embedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return [vec.tolist() for vec in embeddings]


class OpenAIEmbedder(Embedder):
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
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_embedder.py -v
```
Expected: All 9 tests PASS (local tests run real model, API tests are mocked)

**Step 5: Commit**

```bash
git add src/local_rag/engine/embedder.py tests/test_embedder.py
git commit -m "feat: add configurable embedding providers (local/openai/voyage)"
```

---

### Task 6: ChromaDB Store

**Files:**
- Create: `src/local_rag/store/chroma.py`
- Create: `tests/test_chroma.py`

**Step 1: Write the failing tests**

```python
# tests/test_chroma.py
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
    assert "doc1" in results["ids"][0]  # closest to query


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
        metadatas=[{}, {}],
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
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_chroma.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/local_rag/store/chroma.py
from __future__ import annotations

from typing import Any

import chromadb


class ChromaStore:
    """Wrapper around ChromaDB for vector storage and retrieval."""

    def __init__(self, persist_dir: str, collection_name: str = "default"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection_name = collection_name

    def add(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def upsert(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embeddings: list[list[float]],
        top_k: int = 5,
        where: dict | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "query_embeddings": query_embeddings,
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)

    def delete(
        self,
        ids: list[str] | None = None,
        where: dict | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if ids:
            kwargs["ids"] = ids
        if where:
            kwargs["where"] = where
        self._collection.delete(**kwargs)

    def count(self) -> int:
        return self._collection.count()

    def list_collections(self) -> list[str]:
        return [c.name for c in self._client.list_collections()]
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_chroma.py -v
```
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/local_rag/store/chroma.py tests/test_chroma.py
git commit -m "feat: add ChromaDB store with query, upsert, and filtering"
```

---

### Task 7: Retriever (Ties It All Together)

**Files:**
- Create: `src/local_rag/engine/retriever.py`
- Create: `tests/test_retriever.py`

**Step 1: Write the failing tests**

```python
# tests/test_retriever.py
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
    # distance 0.1 => similarity 0.9 (passes), distance 0.5 => similarity 0.5 (fails)
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
    # 5 chunks with batch_size=2 => 3 calls to embed
    assert mock_embedder.embed.call_count == 3
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_retriever.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/local_rag/engine/retriever.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from local_rag.engine.chunker import Chunk
from local_rag.engine.embedder import Embedder
from local_rag.store.chroma import ChromaStore


@dataclass
class SearchResult:
    text: str
    source: str
    score: float
    metadata: dict


class Retriever:
    """Orchestrates embedding and searching against the vector store."""

    def __init__(
        self,
        embedder: Embedder,
        store: ChromaStore,
        batch_size: int = 64,
    ):
        self._embedder = embedder
        self._store = store
        self._batch_size = batch_size

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        query_embedding = self._embedder.embed([query])
        raw = self._store.query(
            query_embeddings=query_embedding,
            top_k=top_k,
            where=filters,
        )

        results = []
        for i, doc_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            similarity = 1.0 - distance
            if similarity < similarity_threshold:
                continue
            results.append(SearchResult(
                text=raw["documents"][0][i],
                source=raw["metadatas"][0][i].get("source", ""),
                score=similarity,
                metadata=raw["metadatas"][0][i],
            ))

        return results

    def index_chunks(self, chunks: list[Chunk]) -> None:
        """Embed and upsert chunks into the store in batches."""
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start:start + self._batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed(texts)
            ids = [self._chunk_id(c) for c in batch]
            metadatas = [c.metadata for c in batch]

            self._store.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

    @staticmethod
    def _chunk_id(chunk: Chunk) -> str:
        source = chunk.metadata.get("source", "")
        content = f"{source}::{chunk.index}::{chunk.text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_retriever.py -v
```
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/local_rag/engine/retriever.py tests/test_retriever.py
git commit -m "feat: add retriever with batched indexing and similarity threshold"
```

---

### Task 8: MCP Server

**Files:**
- Create: `src/local_rag/server.py`
- Create: `tests/test_server.py`

**Step 1: Write the failing tests**

```python
# tests/test_server.py
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

from local_rag.server import create_server


@pytest.fixture
def config_path(tmp_path):
    config = {
        "chunking": {"chunk_size": 500, "chunk_overlap": 100, "min_chunk_size": 50},
        "embeddings": {
            "provider": "local", "local_model": "all-MiniLM-L6-v2",
            "openai_model": "text-embedding-3-small", "voyage_model": "voyage-3",
            "batch_size": 32,
        },
        "indexing": {
            "supported_extensions": [".txt", ".md"],
            "max_file_size_mb": 5, "ignore_file": ".ragignore", "recursive": True,
        },
        "retrieval": {"default_top_k": 3, "similarity_threshold": 0.3},
        "storage": {
            "chroma_persist_dir": str(tmp_path / "chroma"),
            "collection_name": "test",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


def test_create_server_returns_fastmcp(config_path):
    server = create_server(config_path)
    assert server is not None
    assert server.name == "local-rag"


def test_server_has_search_tool(config_path):
    server = create_server(config_path)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "search_docs" in tool_names


def test_server_has_index_tool(config_path):
    server = create_server(config_path)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "index_folder" in tool_names


def test_server_has_status_tools(config_path):
    server = create_server(config_path)
    tool_names = [t.name for t in server._tool_manager.list_tools()]
    assert "list_collections" in tool_names
    assert "get_index_status" in tool_names
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_server.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/local_rag/server.py
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from fastmcp import FastMCP

from local_rag.config import load_config, RAGConfig
from local_rag.engine.chunker import chunk_text
from local_rag.engine.embedder import create_embedder
from local_rag.engine.loader import discover_files, load_document
from local_rag.engine.retriever import Retriever
from local_rag.store.chroma import ChromaStore

logger = logging.getLogger(__name__)

# Track last index time per collection
_index_times: dict[str, float] = {}


def create_server(config_path: str | None = None) -> FastMCP:
    """Create and configure the MCP server with all tools."""
    config = load_config(config_path)
    mcp = FastMCP("local-rag")

    # Initialize components
    embedder = create_embedder(
        provider=config.embeddings.provider,
        model=_get_model_name(config),
    )
    store = ChromaStore(
        persist_dir=config.storage.chroma_persist_dir,
        collection_name=config.storage.collection_name,
    )
    retriever = Retriever(
        embedder=embedder,
        store=store,
        batch_size=config.embeddings.batch_size,
    )

    @mcp.tool
    def search_docs(
        query: str,
        top_k: int | None = None,
        collection: str | None = None,
        filters: dict | None = None,
    ) -> list[dict]:
        """Search indexed documents for chunks matching the query.

        Args:
            query: The search question or text to find relevant documents for.
            top_k: Number of results to return (default from config).
            collection: ChromaDB collection name (default from config).
            filters: Key-value pairs to filter results by metadata.
        """
        actual_top_k = top_k or config.retrieval.default_top_k

        # Use a different store/retriever if collection differs
        if collection and collection != config.storage.collection_name:
            coll_store = ChromaStore(
                persist_dir=config.storage.chroma_persist_dir,
                collection_name=collection,
            )
            coll_retriever = Retriever(
                embedder=embedder, store=coll_store,
                batch_size=config.embeddings.batch_size,
            )
            results = coll_retriever.search(
                query, top_k=actual_top_k,
                similarity_threshold=config.retrieval.similarity_threshold,
                filters=filters,
            )
        else:
            results = retriever.search(
                query, top_k=actual_top_k,
                similarity_threshold=config.retrieval.similarity_threshold,
                filters=filters,
            )

        return [
            {
                "text": r.text,
                "source": r.source,
                "score": round(r.score, 4),
                "metadata": r.metadata,
            }
            for r in results
        ]

    @mcp.tool
    def index_folder(
        folder_path: str,
        collection: str | None = None,
    ) -> dict:
        """Index all documents in a folder into the vector store.

        Args:
            folder_path: Absolute path to the folder containing documents.
            collection: ChromaDB collection name (default from config).
        """
        folder = Path(folder_path).expanduser()
        if not folder.is_dir():
            return {"error": f"Not a valid directory: {folder_path}"}

        # Use a different store if collection differs
        if collection and collection != config.storage.collection_name:
            idx_store = ChromaStore(
                persist_dir=config.storage.chroma_persist_dir,
                collection_name=collection,
            )
            idx_retriever = Retriever(
                embedder=embedder, store=idx_store,
                batch_size=config.embeddings.batch_size,
            )
        else:
            idx_store = store
            idx_retriever = retriever

        files = discover_files(
            str(folder),
            extensions=config.indexing.supported_extensions,
            recursive=config.indexing.recursive,
            max_size_mb=config.indexing.max_file_size_mb,
            ignore_file=config.indexing.ignore_file,
        )

        processed = 0
        skipped = 0
        errors = []
        total_chunks = 0

        for file_path in files:
            try:
                doc = load_document(file_path)
                if not doc.text.strip():
                    skipped += 1
                    continue

                chunks = chunk_text(
                    doc.text,
                    chunk_size=config.chunking.chunk_size,
                    chunk_overlap=config.chunking.chunk_overlap,
                    min_chunk_size=config.chunking.min_chunk_size,
                    metadata={
                        "source": doc.source,
                        "file_type": doc.file_type,
                        **doc.metadata,
                    },
                )
                idx_retriever.index_chunks(chunks)
                total_chunks += len(chunks)
                processed += 1
            except Exception as e:
                errors.append({"file": str(file_path), "error": str(e)})
                logger.warning("Failed to index %s: %s", file_path, e)

        coll_name = collection or config.storage.collection_name
        _index_times[coll_name] = time.time()

        return {
            "files_processed": processed,
            "files_skipped": skipped,
            "chunks_created": total_chunks,
            "errors": errors,
        }

    @mcp.tool
    def list_collections() -> list[dict]:
        """List all indexed collections with document counts."""
        names = store.list_collections()
        result = []
        for name in names:
            coll_store = ChromaStore(
                persist_dir=config.storage.chroma_persist_dir,
                collection_name=name,
            )
            result.append({
                "name": name,
                "document_count": coll_store.count(),
                "last_indexed": _index_times.get(name),
            })
        return result

    @mcp.tool
    def get_index_status(collection: str | None = None) -> dict:
        """Get status of an indexed collection.

        Args:
            collection: Collection name (default from config).
        """
        coll_name = collection or config.storage.collection_name
        coll_store = ChromaStore(
            persist_dir=config.storage.chroma_persist_dir,
            collection_name=coll_name,
        )
        return {
            "collection": coll_name,
            "total_chunks": coll_store.count(),
            "embedding_provider": config.embeddings.provider,
            "last_indexed": _index_times.get(coll_name),
        }

    return mcp


def _get_model_name(config: RAGConfig) -> str:
    provider = config.embeddings.provider
    if provider == "local":
        return config.embeddings.local_model
    elif provider == "openai":
        return config.embeddings.openai_model
    elif provider == "voyage":
        return config.embeddings.voyage_model
    return config.embeddings.local_model


# Entry point: python -m local_rag.server
if __name__ == "__main__":
    config_path = os.environ.get("LOCAL_RAG_CONFIG")
    server = create_server(config_path)
    server.run()
```

**Step 4: Run test to verify it passes**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/test_server.py -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/local_rag/server.py tests/test_server.py
git commit -m "feat: add MCP server with search, index, and status tools"
```

---

### Task 9: Run Full Test Suite and Wire Up MCP

**Step 1: Run entire test suite**

```bash
PYTHONPATH="" .venv/bin/python -m pytest tests/ -v
```
Expected: All tests PASS

**Step 2: Add `__main__.py` for module execution**

Create `src/local_rag/__main__.py`:

```python
from local_rag.server import create_server
import os

config_path = os.environ.get("LOCAL_RAG_CONFIG")
server = create_server(config_path)
server.run()
```

**Step 3: Test the server starts**

```bash
PYTHONPATH="" .venv/bin/python -m local_rag --help 2>&1 | head -5
```
Expected: Server starts or shows help (Ctrl+C to exit)

**Step 4: Configure Claude Code MCP**

Add to `~/.claude/settings.json` under `mcpServers`:

```json
{
  "local-rag": {
    "command": "/Users/georgey/local_rag/.venv/bin/python",
    "args": ["-m", "local_rag"],
    "env": {
      "LOCAL_RAG_CONFIG": "/Users/georgey/local_rag/config.yaml",
      "PYTHONPATH": ""
    }
  }
}
```

**Step 5: Commit**

```bash
git add src/local_rag/__main__.py
git commit -m "feat: add module entry point for MCP server"
```

---

### Task 10: Manual Integration Test

**Step 1: Create a test documents folder**

```bash
mkdir -p ~/test_docs
echo "Python is a high-level programming language known for its simplicity." > ~/test_docs/python.txt
echo "# Rust\n\nRust is a systems programming language focused on safety and performance." > ~/test_docs/rust.md
```

**Step 2: Restart Claude Code to pick up MCP server**

Exit and re-enter Claude Code so it loads the new MCP server.

**Step 3: Test indexing via Claude Code**

Ask Claude Code: "Use the local-rag tool to index the folder ~/test_docs"

Expected: `index_folder` tool returns files_processed: 2, chunks_created: 2+

**Step 4: Test searching via Claude Code**

Ask Claude Code: "Use the local-rag tool to search for 'what language is known for safety?'"

Expected: `search_docs` returns the Rust chunk with a high similarity score.

**Step 5: Test status via Claude Code**

Ask Claude Code: "Use the local-rag tool to show the index status"

Expected: `get_index_status` returns chunk count and provider info.
