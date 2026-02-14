# Local RAG — Design Document

## Overview

A local RAG (Retrieval-Augmented Generation) system exposed as an MCP server so Claude Code can natively search and retrieve context from a local document folder. Built with ChromaDB for vector storage, configurable embedding providers, and support for mixed document formats.

## Requirements

- **Document formats:** PDF, Markdown, plain text, code files, Word (.docx)
- **Scale:** Medium (~50-500 documents)
- **Interface:** MCP server over stdio, integrated with Claude Code
- **Embeddings:** Configurable — local (sentence-transformers) or API (OpenAI, Voyage)
- **Search:** Semantic similarity search with filtering and configurable thresholds

## Architecture

```
Claude Code
    │ MCP Protocol (stdio)
    ▼
MCP Server (FastMCP)
    │ Tools: search_docs, index_folder, list_collections, get_index_status
    ▼
RAG Engine
    │ Loader → Chunker → Embedder → Retriever
    ▼
ChromaDB (embedded, local)
    ~/.local_rag/chroma_db/
```

**Indexing flow:** Documents folder → Loader (extracts text) → Chunker (splits into overlapping chunks) → Embedder (generates vectors) → ChromaDB (stores vectors + metadata + text)

**Query flow:** User question → Claude Code → MCP `search_docs` → Embedder (embed query) → ChromaDB (similarity search) → Return top-k chunks with source info → Claude Code uses chunks as context to answer

## Configuration

All processing parameters are configurable via `config.yaml`:

```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100

embeddings:
  provider: "local"           # "local", "openai", or "voyage"
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

API keys read from environment variables (`OPENAI_API_KEY`, `VOYAGE_API_KEY`), never from config.

## Document Processing

**Loaders:**

| Format | Library | Extracts |
|--------|---------|----------|
| PDF | pymupdf (fitz) | Text + page numbers |
| Markdown | Built-in | Raw text with heading structure |
| Plain text | Built-in | Raw text |
| Code files | Built-in | Raw text with language detection |
| Word (.docx) | python-docx | Text + paragraph structure |

**Chunking:** Recursive character splitter. Split hierarchy: paragraph → sentence → word. Each chunk stores metadata: `source_file`, `page_number`, `chunk_index`, `file_type`, `last_modified`.

**Deduplication:** Hash chunk content + source path for stable IDs. Re-index skips unchanged chunks. Deleted files get chunks removed.

**File discovery:** Recursive walk, respects `.ragignore`, skips binary files and files over `max_file_size_mb`, extension whitelist from config.

## MCP Tools

### `search_docs(query, top_k?, collection?, filters?)`
Embeds the query, runs similarity search. Returns top-k chunks with text, source path, page number, score, metadata. Defaults from config.

### `index_folder(folder_path, collection?)`
Scans folder, processes documents, generates embeddings, stores in ChromaDB. Returns summary of files processed/skipped/errored. Idempotent.

### `list_collections()`
Returns all collections with doc count and last indexed timestamp.

### `get_index_status(collection?)`
Returns total chunks, total files, last index time, embedding provider in use.

## MCP Configuration

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "/Users/georgey/local_rag/.venv/bin/python",
      "args": ["-m", "local_rag.server"],
      "env": {
        "LOCAL_RAG_CONFIG": "/Users/georgey/local_rag/config.yaml"
      }
    }
  }
}
```

## Project Structure

```
local_rag/
├── config.yaml
├── .ragignore
├── pyproject.toml
├── src/
│   └── local_rag/
│       ├── __init__.py
│       ├── server.py
│       ├── config.py
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── loader.py
│       │   ├── chunker.py
│       │   ├── embedder.py
│       │   └── retriever.py
│       └── store/
│           ├── __init__.py
│           └── chroma.py
├── tests/
│   ├── test_loader.py
│   ├── test_chunker.py
│   ├── test_embedder.py
│   ├── test_retriever.py
│   └── test_server.py
└── docs/
    └── plans/
```

## Dependencies

| Package | Purpose |
|---------|---------|
| fastmcp | MCP server framework |
| chromadb | Embedded vector database |
| sentence-transformers | Local embedding models |
| pymupdf | PDF text extraction |
| python-docx | Word document extraction |
| pyyaml | Config file parsing |
| openai | OpenAI embeddings (optional) |
| voyageai | Voyage embeddings (optional) |

Optional dependency groups: `[openai]`, `[voyage]`, `[all]`, `[dev]`

## Testing

- **test_loader.py** — Each loader returns correct text from fixtures. Handles corrupt/empty files.
- **test_chunker.py** — Chunks respect size/overlap config. Metadata correct. Min size filtering.
- **test_embedder.py** — Local embedder returns correct dimensions. Provider switching. API mocked.
- **test_retriever.py** — Results sorted by score. Respects top_k and threshold. Filters work.
- **test_server.py** — MCP tools return expected shapes. Error cases return clean messages.

Test fixtures: small sample docs in `tests/fixtures/`. Config override for temp ChromaDB directory.

## Error Handling

- File read failures: logged and skipped, don't fail the batch
- Embedding API failures: clear error messages with retry guidance
- Invalid folder paths: descriptive errors, not stack traces
