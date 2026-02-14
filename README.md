# Local RAG System

A local Retrieval-Augmented Generation (RAG) system with an MCP server for semantic document search.

## Features

- **Document Loading**: Supports PDF, DOCX, Markdown, text files, and code files
- **Text Chunking**: Recursive text splitting with configurable overlap
- **Embeddings**: Multiple providers (local sentence-transformers, OpenAI, Voyage AI)
- **Vector Storage**: ChromaDB for persistent vector storage
- **MCP Server**: Model Context Protocol server for integration with AI assistants

## Installation

```bash
# Install the package
pip install -e .

# For OpenAI embeddings
pip install -e ".[openai]"

# For Voyage AI embeddings
pip install -e ".[voyage]"

# For development
pip install -e ".[dev]"
```

## Configuration

Create a `config.yaml` file:

```yaml
chunking:
  chunk_size: 1000
  chunk_overlap: 200
  min_chunk_size: 100

embeddings:
  provider: local  # local, openai, or voyage
  local_model: all-MiniLM-L6-v2
  batch_size: 64

indexing:
  supported_extensions:
    - .pdf
    - .md
    - .txt
    - .py
  max_file_size_mb: 10
  ignore_file: .ragignore
  recursive: true

retrieval:
  default_top_k: 5
  similarity_threshold: 0.3

storage:
  chroma_persist_dir: ~/.local_rag/chroma_db
  collection_name: default
```

Set the config path:
```bash
export LOCAL_RAG_CONFIG=/path/to/config.yaml
```

## MCP Server

The MCP server exposes 6 tools for document indexing and search:

### Available Tools

1. **rag_search** - Search indexed documents
   - Parameters: `query` (str), `top_k` (int), `similarity_threshold` (float)
   - Returns: Matching documents with scores and metadata

2. **rag_index_file** - Index a single file
   - Parameters: `file_path` (str)
   - Returns: Success status and number of chunks indexed

3. **rag_index_directory** - Index all files in a directory
   - Parameters: `directory_path` (str), `recursive` (bool)
   - Returns: Summary of files indexed

4. **rag_list_collections** - List all ChromaDB collections
   - Returns: Collection names and counts

5. **rag_get_stats** - Get collection statistics
   - Returns: Document count and configuration

6. **rag_configure** - Reconfigure the RAG system
   - Parameters: `collection_name` (str), `embedding_provider` (str)
   - Returns: Configuration changes

### Running the MCP Server

```bash
# Using FastMCP
fastmcp run src/local_rag/server.py

# Or use the dev server with auto-reload
fastmcp dev src/local_rag/server.py
```

### Adding to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "fastmcp",
      "args": ["run", "/path/to/local_rag/src/local_rag/server.py"],
      "env": {
        "LOCAL_RAG_CONFIG": "/path/to/config.yaml"
      }
    }
  }
}
```

## Development

Run tests:
```bash
pytest tests/
```

Run all tests with coverage:
```bash
pytest tests/ --cov=local_rag --cov-report=html
```

## Architecture

```
src/local_rag/
├── config.py          # Configuration management
├── server.py          # MCP server
├── engine/
│   ├── loader.py      # Document loading
│   ├── chunker.py     # Text chunking
│   ├── embedder.py    # Embedding providers
│   └── retriever.py   # Search orchestration
└── store/
    └── chroma.py      # ChromaDB wrapper
```

## License

MIT
