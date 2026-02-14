# Manual Integration Test Guide

This guide walks through manually testing the MCP server with Claude Desktop or MCP Inspector.

## Prerequisites

1. Install the package:
```bash
pip install -e .
```

2. Create a test documents directory:
```bash
mkdir -p test_docs
```

## Test Documents

Create sample documents for testing:

```bash
# Create Python guide
cat > test_docs/python.md << 'ENDDOC'
# Python Programming

Python is a high-level, interpreted programming language known for its simplicity and readability.

## Key Features
- Easy to learn syntax
- Extensive standard library
- Great for data science and web development
- Large community and ecosystem

## Common Use Cases
- Web development (Django, Flask)
- Data analysis (Pandas, NumPy)
- Machine learning (TensorFlow, PyTorch)
- Automation and scripting
ENDDOC

# Create JavaScript guide
cat > test_docs/javascript.md << 'ENDDOC'
# JavaScript Programming

JavaScript is a versatile programming language primarily used for web development.

## Features
- Runs in browsers and Node.js
- Event-driven and asynchronous
- First-class functions
- Prototype-based inheritance

## Frameworks
- React: UI library
- Vue: Progressive framework
- Angular: Full-featured framework
- Express: Web server framework
ENDDOC

# Create algorithms reference
cat > test_docs/algorithms.txt << 'ENDDOC'
Sorting Algorithms Reference

1. Bubble Sort
   - Time: O(n²)
   - Space: O(1)
   - Simple but inefficient

2. Quick Sort
   - Time: O(n log n) average
   - Space: O(log n)
   - Divide and conquer

3. Merge Sort
   - Time: O(n log n)
   - Space: O(n)
   - Stable sorting
ENDDOC
```

## Option 1: Test with FastMCP Inspector

The fastest way to test:

```bash
# Run the MCP inspector
fastmcp dev src/local_rag/server.py
```

This will open a web interface where you can:
1. See all available tools
2. Call tools with parameters
3. View responses

### Test Scenarios

1. **Get Stats**
   - Tool: `rag_get_stats`
   - Expected: Shows collection name, document count, config

2. **Index Directory**
   - Tool: `rag_index_directory`
   - Params: `{"directory_path": "./test_docs", "recursive": true}`
   - Expected: Success with files_indexed=3, total_chunks>0

3. **Search**
   - Tool: `rag_search`
   - Params: `{"query": "Python web development", "top_k": 3}`
   - Expected: Results mentioning Django, Flask, web

4. **Search with Threshold**
   - Tool: `rag_search`
   - Params: `{"query": "sorting algorithms", "top_k": 3, "similarity_threshold": 0.5}`
   - Expected: High-quality results only

5. **List Collections**
   - Tool: `rag_list_collections`
   - Expected: List of collection names

6. **Configure Collection**
   - Tool: `rag_configure`
   - Params: `{"collection_name": "test_manual"}`
   - Expected: Success, collection switched

## Option 2: Test with Claude Desktop

### Setup

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "local-rag": {
      "command": "fastmcp",
      "args": ["run", "/Users/georgey/local_rag/src/local_rag/server.py"],
      "env": {
        "LOCAL_RAG_CONFIG": "/Users/georgey/local_rag/config.yaml"
      }
    }
  }
}
```

### Restart Claude Desktop

After updating the config, restart Claude Desktop for changes to take effect.

### Test Prompts

1. **Check Tools Available**
   > "What MCP tools do you have available?"

2. **Index Documents**
   > "Use the local-rag MCP server to index all documents in /Users/georgey/local_rag/test_docs"

3. **Search**
   > "Search the indexed documents for information about Python web frameworks"

4. **Get Stats**
   > "Get the stats for the RAG collection"

5. **Semantic Search**
   > "Search for content about sorting algorithms and their complexity"

6. **Collection Management**
   > "List all available RAG collections"

## Expected Results

### After Indexing test_docs/

- 3 files indexed
- ~6-10 chunks created (depending on chunk size)
- Collection contains Python, JavaScript, and algorithms content

### Search Queries

1. "Python web development" → Should find Django, Flask mentions
2. "JavaScript frameworks" → Should find React, Vue, Angular
3. "sorting algorithms complexity" → Should find algorithm details
4. "machine learning" → Should find ML use cases in Python doc

### Error Cases to Test

1. Index non-existent file:
   ```json
   {"file_path": "/nonexistent/file.txt"}
   ```
   Expected: `{"success": false, "error": "File not found: ..."}`

2. Index non-existent directory:
   ```json
   {"directory_path": "/nonexistent/dir"}
   ```
   Expected: `{"success": false, "error": "Directory not found: ..."}`

3. Invalid provider:
   ```json
   {"embedding_provider": "invalid"}
   ```
   Expected: `{"success": false, "error": "Invalid provider: ..."}`

## Verification Checklist

- [ ] MCP server starts without errors
- [ ] All 6 tools are available
- [ ] Can index a directory successfully
- [ ] Can index a single file
- [ ] Search returns relevant results
- [ ] Similarity threshold filters results
- [ ] Can list collections
- [ ] Can get stats
- [ ] Can switch collections
- [ ] Error handling works correctly
- [ ] Results include source file paths
- [ ] Scores are between 0 and 1

## Troubleshooting

### Server Won't Start

- Check Python version: `python --version` (should be >=3.11)
- Check dependencies: `pip list | grep -E "(fastmcp|chromadb|sentence)"`
- Check for errors: `fastmcp dev src/local_rag/server.py` (verbose mode)

### No Results When Searching

- Verify documents are indexed: Use `rag_get_stats` to check document count
- Check similarity threshold: Try with `similarity_threshold: 0.0`
- Verify query is relevant to indexed content

### Wrong Collection

- Use `rag_configure` to switch to correct collection
- Use `rag_list_collections` to see all available collections

## Clean Up

After testing:

```bash
# Remove test docs
rm -rf test_docs/

# (Optional) Remove ChromaDB data
rm -rf ~/.local_rag/chroma_db/
```
