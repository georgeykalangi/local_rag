# Quick Start - Index Your Own Documents

## Option 1: Use the test_docs/ Directory

Already set up with sample documents:

```bash
cd /Users/georgey/local_rag
ls test_docs/
# python.md, javascript.md, algorithms.txt
```

**Index them:**
```python
from local_rag.server import index_directory, search

# Index all test docs
result = index_directory("test_docs", recursive=True)
print(f"Indexed {result['files_indexed']} files")

# Search
results = search("Python web frameworks", top_k=3)
for r in results["results"]:
    print(f"- {r['text'][:100]}...")
```

## Option 2: Create Your Own Documents Directory

```bash
# Create a directory for your documents
mkdir ~/my_documents

# Add some files
cp ~/Downloads/*.pdf ~/my_documents/
cp ~/Documents/*.md ~/my_documents/
```

**Then index:**
```python
from local_rag.server import index_directory

result = index_directory(
    directory_path="/Users/georgey/my_documents",
    recursive=True  # Include subdirectories
)

print(f"Indexed: {result['files_indexed']} files")
print(f"Chunks: {result['total_chunks']}")
```

## Option 3: Index Specific Files

```python
from local_rag.server import index_file

# Index a single PDF
result = index_file("/Users/georgey/Documents/important_doc.pdf")

# Index multiple files
files = [
    "/Users/georgey/Documents/report.pdf",
    "/Users/georgey/Downloads/paper.pdf",
    "/Users/georgey/code/README.md",
]

for file in files:
    result = index_file(file)
    if result["success"]:
        print(f"✓ {file}")
    else:
        print(f"✗ {file}: {result['error']}")
```

## Option 4: With Claude Desktop (via MCP)

Once the MCP server is running, just ask Claude:

```
"Please index all the PDF files in /Users/georgey/Documents"

"Index the markdown files in my code repository at ~/projects/myapp"

"Search the indexed documents for information about API authentication"
```

## Supported File Types

The RAG system can index:

### Documents
- `.pdf` - PDF documents
- `.docx` - Word documents
- `.md` - Markdown files
- `.txt` - Text files
- `.rst` - reStructuredText

### Code Files
- `.py` - Python
- `.js`, `.jsx`, `.ts`, `.tsx` - JavaScript/TypeScript
- `.java` - Java
- `.go` - Go
- `.rs` - Rust
- `.c`, `.cpp`, `.h`, `.hpp` - C/C++
- `.rb` - Ruby
- `.sh` - Shell scripts

### Config Files
- `.yaml`, `.yml` - YAML
- `.toml` - TOML
- `.json` - JSON
- `.css` - CSS
- `.html` - HTML

## Configuration

Edit `config.yaml` to customize:

```yaml
indexing:
  supported_extensions:
    - .pdf
    - .md
    - .txt
    # Add more...
  
  max_file_size_mb: 10  # Skip files larger than this
  ignore_file: .ragignore  # Like .gitignore
  recursive: true  # Index subdirectories
```

## Ignore Files (.ragignore)

Create a `.ragignore` file in your directory to exclude files:

```
# .ragignore example
*.tmp
*.log
node_modules/
.git/
__pycache__/
*.pyc
```

## Quick Test Script

```bash
# Create a test file
echo "Artificial Intelligence is transforming technology" > test.txt

# Index it
python3 << 'PYEOF'
from local_rag.server import index_file, search

# Index
result = index_file("test.txt")
print(f"Indexed: {result}")

# Search
results = search("AI technology", top_k=1)
print(f"\nFound: {results['results'][0]['text']}")
PYEOF
```

## Examples Directory Structure

### Example 1: Personal Knowledge Base
```
~/knowledge_base/
├── articles/
│   ├── ai_research.pdf
│   └── programming_tips.md
├── books/
│   └── clean_code.pdf
└── notes/
    └── meeting_notes.md
```

**Index:**
```python
index_directory("/Users/georgey/knowledge_base", recursive=True)
```

### Example 2: Project Documentation
```
~/projects/myapp/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
└── .ragignore  # Exclude node_modules, etc.
```

**Index:**
```python
index_directory("/Users/georgey/projects/myapp", recursive=True)
```

### Example 3: Research Papers
```
~/papers/
├── nlp/
│   ├── attention_is_all_you_need.pdf
│   └── bert.pdf
└── computer_vision/
    ├── resnet.pdf
    └── vit.pdf
```

**Index:**
```python
index_directory("/Users/georgey/papers", recursive=True)
```

## Common Issues

### "File not found"
- Use **absolute paths**: `/Users/georgey/docs/file.pdf`
- Not relative paths: `~/docs/file.pdf` (expand ~ first)

### "File too large"
- Default max: 10MB
- Change in config.yaml: `max_file_size_mb: 50`

### "Unsupported file type"
- Check `supported_extensions` in config.yaml
- Add your extension if needed

## Next Steps

1. **Choose where to put your files** (or use test_docs/)
2. **Index them** using any method above
3. **Search** with semantic queries
4. **Integrate with Claude Desktop** for conversational search

Try it now:
```bash
cd /Users/georgey/local_rag
python3 -c "from local_rag.server import index_directory; print(index_directory('test_docs'))"
```
