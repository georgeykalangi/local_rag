#!/usr/bin/env python3
"""
Example: Using the MCP server tools directly.

This shows how to use the MCP server functions programmatically,
which is the same interface exposed to Claude via MCP.
"""

from pathlib import Path

from local_rag.server import (
    configure,
    get_stats,
    index_directory,
    index_file,
    list_collections,
    search,
)


def main():
    print("=" * 70)
    print("MCP Server Example Usage")
    print("=" * 70)
    print()

    # Step 1: Get current stats
    print("1. Getting current stats...")
    stats = get_stats()
    print(f"   Collection: {stats['collection_name']}")
    print(f"   Documents: {stats['document_count']}")
    print(f"   Provider: {stats['config']['embedding_provider']}")
    print()

    # Step 2: List available collections
    print("2. Listing collections...")
    collections = list_collections()
    print(f"   Total collections: {collections['total_collections']}")
    print(f"   Collections: {collections['collections']}")
    print()

    # Step 3: Configure to use a specific collection
    print("3. Switching to 'example' collection...")
    config_result = configure(collection_name="example")
    if config_result["success"]:
        print(f"   ✓ {config_result['message']}")
        if "changes" in config_result:
            for change in config_result["changes"]:
                print(f"     - {change}")
    print()

    # Step 4: Create and index a sample file
    print("4. Creating and indexing a sample file...")
    sample_file = Path("mcp_example.md")
    sample_file.write_text("""
# MCP Server Documentation

The Model Context Protocol (MCP) enables AI assistants to interact with
external data sources and tools in a standardized way.

## Features

- **Tools**: Expose functions that the AI can call
- **Resources**: Provide access to data (files, databases, APIs)
- **Prompts**: Pre-defined prompts for common tasks

## Use Cases

MCP is perfect for:
- Document search and retrieval
- Database queries
- API integrations
- File system operations
""")

    result = index_file(str(sample_file))
    if result["success"]:
        print(f"   ✓ Indexed {result['file']}")
        print(f"     Chunks: {result['chunks_indexed']}")
        print(f"     Type: {result['file_type']}")
    else:
        print(f"   ✗ Error: {result['error']}")
    print()

    # Step 5: Create a directory with multiple files
    print("5. Creating and indexing a directory...")
    docs_dir = Path("mcp_docs")
    docs_dir.mkdir(exist_ok=True)

    (docs_dir / "rag.txt").write_text("""
Retrieval-Augmented Generation (RAG)

RAG combines retrieval with language models to provide accurate,
context-aware responses. The system retrieves relevant documents
and uses them to generate better answers.

Benefits:
- Factual accuracy from retrieved documents
- Up-to-date information without retraining
- Transparent sources for verification
""")

    (docs_dir / "vectors.md").write_text("""
# Vector Embeddings

Vector embeddings represent text as numerical vectors in
high-dimensional space. Similar texts have similar vectors,
enabling semantic search.

## Common Models

- sentence-transformers (local, fast)
- OpenAI text-embedding-3 (high quality)
- Voyage AI (optimized for RAG)
""")

    dir_result = index_directory(str(docs_dir), recursive=True)
    if dir_result["success"]:
        print(f"   ✓ Indexed {dir_result['directory']}")
        print(f"     Files found: {dir_result['files_discovered']}")
        print(f"     Files indexed: {dir_result['files_indexed']}")
        print(f"     Total chunks: {dir_result['total_chunks']}")
    else:
        print(f"   ✗ Error: {dir_result['error']}")
    print()

    # Step 6: Perform semantic searches
    print("6. Performing semantic searches...")
    print()

    queries = [
        "What is MCP and what are its features?",
        "Tell me about RAG benefits",
        "Which embedding models are available?",
    ]

    for query in queries:
        print(f"   Query: {query}")
        results = search(query=query, top_k=2, similarity_threshold=0.3)

        if results["num_results"] > 0:
            for i, result in enumerate(results["results"], 1):
                source = Path(result["source"]).name
                score = result["score"]
                text = result["text"][:100].replace("\n", " ")
                print(f"     {i}. [{source}] (score: {score:.3f})")
                print(f"        {text}...")
        else:
            print("     No results found")
        print()

    # Step 7: Get final stats
    print("7. Final stats...")
    final_stats = get_stats()
    print(f"   Total documents: {final_stats['document_count']}")
    print()

    print("=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
