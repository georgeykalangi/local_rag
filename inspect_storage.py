#!/usr/bin/env python3
"""
Deep dive into ChromaDB storage internals.
Shows how data is actually stored.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from local_rag.store.chroma import ChromaStore
import sqlite3
import json

def main():
    print("="*70)
    print("CHROMADB STORAGE DEEP DIVE")
    print("="*70)

    # Connect to ChromaDB
    store = ChromaStore(
        persist_dir="~/.local_rag/chroma_db",
        collection_name="my_docs"
    )

    # Get the actual ChromaDB collection object
    collection = store._collection

    print(f"\n📦 Collection: {collection.name}")
    print(f"   ID: {collection.id}")
    print(f"   Count: {collection.count()}")

    # Get all data including embeddings
    print("\n🔍 Fetching ALL data (including embeddings)...")
    results = collection.get(
        include=['embeddings', 'documents', 'metadatas']
    )

    print(f"\n📊 DATA STRUCTURE:")
    print(f"   Total items: {len(results['ids'])}")
    print(f"   Keys in results: {list(results.keys())}")

    # Show first document in detail
    if results['ids']:
        print(f"\n📄 FIRST DOCUMENT DETAILS:")
        print(f"   ID: {results['ids'][0]}")
        print(f"   Document length: {len(results['documents'][0])} chars")
        print(f"   Document preview: {results['documents'][0][:200]}...")
        print(f"\n   Metadata:")
        for key, value in results['metadatas'][0].items():
            print(f"      {key}: {value}")

        # Show embedding details
        if results['embeddings'] is not None and len(results['embeddings']) > 0:
            embedding = results['embeddings'][0]
            print(f"\n   🧮 EMBEDDING VECTOR:")
            print(f"      Dimensions: {len(embedding)}")
            print(f"      Data type: {type(embedding)}")
            print(f"      First 10 values: {embedding[:10]}")
            print(f"      Min value: {min(embedding):.4f}")
            print(f"      Max value: {max(embedding):.4f}")
            print(f"      Sample mid-range values: {embedding[100:110]}")

    # Now check the SQLite database directly
    print("\n" + "="*70)
    print("SQLITE DATABASE INSPECTION")
    print("="*70)

    db_path = Path.home() / ".local_rag/chroma_db/chroma.sqlite3"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Collections
    print("\n📚 COLLECTIONS TABLE:")
    cursor.execute("SELECT id, name, dimension FROM collections")
    for row in cursor.fetchall():
        print(f"   ID: {row[0]}")
        print(f"   Name: {row[1]}")
        print(f"   Dimension: {row[2]}")

    # Segments
    print("\n🔧 SEGMENTS TABLE:")
    cursor.execute("SELECT id, type, scope FROM segments")
    for row in cursor.fetchall():
        print(f"   ID: {row[0]}")
        print(f"   Type: {row[1]}")
        print(f"   Scope: {row[2]}")

    # Check all table row counts
    print("\n📊 TABLE ROW COUNTS:")
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        if count > 0:
            print(f"   {table_name}: {count} rows")

    conn.close()

    print("\n" + "="*70)
    print("💡 KEY INSIGHTS")
    print("="*70)
    print("""
ChromaDB Storage Architecture:
1. SQLite database: Stores metadata, collection info, segments
2. ChromaDB API: Abstracts storage - data might be in memory or files
3. Embeddings: Stored efficiently by ChromaDB (not directly in SQLite)
4. Documents & Metadata: Retrieved via ChromaDB API
5. HNSW Index: Used for fast similarity search

The exact storage mechanism is abstracted by ChromaDB.
Use the ChromaDB API (as shown above) to access all data.
""")

if __name__ == "__main__":
    main()
