#!/usr/bin/env python3
"""
Visualize your indexed data in ChromaDB.
Shows what's stored and how it's organized.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from local_rag.store.chroma import ChromaStore
import json

def main():
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "my_docs"
    
    print("="*70)
    print(f"VIEWING COLLECTION: {collection_name}")
    print("="*70)
    
    # Connect
    store = ChromaStore(
        persist_dir="~/.local_rag/chroma_db",
        collection_name=collection_name
    )
    
    count = store.count()
    print(f"\n📊 Total documents: {count}")
    
    if count == 0:
        print("   (Collection is empty)")
        return
    
    # Get all data
    results = store.query(
        query_embeddings=[[0.0] * 384],
        top_k=count
    )
    
    print(f"\n📄 DOCUMENTS:")
    print("-"*70)
    
    for i, (doc_id, text, meta) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0]
    ), 1):
        print(f"\n{i}. ID: {doc_id[:16]}...")
        print(f"   Source: {meta.get('source', 'N/A')}")
        print(f"   Type: {meta.get('file_type', 'N/A')}")
        print(f"   Length: {len(text)} chars")
        print(f"   Preview: {text[:120]}...")
    
    # Show vector information
    print(f"\n🧮 EMBEDDINGS INFO:")
    print("-"*70)
    print(f"   Vector dimensions: 384")
    print(f"   Total vectors: {count}")
    print(f"   Storage location: ~/.local_rag/chroma_db/")
    
    # Show some embedding values
    if results.get('embeddings'):
        first_vector = results['embeddings'][0][0]
        print(f"   Sample vector (first 10 dims):")
        print(f"   {[f'{v:.3f}' for v in first_vector[:10]]}")
    
    print("\n" + "="*70)
    print("💡 TIP: Run with different collection:")
    print(f"   python {Path(__file__).name} my_docs")
    print(f"   python {Path(__file__).name} default")
    print("="*70)

if __name__ == "__main__":
    main()
