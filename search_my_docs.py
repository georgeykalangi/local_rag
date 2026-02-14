#!/usr/bin/env python3
"""
Quick script to search your indexed documents.
Run from /Users/georgey/local_rag directory.
"""
import sys
from pathlib import Path

# Ensure we're using the local codebase
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from local_rag.server import configure, search, get_stats

def main():
    print("="*70)
    print("SEARCHING YOUR DOCUMENTS")
    print(f"Using codebase at: {project_root}")
    print("="*70)
    
    # Use the my_docs collection
    configure(collection_name="my_docs")
    
    # Show stats
    stats = get_stats()
    print(f"\nCollection: {stats['collection_name']}")
    print(f"Documents: {stats['document_count']}")
    
    # Get query from command line or use default
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "when is the investor day"
    
    print(f"\nSearching for: '{query}'")
    print("-"*70)
    
    # Search
    results = search(query=query, top_k=3, similarity_threshold=0.2)
    
    if results['num_results'] > 0:
        print(f"\nFound {results['num_results']} results:\n")
        for i, r in enumerate(results['results'], 1):
            print(f"Result {i} (Score: {r['score']:.1%}):")
            print(r['text'][:300] + "...")
            print()
    else:
        print("\nNo results found. Try a different query or lower threshold.")
    
    print("="*70)
    print("TIP: Run with your own query:")
    print(f"  python {Path(__file__).name} your question here")
    print("="*70)

if __name__ == "__main__":
    main()
