#!/usr/bin/env python3
"""
Visualize how different documents have different embedding vectors.
Shows the mathematical similarity between chunks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from local_rag.store.chroma import ChromaStore
import numpy as np

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    return dot_product / (magnitude1 * magnitude2)

def main():
    print("="*70)
    print("EMBEDDING VECTORS VISUALIZATION")
    print("="*70)

    # Connect
    store = ChromaStore(
        persist_dir="~/.local_rag/chroma_db",
        collection_name="my_docs"
    )

    # Get all data with embeddings
    results = store._collection.get(include=['embeddings', 'documents', 'metadatas'])

    print(f"\n📊 Analyzing {len(results['ids'])} document chunks...\n")

    # Show first few dimensions of each vector
    print("📈 EMBEDDING VECTOR COMPARISON")
    print("(First 20 dimensions of each vector)")
    print("-"*70)

    for i, (doc_id, embedding) in enumerate(zip(results['ids'], results['embeddings']), 1):
        preview = results['documents'][i-1][:60].replace('\n', ' ')
        print(f"\nDoc {i}: {preview}...")
        print(f"Vector: [{', '.join([f'{v:6.3f}' for v in embedding[:20]])}...]")

    # Calculate similarity matrix
    print("\n\n" + "="*70)
    print("🔍 SIMILARITY MATRIX")
    print("="*70)
    print("\nHow similar is each document to every other document?")
    print("(1.0 = identical, 0.0 = completely different)\n")

    # Header
    print("      ", end="")
    for i in range(len(results['ids'])):
        print(f"Doc{i+1:2d}  ", end="")
    print()

    # Similarity matrix
    for i, emb1 in enumerate(results['embeddings']):
        print(f"Doc{i+1:2d}  ", end="")
        for j, emb2 in enumerate(results['embeddings']):
            similarity = cosine_similarity(emb1, emb2)
            print(f"{similarity:5.3f}  ", end="")
        print()

    # Find most similar and most different pairs
    print("\n\n" + "="*70)
    print("🎯 INSIGHTS")
    print("="*70)

    similarities = []
    for i in range(len(results['embeddings'])):
        for j in range(i+1, len(results['embeddings'])):
            sim = cosine_similarity(results['embeddings'][i], results['embeddings'][j])
            similarities.append((sim, i, j))

    similarities.sort(reverse=True)

    # Most similar pair
    print("\n✅ MOST SIMILAR CHUNKS:")
    sim, i, j = similarities[0]
    print(f"   Similarity: {sim:.3f}")
    print(f"   Doc {i+1}: {results['documents'][i][:80].replace(chr(10), ' ')}...")
    print(f"   Doc {j+1}: {results['documents'][j][:80].replace(chr(10), ' ')}...")

    # Least similar pair
    print("\n❌ LEAST SIMILAR CHUNKS:")
    sim, i, j = similarities[-1]
    print(f"   Similarity: {sim:.3f}")
    print(f"   Doc {i+1}: {results['documents'][i][:80].replace(chr(10), ' ')}...")
    print(f"   Doc {j+1}: {results['documents'][j][:80].replace(chr(10), ' ')}...")

    # Vector statistics
    print("\n\n" + "="*70)
    print("📊 VECTOR STATISTICS")
    print("="*70)

    all_embeddings = np.array(results['embeddings'])

    print(f"\nShape: {all_embeddings.shape} (5 docs × 384 dimensions)")
    print(f"\nAcross all documents and dimensions:")
    print(f"   Min value:  {all_embeddings.min():7.4f}")
    print(f"   Max value:  {all_embeddings.max():7.4f}")
    print(f"   Mean value: {all_embeddings.mean():7.4f}")
    print(f"   Std dev:    {all_embeddings.std():7.4f}")

    # Distribution of values
    print(f"\n📈 Value Distribution:")
    print(f"   Values < -0.1: {(all_embeddings < -0.1).sum()} ({(all_embeddings < -0.1).sum() / all_embeddings.size * 100:.1f}%)")
    print(f"   Values -0.1 to 0: {((all_embeddings >= -0.1) & (all_embeddings < 0)).sum()} ({((all_embeddings >= -0.1) & (all_embeddings < 0)).sum() / all_embeddings.size * 100:.1f}%)")
    print(f"   Values 0 to 0.1: {((all_embeddings >= 0) & (all_embeddings < 0.1)).sum()} ({((all_embeddings >= 0) & (all_embeddings < 0.1)).sum() / all_embeddings.size * 100:.1f}%)")
    print(f"   Values > 0.1: {(all_embeddings > 0.1).sum()} ({(all_embeddings > 0.1).sum() / all_embeddings.size * 100:.1f}%)")

    print("\n\n" + "="*70)
    print("💡 KEY TAKEAWAYS")
    print("="*70)
    print("""
1. Each document has a unique 384-dimensional fingerprint
2. Similar content → similar vectors → high cosine similarity
3. Different content → different vectors → low cosine similarity
4. The model learned these representations from training data
5. This is how semantic search works - math, not magic!

Vector values are typically between -0.2 and 0.2
The direction of the vector matters more than individual values
Cosine similarity measures the angle between vectors, not distance
""")

if __name__ == "__main__":
    main()
