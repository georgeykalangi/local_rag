#!/usr/bin/env python3
"""
Real RAG Quality Validation

Tests that the RAG system actually retrieves relevant, useful content
for real-world questions.
"""

from pathlib import Path
from local_rag.server import configure, index_directory, search, get_stats

def print_result(query: str, results: dict, expected_in_answer: list[str]):
    """Print search results and validate relevance."""
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")
    
    if results["num_results"] == 0:
        print("❌ NO RESULTS FOUND")
        return False
    
    print(f"\nFound {results['num_results']} results:\n")
    
    # Show top 3 results
    for i, result in enumerate(results["results"][:3], 1):
        print(f"Result #{i} - Score: {result['score']:.3f}")
        print(f"Source: {Path(result['source']).name}")
        print(f"Text: {result['text'][:300]}...")
        print()
    
    # Check if expected content is in top result
    top_text = results["results"][0]["text"].lower()
    found = [term for term in expected_in_answer if term.lower() in top_text]
    missing = [term for term in expected_in_answer if term.lower() not in top_text]
    
    if found:
        print(f"✅ RELEVANT - Found: {', '.join(found)}")
        if missing:
            print(f"   (Missing: {', '.join(missing)})")
        return True
    else:
        print(f"❌ NOT RELEVANT - Expected to find: {', '.join(expected_in_answer)}")
        return False


def main():
    print("="*70)
    print("RAG QUALITY VALIDATION")
    print("="*70)
    
    # Setup - use test collection
    configure(collection_name="rag_quality_test")
    
    # Index test documents
    print("\n1. Indexing test documents...")
    test_docs = Path("test_docs")
    if not test_docs.exists():
        print("❌ test_docs/ directory not found!")
        print("   Run: Create test documents first")
        return
    
    result = index_directory(str(test_docs), recursive=True)
    print(f"   Indexed {result['files_indexed']} files")
    print(f"   Created {result['total_chunks']} chunks")
    
    stats = get_stats()
    print(f"   Total documents in collection: {stats['document_count']}")
    
    # Test Cases - Real questions someone might ask
    print("\n2. Testing RAG Quality with Real Questions...")
    
    test_cases = [
        {
            "query": "What are the main features of Python?",
            "expected": ["easy to learn", "standard library", "community"],
            "explanation": "Should retrieve Python features from python.md"
        },
        {
            "query": "Which frameworks can I use for building web applications with JavaScript?",
            "expected": ["React", "Vue", "Angular", "Express"],
            "explanation": "Should find JavaScript frameworks"
        },
        {
            "query": "What is the time complexity of Quick Sort?",
            "expected": ["O(n log n)", "Quick Sort"],
            "explanation": "Should find algorithm complexity info"
        },
        {
            "query": "How can I use Python for data analysis?",
            "expected": ["Pandas", "NumPy", "data"],
            "explanation": "Should find Python's data science use cases"
        },
        {
            "query": "What makes Merge Sort different from Bubble Sort?",
            "expected": ["Merge Sort", "Bubble Sort", "O(n"],
            "explanation": "Should retrieve sorting algorithm comparisons"
        },
        {
            "query": "Tell me about asynchronous programming in JavaScript",
            "expected": ["asynchronous", "event-driven"],
            "explanation": "Should find JavaScript async features"
        },
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}/{len(test_cases)} ---")
        print(f"Expectation: {test['explanation']}")
        
        results = search(
            query=test["query"],
            top_k=3,
            similarity_threshold=0.3
        )
        
        if print_result(test["query"], results, test["expected"]):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print("\n" + "="*70)
    print("RAG QUALITY SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(test_cases)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    print(f"Success Rate: {(passed/len(test_cases)*100):.1f}%")
    print("="*70)
    
    if passed == len(test_cases):
        print("\n🎉 EXCELLENT - All RAG queries returned relevant content!")
    elif passed >= len(test_cases) * 0.8:
        print("\n✅ GOOD - Most RAG queries returned relevant content")
    elif passed >= len(test_cases) * 0.6:
        print("\n⚠️  ACCEPTABLE - RAG is working but could be improved")
    else:
        print("\n❌ POOR - RAG needs improvement")
    
    print("\nTo improve RAG quality:")
    print("- Adjust chunk_size and chunk_overlap in config.yaml")
    print("- Try different embedding providers (OpenAI, Voyage)")
    print("- Adjust similarity_threshold for queries")
    print("- Add more diverse training documents")


if __name__ == "__main__":
    main()
