import pytest
from local_rag.engine.chunker import chunk_text, Chunk


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=1000, chunk_overlap=200, min_chunk_size=10)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world"


def test_chunk_respects_size():
    text = "word " * 500  # 2500 chars
    chunks = chunk_text(text, chunk_size=500, chunk_overlap=100, min_chunk_size=50)
    for chunk in chunks:
        assert len(chunk.text) <= 600  # some tolerance for not splitting mid-word


def test_chunk_overlap_preserves_context():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=10, min_chunk_size=5)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        end_of_current = chunks[i].text[-10:]
        start_of_next = chunks[i + 1].text[:20]
        assert len(end_of_current) > 0


def test_chunk_index_metadata():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=200, chunk_overlap=50, min_chunk_size=10)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i


def test_min_chunk_size_filters_small_chunks():
    text = "Hi. Hello. Hey."
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=0, min_chunk_size=100)
    assert len(chunks) <= 1


def test_empty_text_returns_empty():
    chunks = chunk_text("", chunk_size=1000, chunk_overlap=200, min_chunk_size=10)
    assert chunks == []


def test_chunk_splits_on_paragraphs_first():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, chunk_size=30, chunk_overlap=5, min_chunk_size=5)
    assert any("First paragraph." in c.text for c in chunks)


def test_chunk_carries_source_metadata():
    text = "Some text content here."
    chunks = chunk_text(
        text, chunk_size=1000, chunk_overlap=200, min_chunk_size=10,
        metadata={"source": "test.txt", "file_type": ".txt"},
    )
    assert chunks[0].metadata["source"] == "test.txt"
