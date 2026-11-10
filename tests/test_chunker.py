"""
Unit tests for token-aware chunker (chunker.py).
Tests token limits, overlap, and metadata preservation (bbox, element_ids, page).
"""
import pytest
from app.chunker import chunk_elements, count_tokens, extract_element_bbox, extract_element_page


def test_token_counter():
    text = "Hello world! This is a test sentence for token counting."
    tokens = count_tokens(text)
    assert tokens > 0
    assert count_tokens("") == 0


def test_chunker_metadata_preservation():
    """Verify that chunk metadata preserves original element ids, page, and bbox."""
    sample_elements = [
        {
            "id": 101,
            "type": "heading",
            "page number": 1,
            "bounding box": [10.0, 20.0, 100.0, 50.0],
            "content": "Introduction to Retrieval Augmented Generation"
        },
        {
            "id": 102,
            "type": "paragraph",
            "page number": 1,
            "bounding box": [10.0, 60.0, 300.0, 150.0],
            "content": "RAG combines retrieval systems with generative language models."
        }
    ]

    chunks = chunk_elements(
        elements=sample_elements,
        upload_id="test-upload-123",
        source_file="test_doc.pdf",
        max_tokens=500,
        overlap_tokens=50
    )

    assert len(chunks) == 1
    c = chunks[0]
    assert c["upload_id"] == "test-upload-123"
    assert c["source_file"] == "test_doc.pdf"
    assert c["page"] == 1
    assert c["bbox"] == [10.0, 20.0, 100.0, 50.0]
    assert 101 in c["element_ids"]
    assert 102 in c["element_ids"]
    assert "chunk_id" in c
    assert "Introduction" in c["text"]
    assert c["extra_metadata"]["heading"] == "Introduction to Retrieval Augmented Generation"


def test_chunker_token_limit_and_overlap():
    """Verify that element grouping respects max_tokens and applies overlap."""
    # Create several paragraph elements
    elements = [
        {
            "id": i,
            "type": "paragraph",
            "page number": 1,
            "bounding box": [10.0, float(i * 30), 200.0, float(i * 30 + 20)],
            "content": f"Paragraph element number {i} containing enough words to measure token boundaries accurately across iterations."
        }
        for i in range(1, 15)
    ]

    max_tokens = 60
    overlap_tokens = 20

    chunks = chunk_elements(
        elements=elements,
        upload_id="test-overlap-upload",
        source_file="overlap.pdf",
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens
    )

    assert len(chunks) > 1

    for c in chunks:
        assert len(c["element_ids"]) > 0
        assert c["page"] == 1
        assert len(c["bbox"]) == 4

    # Verify overlap exists: some element id should appear in consecutive chunks
    overlap_found = False
    for i in range(len(chunks) - 1):
        ids_current = set(chunks[i]["element_ids"])
        ids_next = set(chunks[i + 1]["element_ids"])
        if ids_current.intersection(ids_next):
            overlap_found = True
            break
    assert overlap_found, "Expected at least one element to overlap between consecutive chunks."


def test_chunker_oversized_single_element():
    """Verify that a single element larger than max_tokens gets split properly."""
    long_text = "Word " * 200  # ~200 tokens
    elements = [{
        "id": 999,
        "type": "paragraph",
        "page number": 2,
        "bounding box": [50.0, 50.0, 400.0, 400.0],
        "content": long_text
    }]

    chunks = chunk_elements(
        elements=elements,
        upload_id="large-elem-upload",
        source_file="large.pdf",
        max_tokens=50,
        overlap_tokens=10
    )

    assert len(chunks) > 1
    for c in chunks:
        assert c["page"] == 2
        assert 999 in c["element_ids"]
        assert c["token_count"] <= 60  # close to max_tokens boundary
