"""
Tests for ingest_worker.py:
- Monkeypatches opendataloader_pdf.convert to output representative JSON fixtures
- Mocks Weaviate / vector store to verify upserted chunk objects and metadata
- Optional integration test testing real PDF processing with Java if available
"""
import os
import json
import shutil
import pytest
from unittest.mock import MagicMock, patch

from app import ingest_worker
from app import config


FIXTURE_ODL_JSON = {
    "file name": "sample_contract.pdf",
    "number of pages": 2,
    "author": "Test Author",
    "title": "Service Agreement",
    "kids": [
        {
            "id": 1,
            "type": "heading",
            "page number": 1,
            "bounding box": [50.0, 700.0, 300.0, 740.0],
            "content": "Master Services Agreement"
        },
        {
            "id": 2,
            "type": "paragraph",
            "page number": 1,
            "bounding box": [50.0, 600.0, 500.0, 680.0],
            "content": "This agreement is entered into between Client Inc. and Provider Corp."
        },
        {
            "id": 3,
            "type": "heading",
            "page number": 2,
            "bounding box": [50.0, 720.0, 200.0, 750.0],
            "content": "Payment Terms"
        },
        {
            "id": 4,
            "type": "paragraph",
            "page number": 2,
            "bounding box": [50.0, 620.0, 500.0, 690.0],
            "content": "Invoices shall be payable within thirty (30) days from the receipt date."
        }
    ]
}


def test_ingest_worker_with_mocked_odl_and_weaviate(monkeypatch, tmp_path):
    """
    Test end-to-end ingest_worker pipeline with monkeypatched ODL convert and mocked vector store.
    """
    upload_id = "test-mock-upload-001"
    fake_pdf = tmp_path / "test.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 mock pdf content")

    # 1. Stub opendataloader_pdf.convert to write fixture JSON to output_dir
    def stub_convert(input_path, output_dir=None, format=None, sanitize=True):
        os.makedirs(output_dir, exist_ok=True)
        json_file = os.path.join(output_dir, "test.json")
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(FIXTURE_ODL_JSON, f)

    monkeypatch.setattr("opendataloader_pdf.convert", stub_convert)

    # 2. Mock vector store
    mock_store = MagicMock()
    mock_store.class_name = "PDFChunk"
    monkeypatch.setattr("app.ingest_worker.get_vector_store", lambda: mock_store)

    # 3. Execute process
    res = ingest_worker.process(str(fake_pdf), upload_id, "test.pdf")

    assert res["status"] == "indexed"
    assert res["chunks_count"] > 0

    # 4. Assert vector store upsert was invoked
    mock_store.upsert_chunks.assert_called_once()
    called_chunks, called_vectors = mock_store.upsert_chunks.call_args[0]

    assert len(called_chunks) == res["chunks_count"]
    assert len(called_vectors) == res["chunks_count"]

    first_chunk = called_chunks[0]
    assert first_chunk["upload_id"] == upload_id
    assert first_chunk["source_file"] == "test.pdf"
    assert 1 in first_chunk["element_ids"] or 2 in first_chunk["element_ids"]
    assert len(first_chunk["bbox"]) == 4


@pytest.mark.integration
def test_real_pdf_ingestion_with_java(tmp_path):
    """
    Optional integration test using bundled sample PDF and real Java execution.
    Skipped if sample PDF or Java is not present.
    """
    sample_pdf = r"C:\tmp\opendataloader-pdf\samples\pdf\lorem.pdf"
    if not os.path.exists(sample_pdf):
        pytest.skip("Sample PDF lorem.pdf not found in /tmp/opendataloader-pdf")

    # Check if java is in PATH
    if not shutil.which("java"):
        pytest.skip("Java 11+ is not installed or not in PATH")

    upload_id = "integration-test-lorem"
    res = ingest_worker.process(sample_pdf, upload_id, "lorem.pdf")

    assert res["status"] == "indexed"
    assert res["chunks_count"] >= 1
    assert any("Lorem Ipsum" in log or "Extracted" in log or "Successfully" in log for log in res["logs"])
