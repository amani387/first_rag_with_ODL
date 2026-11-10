"""
API endpoint integration tests using FastAPI TestClient.
Tests /health, /upload_pdf, /status, and /query.
"""
import io
import time
import pytest
from fastapi.testclient import TestClient

from app.ingest_api import app
from app import ingest_worker

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "vector_store" in data


def test_upload_and_status_and_query_flow(monkeypatch):
    upload_id_intercepted = None

    # Stub process so background task runs quickly and deterministically
    def stub_process(pdf_path, upload_id, source_file=None):
        nonlocal upload_id_intercepted
        upload_id_intercepted = upload_id
        ingest_worker.update_status(
            upload_id=upload_id,
            status="indexed",
            log_msg="Stub indexing complete",
            chunks_count=2
        )

    monkeypatch.setattr("app.ingest_worker.process", stub_process)

    # 1. Test upload_pdf
    pdf_bytes = b"%PDF-1.4 mock pdf header content"
    files = {"file": ("demo.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
    upload_resp = client.post("/upload_pdf", files=files)
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert "upload_id" in upload_data
    assert upload_data["status"] == "processing"

    target_upload_id = upload_data["upload_id"]

    # 2. Test status endpoint
    status_resp = client.get(f"/status/{target_upload_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["upload_id"] == target_upload_id
    assert status_data["status"] in ["queued", "indexed"]

    # 3. Test query endpoint with mocked retrieved chunks
    def stub_query_chunks(query_vector, upload_id, top_k=4):
        return [{
            "chunk_id": f"{upload_id}_c1",
            "upload_id": upload_id,
            "source_file": "demo.pdf",
            "text": "OpenDataLoader extracts high quality structured data from PDFs.",
            "page": 1,
            "bbox": [10.0, 20.0, 100.0, 50.0],
            "element_ids": [1],
            "heading": "Overview"
        }]

    class MockStore:
        def query_chunks(self, query_vector, upload_id, top_k=4):
            return stub_query_chunks(query_vector, upload_id, top_k)

    monkeypatch.setattr("app.query_api.get_vector_store", lambda: MockStore())

    query_payload = {
        "upload_id": target_upload_id,
        "question": "What does OpenDataLoader do?",
        "top_k": 2
    }
    query_resp = client.post("/query", json=query_payload)
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert "sources" in query_data
    assert len(query_data["sources"]) == 1
    src = query_data["sources"][0]
    assert src["chunk_id"] == f"{target_upload_id}_c1"
    assert src["page"] == 1
    assert src["bbox"] == [10.0, 20.0, 100.0, 50.0]
