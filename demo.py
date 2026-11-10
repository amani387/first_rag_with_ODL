"""
End-to-End Demo Script for OpenDataLoader + LangChain + Vector Store integration.
Runs a complete ingestion, chunking, embedding, indexing, and retrieval cycle.
Supports Google Gemini, OpenAI, and Local Fallback with detailed error reporting.
"""
import sys
import os
from pathlib import Path

# Ensure app is importable
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from app import ingest_worker
from app import query_api
from app.query_api import QueryRequest
from app import config


def main():
    print("=" * 65)
    print("  OpenDataLoader + LangChain + Vector Store Pipeline Demo")
    print("=" * 65)

    print(f"Configured Gemini Key: {'YES (Set)' if config.GEMINI_API_KEY else 'NO (Unset)'}")
    print(f"Configured OpenAI Key: {'YES (Set)' if config.OPENAI_API_KEY else 'NO (Unset)'}")

    # Use sample PDF from cloned opendataloader-pdf repo
    sample_pdf = os.path.abspath(os.path.join(
        current_dir.parent.parent, "opendataloader-pdf", "samples", "pdf", "lorem.pdf"
    ))
    if not os.path.exists(sample_pdf):
        sample_pdf = r"C:\tmp\opendataloader-pdf\samples\pdf\lorem.pdf"

    if not os.path.exists(sample_pdf):
        print(f"Sample PDF not found at {sample_pdf}. Exiting.")
        return

    print(f"\n[1/3] Ingesting PDF: {sample_pdf}")
    upload_id = "gemini-demo-001"

    status = ingest_worker.process(
        pdf_path=sample_pdf,
        upload_id=upload_id,
        source_file="lorem.pdf"
    )

    print(f"Ingestion Status: {status.get('status')}")
    print(f"Total Chunks Indexed: {status.get('chunks_count')}")
    print(f"Active Vector Store: {status.get('vector_store', 'ChromaDB/Weaviate')}")
    print("Ingestion Logs:")
    for log in status.get("logs", []):
        print(f"  - {log}")

    print("\n[2/3] Querying Indexed Content...")
    test_question = "What is this document about, and what does it discuss?"
    print(f"Question: {test_question}")

    req = QueryRequest(
        upload_id=upload_id,
        question=test_question,
        top_k=2
    )

    res = query_api.query_documents(req)

    print("\n[3/3] Retrieval & QA Result:")
    print(f"Model Used: {res.model_used}")
    if res.error:
        print(f"\nAPI Error Reported:\n{res.error}\n")
    print(f"\nAnswer:\n{res.answer}\n")

    print("Sources with Bounding Box & Page Metadata:")
    for i, src in enumerate(res.sources, 1):
        print(f"\n  Source #{i}:")
        print(f"    - Chunk ID:    {src.chunk_id}")
        print(f"    - Page:        {src.page}")
        print(f"    - BoundingBox: {src.bbox}")
        print(f"    - Element IDs: {src.element_ids}")
        print(f"    - Snippet:     {src.text_snippet.strip()}")

    print("\n" + "=" * 65)
    print("  Demo run finished!")
    print("=" * 65)


if __name__ == "__main__":
    main()
