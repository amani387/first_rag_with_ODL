"""
Ingestion worker module.
Executes OpenDataLoader PDF conversion, parses elements, chunks them, computes embeddings,
and indexes vectors into Weaviate (or local Chroma fallback).
Handles edge cases: password-protected PDFs, malformed structure, empty content.
"""
import os
import sys
import io
import json
import logging
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional

import opendataloader_pdf

from app import config
from app.chunker import chunk_elements
from app.vector_store import PluggableEmbedding, get_vector_store

logger = logging.getLogger("ingest_worker")

# In-memory status registry; also backed up in files if needed
UPLOAD_STATUS: Dict[str, Dict[str, Any]] = {}


def get_status(upload_id: str) -> Dict[str, Any]:
    """Retrieves status and logs for an upload_id."""
    if upload_id in UPLOAD_STATUS:
        return UPLOAD_STATUS[upload_id]
    
    # Check if persisted on disk
    status_file = os.path.join(config.ODL_OUTPUT_DIR, upload_id, "status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {"upload_id": upload_id, "status": "unknown", "logs": ["Upload ID not found"]}


def update_status(upload_id: str, status: str, log_msg: Optional[str] = None, **kwargs):
    """Updates status dictionary and writes status.json to the output folder."""
    if upload_id not in UPLOAD_STATUS:
        UPLOAD_STATUS[upload_id] = {
            "upload_id": upload_id,
            "status": "queued",
            "logs": [],
            "chunks_count": 0
        }
    entry = UPLOAD_STATUS[upload_id]
    entry["status"] = status
    if log_msg:
        entry["logs"].append(log_msg)
        logger.info(f"[{upload_id}] {log_msg}")
    entry.update(kwargs)

    out_dir = os.path.join(config.ODL_OUTPUT_DIR, upload_id)
    os.makedirs(out_dir, exist_ok=True)
    status_file = os.path.join(out_dir, "status.json")
    try:
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
    except Exception:
        pass


def process(pdf_path: str, upload_id: str, source_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Core ingestion process:
    1. Validates file existence and size
    2. Converts PDF using opendataloader_pdf.convert
    3. Detects password-protection or encryption
    4. Parses JSON output
    5. Runs token-aware chunking
    6. Computes embeddings
    7. Indexes vectors into vector store
    """
    if not source_file:
        source_file = os.path.basename(pdf_path)

    out_dir = os.path.join(config.ODL_OUTPUT_DIR, upload_id)
    os.makedirs(out_dir, exist_ok=True)

    # Edge Case: Check file exists and has size
    if not os.path.exists(pdf_path) or os.path.getsize(pdf_path) == 0:
        update_status(upload_id, "failed", "Uploaded file is missing or empty (0 bytes).")
        return get_status(upload_id)

    update_status(upload_id, "processing", f"Starting conversion for file: {source_file}")

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            opendataloader_pdf.convert(
                input_path=[pdf_path],
                output_dir=out_dir,
                format="json,markdown",
                sanitize=True
            )
        conv_stdout = stdout_capture.getvalue()
        conv_stderr = stderr_capture.getvalue()

        # Edge Case: Password / Encryption Detection
        combined_logs = (conv_stdout + " " + conv_stderr).lower()
        if "password" in combined_logs or "encrypted" in combined_logs or "cryptography" in combined_logs:
            update_status(upload_id, "failed", "PDF is password-protected or encrypted. Please provide an unencrypted document.")
            return get_status(upload_id)

        if conv_stdout:
            update_status(upload_id, "processing", f"ODL Stdout: {conv_stdout.strip()}")
        if conv_stderr:
            update_status(upload_id, "processing", f"ODL Stderr: {conv_stderr.strip()}")
    except Exception as e:
        err_detail = traceback.format_exc()
        err_msg = str(e)
        if "password" in err_msg.lower() or "encrypt" in err_msg.lower():
            update_status(upload_id, "failed", "PDF is password-protected or encrypted. Please provide an unencrypted document.")
        else:
            update_status(upload_id, "failed", f"OpenDataLoader parser failed (possibly corrupted PDF): {err_msg}", error=err_detail)
        return get_status(upload_id)

    # Locate generated JSON file
    json_files = [f for f in os.listdir(out_dir) if f.endswith(".json") and f != "status.json"]
    if not json_files:
        update_status(upload_id, "failed", "No parseable JSON produced by OpenDataLoader. The PDF might be corrupted or image-only without OCR.")
        return get_status(upload_id)

    json_path = os.path.join(out_dir, json_files[0])
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        update_status(upload_id, "failed", f"Failed to parse ODL JSON output: {str(e)}")
        return get_status(upload_id)

    # Extract elements from ODL structure
    elements = []
    if isinstance(data, dict):
        elements = data.get("kids") or []
    elif isinstance(data, list):
        elements = data

    # Edge Case: PDF had 0 text/content elements
    if not elements:
        update_status(upload_id, "failed", "No extractable elements found in PDF (empty document or scanned images without digital text layer).")
        return get_status(upload_id)

    update_status(upload_id, "processing", f"Extracted {len(elements)} raw elements from ODL JSON")

    # Chunk elements preserving metadata
    chunks = chunk_elements(
        elements=elements,
        upload_id=upload_id,
        source_file=source_file,
        max_tokens=config.CHUNK_MAX_TOKENS,
        overlap_tokens=config.CHUNK_OVERLAP_TOKENS
    )

    if not chunks:
        update_status(upload_id, "failed", "No valid text chunks could be created from document elements.")
        return get_status(upload_id)

    update_status(upload_id, "processing", f"Generated {len(chunks)} token-bounded chunks with metadata")

    # Compute embeddings
    try:
        embedding_model = PluggableEmbedding()
        texts = [c["text"] for c in chunks]
        update_status(upload_id, "processing", f"Computing embeddings using {embedding_model.provider} provider...")
        vectors = embedding_model.embed_documents(texts)
    except Exception as e:
        err_detail = traceback.format_exc()
        err_summary = str(e)
        update_status(upload_id, "failed", f"Embedding computation failed: {err_summary}", error=err_detail)
        return get_status(upload_id)

    # Index into Vector Store (Weaviate or Chroma fallback)
    try:
        store = get_vector_store()
        store_type = "Weaviate" if getattr(store, "class_name", None) else "ChromaDB"
        update_status(upload_id, "processing", f"Indexing {len(chunks)} chunks into {store_type}...")
        store.upsert_chunks(chunks, vectors)
        update_status(upload_id, "indexed", f"Successfully indexed {len(chunks)} chunks into {store_type}", chunks_count=len(chunks), vector_store=store_type)
    except Exception as e:
        err_detail = traceback.format_exc()
        update_status(upload_id, "failed", f"Vector indexing failed: {str(e)}", error=err_detail)
        return get_status(upload_id)

    return get_status(upload_id)
