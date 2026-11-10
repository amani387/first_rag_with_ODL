"""
Main FastAPI application module.
Provides /upload_pdf, /status/{upload_id}, /query endpoints and background worker dispatch.
Serves BeautifulUI web frontend and handles HTTP errors (400, 404, 500, 503, file validation edge cases).
"""
import os
import uuid
import logging
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import config
from app import ingest_worker
from app.query_api import router as query_router
from app.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest_api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

app = FastAPI(
    title="OpenDataLoader PDF + LangChain + Weaviate RAG API",
    description="Local integration demo ingesting PDFs with OpenDataLoader, chunking with LangChain, and storing in Weaviate.",
    version="1.0.0"
)

# Enable CORS for local frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include query endpoints
app.include_router(query_router)

# Mount static files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def render_error_html(status_code: int, title: str, description: str) -> str:
    """Renders the BeautifulUI styled error page."""
    error_file = STATIC_DIR / "error.html"
    if error_file.exists():
        with open(error_file, "r", encoding="utf-8") as f:
            template = f.read()
        badge_class = "amber" if status_code < 500 else ""
        return (
            template.replace("{{status_code}}", str(status_code))
                    .replace("{{error_title}}", title)
                    .replace("{{error_description}}", description)
                    .replace("{{badge_class}}", badge_class)
        )
    return f"<h1>{status_code} - {title}</h1><p>{description}</p><p><a href='/'>Return to Home</a></p>"


# --- Global Exception Handlers ---

@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handles HTTP 400, 404, 503 and other client/server status codes."""
    accept_header = request.headers.get("accept", "").lower()
    titles = {
        400: "Bad Request",
        404: "Page or Resource Not Found",
        500: "Internal Server Error",
        503: "Service Unavailable"
    }
    title = titles.get(exc.status_code, "Error")

    if "text/html" in accept_header and request.method == "GET":
        return HTMLResponse(
            content=render_error_html(exc.status_code, title, str(exc.detail)),
            status_code=exc.status_code
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": title, "status_code": exc.status_code, "detail": str(exc.detail)}
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Catches unhandled exceptions and returns clean 500 responses."""
    logger.error(f"Unhandled 500 Exception: {str(exc)}", exc_info=True)
    accept_header = request.headers.get("accept", "").lower()
    detail = "An unexpected server error occurred. Please verify backend logs or retry."

    if "text/html" in accept_header and request.method == "GET":
        return HTMLResponse(
            content=render_error_html(500, "Internal Server Error", detail),
            status_code=500
        )
    return JSONResponse(
        status_code=500,
        content={"error": "Internal Server Error", "status_code": 500, "detail": str(exc)}
    )


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the BeautifulUI-inspired interactive web application."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "service": "OpenDataLoader + LangChain + Weaviate RAG API",
        "ui": "Place index.html in app/static to view the web interface."
    }


@app.get("/ui", response_class=HTMLResponse)
def serve_ui_alias():
    """Alias to access the web UI at /ui."""
    return serve_ui()


@app.get("/health")
def health_check():
    store = get_vector_store()
    is_weaviate = getattr(store, "class_name", None) is not None
    return {
        "status": "ok",
        "vector_store": "Weaviate" if is_weaviate else "ChromaDB (Local Fallback)",
        "gemini_configured": bool(config.GEMINI_API_KEY),
        "openai_configured": bool(config.OPENAI_API_KEY),
        "gemini_model": config.GEMINI_LLM_MODEL
    }


@app.post("/upload_pdf")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Accepts a PDF file upload, runs deep content validation (extension, size, magic bytes),
    saves it with a UUID upload_id, and schedules ingest_worker.process in background.
    """
    # 1. Edge Case: File extension check
    filename = file.filename or "unknown.pdf"
    if not filename.lower().endswith(".pdf"):
        ext = filename.split(".")[-1] if "." in filename else "none"
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type (detected '.{ext}'). Only valid .pdf documents are supported."
        )

    # Read incoming bytes
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read incoming file stream: {str(e)}")

    # 2. Edge Case: Empty file (0 bytes)
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty (0 bytes). Please upload a valid PDF document."
        )

    # 3. Edge Case: File too large (> 50MB)
    if len(content) > MAX_FILE_SIZE:
        size_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size ({size_mb:.1f} MB) exceeds maximum allowed limit of 50 MB."
        )

    # 4. Edge Case: Malformed or Spoofed PDF (Magic Bytes Validation)
    # A genuine PDF must have %PDF- in the first 1024 bytes
    header_chunk = content[:1024]
    if b"%PDF-" not in header_chunk:
        # Check if user renamed an image, office doc, or binary to .pdf
        if content.startswith(b"\x89PNG"):
            detail = "Malformed file: This file is a PNG image renamed with a .pdf extension."
        elif content.startswith(b"\xff\xd8\xff"):
            detail = "Malformed file: This file is a JPEG image renamed with a .pdf extension."
        elif content.startswith(b"PK\x03\x04"):
            detail = "Malformed file: This file is a ZIP/Office archive (.docx/.xlsx) renamed with a .pdf extension."
        elif content.startswith(b"MZ"):
            detail = "Security rejection: Binary executable (.exe/.dll) cannot be uploaded."
        else:
            detail = "Malformed PDF: The file header does not match the PDF standard specification (%PDF- missing)."
        raise HTTPException(status_code=400, detail=detail)

    upload_id = str(uuid.uuid4())
    saved_filename = f"{upload_id}_{filename}"
    saved_path = os.path.join(config.UPLOAD_DIR, saved_filename)

    try:
        with open(saved_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file to server storage: {str(e)}")

    # Initialize status
    ingest_worker.update_status(
        upload_id=upload_id,
        status="queued",
        log_msg=f"File {filename} ({len(content) / 1024:.1f} KB) verified and queued for ingestion."
    )

    # Dispatch background worker
    background_tasks.add_task(
        ingest_worker.process,
        pdf_path=saved_path,
        upload_id=upload_id,
        source_file=filename
    )

    return {
        "upload_id": upload_id,
        "filename": filename,
        "size_bytes": len(content),
        "status": "processing"
    }


@app.get("/status/{upload_id}")
def get_upload_status(upload_id: str):
    """
    Returns current processing state (queued / processing / indexed / failed) and detailed logs.
    """
    status_info = ingest_worker.get_status(upload_id)
    if not status_info or status_info.get("status") == "unknown":
        raise HTTPException(
            status_code=404,
            detail=f"Upload ID '{upload_id}' not found. It may have expired or was never uploaded."
        )
    return status_info
