# OpenDataLoader PDF + LangChain + Weaviate Local Integration

This project is a local RAG integration that:
1. Ingests user-uploaded PDFs using **OpenDataLoader PDF**.
2. Performs token-aware chunking preserving bounding boxes (`bbox`), page numbers, and original element IDs.
3. Computes embeddings (via **LangChain OpenAIEmbeddings** or pluggable local embeddings) and stores vectors in **Weaviate** (with automated **ChromaDB fallback** when Docker is not running).
4. Exposes a **FastAPI** service with background processing (`POST /upload_pdf`, `GET /status/{upload_id}`) and semantic QA retrieval (`POST /query`).

---

## Workspace Structure

```
integration/
├── app/
│   ├── __init__.py
│   ├── config.py           # Environment variables and path configurations
│   ├── chunker.py          # Token-aware chunker with bbox & element metadata preservation
│   ├── vector_store.py     # Pluggable vector store (Weaviate + ChromaDB fallback) and embeddings
│   ├── ingest_worker.py    # OpenDataLoader PDF extraction, chunking, and indexing worker
│   ├── query_api.py        # POST /query endpoint with vector search & LangChain QA
│   ├── ingest_api.py       # Main FastAPI application with /upload_pdf and /status endpoints
│   └── requirements.txt    # Application dependencies
├── docker/
│   ├── Dockerfile          # Container build with Java 17/11+ and Python runtime
│   └── docker-compose.yml  # Weaviate vector DB & optional API service
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py     # Token limits, overlap, and metadata preservation tests
│   ├── test_ingest_worker.py # Mocked & real PDF ingestion tests
│   └── test_api.py         # FastAPI endpoint integration tests
├── pytest.ini
└── README.md
```

---

## Prerequisites

- **Python 3.10+** (tested with Python 3.12)
- **Java 11+** (required to run `opendataloader-pdf` JAR execution)
- **Docker & Docker Compose** (optional for Weaviate; if Docker is stopped, the application seamlessly falls back to local ChromaDB)
- **OpenAI API Key** (optional; if not set, local embeddings and synthesis are used)

---

## Quickstart & Local Setup

### 1. Start Weaviate (Docker)
```bash
cd integration/docker
docker compose up -d weaviate
```
> *Note: If Docker is not running, the system will automatically fall back to local persistent ChromaDB without throwing errors.*

### 2. Create Virtual Environment & Install Dependencies
```bash
# From the workspace root:
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install requirements
pip install -r integration/app/requirements.txt
```

### 3. Configure Environment Variables (Optional)
```bash
# Linux/macOS
export OPENAI_API_KEY="your-openai-api-key"

# Windows PowerShell
$env:OPENAI_API_KEY="your-openai-api-key"
```

### 4. Run the FastAPI Server
```bash
# Navigate to the integration folder or ensure PYTHONPATH includes it
cd integration
uvicorn app.ingest_api:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage Examples

### 1. Upload a PDF
```bash
curl -F "file=@sample.pdf" http://localhost:8000/upload_pdf
```
**Response:**
```json
{
  "upload_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "filename": "sample.pdf",
  "status": "processing"
}
```

### 2. Check Ingestion Status
```bash
curl http://localhost:8000/status/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d
```
**Response:**
```json
{
  "upload_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "status": "indexed",
  "chunks_count": 4,
  "vector_store": "Weaviate",
  "logs": [
    "File sample.pdf queued for ingestion.",
    "Starting conversion for file: sample.pdf",
    "Extracted 8 raw elements from ODL JSON",
    "Generated 4 token-bounded chunks with metadata",
    "Successfully indexed 4 chunks into Weaviate"
  ]
}
```

### 3. Query the Document
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "upload_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
    "question": "What is covered in the agreement?",
    "top_k": 3
  }'
```
**Response:**
```json
{
  "answer": "The agreement outlines terms and conditions between the parties...",
  "sources": [
    {
      "chunk_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d_chunk_1",
      "page": 1,
      "bbox": [50.0, 700.0, 300.0, 740.0],
      "element_ids": [1, 2],
      "source_file": "sample.pdf",
      "text_snippet": "Master Services Agreement\nThis agreement is entered into..."
    }
  ]
}
```

---

## Running Tests

Run the test suite using `pytest`:

```bash
cd integration
pytest -v
```

To run only fast unit tests without Java integration:
```bash
pytest -v -m "not integration"
```

---

## Troubleshooting & Build Notes

- **Java Requirement:** OpenDataLoader executes a bundled Java CLI JAR. If `java -version` is missing or below 11, install OpenJDK 11+ (or Temurin).
- **Missing JAR Build from Local Source:** When cloning `opendataloader-pdf`, the Java module `opendataloader-pdf-cli` requires `mvn package` in `java/` to produce the JAR. If Maven is not installed on the system, the project seamlessly uses the pre-built distribution from PyPI (`pip install opendataloader-pdf`) which bundles the tested JAR.
- **Docker Unavailable:** If the Docker daemon is offline or Weaviate cannot be reached at `WEAVIATE_URL`, the vector store automatically uses local `ChromaDB` persisted at `data/chroma`, ensuring 100% uninterrupted operation.
