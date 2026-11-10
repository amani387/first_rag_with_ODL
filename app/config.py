"""
Configuration module for the LangChain + OpenDataLoader + Weaviate integration.
Loads settings from environment variables and .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# API Keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Weaviate Settings
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_CLASS_NAME = os.getenv("WEAVIATE_CLASS_NAME", "PDFChunk")

# Directories
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads"))
ODL_OUTPUT_DIR = os.getenv("ODL_OUTPUT_DIR", str(BASE_DIR / "data" / "odl_output"))
CHROMA_DIR = os.getenv("CHROMA_DIR", str(BASE_DIR / "data" / "chroma"))

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ODL_OUTPUT_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Chunking Parameters
CHUNK_MAX_TOKENS = int(os.getenv("CHUNK_MAX_TOKENS", "700"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

# Models
GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-3.6-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
