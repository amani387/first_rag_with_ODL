"""
Vector Store abstraction supporting Weaviate with automatic ChromaDB fallback.
Provides pluggable embedding computation (Google Gemini, OpenAI, or Local).
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional
import hashlib

from app import config

logger = logging.getLogger("vector_store")


class PluggableEmbedding:
    """
    Pluggable embedding provider:
    - Uses Google Gemini embeddings if GEMINI_API_KEY is configured.
    - Uses LangChain OpenAIEmbeddings if OPENAI_API_KEY is configured.
    - Uses SentenceTransformers or deterministic local embedding if no API key is provided.
    """
    def __init__(self):
        self.gemini_key = config.GEMINI_API_KEY
        self.openai_key = config.OPENAI_API_KEY
        self.provider = "local"
        self._gemini_client = None
        self._openai_client = None
        self._local_model = None
        self.last_error = None

        # 1. Check Gemini
        if self.gemini_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self._gemini_client = GoogleGenerativeAIEmbeddings(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    google_api_key=self.gemini_key
                )
                self.provider = "gemini"
                logger.info(f"Initialized Gemini Embeddings with model={config.GEMINI_EMBEDDING_MODEL}")
            except Exception as e:
                self.last_error = f"Gemini embedding init error: {str(e)}"
                logger.warning(self.last_error)

        # 2. Check OpenAI
        if self.provider == "local" and self.openai_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                self._openai_client = OpenAIEmbeddings(
                    model=config.OPENAI_EMBEDDING_MODEL,
                    api_key=self.openai_key
                )
                self.provider = "openai"
                logger.info(f"Initialized OpenAIEmbeddings with model={config.OPENAI_EMBEDDING_MODEL}")
            except Exception as e:
                self.last_error = f"OpenAI embedding init error: {str(e)}"
                logger.warning(self.last_error)

        # 3. Local fallback
        if self.provider == "local":
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
                logger.info(f"Initialized local SentenceTransformer with model={config.LOCAL_EMBEDDING_MODEL}")
            except Exception as e:
                logger.warning(f"Could not load SentenceTransformer ({e}). Using deterministic offline embedding.")
                self._local_model = None

    def _fallback_vector(self, text: str, dim: int = 384) -> List[float]:
        """Generates a normalized deterministic embedding vector from text."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = []
        for i in range(dim):
            byte_val = h[i % len(h)]
            vec.append((byte_val - 128.0) / 128.0)
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        
        # Try Gemini
        if self.provider == "gemini" and self._gemini_client:
            try:
                return self._gemini_client.embed_documents(texts)
            except Exception as e:
                err_msg = f"[Gemini Embeddings Error]: {type(e).__name__}: {str(e)}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        # Try OpenAI
        if self.provider == "openai" and self._openai_client:
            try:
                return self._openai_client.embed_documents(texts)
            except Exception as e:
                err_msg = f"[OpenAI Embeddings Error]: {type(e).__name__}: {str(e)}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        # Local
        if self._local_model:
            embeddings = self._local_model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        else:
            return [self._fallback_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if not text:
            return self._fallback_vector("")
        
        # Try Gemini
        if self.provider == "gemini" and self._gemini_client:
            try:
                return self._gemini_client.embed_query(text)
            except Exception as e:
                err_msg = f"[Gemini Embeddings Error]: {type(e).__name__}: {str(e)}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        # Try OpenAI
        if self.provider == "openai" and self._openai_client:
            try:
                return self._openai_client.embed_query(text)
            except Exception as e:
                err_msg = f"[OpenAI Embeddings Error]: {type(e).__name__}: {str(e)}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

        # Local
        if self._local_model:
            embedding = self._local_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        else:
            return self._fallback_vector(text)


class WeaviateStore:
    """Weaviate client wrapper using weaviate-client v4."""
    def __init__(self, client):
        self.client = client
        self.class_name = config.WEAVIATE_CLASS_NAME
        self._ensure_schema()

    def _ensure_schema(self):
        try:
            import weaviate.classes as wvc
            if not self.client.collections.exists(self.class_name):
                self.client.collections.create(
                    name=self.class_name,
                    properties=[
                        wvc.config.Property(name="text", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="upload_id", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="source_file", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="page", data_type=wvc.config.DataType.INT),
                        wvc.config.Property(name="bbox", data_type=wvc.config.DataType.NUMBER_ARRAY),
                        wvc.config.Property(name="element_ids", data_type=wvc.config.DataType.INT_ARRAY),
                        wvc.config.Property(name="chunk_id", data_type=wvc.config.DataType.TEXT),
                        wvc.config.Property(name="heading", data_type=wvc.config.DataType.TEXT),
                    ]
                )
                logger.info(f"Created Weaviate collection: {self.class_name}")
        except Exception as e:
            logger.error(f"Error ensuring Weaviate schema: {e}")

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        collection = self.client.collections.get(self.class_name)
        with collection.batch.dynamic() as batch:
            for chunk, vector in zip(chunks, vectors):
                props = {
                    "text": chunk["text"],
                    "upload_id": chunk["upload_id"],
                    "source_file": chunk["source_file"],
                    "page": int(chunk["page"]),
                    "bbox": [float(x) for x in chunk["bbox"]],
                    "element_ids": [int(x) for x in chunk["element_ids"]],
                    "chunk_id": chunk["chunk_id"],
                    "heading": chunk.get("extra_metadata", {}).get("heading", "")
                }
                batch.add_object(
                    properties=props,
                    vector=vector
                )

    def query_chunks(self, query_vector: List[float], upload_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        import weaviate.classes as wvc
        collection = self.client.collections.get(self.class_name)
        response = collection.query.near_vector(
            near_vector=query_vector,
            limit=top_k,
            filters=wvc.query.Filter.by_property("upload_id").equal(upload_id),
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        results = []
        for obj in response.objects:
            p = obj.properties
            results.append({
                "chunk_id": p.get("chunk_id"),
                "upload_id": p.get("upload_id"),
                "source_file": p.get("source_file"),
                "text": p.get("text"),
                "page": p.get("page"),
                "bbox": p.get("bbox", []),
                "element_ids": p.get("element_ids", []),
                "heading": p.get("heading", ""),
                "distance": getattr(obj.metadata, "distance", None)
            })
        return results


class ChromaStore:
    """
    Local ChromaDB fallback vector store.
    Namespaces collections by embedding dimension/provider to prevent dimension mismatch errors.
    """
    def __init__(self, persist_dir: str):
        import chromadb
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=persist_dir)

    def _get_collection(self, dimension: int):
        col_name = f"pdf_chunks_dim_{dimension}"
        return self.client.get_or_create_collection(name=col_name)

    def upsert_chunks(self, chunks: List[Dict[str, Any]], vectors: List[List[float]]):
        if not chunks or not vectors:
            return
        dim = len(vectors[0])
        col = self._get_collection(dim)

        ids = [c["chunk_id"] for c in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = []
        for c in chunks:
            metadatas.append({
                "upload_id": str(c["upload_id"]),
                "source_file": str(c["source_file"]),
                "page": int(c["page"]),
                "bbox": json.dumps(c.get("bbox", [])),
                "element_ids": json.dumps(c.get("element_ids", [])),
                "heading": str(c.get("extra_metadata", {}).get("heading", "")),
                "chunk_id": str(c["chunk_id"])
            })
        col.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas
        )

    def query_chunks(self, query_vector: List[float], upload_id: str, top_k: int = 4) -> List[Dict[str, Any]]:
        dim = len(query_vector)
        col = self._get_collection(dim)
        results = col.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where={"upload_id": upload_id}
        )
        items = []
        if results and results.get("ids") and results["ids"][0]:
            count = len(results["ids"][0])
            for i in range(count):
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                doc = results["documents"][0][i] if results.get("documents") else ""
                dist = results["distances"][0][i] if results.get("distances") else None
                
                bbox = []
                if "bbox" in meta:
                    try:
                        bbox = json.loads(meta["bbox"])
                    except Exception:
                        pass
                
                element_ids = []
                if "element_ids" in meta:
                    try:
                        element_ids = json.loads(meta["element_ids"])
                    except Exception:
                        pass

                items.append({
                    "chunk_id": meta.get("chunk_id", results["ids"][0][i]),
                    "upload_id": meta.get("upload_id", upload_id),
                    "source_file": meta.get("source_file", ""),
                    "text": doc,
                    "page": meta.get("page", 1),
                    "bbox": bbox,
                    "element_ids": element_ids,
                    "heading": meta.get("heading", ""),
                    "distance": dist
                })
        return items


def get_vector_store():
    """
    Attempts to connect to Weaviate. If unreachable, falls back to ChromaDB.
    """
    try:
        import weaviate
        from urllib.parse import urlparse
        parsed = urlparse(config.WEAVIATE_URL)
        host = parsed.hostname or "localhost"
        port = parsed.port or 8080

        client = weaviate.connect_to_custom(
            http_host=host,
            http_port=port,
            http_secure=parsed.scheme == "https",
            grpc_host=host,
            grpc_port=config.WEAVIATE_GRPC_PORT,
            grpc_secure=parsed.scheme == "https",
        )
        if client.is_ready():
            logger.info("Connected to Weaviate successfully.")
            return WeaviateStore(client)
    except Exception as e:
        logger.info(f"Weaviate not accessible ({e}). Using local ChromaDB fallback.")

    return ChromaStore(persist_dir=config.CHROMA_DIR)
