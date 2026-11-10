"""
Query API module.
Handles semantic retrieval from Weaviate/Chroma and generates answers with Gemini / LangChain.
Provides exact error feedback if API keys fail.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import config
from app.vector_store import PluggableEmbedding, get_vector_store

logger = logging.getLogger("query_api")
router = APIRouter()


class QueryRequest(BaseModel):
    upload_id: str = Field(..., description="ID of the uploaded PDF to query against")
    question: str = Field(..., description="User's natural language question")
    top_k: int = Field(default=4, description="Number of nearest chunks to retrieve")


class SourceItem(BaseModel):
    chunk_id: str
    page: int
    bbox: List[float] = []
    element_ids: List[Any] = []
    source_file: str = ""
    text_snippet: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    model_used: Optional[str] = None
    error: Optional[str] = None


def answer_with_llm(question: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates answer using Google Gemini (preferred) or OpenAI if keys are provided.
    If the API call fails, captures and returns the exact error message.
    Falls back to local extractive summary when no cloud keys are provided.
    """
    if not chunks:
        return {
            "answer": "I could not find any relevant information in the uploaded document.",
            "model": "none",
            "error": None
        }

    context_str = "\n\n---\n\n".join(
        f"[Source: {c.get('source_file')}, Page {c.get('page')}, Chunk {c.get('chunk_id')}]\n{c.get('text')}"
        for c in chunks
    )

    # 1. Try Google Gemini
    if config.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_LLM_MODEL,
                temperature=0.2,
                google_api_key=config.GEMINI_API_KEY
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an AI assistant answering questions based solely on the provided document excerpts. "
                           "Be accurate, concise, and helpful. If the answer cannot be determined from the excerpts, say so."),
                ("human", "Document Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"context": context_str, "question": question})
            return {
                "answer": result,
                "model": f"Google Gemini ({config.GEMINI_LLM_MODEL})",
                "error": None
            }
        except Exception as e:
            err_msg = f"[Gemini API Error]: {type(e).__name__}: {str(e)}"
            logger.error(err_msg)
            # Return the exact error so the user knows why the key failed
            top_chunk = chunks[0]
            fallback_ans = (
                f"{err_msg}\n\n"
                f"--- [Local Fallback Response from Page {top_chunk.get('page')}] ---\n"
                f"{top_chunk.get('text')[:300]}..."
            )
            return {
                "answer": fallback_ans,
                "model": "Fallback Extractive (Gemini Failed)",
                "error": err_msg
            }

    # 2. Try OpenAI
    if config.OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatOpenAI(
                model=config.OPENAI_LLM_MODEL,
                temperature=0.0,
                api_key=config.OPENAI_API_KEY
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are an assistant answering questions based solely on the provided document excerpts."),
                ("human", "Context excerpts:\n{context}\n\nQuestion: {question}\n\nAnswer:")
            ])
            chain = prompt | llm | StrOutputParser()
            result = chain.invoke({"context": context_str, "question": question})
            return {
                "answer": result,
                "model": f"OpenAI ({config.OPENAI_LLM_MODEL})",
                "error": None
            }
        except Exception as e:
            err_msg = f"[OpenAI API Error]: {type(e).__name__}: {str(e)}"
            logger.error(err_msg)
            top_chunk = chunks[0]
            fallback_ans = (
                f"{err_msg}\n\n"
                f"--- [Local Fallback Response from Page {top_chunk.get('page')}] ---\n"
                f"{top_chunk.get('text')[:300]}..."
            )
            return {
                "answer": fallback_ans,
                "model": "Fallback Extractive (OpenAI Failed)",
                "error": err_msg
            }

    # 3. Local / Offline Extractive Synthesis
    top_chunk = chunks[0]
    return {
        "answer": (
            f"[Local Extractive Synthesis - No API Key Set]: Based on Page {top_chunk.get('page')}: "
            f"\"{top_chunk.get('text')[:300]}...\""
        ),
        "model": "Local Extractive",
        "error": None
    }


@router.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    """
    Retrieves top_k chunks for upload_id and generates QA response with source metadata.
    """
    store = get_vector_store()
    embedding_model = PluggableEmbedding()

    # 1. Embed query
    try:
        query_vector = embedding_model.embed_query(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to embed query: {str(e)}")

    # 2. Vector search
    try:
        retrieved = store.query_chunks(query_vector=query_vector, upload_id=req.upload_id, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

    if not retrieved:
        return QueryResponse(
            answer="No indexed chunks found for this upload_id. Please verify the upload status.",
            sources=[],
            model_used="None"
        )

    # 3. Generate Answer
    llm_output = answer_with_llm(req.question, retrieved)

    # 4. Format sources with chunk_id, page, bbox
    sources = []
    for item in retrieved:
        sources.append(SourceItem(
            chunk_id=item.get("chunk_id", ""),
            page=item.get("page", 1),
            bbox=item.get("bbox", []),
            element_ids=item.get("element_ids", []),
            source_file=item.get("source_file", ""),
            text_snippet=item.get("text", "")[:250]
        ))

    return QueryResponse(
        answer=llm_output["answer"],
        sources=sources,
        model_used=llm_output.get("model"),
        error=llm_output.get("error")
    )
