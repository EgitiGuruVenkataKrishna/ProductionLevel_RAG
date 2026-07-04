"""
Vercel Serverless Function: /api/ask

Main query endpoint for the Legal RAG assistant.
Now uses shared pipeline logic.
"""
import sys
import os
import logging
import threading
from pathlib import Path

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    FAISS_INDEX_PATH, BM25_INDEX_PATH, CORS_ORIGINS
)
from app.models import QueryRequest, QueryResponse
from app.services.bm25_index import bm25_index
from app.services.vector_index import vector_index
from app.services.hybrid_retriever import load_chunks_metadata
from app.services.pipeline import run_ask_pipeline, run_ask_pipeline_stream
from app.rate_limiter import rate_limit
from app.middleware.auth import get_current_user_session, JWT_SECRET, JWT_ALGORITHM
from sse_starlette.sse import EventSourceResponse
import jwt
import datetime
import uuid

app = FastAPI(
    title="Legal RAG Assistant API",
    description="Junior Legal Assistant — Indian Law Q&A with hybrid search, reranking, grounding, and citations",
    version="2.1.0"
)

# Parse CORS origins
origin_list = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()]
if not origin_list:
    origin_list = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origin_list,
    allow_credentials="*" not in origin_list, # Prevent allow_credentials with wildcard
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== STARTUP: Load Indices ====================
_initialized = False
_init_lock = threading.Lock()

def _ensure_initialized():
    """Lazy initialization of indices (for serverless cold starts) with thread safety."""
    global _initialized
    if _initialized:
        return
        
    with _init_lock:
        if _initialized:
            return
            
        logger.info("Initializing indices...")
        
        faiss_path = str(FAISS_INDEX_PATH)
        if Path(faiss_path).exists():
            vector_index.load(faiss_path)
        else:
            logger.warning(f"FAISS index not found at {faiss_path}")
        
        bm25_path = str(BM25_INDEX_PATH)
        if Path(bm25_path).exists():
            bm25_index.load(bm25_path)
        else:
            logger.warning(f"BM25 index not found at {bm25_path}")
        
        load_chunks_metadata()
        _initialized = True
        logger.info("Initialization complete.")


# ==================== ENDPOINTS ====================

@app.get("/api/auth/token")
async def generate_dev_token():
    """
    DEV ONLY: Generate a valid JWT token for testing.
    In a real app, this would be your login endpoint.
    """
    payload = {
        "sub": f"user_{uuid.uuid4().hex[:8]}",
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/api/ask", response_model=QueryResponse)
@rate_limit
async def ask_question(request: Request, query: QueryRequest, session_id: str = Depends(get_current_user_session)):
    """
    Ask a legal question — full 8-step pipeline.
    """
    _ensure_initialized()
    
    # Override client-provided session_id with the secure one from JWT
    query.session_id = session_id
    
    try:
        return await run_ask_pipeline(query)
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        # Mask sensitive error details for client
        return JSONResponse(
            status_code=500, 
            content={"detail": "An internal error occurred while processing your request. Please try again later."}
        )

@app.post("/api/ask/stream")
@rate_limit
async def ask_question_stream(request: Request, query: QueryRequest, session_id: str = Depends(get_current_user_session)):
    """
    Ask a legal question — streams back response chunks and JSON updates.
    """
    _ensure_initialized()
    
    # Override client-provided session_id with the secure one from JWT
    query.session_id = session_id
    
    async def event_generator():
        try:
            async for event_json in run_ask_pipeline_stream(query):
                yield {"data": event_json}
        except Exception as e:
            logger.error(f"Stream Query error: {e}", exc_info=True)
            yield {"data": '{"status": "error", "message": "An internal error occurred."}'}

    return EventSourceResponse(event_generator())
