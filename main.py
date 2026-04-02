"""
Legal RAG System — Local Development Entry Point

Run with: python main.py
This starts a local server with both API and frontend.
For Vercel deployment, this file is not used (Vercel uses api/ endpoints directly).
"""
import sys
import os
import logging
import json
from pathlib import Path
from contextlib import asynccontextmanager

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.config import (
    FAISS_INDEX_PATH, BM25_INDEX_PATH, CHUNKS_METADATA_PATH,
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL, CORS_ORIGINS
)
from app.models import QueryRequest, QueryResponse, HealthResponse
from app.services.bm25_index import bm25_index
from app.services.vector_index import vector_index
from app.services.hybrid_retriever import load_chunks_metadata
from app.services.pipeline import run_ask_pipeline
from app.rate_limiter import rate_limit

# ==================== STARTUP ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading indices...")
    faiss_path = str(FAISS_INDEX_PATH)
    if Path(faiss_path).exists():
        vector_index.load(faiss_path)
    else:
        logger.warning(f"FAISS index not found. Run: python scripts/build_index.py")
    
    bm25_path = str(BM25_INDEX_PATH)
    if Path(bm25_path).exists():
        bm25_index.load(bm25_path)
    else:
        logger.warning(f"BM25 index not found. Run: python scripts/build_index.py")
    
    load_chunks_metadata()
    logger.info("Startup complete.")
    yield
    # Shutdown logic if any
    logger.info("Shutting down...")

# ==================== APP ====================
app = FastAPI(
    title="⚖️ Legal RAG Assistant",
    description="Junior Legal Assistant — Indian Law Q&A with Hybrid Search, Reranking, Grounding & Citations",
    version="2.1.0",
    lifespan=lifespan
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


# ==================== MAIN QUERY ENDPOINT ====================
@app.post("/api/ask", response_model=QueryResponse)
@rate_limit
async def ask_question(request: Request, query: QueryRequest):
    """Ask a legal question — full 8-step pipeline."""
    try:
        if query.min_confidence > 0.0:
            # We enforce returning the pipeline's decision on min_confidence inside pipeline.py
            pass
        return await run_ask_pipeline(query)
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal error occurred while processing your request. Please try again later."}
        )
        


# ==================== HEALTH ENDPOINT ====================
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health and status check."""
    total_chunks = 0
    meta_path = Path(CHUNKS_METADATA_PATH)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_chunks = len(data)
    
    return HealthResponse(
        status="healthy",
        total_chunks_indexed=total_chunks,
        bm25_index_loaded=bm25_index.is_loaded,
        faiss_index_loaded=vector_index.is_loaded,
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=LLM_MODEL
    )


# ==================== SERVE FRONTEND ====================
frontend_dir = Path(__file__).parent / "frontend"

@app.get("/")
async def serve_index():
    return FileResponse(str(frontend_dir / "index.html"))

if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ==================== RUN ====================
PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)