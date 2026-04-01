"""
Vercel Serverless Function: /api/health

Health check and system status endpoint.
"""
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL,
    FAISS_INDEX_PATH, BM25_INDEX_PATH, CHUNKS_METADATA_PATH
)
from app.models import HealthResponse
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """System health and status check."""
    
    # Count chunks
    total_chunks = 0
    meta_path = Path(CHUNKS_METADATA_PATH)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            total_chunks = len(data)
    
    return HealthResponse(
        status="healthy",
        total_chunks_indexed=total_chunks,
        bm25_index_loaded=Path(BM25_INDEX_PATH).exists(),
        faiss_index_loaded=(Path(FAISS_INDEX_PATH) / "index.bin").exists(),
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=LLM_MODEL
    )
