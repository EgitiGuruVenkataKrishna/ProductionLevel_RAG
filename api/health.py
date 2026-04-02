"""
Vercel Serverless Function: /api/health

Health check and system status endpoint.
"""
import sys
import os
import json
import httpx
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import (
    EMBEDDING_MODEL, RERANKER_MODEL, LLM_MODEL,
    FAISS_INDEX_PATH, BM25_INDEX_PATH, CHUNKS_METADATA_PATH,
    CORS_ORIGINS, GROQ_API_KEY, HF_API_TOKEN, HF_EMBEDDING_URL
)

app = FastAPI()

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


class HealthResponse(BaseModel):
    status: str
    total_chunks_indexed: int
    bm25_index_loaded: bool
    faiss_index_loaded: bool
    embedding_model: str
    reranker_model: str
    llm_model: str
    # Deep health status
    groq_api_status: str = Field(default="unknown")
    hf_api_status: str = Field(default="unknown")


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
            
    # Check APIs
    groq_status = "ok"
    if not GROQ_API_KEY:
        groq_status = "missing_key"
    else:
        # Just check if we can connect to Groq
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                res = await client.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {GROQ_API_KEY}"})
                if not res.is_success:
                    groq_status = f"error: {res.status_code}"
        except Exception:
            groq_status = "unreachable"
            
    hf_status = "ok"
    if HF_API_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                # Do a tiny check
                res = await client.post(HF_EMBEDDING_URL, json={"inputs": ["test"]}, headers={"Authorization": f"Bearer {HF_API_TOKEN}"})
                if not res.is_success:
                    hf_status = f"error: {res.status_code}"
        except Exception:
            hf_status = "unreachable"
    else:
        hf_status = "missing_key_warning"
    
    return HealthResponse(
        status="healthy",
        total_chunks_indexed=total_chunks,
        bm25_index_loaded=Path(BM25_INDEX_PATH).exists(),
        faiss_index_loaded=(Path(FAISS_INDEX_PATH) / "index.bin").exists(),
        embedding_model=EMBEDDING_MODEL,
        reranker_model=RERANKER_MODEL,
        llm_model=LLM_MODEL,
        groq_api_status=groq_status,
        hf_api_status=hf_status
    )
