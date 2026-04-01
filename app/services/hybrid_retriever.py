"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF).

Merges results from BM25 (keyword) and FAISS (semantic) search.
"""
import logging
import json
import numpy as np
import httpx
from pathlib import Path
from typing import Optional

from app.config import (
    SEMANTIC_TOP_K, BM25_TOP_K, RRF_K, RERANK_TOP_N,
    HF_EMBEDDING_URL, HF_API_TOKEN, CHUNKS_METADATA_PATH
)
from app.services.bm25_index import bm25_index
from app.services.vector_index import vector_index

logger = logging.getLogger(__name__)

# ==================== CHUNK METADATA STORE ====================
_chunks_metadata: list[dict] = []


def load_chunks_metadata(path: str = None):
    """Load chunk texts and metadata from JSON."""
    global _chunks_metadata
    meta_path = Path(path) if path else CHUNKS_METADATA_PATH
    
    if not meta_path.exists():
        logger.warning(f"Chunks metadata not found at {meta_path}")
        return
    
    with open(meta_path, "r", encoding="utf-8") as f:
        _chunks_metadata = json.load(f)
    
    logger.info(f"Loaded metadata for {len(_chunks_metadata)} chunks")


def get_chunks_metadata() -> list[dict]:
    """Get the loaded chunks metadata."""
    return _chunks_metadata


def get_chunk_by_id(chunk_id: int) -> Optional[dict]:
    """Get a single chunk by its ID."""
    if 0 <= chunk_id < len(_chunks_metadata):
        return _chunks_metadata[chunk_id]
    return None


# ==================== QUERY EMBEDDING ====================
async def embed_query(query: str) -> Optional[np.ndarray]:
    """
    Get query embedding via HuggingFace Inference API.
    
    Falls back to None if API is unavailable.
    """
    headers = {}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                HF_EMBEDDING_URL,
                headers=headers,
                json={"inputs": query, "options": {"wait_for_model": True}}
            )
            
            if response.status_code == 200:
                embedding = np.array(response.json(), dtype=np.float32)
                return embedding
            else:
                logger.error(f"HF Embedding API error {response.status_code}: {response.text[:200]}")
                return None
                
    except Exception as e:
        logger.error(f"HF Embedding API call failed: {e}")
        return None


# ==================== RECIPROCAL RANK FUSION ====================
def reciprocal_rank_fusion(
    semantic_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = RRF_K
) -> list[tuple[int, float]]:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    
    RRF score for doc d = sum(1 / (k + rank_i(d))) for each ranker i
    
    Args:
        semantic_results: List of (doc_id, score) from semantic search
        bm25_results: List of (doc_id, score) from BM25 search
        k: Fusion constant (default 60, standard from the RRF paper)
    
    Returns:
        Merged list of (doc_id, fused_score) sorted descending
    """
    fused_scores = {}
    
    # Add semantic search contributions
    for rank, (doc_id, _score) in enumerate(semantic_results):
        rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
    
    # Add BM25 search contributions
    for rank, (doc_id, _score) in enumerate(bm25_results):
        rrf_score = 1.0 / (k + rank + 1)
        fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + rrf_score
    
    # Sort by fused score descending
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked


# ==================== HYBRID SEARCH ====================
async def hybrid_search(
    query: str,
    mode: str = "hybrid",
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K
) -> list[tuple[int, float]]:
    """
    Perform hybrid search combining semantic + BM25.
    
    Args:
        query: User's question
        mode: 'hybrid', 'semantic', or 'keyword'
        semantic_top_k: Top-K for semantic search
        bm25_top_k: Top-K for BM25 search
    
    Returns:
        List of (chunk_id, score) tuples, sorted by relevance
    """
    semantic_results = []
    bm25_results = []
    
    # Semantic search
    if mode in ("hybrid", "semantic"):
        query_embedding = await embed_query(query)
        if query_embedding is not None:
            semantic_results = vector_index.search(query_embedding, top_k=semantic_top_k)
            logger.info(f"Semantic search returned {len(semantic_results)} results")
        else:
            logger.warning("Semantic search skipped — embedding failed")
    
    # BM25 keyword search
    if mode in ("hybrid", "keyword"):
        bm25_results = bm25_index.search(query, top_k=bm25_top_k)
        logger.info(f"BM25 search returned {len(bm25_results)} results")
    
    # Fusion
    if mode == "hybrid" and semantic_results and bm25_results:
        fused = reciprocal_rank_fusion(semantic_results, bm25_results)
        logger.info(f"RRF fusion produced {len(fused)} unique candidates")
        return fused
    elif semantic_results:
        return semantic_results
    elif bm25_results:
        return bm25_results
    else:
        logger.warning("No results from any search method")
        return []


# ==================== MULTI-QUERY HYBRID SEARCH ====================
async def multi_query_hybrid_search(
    queries: list[str],
    mode: str = "hybrid",
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K
) -> list[tuple[int, float]]:
    """
    Run hybrid search across multiple expanded queries and merge results.
    
    Each query's results contribute additively to the final score,
    so documents appearing in multiple query results rank higher.
    
    Args:
        queries: List of query strings (original + expansions)
        mode: Search mode
        semantic_top_k: Top-K per query for semantic
        bm25_top_k: Top-K per query for BM25
    
    Returns:
        Merged list of (chunk_id, accumulated_score) sorted descending
    """
    if not queries:
        return []
    
    accumulated_scores = {}
    
    for i, query in enumerate(queries):
        results = await hybrid_search(query, mode, semantic_top_k, bm25_top_k)
        
        # Weight: original query gets full weight, expansions get 0.7x
        weight = 1.0 if i == 0 else 0.7
        
        for chunk_id, score in results:
            accumulated_scores[chunk_id] = (
                accumulated_scores.get(chunk_id, 0.0) + score * weight
            )
    
    # Sort by accumulated score descending
    merged = sorted(accumulated_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(
        f"Multi-query search: {len(queries)} queries → "
        f"{len(merged)} unique candidates"
    )
    
    return merged

