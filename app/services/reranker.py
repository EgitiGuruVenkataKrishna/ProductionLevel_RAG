"""
Cross-Encoder Reranker via Cohere API.

Reranks the top candidates from hybrid search using Cohere's
rerank-v3.0 model for maximum syntactic relevance scoring.
"""
import logging
import os
import asyncio
import numpy as np
import httpx
from langfuse import observe
from typing import Optional

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

from app.config import COHERE_API_KEY, RERANKER_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)

@observe(name="rerank_passages")
async def rerank_passages(
    query: str,
    passages: list[dict],
    top_n: int = RERANK_TOP_N
) -> list[dict]:
    """
    Rerank passages using Cohere API.
    
    Args:
        query: The user's original question
        passages: List of dicts with at least 'text' and 'chunk_id' keys
        top_n: Number of top results to return after reranking
    
    Returns:
        Reranked list of passage dicts with added 'rerank_score'
    """
    if not passages:
        return []
        
    api_key = COHERE_API_KEY or os.getenv("COHERE_API_KEY", "")
    
    if not COHERE_AVAILABLE or not api_key:
        logger.warning("Cohere not available or API key missing. Using fallback ordering.")
        return _fallback_rerank(passages, top_n)
    
    try:
        co = cohere.Client(api_key=api_key)
        
        # Cohere expects a list of text strings
        docs = [p["text"] for p in passages]
        
        def _call_cohere():
            return co.rerank(
                query=query,
                documents=docs,
                top_n=top_n,
                model=RERANKER_MODEL
            )
            
        response = await asyncio.to_thread(_call_cohere)
        
        # Create a new list for the reranked results
        reranked = []
        for result in response.results:
            idx = result.index
            passage = passages[idx].copy()
            passage["rerank_score"] = float(result.relevance_score)
            reranked.append(passage)
            
        logger.info(f"Cohere reranked {len(passages)} → top {len(reranked)} | "
                    f"Best score: {reranked[0].get('rerank_score', 0):.4f}")
        return reranked
    
    except Exception as e:
        logger.error(f"Cohere reranker failed: {e}")
        return _fallback_rerank(passages, top_n)


def _fallback_rerank(passages: list[dict], top_n: int) -> list[dict]:
    """
    Fallback: Use the existing RRF/similarity scores when reranker is unavailable.
    """
    logger.info(f"Using fallback reranking (RRF order) → top {top_n}")
    for p in passages:
        p["rerank_score"] = p.get("fusion_score", p.get("similarity_score", 0.0))
    return passages[:top_n]
