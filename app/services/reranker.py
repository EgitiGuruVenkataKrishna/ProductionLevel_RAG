"""
Cross-Encoder Reranker via HuggingFace Inference API.

Reranks the top candidates from hybrid search using a
cross-encoder model for more accurate relevance scoring.
"""
import logging
import httpx
from typing import Optional

from app.config import HF_RERANKER_URL, HF_API_TOKEN, RERANK_TOP_N

logger = logging.getLogger(__name__)


async def rerank_passages(
    query: str,
    passages: list[dict],
    top_n: int = RERANK_TOP_N
) -> list[dict]:
    """
    Rerank passages using cross-encoder model via HF Inference API.
    
    Args:
        query: The user's question
        passages: List of dicts with at least 'text' and 'chunk_id' keys
        top_n: Number of top results to return after reranking
    
    Returns:
        Reranked list of passage dicts with added 'rerank_score'
    """
    if not passages:
        return []
    
    # Prepare pairs for cross-encoder
    inputs = {
        "inputs": {
            "source_sentence": query,
            "sentences": [p["text"][:512] for p in passages]  # Truncate for model
        }
    }
    
    headers = {"Content-Type": "application/json"}
    if HF_API_TOKEN:
        headers["Authorization"] = f"Bearer {HF_API_TOKEN}"
    
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(
                HF_RERANKER_URL,
                headers=headers,
                json=inputs
            )
            
            if response.status_code == 200:
                scores = response.json()
                
                # Handle different response formats
                if isinstance(scores, list):
                    # Attach scores to passages
                    for i, passage in enumerate(passages):
                        if i < len(scores):
                            score = scores[i]
                            # Handle nested format
                            if isinstance(score, dict):
                                passage["rerank_score"] = float(score.get("score", 0.0))
                            else:
                                passage["rerank_score"] = float(score)
                        else:
                            passage["rerank_score"] = 0.0
                else:
                    logger.warning(f"Unexpected reranker response format: {type(scores)}")
                    return _fallback_rerank(passages, top_n)
                
                # Sort by rerank score descending
                reranked = sorted(passages, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
                
                logger.info(f"Reranked {len(passages)} → top {top_n} | "
                          f"Best score: {reranked[0].get('rerank_score', 0):.4f}")
                
                return reranked[:top_n]
            
            elif response.status_code == 503:
                logger.warning("Reranker model loading. Using fallback ordering.")
                return _fallback_rerank(passages, top_n)
            else:
                logger.error(f"Reranker API error {response.status_code}: {response.text[:200]}")
                return _fallback_rerank(passages, top_n)
    
    except httpx.TimeoutException:
        logger.warning("Reranker API timeout. Using fallback ordering.")
        return _fallback_rerank(passages, top_n)
    except Exception as e:
        logger.error(f"Reranker failed: {e}")
        return _fallback_rerank(passages, top_n)


def _fallback_rerank(passages: list[dict], top_n: int) -> list[dict]:
    """
    Fallback: Use the existing RRF/similarity scores when reranker is unavailable.
    """
    logger.info(f"Using fallback reranking (RRF order) → top {top_n}")
    for p in passages:
        p["rerank_score"] = p.get("fusion_score", p.get("similarity_score", 0.0))
    return passages[:top_n]
