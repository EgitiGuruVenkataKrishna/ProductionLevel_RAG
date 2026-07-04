import json
import hashlib
import re
import numpy as np
import logging
from typing import Optional
from app.db.redis_client import get_redis
from app.config import INDEX_VERSION, REDIS_TTL_EXACT, REDIS_TTL_SEMANTIC, REDIS_TTL_HYDE

logger = logging.getLogger(__name__)

def normalize_query(query: str) -> str:
    """Normalize query: lowercase, strip whitespace, strip trailing punctuation."""
    q = query.lower().strip()
    q = re.sub(r'[^\w\s]+$', '', q)  # Strip trailing punctuation
    return q

def hash_query(normalized_query: str) -> str:
    """Return SHA256 hash of the normalized query."""
    return hashlib.sha256(normalized_query.encode('utf-8')).hexdigest()

def get_exact_cache_key(search_mode: str, query_hash: str) -> str:
    """Generate exact cache key namespaced by index version."""
    return f"cache:exact:{INDEX_VERSION}:{search_mode}:{query_hash}"

async def get_exact_cache(search_mode: str, query: str) -> Optional[dict]:
    """Retrieve exact match response from Redis."""
    try:
        r = await get_redis()
        normalized = normalize_query(query)
        query_hash = hash_query(normalized)
        key = get_exact_cache_key(search_mode, query_hash)
        
        cached_data = await r.get(key)
        if cached_data:
            logger.info(f"Exact cache HIT for query: {query[:30]}...")
            return json.loads(cached_data)
        return None
    except Exception as e:
        logger.error(f"Error reading exact cache: {e}")
        return None

async def set_exact_cache(search_mode: str, query: str, response_data: dict):
    """Store exact match response in Redis."""
    try:
        r = await get_redis()
        normalized = normalize_query(query)
        query_hash = hash_query(normalized)
        key = get_exact_cache_key(search_mode, query_hash)
        
        await r.setex(key, REDIS_TTL_EXACT, json.dumps(response_data))
        logger.info(f"Exact cache SET for query: {query[:30]}...")
    except Exception as e:
        logger.error(f"Error setting exact cache: {e}")

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return float(dot / norm)

async def get_semantic_cache(search_mode: str, query_embedding: np.ndarray, threshold: float = 0.94) -> Optional[dict]:
    """Check for semantically similar queries in the cache."""
    try:
        r = await get_redis()
        pattern = f"cache:semantic:{INDEX_VERSION}:{search_mode}:*"
        
        # Use SCAN to iterate over semantic cache keys (non-blocking)
        # Note: If there are too many keys, this could be slow, but for a lightweight fallback it's acceptable.
        cursor = '0'
        while cursor != 0:
            cursor, keys = await r.scan(cursor=cursor, match=pattern, count=100)
            if keys:
                # Fetch all values in one go
                values = await r.mget(keys)
                for key, val in zip(keys, values):
                    if not val:
                        continue
                    data = json.loads(val)
                    cached_embedding = np.array(data["embedding"], dtype=np.float32)
                    sim = cosine_similarity(query_embedding, cached_embedding)
                    
                    if sim >= threshold:
                        logger.info(f"Semantic cache HIT (sim: {sim:.4f}) for key: {key}")
                        return data["response"]
    except Exception as e:
        logger.error(f"Error reading semantic cache: {e}")
    return None

async def set_semantic_cache(search_mode: str, query: str, query_embedding: np.ndarray, response_data: dict):
    """Store the query embedding and the response pointer in the semantic cache."""
    try:
        r = await get_redis()
        normalized = normalize_query(query)
        query_hash = hash_query(normalized)
        key = f"cache:semantic:{INDEX_VERSION}:{search_mode}:{query_hash}"
        
        payload = {
            "embedding": query_embedding.tolist(),
            "response": response_data
        }
        await r.setex(key, REDIS_TTL_SEMANTIC, json.dumps(payload))
        logger.info(f"Semantic cache SET for query: {query[:30]}...")
    except Exception as e:
        logger.error(f"Error setting semantic cache: {e}")

async def get_hyde_cache(query: str) -> Optional[str]:
    """Retrieve HyDE expanded query from Redis."""
    try:
        r = await get_redis()
        normalized = normalize_query(query)
        query_hash = hash_query(normalized)
        key = f"cache:hyde:{query_hash}"
        
        cached_data = await r.get(key)
        if cached_data:
            logger.info(f"HyDE cache HIT for query: {query[:30]}...")
            return cached_data
        return None
    except Exception as e:
        logger.error(f"Error reading HyDE cache: {e}")
        return None

async def set_hyde_cache(query: str, hyde_text: str):
    """Store HyDE expanded query in Redis."""
    try:
        r = await get_redis()
        normalized = normalize_query(query)
        query_hash = hash_query(normalized)
        key = f"cache:hyde:{query_hash}"
        
        await r.setex(key, REDIS_TTL_HYDE, hyde_text)
        logger.info(f"HyDE cache SET for query: {query[:30]}...")
    except Exception as e:
        logger.error(f"Error setting HyDE cache: {e}")
