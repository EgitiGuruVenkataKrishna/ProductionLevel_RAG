"""
Hybrid Retriever with Reciprocal Rank Fusion (RRF).

Merges results from BM25 (keyword) and FAISS (semantic) search.
"""
import logging
import json
import asyncio
import numpy as np
import httpx
from langfuse import observe
from pathlib import Path
from typing import Optional

from app.config import (
    SEMANTIC_TOP_K, BM25_TOP_K, RRF_K, HF_EMBEDDING_URL, HF_API_TOKEN, CHUNKS_METADATA_PATH,
    USE_LOCAL_MODELS, USE_SQLITE_METADATA
)
from app.services.bm25_index import bm25_index
from app.services.pinecone_index import pinecone_service

# Import metadata store for SQLite chunks (Priority 4)
if USE_SQLITE_METADATA:
    from app.db.metadata_store import metadata_store

logger = logging.getLogger(__name__)

# Initialize local embedding model if configured
_local_embedding_model = None
if USE_LOCAL_MODELS:
    try:
        from fastembed import TextEmbedding
        from app.config import EMBEDDING_MODEL
        logger.info(f"Initializing FastEmbed TextEmbedding ({EMBEDDING_MODEL}) for local inference...")
        _local_embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL)
    except ImportError:
        logger.error("fastembed not installed. Falling back to HF Inference API. Run `pip install fastembed`.")
        USE_LOCAL_MODELS = False

# ==================== CHUNK METADATA STORE ====================
_chunks_metadata: list[dict] = []


def load_chunks_metadata(path: str = None):
    """Load chunk texts and metadata from JSON (if SQLite not used)."""
    global _chunks_metadata
    
    if USE_SQLITE_METADATA:
        logger.info("USE_SQLITE_METADATA is true. Skipping JSON load in memory.")
        return
        
    meta_path = Path(path) if path else CHUNKS_METADATA_PATH
    
    if not meta_path.exists():
        logger.warning(f"Chunks metadata not found at {meta_path}")
        return
    
    with open(meta_path, "r", encoding="utf-8") as f:
        _chunks_metadata = json.load(f)
    
    logger.info(f"Loaded metadata for {len(_chunks_metadata)} chunks")


def get_chunks_metadata() -> list[dict]:
    """Get the loaded chunks metadata (in-memory only)."""
    return _chunks_metadata


def get_chunk_by_id(chunk_id: int) -> Optional[dict]:
    """Get a single chunk by its ID (in-memory only). DEPRECATED."""
    if 0 <= chunk_id < len(_chunks_metadata):
        return _chunks_metadata[chunk_id]
    return None


async def get_chunks(chunk_ids: list[int]) -> list[dict]:
    """
    Fetch a batch of chunks either from SQLite or in-memory JSON.
    """
    if USE_SQLITE_METADATA:
        return await metadata_store.get_chunks_batch(chunk_ids)
    
    # In-memory fallback
    chunks = []
    for cid in chunk_ids:
        chunk = get_chunk_by_id(cid)
        if chunk:
            # Inject id since SQLite dicts will have it
            chunk_copy = chunk.copy()
            chunk_copy["id"] = cid
            chunks.append(chunk_copy)
    return chunks


# ==================== QUERY EMBEDDING ====================
async def embed_query(query: str) -> Optional[np.ndarray]:
    """
    Get query embedding via FastEmbed (local) or HuggingFace API (remote).
    
    Falls back to None if unavailable.
    """
    if USE_LOCAL_MODELS and _local_embedding_model:
        try:
            # FastEmbed embed() returns a generator of arrays
            embedding_gen = _local_embedding_model.embed([query])
            embedding = next(embedding_gen).astype(np.float32)
            return embedding
        except Exception as e:
            logger.error(f"Local FastEmbed call failed: {e}")
            return None

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



CATEGORIES = {
    'Civil Torts': "tort sue negligence liable damage injury nuisance trespass defamation",
    'Criminal Law': "crime ipc murder theft arrest penal nyaya sanhita bns culpable robbery cheating",
    'Contract Law': "contract agree sign breach consideration indemnity bailment pledge",
    'Constitutional Law': "constitution fundamental right directive amendment preamble article",
    'Evidence Law': "evidence witness testimony confession hearsay burden of proof sakshya",
    'Civil Procedure': "civil procedure suit decree plaint civil court",
    'Criminal Procedure': "criminal procedure bail fir charge crpc bnss investigation cognizable",
    'Property Law': "transfer of property mortgage lease sale deed easement immovable",
    'RTI': "rti right to information public authority information commission",
    'Statutory Interpretation': "interpretation statute statutory maxim",
    'Corporate Law': "corporate company nclt ibc insolvency bankruptcy shares shareholder director board oppression mismanagement"
}

_category_embeddings = {}

async def _get_category_embeddings():
    if _category_embeddings:
        return _category_embeddings
    for cat, text in CATEGORIES.items():
        emb = await embed_query(text)
        if emb is not None:
            _category_embeddings[cat] = emb
    return _category_embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def detect_category(query: str) -> Optional[str]:
    """Detect legal domain using embedding cosine similarity for Pinecone metadata filtering."""
    query_emb = await embed_query(query)
    if query_emb is None:
        return None
        
    cat_embs = await _get_category_embeddings()
    if not cat_embs:
        return None
        
    best_cat = None
    best_score = 0.0
    for cat, emb in cat_embs.items():
        score = cosine_similarity(query_emb, emb)
        if score > best_score:
            best_score = score
            best_cat = cat
            
    if best_score > 0.65:
        return best_cat
    return None

# ==================== HYBRID SEARCH ====================
async def hybrid_search(
    query_to_embed: str,
    original_query: str,
    mode: str = "hybrid",
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    precomputed_embedding: Optional[np.ndarray] = None
) -> list[tuple[int, float]]:
    """
    Perform hybrid search combining semantic (Pinecone) + BM25.
    
    Args:
        query_to_embed: The HyDE paragraph
        original_query: The raw user query (for intent detection)
        mode: 'hybrid', 'semantic', or 'keyword'
        semantic_top_k: Top-K for semantic search
        bm25_top_k: Top-K for BM25 search
        
    Returns:
        List of (chunk_id, score) tuples, sorted by relevance
    """
    semantic_results = []
    bm25_results = []
    
    # Semantic search (Pinecone)
    if mode in ("hybrid", "semantic"):
        category_filter = await detect_category(original_query)
        if category_filter:
            logger.info(f"Embedding intent matched. Applying Pinecone filter: {category_filter}")
            
        if precomputed_embedding is not None and query_to_embed == original_query:
            query_embedding = precomputed_embedding
        else:
            query_embedding = await embed_query(query_to_embed)
            
        if query_embedding is not None:
            # Pinecone requires python native floats
            query_embedding_list = query_embedding.tolist()
            semantic_results = await pinecone_service.search(
                vector=query_embedding_list, 
                top_k=semantic_top_k, 
                category_filter=category_filter
            )
            logger.info(f"Semantic search returned {len(semantic_results)} results")
        else:
            logger.warning("Semantic search skipped - embedding failed")
            # We degrade seamlessly. Handled at pipeline level.
    
    # BM25 keyword search
    if mode in ("hybrid", "keyword"):
        bm25_results = bm25_index.search(original_query, top_k=bm25_top_k)
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
@observe(name="multi_query_hybrid_search")
async def multi_query_hybrid_search(
    queries: list[str],
    original_query: str,
    mode: str = "hybrid",
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K,
    precomputed_embedding: Optional[np.ndarray] = None
) -> list[tuple[int, float]]:
    """
    Run hybrid search across multiple expanded queries CONCURRENTLY and merge results.
    
    All queries are dispatched in parallel via asyncio.gather, then results
    are merged with additive scoring so documents appearing in multiple
    query results rank higher.
    
    Latency: O(1 x single_query_latency) instead of O(N x single_query_latency).
    
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
    
    # Dispatch all queries concurrently
    tasks = [
        hybrid_search(query, original_query, mode, semantic_top_k, bm25_top_k, precomputed_embedding)
        for query in queries
    ]
    all_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    accumulated_scores = {}
    
    for i, results in enumerate(all_results):
        # If a single query failed, log and skip it rather than crashing
        if isinstance(results, Exception):
            logger.error(f"Parallel query {i} failed: {results}")
            continue
        
        # Weight: original query gets full weight, expansions get 0.7x
        weight = 1.0 if i == 0 else 0.7
        
        for chunk_id, score in results:
            accumulated_scores[chunk_id] = (
                accumulated_scores.get(chunk_id, 0.0) + score * weight
            )
    
    # Sort by accumulated score descending
    sorted_results = sorted(accumulated_scores.items(), key=lambda x: x[1], reverse=True)
    
    logger.info(
        f"Multi-query search: {len(queries)} queries (parallel) -> "
        f"{len(sorted_results)} unique candidates"
    )
    
    return sorted_results


def multi_query_hybrid_search_sync(
    queries: list[str],
    original_query: str,
    mode: str = "hybrid",
    semantic_top_k: int = SEMANTIC_TOP_K,
    bm25_top_k: int = BM25_TOP_K
) -> list[tuple[int, float]]:
    """
    Synchronous wrapper for multi_query_hybrid_search.
    
    For backward compatibility with sync callers.
    Do NOT call from within an already-running event loop - use the async
    variant directly instead.
    """
    return asyncio.run(
        multi_query_hybrid_search(queries, original_query, mode, semantic_top_k, bm25_top_k)
    )
