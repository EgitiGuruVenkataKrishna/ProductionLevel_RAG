"""
Context Filter & Sanitization Service.

Cleans up retrieved chunks before feeding to the LLM:
- Removes duplicate/overlapping content
- Filters low-relevance noise
- Caps total context length
- Orders by relevance
"""
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

# Max total characters of context to send to LLM
MAX_CONTEXT_CHARS = 4000

# Similarity threshold for deduplication (0-1)
DEDUP_THRESHOLD = 0.75

# Minimum score to keep a passage (filters noise)
MIN_PASSAGE_SCORE = 0.005


def text_similarity(text1: str, text2: str) -> float:
    """Quick similarity ratio between two texts using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    # Compare first 300 chars for speed
    return SequenceMatcher(None, text1[:300], text2[:300]).ratio()


def filter_and_sanitize(passages: list[dict]) -> list[dict]:
    """
    Clean up retrieved passages before LLM generation.
    
    Steps:
    1. Remove passages below minimum score threshold
    2. Deduplicate overlapping content
    3. Cap total context length
    4. Ensure relevance ordering
    
    Args:
        passages: List of passage dicts with 'text', 'rerank_score'/'fusion_score'
    
    Returns:
        Cleaned list of passages
    """
    if not passages:
        return []
    
    original_count = len(passages)
    
    # ---- Step 1: Filter low-score noise ----
    scored_passages = []
    for p in passages:
        score = p.get("rerank_score", p.get("fusion_score", 0.0))
        if score >= MIN_PASSAGE_SCORE:
            scored_passages.append(p)
    
    if not scored_passages:
        # If all filtered out, keep the top passage anyway
        scored_passages = passages[:1]
    
    # ---- Step 2: Deduplicate overlapping content ----
    deduplicated = []
    for p in scored_passages:
        is_duplicate = False
        for existing in deduplicated:
            sim = text_similarity(p.get("text", ""), existing.get("text", ""))
            if sim > DEDUP_THRESHOLD:
                is_duplicate = True
                # Keep the one with higher score
                existing_score = existing.get("rerank_score", existing.get("fusion_score", 0))
                new_score = p.get("rerank_score", p.get("fusion_score", 0))
                if new_score > existing_score:
                    # Replace with higher-scored version
                    existing.update(p)
                break
        
        if not is_duplicate:
            deduplicated.append(p)
    
    # ---- Step 3: Sort by relevance score ----
    deduplicated.sort(
        key=lambda x: x.get("rerank_score", x.get("fusion_score", 0.0)),
        reverse=True
    )
    
    # ---- Step 4: Cap total context length ----
    capped = []
    total_chars = 0
    for p in deduplicated:
        text_len = len(p.get("text", ""))
        if total_chars + text_len > MAX_CONTEXT_CHARS and capped:
            break  # Already have some context, stop here
        capped.append(p)
        total_chars += text_len
    
    removed = original_count - len(capped)
    if removed > 0:
        logger.info(
            f"Context filter: {original_count} → {len(capped)} passages "
            f"(removed {removed}: dedup/noise/cap)"
        )
    
    return capped
