"""
Query Expansion Service.

Generates multiple rephrasings of the user's legal question
to improve retrieval recall. Uses Groq LLM for expansion.
"""
import logging
import os
from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

EXPANSION_PROMPT = """You are a legal search query expander for Indian law.

Given a user's legal question, generate exactly 3 alternative search queries that:
1. Rephrase the question using different legal terminology
2. Include relevant Article numbers, Section numbers, or Act names if applicable
3. Cover broader or narrower aspects of the same legal concept

RULES:
- Output ONLY the 3 queries, one per line, numbered 1-3
- Keep each query concise (under 50 words)
- Focus on Indian law: Constitution, IPC, BNS, Acts
- Do NOT add explanations or commentary

USER QUESTION: {question}

ALTERNATIVE QUERIES:"""


async def expand_query(question: str) -> list[str]:
    """
    Generate alternative search queries for better retrieval recall.
    
    Args:
        question: Original user question
    
    Returns:
        List of queries (original + 3 expansions). Returns [question] on failure.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        logger.warning("No GROQ_API_KEY — skipping query expansion")
        return [question]
    
    try:
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": EXPANSION_PROMPT.format(question=question)}
            ],
            temperature=0.4,
            max_tokens=200,
        )
        
        raw = response.choices[0].message.content.strip()
        
        # Parse numbered lines
        expanded = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering: "1. ", "1) ", "1: "
            cleaned = line
            for prefix_len in range(1, 5):
                if len(line) > prefix_len and line[prefix_len] in ".):- ":
                    cleaned = line[prefix_len + 1:].strip()
                    break
            if cleaned and len(cleaned) > 5:
                expanded.append(cleaned)
        
        # Always include the original query first
        all_queries = [question] + expanded[:3]
        
        logger.info(f"Query expanded: 1 original + {len(expanded[:3])} alternatives")
        for i, q in enumerate(all_queries):
            logger.info(f"  Q{i}: {q[:80]}")
        
        return all_queries
    
    except Exception as e:
        logger.error(f"Query expansion failed: {e}")
        return [question]
