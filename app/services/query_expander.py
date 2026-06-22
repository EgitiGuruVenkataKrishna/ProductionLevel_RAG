"""
Query Expansion Service.

Generates a dense HyDE paragraph from the user's legal question
to improve retrieval recall. Uses Groq LLM for expansion.
"""
import logging
import os
import asyncio
from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

HYDE_PROMPT = """You are a Legal RAG expansion layer. Convert the user's conversational query into a dense paragraph containing formal common law terminology, civil tort/criminal doctrines, and relevant Latin legal maxims that would appear in a classic textbook commentary. 
CRITICAL: Do NOT invent, guess, or append arbitrary statutory section numbers, article numbers, or clause numbers unless they are explicitly provided in the user's input.

USER QUESTION: {question}

DENSE LEGAL PARAGRAPH:"""


async def expand_query(question: str) -> list[str]:
    """
    Generate a constrained HyDE paragraph for better retrieval recall.
    
    Args:
        question: Original user question
    
    Returns:
        List containing the HyDE paragraph. Returns [question] on failure.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        logger.warning("No GROQ_API_KEY — skipping query expansion")
        return [question]
    
    try:
        client = Groq(api_key=api_key)
        
        def _call_groq():
            return client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": HYDE_PROMPT.format(question=question)}
                ],
                temperature=0.2,
                max_tokens=300,
                timeout=15.0
            )
            
        response = await asyncio.to_thread(_call_groq)
        
        hyde_paragraph = response.choices[0].message.content.strip()
        
        logger.info(f"Constrained HyDE generated: {hyde_paragraph[:80]}...")
        
        return [hyde_paragraph]
    
    except Exception as e:
        logger.error(f"Query expansion (HyDE) failed: {e}")
        return [question]
