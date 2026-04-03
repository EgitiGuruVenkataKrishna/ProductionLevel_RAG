"""
LLM Generator with Legal System Prompt.

Uses Groq API for fast inference with a strict legal persona.
"""
import logging
import os
import asyncio
from groq import Groq

from app.config import (
    LEGAL_SYSTEM_PROMPT, STRATEGY_SYSTEM_PROMPT, GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, VERY_LOW_CONFIDENCE
)

logger = logging.getLogger(__name__)


def get_confidence_level(score: float) -> tuple[str, str | None]:
    """
    Convert a similarity/rerank score to confidence level and optional warning.
    
    Args:
        score: Float between 0 and 1
        
    Returns:
        Tuple of (confidence_label, warning_message_or_None)
    """
    if score >= HIGH_CONFIDENCE:
        return "high", None
    elif score >= MEDIUM_CONFIDENCE:
        return "medium", "Moderate confidence — verify important legal details with a qualified advocate."
    elif score >= LOW_CONFIDENCE:
        return "low", "Low confidence — the retrieved information may not fully address your question."
    elif score >= VERY_LOW_CONFIDENCE:
        return "very_low", "Very low confidence — the system could not find closely relevant legal provisions."
    else:
        return "rejected", "Confidence too low to provide a safe answer. Please contact a Senior Advocate."


def build_context(passages: list[dict]) -> str:
    """
    Build the context string from reranked passages for the LLM prompt.
    
    Each passage includes its legal metadata for precise citation.
    """
    context_parts = []
    
    for i, passage in enumerate(passages, 1):
        # Build a header with legal metadata
        header_parts = []
        if passage.get("article_number"):
            header_parts.append(passage["article_number"])
        if passage.get("section"):
            header_parts.append(passage["section"])
        if passage.get("act_name"):
            header_parts.append(passage["act_name"])
        if passage.get("part"):
            header_parts.append(passage["part"])
        if passage.get("source_file"):
            header_parts.append(f"Source: {passage['source_file']}")
        
        header = " | ".join(header_parts) if header_parts else f"Source {i}"
        
        context_parts.append(
            f"[Source {i}: {header}]\n{passage['text']}"
        )
    
    return "\n\n---\n\n".join(context_parts)


async def generate_legal_answer(
    question: str,
    passages: list[dict],
    is_strategy: bool = False
) -> str:
    """
    Generate a legal answer using Groq LLM with strict legal prompt.
    
    Args:
        question: User's legal question
        passages: Reranked passages with text + metadata
        is_strategy: True to use deep 'Devil's Advocate' analytical prompt.
        
    Returns:
        Generated answer string
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. "
            "Add it to your .env file or set it as an environment variable."
        )
    
    # Build context from passages
    context = build_context(passages)
    
    # Format the prompt
    active_prompt = STRATEGY_SYSTEM_PROMPT if is_strategy else LEGAL_SYSTEM_PROMPT
    prompt = active_prompt.format(
        context=context,
        question=question
    )
    
    try:
        client = Groq(api_key=api_key)
        
        def _call_groq():
            return client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Senior Legal Assistant specializing in Indian Law. You MUST use the STRICT IRAC (Issue, Rule, Application, Conclusion) framework for complex hypotheticals. Always check for specific conditions (e.g., number of perpetrators for dacoity). Never allow criminals to claim private defence against lawful force or self-defence (The Aggressor Rule). Base your entire application EXCLUSIVELY on the provided context."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=1024,
                timeout=15.0
            )
            
        chat_completion = await asyncio.to_thread(_call_groq)
        
        answer = chat_completion.choices[0].message.content
        logger.info(f"LLM generated answer ({len(answer)} chars)")
        return answer
    
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        raise RuntimeError(f"Failed to generate answer: {e}")
