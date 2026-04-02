"""
Answer Grounding Check & Real Confidence Scoring.

Verifies that every claim in the LLM's answer is actually
supported by the retrieved context. Computes real metrics:
- Faithfulness: % of claims grounded in context
- Relevance: how well the answer addresses the question
- Coverage: how much of the context was utilized
"""
import logging
import os
import re
import asyncio
from groq import Groq

from app.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)

GROUNDING_PROMPT = """You are a legal answer verification system. Your job is to check if an AI-generated answer about Indian law is factually grounded in the provided source context.

CONTEXT (source documents):
{context}

QUESTION:
{question}

AI-GENERATED ANSWER:
{answer}

VERIFICATION TASK:
Analyze the answer and rate each of these metrics from 0.0 to 1.0:

1. **faithfulness**: What fraction of claims in the answer are directly supported by the context? (1.0 = all claims grounded, 0.0 = all fabricated)
2. **relevance**: How well does the answer address the actual question? (1.0 = perfectly relevant, 0.0 = completely off-topic)
3. **coverage**: How much of the relevant context information was used in the answer? (1.0 = all relevant info used, 0.0 = none used)

Also identify any UNGROUNDED claims (statements in the answer NOT supported by the context).

OUTPUT FORMAT (exactly this, no other text):
faithfulness: <score>
relevance: <score>
coverage: <score>
ungrounded: <comma-separated list of ungrounded claims, or "none">"""


class GroundingResult:
    """Holds grounding check results."""
    def __init__(
        self,
        faithfulness: float = 0.0,
        relevance: float = 0.0,
        coverage: float = 0.0,
        ungrounded_claims: list[str] = None,
        overall_score: float = 0.0,
        is_grounded: bool = False
    ):
        self.faithfulness = faithfulness
        self.relevance = relevance
        self.coverage = coverage
        self.ungrounded_claims = ungrounded_claims or []
        self.overall_score = overall_score
        self.is_grounded = is_grounded
    
    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 3),
            "relevance": round(self.relevance, 3),
            "coverage": round(self.coverage, 3),
            "overall_score": round(self.overall_score, 3),
            "is_grounded": self.is_grounded,
            "ungrounded_claims": self.ungrounded_claims
        }


async def check_grounding(
    question: str,
    answer: str,
    context: str
) -> GroundingResult:
    """
    Verify that the answer is grounded in the retrieved context.
    
    Args:
        question: User's original question
        answer: LLM-generated answer
        context: The source context that was provided to the LLM
    
    Returns:
        GroundingResult with faithfulness, relevance, coverage scores
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        logger.warning("No GROQ_API_KEY — skipping grounding check")
        return _default_result()
    
    try:
        client = Groq(api_key=api_key)
        
        prompt = GROUNDING_PROMPT.format(
            context=context[:4000],  # Match MAX_CONTEXT_CHARS
            question=question,
            answer=answer[:1500]
        )
        
        def _call_groq():
            return client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=300,
                timeout=15.0
            )
            
        response = await asyncio.to_thread(_call_groq)
        
        raw = response.choices[0].message.content.strip()
        return _parse_grounding_response(raw)
    
    except Exception as e:
        logger.error(f"Grounding check failed: {e}")
        return _default_result()


def _parse_grounding_response(raw: str) -> GroundingResult:
    """Parse the structured grounding check response."""
    result = GroundingResult()
    
    try:
        lines = raw.strip().split("\n")
        
        for line in lines:
            line = line.strip().lower()
            
            if line.startswith("faithfulness:"):
                result.faithfulness = _extract_score(line)
            elif line.startswith("relevance:"):
                result.relevance = _extract_score(line)
            elif line.startswith("coverage:"):
                result.coverage = _extract_score(line)
            elif line.startswith("ungrounded:"):
                claims_text = line.split(":", 1)[1].strip()
                if claims_text and claims_text != "none":
                    result.ungrounded_claims = [
                        c.strip() for c in claims_text.split(",")
                        if c.strip() and c.strip().lower() != "none"
                    ]
        
        # Weighted overall score: faithfulness matters most for legal
        result.overall_score = (
            result.faithfulness * 0.50 +
            result.relevance * 0.30 +
            result.coverage * 0.20
        )
        
        # Consider grounded if faithfulness >= 0.7 and overall >= 0.5
        result.is_grounded = (
            result.faithfulness >= 0.7 and 
            result.overall_score >= 0.5
        )
        
        logger.info(
            f"Grounding check: faith={result.faithfulness:.2f} "
            f"rel={result.relevance:.2f} cov={result.coverage:.2f} "
            f"overall={result.overall_score:.2f} grounded={result.is_grounded}"
        )
        
    except Exception as e:
        logger.error(f"Failed to parse grounding response: {e}")
        return _default_result()
    
    return result


def _extract_score(line: str) -> float:
    """Extract a float score from a line like 'faithfulness: 0.85'."""
    try:
        match = re.search(r'(\d+\.?\d*)', line.split(":", 1)[1])
        if match:
            score = float(match.group(1))
            return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    except (IndexError, ValueError):
        pass
    return 0.0


def _default_result() -> GroundingResult:
    """Return a neutral result when grounding check can't run."""
    return GroundingResult(
        faithfulness=0.5,
        relevance=0.5,
        coverage=0.5,
        overall_score=0.5,
        is_grounded=False,  # DO NOT Assume grounded if we can't check
        ungrounded_claims=["System was unable to verify grounding. Please consult a qualified legal professional."]
    )
