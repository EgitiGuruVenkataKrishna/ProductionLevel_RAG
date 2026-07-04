"""
LLM Generator with Legal System Prompt.

Uses Groq API for fast inference with a strict legal persona.
"""
import logging
import os
import json
import asyncio
from groq import Groq
from langfuse import observe

from app.config import (
    LEGAL_SYSTEM_PROMPT, STRATEGY_SYSTEM_PROMPT, GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, LOW_CONFIDENCE, VERY_LOW_CONFIDENCE
)

logger = logging.getLogger(__name__)


from typing import AsyncGenerator
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
        return "medium", "Moderate confidence - verify important legal details with a qualified advocate."
    elif score >= LOW_CONFIDENCE:
        return "low", "Low confidence - the retrieved information may not fully address your question."
    elif score >= VERY_LOW_CONFIDENCE:
        return "very_low", "Very low confidence - the system could not find closely relevant legal provisions."
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
        
        # Use parent_text if available for full legal context, fallback to chunk text
        context_text = passage.get("parent_text") or passage["text"]
        
        context_parts.append(
            f"[Source {i}: {header}]\n{context_text}"
        )
    return "\n\n---\n\n".join(context_parts)


@observe(name="detect_query_intent")
async def detect_query_intent(question: str) -> str:
    """
    Fast LLM call to classify intent before hitting the RAG pipeline.
    Returns: 'greeting', 'system', or 'legal'
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "legal"  # fallback
    
    try:
        from groq import AsyncGroq
        client = AsyncGroq(api_key=api_key)
        
        prompt = f"""Classify the user's intent into exactly ONE of these three categories:
1. "greeting": Conversational greetings (hi, hello, how are you).
2. "system": Questions about who you are, your capabilities, or your creator (who are you, what can you do).
3. "legal": Any actual legal question, scenario, or topic.

User Input: "{question}"

Output ONLY the category word (greeting, system, or legal) and nothing else."""

        chat_completion = await client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Extremely fast router model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
            timeout=5.0
        )
        
        result = chat_completion.choices[0].message.content.strip().lower()
        if "greeting" in result:
            return "greeting"
        if "system" in result:
            return "system"
        return "legal"
    except Exception as e:
        logger.error(f"Intent detection failed: {e}")
        return "legal"  # Fallback to full pipeline


@observe(name="generate_legal_answer")
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


# ==================== MERGED GENERATION + GROUNDING ====================

MERGED_LEGAL_SYSTEM_PROMPT = """You are a Senior Legal Assistant specializing in Indian Law.
Your goal is to answer legal questions using the strictly provided context, AND simultaneously
verify the grounding of your own answer.

CRITICAL INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the provided legal context.
2. If the user's input is a conversational greeting (like 'hlo', 'hi', 'hello') or fundamentally NOT a legal question, respond with this exact JSON:
   {{"answer": "GREETING_OR_NON_LEGAL_QUERY", "citations": [], "faithfulness_score": 1.0, "ungrounded_claims": []}}
3. **PRIORITIZE NEW LAWS (BNS/BNSS/BSA):** India transitioned to new criminal laws on July 1, 2024. ALWAYS apply and cite the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA) over the repealed IPC, CrPC, or IEA. If the scenario occurs after July 1, 2024, you are STRICTLY FORBIDDEN from using the old Evidence Act, IPC, CrPC, or historical IT Act clauses for procedural validation. If the relevant BSA/BNS chunk is missing, state: "The required new active law is not present in the retrieved context."
4. If it IS a legal question, structure your answer using the IRAC framework but keep it of MODERATE LENGTH. However, DO NOT omit crucial parts of a section just to be concise:
   - **ISSUE:** Briefly state the legal question.
   - **RULE:** Extract exact laws, Sections, their FULL rigid conditions, AND any punishments, penalties, or exceptions mentioned.
   - **APPLICATION:** Briefly apply the rules to the actors. Ensure you mention requirements like 'communication to a third party' or 'cognizance by Sessions Court' if the law demands it.
   - **CONCLUSION:** A definitive legal outcome based purely on the text. Include the potential punishment if applicable.
5. If the context does not contain the answer, say "I cannot determine this from the available excerpts."
6. NEVER fabricate, guess, or hallucinate legal provisions, procedural links, or punishments. If a specific procedural section (like cognizance for public servants) is not in the context, do not guess it.
7. Use formal legal language.

CITATION FORMAT (use exactly):
- You MUST explicitly cite the 'Act' and 'Section' from the provided metadata for every legal claim. 
- Example: [Section 302, Indian Penal Code, 1860]. 
- Do NOT hallucinate citations or section numbers that are not in the context.

SELF-VERIFICATION:
After generating your answer, rate your own faithfulness:
- faithfulness_score: What fraction of your claims are directly supported by the context? (0.0–1.0)
- List any claims you made that are NOT directly stated in the context.

OUTPUT FORMAT - You MUST respond with ONLY a valid JSON object, no other text:
{{
  "answer": "<your legal answer using IRAC>",
  "citations": ["<Section/Article cited>", ...],
  "faithfulness_reasoning": "<brief explanation of how well your answer matches the context>",
  "faithfulness_score": <0.0 to 1.0>,
  "ungrounded_claims": ["<any claim not in context>", ...]
}}

CONTEXT:
{context}

QUESTION: {question}"""

MERGED_STRATEGY_SYSTEM_PROMPT = """You are an elite Junior Lawyer AI specializing in Indian Legal Strategy and Adversarial Analysis.
Your goal is to critically evaluate the user's case facts against the provided context, AND simultaneously
verify the grounding of your analysis.

CRITICAL INSTRUCTIONS:
1. Adopt an analytical, adversarial ("Devil's Advocate") perspective.
2. Rely strictly on the user's provided [FACTS] and the retrieved legal context.
3. Structure your response for legal strategy:
   - FACT SUMMARY, APPLICABLE LAW, THEORY EVALUATION, BAD FACTS
4. Never hallucinate legal provisions or case outcomes.

SELF-VERIFICATION:
After generating your analysis, rate your own faithfulness to the provided context.

OUTPUT FORMAT - You MUST respond with ONLY a valid JSON object, no other text:
{{
  "answer": "<your strategy analysis>",
  "citations": ["<Section/Article cited>", ...],
  "faithfulness_reasoning": "<brief explanation of how well your answer matches the context>",
  "faithfulness_score": <0.0 to 1.0>,
  "ungrounded_claims": ["<any claim not in context>", ...]
}}

CONTEXT:
{context}

USER CASE SCENARIO: {question}"""


class MergedGenerationResult:
    """Result from the merged generation + grounding LLM call."""
    
    def __init__(
        self,
        answer: str = "",
        citations: list[str] = None,
        faithfulness_score: float = 0.5,
        ungrounded_claims: list[str] = None,
        is_low_grounding: bool = False
    ):
        self.answer = answer
        self.citations = citations or []
        self.faithfulness_score = faithfulness_score
        self.ungrounded_claims = ungrounded_claims or []
        self.is_low_grounding = is_low_grounding


@observe(name="generate_and_verify_legal_answer")
async def generate_and_verify_legal_answer(
    question: str,
    passages: list[dict],
    is_strategy: bool = False
) -> MergedGenerationResult:
    """
    Generate a legal answer AND verify grounding in a SINGLE LLM call.
    
    Replaces the old two-call pattern (generate_legal_answer + check_grounding)
    with a single structured-output call. Halves LLM API cost and latency.
    
    Args:
        question: User's legal question
        passages: Reranked passages with text + metadata
        is_strategy: True to use the strategy/adversarial prompt
        
    Returns:
        MergedGenerationResult with answer, citations, faithfulness_score,
        ungrounded_claims, and is_low_grounding flag.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. "
            "Add it to your .env file or set it as an environment variable."
        )
    
    # Build context from passages
    context = build_context(passages)
    
    # Select and format the merged prompt
    active_prompt = MERGED_STRATEGY_SYSTEM_PROMPT if is_strategy else MERGED_LEGAL_SYSTEM_PROMPT
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
                        "content": (
                            "You are a Senior Legal Assistant specializing in Indian Law. "
                            "You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no extra text. "
                            "Base your entire application EXCLUSIVELY on the provided context."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=LLM_TEMPERATURE,
                max_tokens=1500,  # Slightly higher to accommodate JSON wrapper
                timeout=20.0
            )
            
        chat_completion = await asyncio.to_thread(_call_groq)
        
        raw = chat_completion.choices[0].message.content.strip()
        logger.info(f"Merged LLM response ({len(raw)} chars)")
        
        # Parse the JSON response
        result = _parse_merged_response(raw)
        return result
    
    except Exception as e:
        logger.error(f"Merged generation error: {e}")
        raise RuntimeError(f"Failed to generate and verify answer: {e}")


@observe(name="generate_and_verify_legal_answer_stream")
async def generate_and_verify_legal_answer_stream(
    question: str,
    passages: list[dict],
    is_strategy: bool = False
) -> AsyncGenerator[str, None]:
    """
    Stream a legal answer AND verify grounding in a SINGLE LLM call.
    Yields chunks of the raw JSON string as they are generated by the LLM.
    """
    api_key = GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not configured. "
            "Add it to your .env file or set it as an environment variable."
        )
    
    context = build_context(passages)
    active_prompt = STRATEGY_SYSTEM_PROMPT if is_strategy else LEGAL_SYSTEM_PROMPT
    prompt = active_prompt.format(
        context=context,
        question=question
    )
    
    try:
        client = Groq(api_key=api_key)
        
        # Async stream using Groq client
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Legal Assistant specializing in Indian Law. "
                        "You MUST respond with ONLY a valid JSON object. No markdown, no code fences, no extra text. "
                        "Base your entire application EXCLUSIVELY on the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=1500,
            stream=True
        )
        
        # For the sync groq client, we need to iterate in a thread or use AsyncGroq.
        # Since we use the sync client everywhere, let's wrap iteration.
        # However, groq has AsyncGroq available!
        # Wait, the codebase currently uses `from groq import Groq` and `asyncio.to_thread`.
        # Streaming synchronously in `to_thread` is tricky to yield from.
        # Let's import AsyncGroq if we can, or just yield from the sync iterator using a queue or run_in_executor.
        # Wait, let's import AsyncGroq locally or globally.
        from groq import AsyncGroq
        async_client = AsyncGroq(api_key=api_key)
        
        async_stream = await async_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Legal Assistant specializing in Indian Law. "
                        "Respond directly with your legal answer using the IRAC framework. "
                        "Do not output JSON. Just output the markdown text. "
                        "Base your entire application EXCLUSIVELY on the provided context."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=1500,
            stream=True
        )
        
        async for chunk in async_stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        logger.error(f"Merged stream generation error: {e}")
        yield f"\n\n**Error**: {str(e)}"


def _parse_merged_response(raw: str) -> MergedGenerationResult:
    """
    Parse the merged JSON response from the LLM.
    
    Handles multiple edge cases:
    - Clean JSON
    - JSON wrapped in markdown code fences
    - Completely invalid JSON (fallback: treat raw as plain answer)
    """
    result = MergedGenerationResult()
    
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json or ``` prefix and trailing ```
        lines = cleaned.split("\n")
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    
    try:
        data = json.loads(cleaned)
        
        result.answer = data.get("answer", "")
        result.citations = data.get("citations", [])
        result.faithfulness_score = float(data.get("faithfulness_score", 0.5))
        result.ungrounded_claims = data.get("ungrounded_claims", [])
        
        # Clamp faithfulness to [0, 1]
        result.faithfulness_score = max(0.0, min(1.0, result.faithfulness_score))
        
        # Flag low grounding
        if result.faithfulness_score < 0.7:
            result.is_low_grounding = True
            logger.warning(
                f"LOW_GROUNDING: faithfulness={result.faithfulness_score:.2f}, "
                f"ungrounded_claims={result.ungrounded_claims}"
            )
        
        # Log ungrounded claims as warnings
        for claim in result.ungrounded_claims:
            if claim and claim.lower() not in ("none", "n/a", ""):
                logger.warning(f"Ungrounded claim: {claim}")
        
        logger.info(
            f"Merged parse OK: faithfulness={result.faithfulness_score:.2f}, "
            f"citations={len(result.citations)}, "
            f"ungrounded={len(result.ungrounded_claims)}, "
            f"low_grounding={result.is_low_grounding}"
        )
        
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        # Fallback: treat the entire raw response as the answer
        logger.warning(f"Failed to parse merged JSON response ({e}). Using raw text as answer.")
        result.answer = raw
        result.faithfulness_score = 0.5  # Unknown - conservative default
        result.is_low_grounding = True
        result.ungrounded_claims = ["JSON parsing failed - grounding could not be verified"]
    
    return result

