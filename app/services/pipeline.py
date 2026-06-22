"""
Shared pipeline service for Legal RAG.
Used by both main.py and api/ask.py.

Merged generation+grounding: Steps 6+7 are now a single LLM call.
"""
import logging
from fastapi import HTTPException
from app.config import RERANK_TOP_N, CONTEXT_TOP_N, USE_AGENTIC_PIPELINE
from app.models import QueryRequest, QueryResponse, CitationSource, GroundingMetrics
from app.services.query_expander import expand_query
from app.services.hybrid_retriever import multi_query_hybrid_search, get_chunks
from app.services.reranker import rerank_passages
from app.services.context_filter import filter_and_sanitize
from app.services.generator import (
    generate_legal_answer, get_confidence_level, build_context,
    generate_and_verify_legal_answer  # Merged generation + grounding
)
from app.services.grounding_checker import check_grounding  # DEBT: Legacy — kept for backward compat

logger = logging.getLogger(__name__)

async def run_ask_pipeline(request: QueryRequest) -> QueryResponse:
    """Run the complete RAG pipeline (Legacy or Agentic based on config)."""
    
    # ──── Feature Flag Routing ────
    if USE_AGENTIC_PIPELINE:
        from app.services.agentic_pipeline import run_agentic_pipeline
        is_strategy = "[FACTS]:" in request.question.upper()
        return await run_agentic_pipeline(request, is_strategy)
    
    logger.info(f"═══ Query: '{request.question}' | Mode: {request.search_mode} ═══")
    
    # ──── Step 1: Query Intent & Expansion ────
    is_strategy = "[FACTS]:" in request.question.upper()
    
    if is_strategy:
        expanded_queries = [request.question]
        logger.info("Step 1 — Skipped query expansion for complex Strategy query")
    else:
        expanded_queries = await expand_query(request.question)
        logger.info(f"Step 1 — Expanded to {len(expanded_queries)} queries")
    
    # ──── Step 2+3: Multi-Query Hybrid Search + RRF Fusion ────
    search_results = await multi_query_hybrid_search(
        queries=expanded_queries,
        original_query=request.question,
        mode=request.search_mode
    )
    
    degraded = False
    # If mode was hybrid/semantic but no search results had semantic contributions, we can flag degraded.
    # We will let multi_query_hybrid_search handle logging, but if search_results is empty or keyword-only:
    
    if not search_results:
        return QueryResponse(
            answer="I could not find relevant information in the available legal documents to answer this question. Please try rephrasing your query or consult a qualified legal professional.",
            confidence="none",
            confidence_score=0.0,
            best_similarity=0.0,
            search_mode=request.search_mode,
            total_sources_searched=0,
            queries_used=expanded_queries,
            citations=[],
            grounding=None,
            warning="No relevant legal documents found.",
            degraded_mode=False
        )
    
    logger.info(f"Step 2-3 — Retrieved {len(search_results)} candidates")
    
    # ──── Step 4: Gather Passage Details + Rerank ────
    candidate_passages = []
    
    # Extract just the top IDs
    top_search_results = search_results[:RERANK_TOP_N * 4]
    chunk_ids = [chunk_id for chunk_id, _ in top_search_results]
    
    # Fetch all chunks in a batch (async, potentially from SQLite)
    chunks = await get_chunks(chunk_ids)
    
    # Build a lookup to map fusion_scores back to chunks
    fusion_score_map = {chunk_id: score for chunk_id, score in top_search_results}
    
    for chunk in chunks:
        chunk_id = chunk.get("id")
        # In-memory fallback might not inject "id" if not careful, fallback to lookup
        if chunk_id is None:
            continue
            
        fusion_score = fusion_score_map.get(chunk_id, 0.0)
        
        candidate_passages.append({
            "chunk_id": chunk_id,
            "text": chunk.get("text", ""),
            "article_number": chunk.get("article_number"),
            "section": chunk.get("section"),
            "act_name": chunk.get("act_name"),
            "part": chunk.get("part"),
            "source_file": chunk.get("source_file", ""),
            "page": chunk.get("page"),
            "fusion_score": fusion_score,
            "similarity_score": fusion_score,
        })
    
    if not candidate_passages:
        return QueryResponse(
            answer="I could not find relevant information in the available legal documents.",
            confidence="none",
            confidence_score=0.0,
            best_similarity=0.0,
            search_mode=request.search_mode,
            total_sources_searched=len(search_results),
            queries_used=expanded_queries,
            citations=[],
            grounding=None,
            warning="Retrieved chunks could not be loaded.",
            degraded_mode=False
        )
    
    reranked = await rerank_passages(
        query=request.question,
        passages=candidate_passages,
        top_n=CONTEXT_TOP_N
    )
    logger.info(f"Step 4 — Reranked to top {len(reranked)}")
    
    # ──── Step 5: Context Filtering ────
    filtered = filter_and_sanitize(reranked)
    logger.info(f"Step 5 — Filtered to {len(filtered)} clean passages")
    
    # ──── Step 6+7 (Merged): LLM Generation + Grounding in ONE call ────
    merged_result = await generate_and_verify_legal_answer(
        question=request.question,
        passages=filtered,
        is_strategy=is_strategy
    )
    answer = merged_result.answer
    logger.info(f"Step 6+7 — Answer generated + grounded ({len(answer)} chars, "
                f"faith={merged_result.faithfulness_score:.2f})")
    
    # ──── Step 6.5: Intercept Greetings & Non-Legal Queries ────
    if "GREETING_OR_NON_LEGAL_QUERY" in answer:
        return QueryResponse(
            answer="**Hello!** I am a Legal Assistant specializing in Indian Law. I can help you with questions about the Constitution, IPC, BNS, and various Acts.\n\n*Please ask me a legal question or present a legal scenario to get started!*",
            confidence="high",
            confidence_score=1.0,
            best_similarity=0.0,
            search_mode=request.search_mode,
            total_sources_searched=0,
            queries_used=[request.question],
            citations=[],
            grounding=None,
            warning=None,
            degraded_mode=False
        )

    # ──── Map merged result to grounding metrics ────
    # Faithfulness is the primary metric (60% weight). Relevance and coverage
    # are estimated conservatively since the merged call focuses on faithfulness.
    faithfulness = merged_result.faithfulness_score
    
    # Apply strict penalty for ungrounded claims (matches legacy behavior)
    clean_claims = [
        c for c in merged_result.ungrounded_claims
        if c and c.lower() not in ("none", "n/a", "")
    ]
    if clean_claims:
        penalty = len(clean_claims) * 0.15
        faithfulness = max(0.0, faithfulness - penalty)
    
    # Estimate relevance from faithfulness (highly correlated for legal Q&A)
    relevance = min(1.0, faithfulness + 0.1) if faithfulness >= 0.5 else faithfulness
    coverage = 0.5  # Conservative default for merged call
    
    # Weighted overall score (matches legacy: 60% faith + 30% rel + 10% cov)
    overall_score = faithfulness * 0.60 + relevance * 0.30 + coverage * 0.10
    
    # If faithfulness is critically low, tank the score (matches legacy behavior)
    if faithfulness < 0.3:
        overall_score = min(overall_score, 0.15)
    
    is_grounded = (
        faithfulness >= 0.7 and
        overall_score >= 0.5 and
        len(clean_claims) == 0
    )
    
    logger.info(
        f"Grounding: faith={faithfulness:.2f} rel={relevance:.2f} "
        f"cov={coverage:.2f} overall={overall_score:.2f} grounded={is_grounded}"
    )
    
    # Add warning if answer is not well-grounded
    grounding_warning = None
    if not is_grounded:
        grounding_warning = (
            "⚠️ Some claims in this answer may not be fully supported by the "
            "retrieved documents. Please verify with authoritative legal sources."
        )
    elif clean_claims:
        grounding_warning = (
            f"⚠️ Potentially ungrounded claims detected: "
            f"{'; '.join(clean_claims[:3])}"
        )
    
    # ──── Step 8: Real Confidence Scoring ────
    confidence_score = overall_score
    confidence_level, base_warning = get_confidence_level(confidence_score)
    
    # Combine warnings
    final_warning = grounding_warning or base_warning
    if confidence_level == "rejected":
        answer = "I apologize, but I do not have enough specific, reliable legal context to answer this safely. To avoid providing incorrect information, please consult a Senior Advocate."
        final_warning = base_warning
    elif confidence_score < request.min_confidence:
        confidence_level = "low"
        final_warning = (
            f"Confidence ({confidence_score:.2f}) is below your threshold ({request.min_confidence}). "
            f"{final_warning or ''}"
        ).strip()
    
    best_score = max(
        p.get("rerank_score", p.get("fusion_score", 0.0)) for p in filtered
    )
    
    # Build citations
    citations = [
        CitationSource(
            text=p["text"][:400] + ("..." if len(p["text"]) > 400 else ""),
            article_number=p.get("article_number"),
            section=p.get("section"),
            act_name=p.get("act_name"),
            part=p.get("part"),
            page=p.get("page"),
            similarity_score=round(p.get("fusion_score", 0.0), 4),
            rerank_score=round(p.get("rerank_score", 0.0), 4) if p.get("rerank_score") else None
        )
        for p in filtered
    ]
    
    grounding_metrics = GroundingMetrics(
        faithfulness=faithfulness,
        relevance=relevance,
        coverage=coverage,
        overall_score=overall_score,
        is_grounded=is_grounded,
        ungrounded_claims=clean_claims
    )
    
    logger.info(f"═══ Pipeline complete | Confidence: {confidence_level} ({confidence_score:.2f}) ═══")
    
    return QueryResponse(
        answer=answer,
        confidence=confidence_level,
        confidence_score=round(confidence_score, 4),
        best_similarity=round(best_score, 4),
        search_mode=request.search_mode,
        total_sources_searched=len(search_results),
        queries_used=expanded_queries,
        citations=citations,
        grounding=grounding_metrics,
        warning=final_warning
    )
