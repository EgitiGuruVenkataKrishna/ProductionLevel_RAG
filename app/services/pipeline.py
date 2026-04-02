"""
Shared 8-step pipeline service for Legal RAG.
Used by both main.py and api/ask.py
"""
import logging
from fastapi import HTTPException
from app.config import RERANK_TOP_N, CONTEXT_TOP_N
from app.models import QueryRequest, QueryResponse, CitationSource, GroundingMetrics
from app.services.query_expander import expand_query
from app.services.hybrid_retriever import multi_query_hybrid_search, get_chunk_by_id
from app.services.reranker import rerank_passages
from app.services.context_filter import filter_and_sanitize
from app.services.generator import generate_legal_answer, get_confidence_level, build_context
from app.services.grounding_checker import check_grounding

logger = logging.getLogger(__name__)

async def run_ask_pipeline(request: QueryRequest) -> QueryResponse:
    """Run the complete 8-step RAG pipeline."""
    logger.info(f"═══ Query: '{request.question}' | Mode: {request.search_mode} ═══")
    
    # ──── Step 1: Query Expansion ────
    expanded_queries = await expand_query(request.question)
    logger.info(f"Step 1 — Expanded to {len(expanded_queries)} queries")
    
    # ──── Step 2+3: Multi-Query Hybrid Search + RRF Fusion ────
    search_results = await multi_query_hybrid_search(
        queries=expanded_queries,
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
    for chunk_id, fusion_score in search_results[:RERANK_TOP_N * 4]:
        chunk = get_chunk_by_id(chunk_id)
        if chunk:
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
    
    # ──── Step 6: LLM Generation ────
    answer = await generate_legal_answer(
        question=request.question,
        passages=filtered
    )
    logger.info(f"Step 6 — Answer generated ({len(answer)} chars)")
    
    # ──── Step 7: Answer Grounding Check ────
    context_text = build_context(filtered)
    grounding_result = await check_grounding(
        question=request.question,
        answer=answer,
        context=context_text
    )
    logger.info(f"Step 7 — Grounding: faith={grounding_result.faithfulness:.2f} "
                f"grounded={grounding_result.is_grounded}")
    
    # Add warning if answer is not well-grounded
    grounding_warning = None
    if not grounding_result.is_grounded:
        grounding_warning = (
            "⚠️ Some claims in this answer may not be fully supported by the "
            "retrieved documents. Please verify with authoritative legal sources."
        )
    elif grounding_result.ungrounded_claims:
        grounding_warning = (
            f"⚠️ Potentially ungrounded claims detected: "
            f"{'; '.join(grounding_result.ungrounded_claims[:3])}"
        )
    
    # ──── Step 8: Real Confidence Scoring ────
    # Use grounding overall_score as the primary confidence metric
    confidence_score = grounding_result.overall_score
    confidence_level, base_warning = get_confidence_level(confidence_score)
    
    # Combine warnings
    final_warning = grounding_warning or base_warning
    if confidence_score < request.min_confidence:
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
        faithfulness=grounding_result.faithfulness,
        relevance=grounding_result.relevance,
        coverage=grounding_result.coverage,
        overall_score=grounding_result.overall_score,
        is_grounded=grounding_result.is_grounded,
        ungrounded_claims=grounding_result.ungrounded_claims
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
