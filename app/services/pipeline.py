"""
Shared pipeline service for Legal RAG.
Used by both main.py and api/ask.py.

Merged generation+grounding: Steps 6+7 are now a single LLM call.
"""
import logging
from fastapi import HTTPException
from langfuse import observe
from app.config import RERANK_TOP_N, CONTEXT_TOP_N, USE_AGENTIC_PIPELINE
from app.models import QueryRequest, QueryResponse, CitationSource, GroundingMetrics
from app.services.query_expander import expand_query
from app.services.hybrid_retriever import multi_query_hybrid_search, get_chunks, embed_query
from app.services.reranker import rerank_passages
from app.services.context_filter import filter_and_sanitize
from app.services.generator import (
    generate_legal_answer, get_confidence_level, build_context,
    generate_and_verify_legal_answer,  # Merged generation + grounding
    generate_and_verify_legal_answer_stream,
    detect_query_intent
)
from app.services.cache import (
    get_exact_cache, set_exact_cache,
    get_semantic_cache, set_semantic_cache
)
from app.utils.pii import redact_pii, restore_pii
from app.services.memory import get_history, add_message

logger = logging.getLogger(__name__)

@observe(name="run_ask_pipeline")
async def run_ask_pipeline(request: QueryRequest) -> QueryResponse:
    """Run the complete RAG pipeline (Legacy or Agentic based on config)."""
    
    # ──── Feature Flag Routing ────
    if USE_AGENTIC_PIPELINE:
        from app.services.agentic_pipeline import run_agentic_pipeline
        is_strategy = "[FACTS]:" in request.question.upper()
        return await run_agentic_pipeline(request, is_strategy)
    
    # ──── Step 0: PII Redaction & Memory ────
    redacted_question, pii_map = redact_pii(request.question)
    
    memory_context = ""
    if request.session_id:
        history = await get_history(request.session_id)
        if history:
            memory_lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                memory_lines.append(f"{role}: {msg['content']}")
            memory_context = "Conversation History:\n" + "\n".join(memory_lines) + "\n\nCurrent Query: "
            
    # Apply memory to query for search/generation
    processed_question = memory_context + redacted_question if memory_context else redacted_question
    
    logger.info(f"═══ Query: '{processed_question}' | Mode: {request.search_mode} ═══")
    
    # ──── Check Exact Cache ────
    cached_response_dict = await get_exact_cache(request.search_mode, processed_question)
    if cached_response_dict:
        cached_response_dict["answer"] = restore_pii(cached_response_dict["answer"], pii_map)
        return QueryResponse(**cached_response_dict)
    
    # ──── Check Semantic Cache ────
    query_embedding = await embed_query(processed_question)
    if query_embedding is not None:
        semantic_cached_dict = await get_semantic_cache(request.search_mode, query_embedding)
        if semantic_cached_dict:
            semantic_cached_dict["answer"] = restore_pii(semantic_cached_dict["answer"], pii_map)
            return QueryResponse(**semantic_cached_dict)
            
    # ──── Step 0.5: Fast Intent Classification ────
    intent = await detect_query_intent(processed_question)
    
    if intent == "greeting":
        return QueryResponse(
            answer="**Hello!** I am a Legal Assistant specializing in Indian Law. I can help you with questions about the Constitution, IPC, BNS, and various Acts.\n\n*Please ask me a legal question or present a legal scenario to get started!*",
            confidence="high", confidence_score=1.0, best_similarity=1.0,
            search_mode=request.search_mode, total_sources_searched=0, queries_used=[request.question],
            citations=[], grounding=None, warning=None, degraded_mode=False
        )
        
    if intent == "system":
        return QueryResponse(
            answer="I am an advanced AI Legal Assistant designed to help with Indian Law. I can analyze legal scenarios, find relevant acts and sections, and provide structured legal analysis using the IRAC framework. How can I assist you today?",
            confidence="high", confidence_score=1.0, best_similarity=1.0,
            search_mode=request.search_mode, total_sources_searched=0, queries_used=[request.question],
            citations=[], grounding=None, warning=None, degraded_mode=False
        )
    
    # ──── Step 1: Query Intent & Expansion ────
    is_strategy = "[FACTS]:" in processed_question.upper()
    
    if is_strategy:
        expanded_queries = [processed_question]
        logger.info("Step 1 — Skipped query expansion for complex Strategy query")
    else:
        expanded_queries = await expand_query(processed_question)
        logger.info(f"Step 1 — Expanded to {len(expanded_queries)} queries")
    
    # ──── Step 2+3: Multi-Query Hybrid Search + RRF Fusion ────
    search_results = await multi_query_hybrid_search(
        queries=expanded_queries,
        original_query=processed_question,
        mode=request.search_mode,
        precomputed_embedding=query_embedding
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
        query=processed_question,
        passages=candidate_passages,
        top_n=CONTEXT_TOP_N
    )
    logger.info(f"Step 4 — Reranked to top {len(reranked)}")
    
    # ──── Step 5: Context Filtering ────
    filtered = filter_and_sanitize(reranked)
    logger.info(f"Step 5 — Filtered to {len(filtered)} clean passages")
    
    # ──── Step 6+7 (Merged): LLM Generation + Grounding in ONE call ────
    merged_result = await generate_and_verify_legal_answer(
        question=processed_question,
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
    best_score = min(best_score, 1.0)
    
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
    
    # Restore PII in the generated answer
    final_answer = restore_pii(answer, pii_map)
    
    response = QueryResponse(
        answer=final_answer,
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
    
    # ──── Store in Exact Cache ────
    # Cache the original answer (without PII restored) or with PII? 
    # Caching the un-restored answer prevents leaking PII if hashes collide.
    cache_payload = response.model_dump()
    cache_payload["answer"] = answer  # Store redacted answer in cache
    
    await set_exact_cache(request.search_mode, processed_question, cache_payload)
    
    # ──── Store in Semantic Cache ────
    if query_embedding is not None:
        await set_semantic_cache(request.search_mode, processed_question, query_embedding, cache_payload)
        
    # ──── Store in Memory ────
    if request.session_id:
        await add_message(request.session_id, "user", redacted_question)
        await add_message(request.session_id, "assistant", answer)  # Store redacted answer
    
    return response


import json
from typing import AsyncGenerator

@observe(name="run_ask_pipeline_stream")
async def run_ask_pipeline_stream(request: QueryRequest) -> AsyncGenerator[str, None]:
    """Stream the RAG pipeline steps and generated answer."""
    yield json.dumps({"status": "starting", "message": "Initializing pipeline..."})
    
    # ──── Step 0: PII Redaction & Memory ────
    redacted_question, pii_map = redact_pii(request.question)
    
    memory_context = ""
    if request.session_id:
        history = await get_history(request.session_id)
        if history:
            memory_lines = []
            for msg in history:
                role = "User" if msg["role"] == "user" else "Assistant"
                memory_lines.append(f"{role}: {msg['content']}")
            memory_context = "Conversation History:\n" + "\n".join(memory_lines) + "\n\nCurrent Query: "
            
    processed_question = memory_context + redacted_question if memory_context else redacted_question
    
    yield json.dumps({"status": "cache_check", "message": "Checking caches..."})
    
    # ──── Check Exact Cache ────
    cached_response_dict = await get_exact_cache(request.search_mode, processed_question)
    if cached_response_dict:
        cached_response_dict["answer"] = restore_pii(cached_response_dict["answer"], pii_map)
        yield json.dumps({"status": "complete", "response": cached_response_dict})
        return
        
    query_embedding = await embed_query(processed_question)
    if query_embedding is not None:
        semantic_cached_dict = await get_semantic_cache(request.search_mode, query_embedding)
        if semantic_cached_dict:
            semantic_cached_dict["answer"] = restore_pii(semantic_cached_dict["answer"], pii_map)
            yield json.dumps({"status": "complete", "response": semantic_cached_dict})
            return
            
    # ──── Fast Intent Classification ────
    intent = await detect_query_intent(processed_question)
    
    if intent == "greeting":
        yield json.dumps({
            "status": "complete",
            "response": {
                "answer": "**Hello!** I am a Legal Assistant specializing in Indian Law. I can help you with questions about the Constitution, IPC, BNS, and various Acts.\n\n*Please ask me a legal question or present a legal scenario to get started!*",
                "confidence": "high", "confidence_score": 1.0, "best_similarity": 1.0,
                "search_mode": request.search_mode, "total_sources_searched": 0, "queries_used": [request.question],
                "citations": [], "grounding": None, "warning": None, "degraded_mode": False
            }
        })
        return
        
    if intent == "system":
        yield json.dumps({
            "status": "complete",
            "response": {
                "answer": "I am an advanced AI Legal Assistant designed to help with Indian Law. I can analyze legal scenarios, find relevant acts and sections, and provide structured legal analysis using the IRAC framework. How can I assist you today?",
                "confidence": "high", "confidence_score": 1.0, "best_similarity": 1.0,
                "search_mode": request.search_mode, "total_sources_searched": 0, "queries_used": [request.question],
                "citations": [], "grounding": None, "warning": None, "degraded_mode": False
            }
        })
        return
            
    yield json.dumps({"status": "retrieving", "message": "Searching legal knowledge base..."})
    
    is_strategy = "[FACTS]:" in processed_question.upper()
    if is_strategy:
        expanded_queries = [processed_question]
    else:
        expanded_queries = await expand_query(processed_question)
        
    search_results = await multi_query_hybrid_search(
        queries=expanded_queries,
        original_query=processed_question,
        mode=request.search_mode,
        precomputed_embedding=query_embedding
    )
    
    if not search_results:
        yield json.dumps({
            "status": "error", 
            "message": "No relevant legal documents found."
        })
        return
        
    top_search_results = search_results[:RERANK_TOP_N * 4]
    chunk_ids = [chunk_id for chunk_id, _ in top_search_results]
    chunks = await get_chunks(chunk_ids)
    
    fusion_score_map = {chunk_id: score for chunk_id, score in top_search_results}
    candidate_passages = []
    
    for chunk in chunks:
        chunk_id = chunk.get("id")
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
        
    yield json.dumps({"status": "reranking", "message": "Reranking documents..."})
    
    reranked = await rerank_passages(
        query=processed_question,
        passages=candidate_passages,
        top_n=CONTEXT_TOP_N
    )
    filtered = filter_and_sanitize(reranked)
    
    yield json.dumps({"status": "generating", "message": "Generating answer..."})
    
    # We yield the stream chunks as they arrive
    full_answer = ""
    async for chunk in generate_and_verify_legal_answer_stream(processed_question, filtered, is_strategy):
        full_answer += chunk
        # Escape the chunk for JSON streaming
        yield json.dumps({"status": "chunk", "chunk": chunk})
        
    final_answer = restore_pii(full_answer, pii_map)
    
    # Calculate base confidence from retrieved passages (since we skip grounding for stream)
    best_score = max(
        (p.get("rerank_score") or p.get("fusion_score", 0.0)) for p in filtered
    ) if filtered else 0.0
    best_score = min(best_score, 1.0)
    
    # Simple heuristic for streaming confidence
    confidence_level, warning = get_confidence_level(best_score)
    
    citations = [
        {
            "text": p["text"][:400] + ("..." if len(p["text"]) > 400 else ""),
            "article_number": p.get("article_number"),
            "section": p.get("section"),
            "act_name": p.get("act_name"),
            "part": p.get("part"),
            "page": p.get("page"),
            "similarity_score": round(p.get("fusion_score", 0.0), 4),
            "rerank_score": round(p.get("rerank_score", 0.0), 4) if p.get("rerank_score") else None
        }
        for p in filtered
    ]
    
    # Store in cache and memory
    cache_payload = {
        "answer": full_answer,  # Redacted version
        "confidence": confidence_level,
        "confidence_score": round(best_score, 4),
        "best_similarity": round(best_score, 4),
        "search_mode": request.search_mode,
        "total_sources_searched": len(search_results),
        "queries_used": expanded_queries,
        "citations": citations,
        "grounding": None,
        "warning": warning,
        "degraded_mode": False
    }
    
    await set_exact_cache(request.search_mode, processed_question, cache_payload)
    if query_embedding is not None:
        await set_semantic_cache(request.search_mode, processed_question, query_embedding, cache_payload)
        
    if request.session_id:
        await add_message(request.session_id, "user", redacted_question)
        await add_message(request.session_id, "assistant", full_answer)
        
    yield json.dumps({
        "status": "complete",
        "response": {
            **cache_payload,
            "answer": final_answer
        }
    })
