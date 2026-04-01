"""
Vercel Serverless Function: /api/ask

Main query endpoint for the Legal RAG assistant.
Full Pipeline:
  1. Query Expansion (multi-query via Groq)
  2. Hybrid Retrieval (BM25 + FAISS) × N queries
  3. RRF Fusion (merge all results)
  4. Cross-Encoder Reranking (top candidates)
  5. Context Filtering (dedup + sanitize)
  6. LLM Generation (strict legal prompt)
  7. Answer Grounding Check (verify citations)
  8. Real Confidence Scoring → Final Response
"""
import sys
import os
import logging
from pathlib import Path

# Add project root to path for imports
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import (
    FAISS_INDEX_PATH, BM25_INDEX_PATH, RERANK_TOP_N, CONTEXT_TOP_N
)
from app.models import QueryRequest, QueryResponse, CitationSource, GroundingMetrics
from app.services.bm25_index import bm25_index
from app.services.vector_index import vector_index
from app.services.query_expander import expand_query
from app.services.hybrid_retriever import (
    multi_query_hybrid_search, load_chunks_metadata, get_chunk_by_id
)
from app.services.reranker import rerank_passages
from app.services.context_filter import filter_and_sanitize
from app.services.generator import generate_legal_answer, get_confidence_level, build_context
from app.services.grounding_checker import check_grounding

app = FastAPI(
    title="Legal RAG Assistant API",
    description="Junior Legal Assistant — Indian Law Q&A with hybrid search, reranking, grounding, and citations",
    version="2.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== STARTUP: Load Indices ====================
_initialized = False


def _ensure_initialized():
    """Lazy initialization of indices (for serverless cold starts)."""
    global _initialized
    if _initialized:
        return
    
    logger.info("Initializing indices...")
    
    faiss_path = str(FAISS_INDEX_PATH)
    if Path(faiss_path).exists():
        vector_index.load(faiss_path)
    else:
        logger.warning(f"FAISS index not found at {faiss_path}")
    
    bm25_path = str(BM25_INDEX_PATH)
    if Path(bm25_path).exists():
        bm25_index.load(bm25_path)
    else:
        logger.warning(f"BM25 index not found at {bm25_path}")
    
    load_chunks_metadata()
    _initialized = True
    logger.info("Initialization complete.")


# ==================== ENDPOINTS ====================

@app.post("/api/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a legal question — full 8-step pipeline.
    
    1. Query Expansion → 2. Multi-Query Hybrid Search →
    3. RRF Fusion → 4. Cross-Encoder Rerank →
    5. Context Filter → 6. LLM Generate →
    7. Grounding Check → 8. Confidence Score → Response
    """
    _ensure_initialized()
    
    try:
        logger.info(f"═══ Query: '{request.question}' | Mode: {request.search_mode} ═══")
        
        # ──── Step 1: Query Expansion ────
        expanded_queries = await expand_query(request.question)
        logger.info(f"Step 1 — Expanded to {len(expanded_queries)} queries")
        
        # ──── Step 2+3: Multi-Query Hybrid Search + RRF Fusion ────
        search_results = await multi_query_hybrid_search(
            queries=expanded_queries,
            mode=request.search_mode
        )
        
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
                warning="No relevant legal documents found."
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
                warning="Retrieved chunks could not be loaded."
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
        if grounding_result.ungrounded_claims:
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
                f"Confidence ({confidence_score:.2f}) is below your threshold "
                f"({request.min_confidence}). {final_warning or ''}"
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
    
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")
