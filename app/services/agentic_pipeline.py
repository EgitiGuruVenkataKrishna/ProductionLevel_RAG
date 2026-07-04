"""
Agentic RAG Pipeline with CRAG (Corrective Retrieval-Augmented Generation).

Implements a LangGraph-based state machine for dynamic routing,
retrieval grading, and web fallback for the Legal RAG system.
"""
import logging
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

from app.models import QueryRequest, QueryResponse, CitationSource, GroundingMetrics
from app.services.query_expander import expand_query
from app.services.hybrid_retriever import multi_query_hybrid_search, get_chunks
from app.services.reranker import rerank_passages
from app.services.context_filter import filter_and_sanitize
from app.services.generator import generate_and_verify_legal_answer
from app.config import RERANK_TOP_N, AGENT_FALLBACK_THRESHOLD

logger = logging.getLogger(__name__)

# ==================== STATE DEFINITION ====================
class AgentState(TypedDict):
    request: QueryRequest
    is_strategy: bool
    queries: list[str]
    retrieved_chunks: list[dict]
    filtered_passages: list[dict]
    is_grounded: bool
    web_fallback_needed: bool
    final_response: QueryResponse


# ==================== NODE: ROUTER ====================
def route_query(state: AgentState) -> dict:
    """Analyze query complexity to decide pipeline depth."""
    question = state["request"].question.lower()
    
    # Simple heuristics for "naive" routing
    simple_keywords = ["what is", "define", "meaning of", "section", "article"]
    is_simple = any(question.startswith(kw) for kw in simple_keywords) and len(question.split()) < 15
    
    logger.info(f"Router: is_simple={is_simple}")
    return {"queries": [state["request"].question]} if is_simple else {}


# ==================== NODE: NAIVE PIPELINE ====================
async def fast_naive_pipeline(state: AgentState) -> dict:
    """Fast execution path for simple dictionary/definition queries."""
    req = state["request"]
    logger.info("Executing Fast Naive Pipeline...")
    
    # Fast retrieval without query expansion
    search_results = await multi_query_hybrid_search([req.question], req.question, req.search_mode, 5, 5)
    chunk_ids = [cid for cid, _ in search_results[:RERANK_TOP_N]]
    chunks = await get_chunks(chunk_ids)
    
    # Simple score map
    score_map = {cid: score for cid, score in search_results}
    passages = []
    for c in chunks:
        if c.get("id") is not None:
            c["fusion_score"] = score_map.get(c["id"], 0.0)
            c["rerank_score"] = c["fusion_score"]
            passages.append(c)
            
    filtered = filter_and_sanitize(passages)
    return {"filtered_passages": filtered}


# ==================== NODE: DEEP PIPELINE ====================
async def deep_hybrid_pipeline(state: AgentState) -> dict:
    """Deep execution path with query expansion and reranking."""
    req = state["request"]
    logger.info("Executing Deep Hybrid Pipeline...")
    
    # 1. Expand query
    expanded = await expand_query(req.question)
    queries_to_use = [req.question] + expanded
    
    # 2. Parallel retrieval
    search_results = await multi_query_hybrid_search(
        queries_to_use, req.question, req.search_mode
    )
    
    # 3. Fetch chunks
    top_search = search_results[:RERANK_TOP_N * 4]
    chunk_ids = [cid for cid, _ in top_search]
    chunks = await get_chunks(chunk_ids)
    
    # 4. Map & Rerank
    score_map = {cid: score for cid, score in top_search}
    passages = []
    for c in chunks:
        if c.get("id") is not None:
            c["fusion_score"] = score_map.get(c["id"], 0.0)
            c["chunk_id"] = c["id"]  # Ensure chunk_id exists for reranker
            passages.append(c)
            
    reranked = await rerank_passages(req.question, passages)
    filtered = filter_and_sanitize(reranked)
    
    return {"filtered_passages": filtered, "queries": queries_to_use}


# ==================== NODE: GRADER ====================
def grade_retrieval(state: AgentState) -> dict:
    """Determine if retrieved context is sufficient."""
    passages = state["filtered_passages"]
    if not passages:
        return {"web_fallback_needed": True}
        
    best_score = max(p.get("rerank_score", p.get("fusion_score", 0.0)) for p in passages)
    logger.info(f"Grader: best passage score = {best_score:.4f}")
    
    needs_fallback = best_score < AGENT_FALLBACK_THRESHOLD
    return {"web_fallback_needed": needs_fallback}


# ==================== NODE: WEB FALLBACK ====================
def web_fallback(state: AgentState) -> dict:
    """Fallback logic when local documents fail to meet threshold."""
    logger.warning("Agentic pipeline triggered fallback due to low confidence.")
    
    # Empty out the filtered passages to force a deterministic refusal in generation
    return {"filtered_passages": []}


# ==================== NODE: GENERATOR ====================
async def generate_response(state: AgentState) -> dict:
    """Generate final LLM answer."""
    req = state["request"]
    passages = state["filtered_passages"]
    
    if not passages:
        # Build empty response
        resp = QueryResponse(
            answer="I could not find relevant legal provisions for this query.",
            confidence="none",
            confidence_score=0.0,
            best_similarity=0.0,
            search_mode=req.search_mode,
            total_sources_searched=0,
            queries_used=state.get("queries", [req.question]),
            citations=[],
            grounding=None,
            warning="No relevant legal documents found.",
            degraded_mode=False
        )
        return {"final_response": resp, "is_grounded": False}
        
    merged_result = await generate_and_verify_legal_answer(
        question=req.question,
        passages=passages,
        is_strategy=state["is_strategy"]
    )
    
    # (Mapping logic mirrors the updated pipeline.py)
    faithfulness = merged_result.faithfulness_score
    clean_claims = [c for c in merged_result.ungrounded_claims if c and c.lower() not in ("none", "n/a", "")]
    if clean_claims:
        faithfulness = max(0.0, faithfulness - (len(clean_claims) * 0.15))
        
    relevance = min(1.0, faithfulness + 0.1) if faithfulness >= 0.5 else faithfulness
    coverage = 0.5
    overall = faithfulness * 0.6 + relevance * 0.3 + coverage * 0.1
    if faithfulness < 0.3:
        overall = min(overall, 0.15)
    
    is_grounded = (faithfulness >= 0.7 and overall >= 0.5 and not clean_claims)
    
    # Confidence Warning
    conf_level = "high" if overall >= 0.8 else "medium" if overall >= 0.5 else "low"
    warning = None
    if not is_grounded:
        warning = "⚠️ This answer may contain ungrounded claims."
        
    citations = [
        CitationSource(
            text=p["text"][:400],
            article_number=p.get("article_number"),
            section=p.get("section"),
            act_name=p.get("act_name"),
            similarity_score=round(p.get("fusion_score", 0.0), 4),
            rerank_score=round(p.get("rerank_score", 0.0), 4) if p.get("rerank_score") else None
        )
        for p in passages
    ]
    
    resp = QueryResponse(
        answer=merged_result.answer,
        confidence=conf_level,
        confidence_score=overall,
        best_similarity=max(p.get("rerank_score", p.get("fusion_score", 0.0)) for p in passages),
        search_mode=req.search_mode,
        total_sources_searched=0,
        queries_used=state.get("queries", [req.question]),
        citations=citations,
        grounding=GroundingMetrics(
            faithfulness=faithfulness, relevance=relevance, coverage=coverage,
            overall_score=overall, is_grounded=is_grounded, ungrounded_claims=clean_claims
        ),
        warning=warning,
        degraded_mode=False
    )
    
    return {"final_response": resp, "is_grounded": is_grounded}


# ==================== EDGE ROUTING ====================
def decide_pipeline_depth(state: AgentState) -> Literal["fast_naive_pipeline", "deep_hybrid_pipeline"]:
    question = state["request"].question.lower()
    simple_keywords = ["what is", "define", "meaning of", "section", "article"]
    if any(question.startswith(kw) for kw in simple_keywords) and len(question.split()) < 15:
        return "fast_naive_pipeline"
    return "deep_hybrid_pipeline"

def check_fallback(state: AgentState) -> Literal["web_fallback", "generate_response"]:
    if state.get("web_fallback_needed"):
        return "web_fallback"
    return "generate_response"


# ==================== GRAPH COMPILATION ====================
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("route_query", route_query)
workflow.add_node("fast_naive_pipeline", fast_naive_pipeline)
workflow.add_node("deep_hybrid_pipeline", deep_hybrid_pipeline)
workflow.add_node("grade_retrieval", grade_retrieval)
workflow.add_node("web_fallback", web_fallback)
workflow.add_node("generate_response", generate_response)

# Set Entry
workflow.set_entry_point("route_query")

# Add Conditional Edges
workflow.add_conditional_edges(
    "route_query",
    decide_pipeline_depth,
)

# Standard Edges
workflow.add_edge("fast_naive_pipeline", "grade_retrieval")
workflow.add_edge("deep_hybrid_pipeline", "grade_retrieval")

# Conditional Edge from Grader
workflow.add_conditional_edges(
    "grade_retrieval",
    check_fallback,
)

# Fallback goes back to generate
workflow.add_edge("web_fallback", "generate_response")
workflow.add_edge("generate_response", END)

# Compile
crag_pipeline = workflow.compile()


async def run_agentic_pipeline(request: QueryRequest, is_strategy: bool = False) -> QueryResponse:
    """Main entry point for the new CRAG agentic pipeline."""
    logger.info("--- Starting Agentic RAG Pipeline (CRAG) ---")
    
    initial_state = {
        "request": request,
        "is_strategy": is_strategy,
        "queries": [request.question],
        "retrieved_chunks": [],
        "filtered_passages": [],
        "is_grounded": False,
        "web_fallback_needed": False,
        "final_response": None
    }
    
    # Run Graph
    final_state = await crag_pipeline.ainvoke(initial_state)
    
    return final_state["final_response"]
