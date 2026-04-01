"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class QueryRequest(BaseModel):
    """Request model for legal questions."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Legal question to ask"
    )
    search_mode: str = Field(
        default="hybrid",
        description="Search mode: 'hybrid', 'semantic', or 'keyword'"
    )
    min_confidence: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold"
    )


class CitationSource(BaseModel):
    """Individual source citation."""
    text: str
    article_number: Optional[str] = None
    section: Optional[str] = None
    act_name: Optional[str] = None
    part: Optional[str] = None
    page: Optional[int] = None
    similarity_score: float
    rerank_score: Optional[float] = None


class GroundingMetrics(BaseModel):
    """Real confidence metrics from answer grounding verification."""
    faithfulness: float = Field(description="Fraction of claims grounded in context (0-1)")
    relevance: float = Field(description="How well the answer addresses the question (0-1)")
    coverage: float = Field(description="How much relevant context was utilized (0-1)")
    overall_score: float = Field(description="Weighted composite: 50% faith + 30% rel + 20% cov")
    is_grounded: bool = Field(description="True if answer passes grounding threshold")
    ungrounded_claims: list[str] = Field(default_factory=list, description="Claims not supported by context")


class QueryResponse(BaseModel):
    """Response model with legal answer, citations, and scoring."""
    answer: str
    confidence: str           # "high", "medium", "low", "very_low", "none"
    confidence_score: float   # 0.0 - 1.0
    best_similarity: float
    search_mode: str
    total_sources_searched: int
    queries_used: list[str] = Field(default_factory=list, description="Expanded queries used for retrieval")
    citations: list[CitationSource]
    grounding: Optional[GroundingMetrics] = None
    warning: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    total_chunks_indexed: int
    bm25_index_loaded: bool
    faiss_index_loaded: bool
    embedding_model: str
    reranker_model: str
    llm_model: str


class LegalChunkMetadata(BaseModel):
    """Metadata for a legal document chunk."""
    chunk_id: int
    text: str
    source_file: str
    page: Optional[int] = None
    article_number: Optional[str] = None
    section: Optional[str] = None
    act_name: Optional[str] = None
    part: Optional[str] = None
    chapter: Optional[str] = None
    schedule: Optional[str] = None
    amendment: Optional[str] = None
