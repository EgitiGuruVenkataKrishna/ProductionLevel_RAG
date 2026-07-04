import pytest
import asyncio
from app.models import QueryRequest
from app.services.pipeline import run_ask_pipeline

@pytest.mark.asyncio
async def test_run_ask_pipeline_mocked(mocker):
    """
    Mock the pipeline steps to test the orchestration flow without hitting real APIs.
    """
    mocker.patch("app.services.pipeline.get_exact_cache", return_value=None)
    mocker.patch("app.services.pipeline.embed_query", return_value=None)
    mocker.patch("app.services.pipeline.expand_query", return_value=["Mock expanded"])
    mocker.patch("app.services.pipeline.multi_query_hybrid_search", return_value=[("chunk_1", 0.9)])
    mocker.patch("app.services.pipeline.get_chunks", return_value=[
        {"id": "chunk_1", "text": "Mock law passage.", "article_number": "1"}
    ])
    mocker.patch("app.services.pipeline.rerank_passages", return_value=[
        {"chunk_id": "chunk_1", "text": "Mock law passage.", "fusion_score": 0.9, "rerank_score": 0.9}
    ])
    
    # Mock generator response
    class MockResult:
        answer = "Mock answer."
        citations = []
        faithfulness_score = 0.9
        ungrounded_claims = []
        is_low_grounding = False
    
    mocker.patch("app.services.pipeline.generate_and_verify_legal_answer", return_value=MockResult())
    mocker.patch("app.services.pipeline.set_exact_cache")
    
    req = QueryRequest(question="What is the law?", search_mode="hybrid")
    
    response = await run_ask_pipeline(req)
    
    assert response.answer == "Mock answer."
    assert response.confidence_score >= 0.5
    assert len(response.citations) == 1
    assert response.citations[0].text == "Mock law passage."
