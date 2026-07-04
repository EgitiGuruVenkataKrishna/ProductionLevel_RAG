from fastapi.testclient import TestClient
from api.ask import app

client = TestClient(app)

def test_api_auth_token():
    """Test generating a dev JWT token"""
    response = client.get("/api/auth/token")
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"

def test_ask_endpoint_unauthorized():
    """Test that /api/ask rejects unauthenticated requests"""
    response = client.post(
        "/api/ask",
        json={"question": "Test", "search_mode": "keyword"}
    )
    assert response.status_code == 403 or response.status_code == 401

# Note: Integration tests requiring the full model and vector DB
# should ideally use mocked dependencies or run in a specific environment.
