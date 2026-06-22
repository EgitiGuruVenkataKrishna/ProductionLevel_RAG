import asyncio
from fastapi.testclient import TestClient
from api.ask import app
import json

client = TestClient(app)

def test():
    print("Sending request to /api/ask...")
    response = client.post(
        "/api/ask",
        json={
            "question": "A homeowner enters into a written contract with a contractor to build a backyard deck for 300000. Halfway through the project, the contractor stops working and demands an extra 500000. If the contractor sues for the additional amount, what legal doctrine protects the homeowner?",
            "search_mode": "hybrid"
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success:", response.json()["answer"][:100])
    else:
        print("Error response:", response.text)

if __name__ == "__main__":
    test()
