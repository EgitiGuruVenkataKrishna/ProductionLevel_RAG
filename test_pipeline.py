import asyncio
import sys
from pathlib import Path

# Ensure root is in path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from app.models import QueryRequest
from app.services.pipeline import run_ask_pipeline

async def test():
    req = QueryRequest(
        question="A homeowner enters into a written contract with a contractor to build a backyard deck for ₹3,00,000. Halfway through the project, the contractor stops working and demands an extra ₹5,00,000. If the contractor sues for the additional amount, what legal doctrine protects the homeowner?",
        search_mode="hybrid"
    )
    print(f"Executing RAG pipeline for question: {req.question[:50]}...")
    try:
        response = await run_ask_pipeline(req)
        print("\n\n--- PIPELINE SUCCESS ---")
        print(f"Answer: {response.answer[:100]}...")
        print(f"Faithfulness Score: {response.faithfulness_score}")
    except Exception:
        print("\n\n--- PIPELINE CRASH ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test())
