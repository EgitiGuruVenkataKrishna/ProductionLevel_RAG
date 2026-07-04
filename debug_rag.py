import asyncio
import sys
from pathlib import Path

# Ensure root is in path
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

from app.services.query_expander import expand_query
from app.services.hybrid_retriever import multi_query_hybrid_search, get_chunks

async def debug():
    question = "A homeowner enters into a written contract with a contractor to build a backyard deck for ₹3,00,000. Halfway through the project, the contractor stops working and demands an extra ₹5,00,000 to finish, citing a sudden rise in the market price of lumber. The homeowner, desperate to have the deck ready for a scheduled family wedding, signs a new agreement promising the extra money. After the deck is completed, the homeowner refuses to pay the extra ₹5,00,000.If the contractor sues for the additional amount, what legal doctrine protects the homeowner from paying the extra money?"
    
    print("1. Testing Query Expansion (HyDE)")
    expanded = await expand_query(question)
    print("Expanded Queries:")
    for e in expanded:
        print(f" - {e}")
        
    print("\n2. Testing Retrieval")
    queries_to_use = [question] + expanded
    search_results = await multi_query_hybrid_search(queries_to_use, question, "hybrid")
    print(f"Retrieved {len(search_results)} candidates.")
    
    if search_results:
        top_chunks = await get_chunks([cid for cid, _ in search_results[:5]])
        for i, chunk in enumerate(top_chunks):
            print(f"\n--- TOP {i+1} CHUNK ---")
            print(f"Text: {chunk.get('text', '')[:300]}...")
            print(f"Metadata: {chunk.get('metadata', {})}")

if __name__ == "__main__":
    asyncio.run(debug())
