import logging
import os
from pinecone import Pinecone
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME

logger = logging.getLogger(__name__)

class PineconeService:
    def __init__(self):
        self.pc = None
        self.index = None
        if PINECONE_API_KEY:
            try:
                self.pc = Pinecone(api_key=PINECONE_API_KEY)
                self.index = self.pc.Index(PINECONE_INDEX_NAME)
                logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")

    async def search(self, vector: list[float], top_k: int = 25, category_filter: str = None) -> list[tuple[int, float]]:
        """
        Search Pinecone with metadata filter.
        Returns list of (chunk_id, similarity_score).
        """
        if not self.index:
            logger.warning("Pinecone is not initialized. Cannot perform search.")
            return []

        filter_dict = {}
        if category_filter:
            filter_dict = {"category": {"$eq": category_filter}}

        try:
            # Note: Pinecone python client query is synchronous, so we could wrap in to_thread, 
            # but for now we execute directly as it's typically fast.
            response = self.index.query(
                vector=vector,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            results = []
            for match in response.matches:
                # We assume the Pinecone ID is the chunk_id string, e.g., "1", "2"
                # Or chunk_id is stored in metadata.
                chunk_id_str = match.id
                try:
                    chunk_id = int(chunk_id_str)
                except ValueError:
                    # Fallback to metadata if ID is not an integer
                    chunk_id = match.metadata.get("chunk_id", -1)
                    
                score = match.score
                results.append((chunk_id, score))
            
            logger.info(f"Pinecone retrieved {len(results)} chunks for category {category_filter}")
            return results
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

pinecone_service = PineconeService()
