"""
FAISS Vector Index Service.

Uses FAISS for dense vector similarity search.
Index is pre-built offline and loaded at startup.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("faiss-cpu not installed. Semantic search will be unavailable.")


class VectorIndex:
    """Manages FAISS flat index for semantic similarity search."""
    
    def __init__(self):
        self.index: Optional[object] = None
        self.dimension: int = 0
        self.is_loaded = False
    
    def build(self, embeddings: np.ndarray, save_path: str):
        """
        Build FAISS index from pre-computed embeddings.
        Called during offline index building.
        
        Args:
            embeddings: np.ndarray of shape (n_docs, embedding_dim)
            save_path: Directory to save the index
        """
        if not FAISS_AVAILABLE:
            raise RuntimeError("faiss-cpu is required. Install with: pip install faiss-cpu")
        
        n_docs, dim = embeddings.shape
        logger.info(f"Building FAISS index: {n_docs} vectors × {dim} dimensions")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Use IndexFlatIP (inner product = cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.dimension = dim
        
        # Save to disk
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        index_file = save_dir / "index.bin"
        faiss.write_index(self.index, str(index_file))
        
        self.is_loaded = True
        logger.info(f"FAISS index saved to {index_file} ({n_docs} vectors)")
    
    def load(self, load_path: str):
        """
        Load a pre-built FAISS index from disk.
        
        Args:
            load_path: Directory containing the saved index
        """
        if not FAISS_AVAILABLE:
            logger.warning("faiss-cpu not available. Skipping index load.")
            return
        
        index_file = Path(load_path) / "index.bin"
        if not index_file.exists():
            logger.warning(f"FAISS index not found at {index_file}")
            return
        
        try:
            self.index = faiss.read_index(str(index_file))
            self.dimension = self.index.d
            self.is_loaded = True
            logger.info(f"FAISS index loaded: {self.index.ntotal} vectors × {self.dimension}d")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            self.is_loaded = False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search FAISS index for nearest neighbors.
        
        Args:
            query_embedding: np.ndarray of shape (embedding_dim,)
            top_k: Number of results
            
        Returns:
            List of (doc_index, similarity_score) tuples
        """
        if not self.is_loaded or self.index is None:
            logger.warning("FAISS index not loaded. Returning empty results.")
            return []
        
        try:
            # Reshape and normalize query
            query = query_embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(query)
            
            # Search
            scores, indices = self.index.search(query, top_k)
            
            # Convert to list of (index, score) tuples
            ranked = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0:  # FAISS returns -1 for not found
                    ranked.append((int(idx), float(score)))
            
            return ranked
        
        except Exception as e:
            logger.error(f"FAISS search error: {e}")
            return []


# Singleton instance
vector_index = VectorIndex()
