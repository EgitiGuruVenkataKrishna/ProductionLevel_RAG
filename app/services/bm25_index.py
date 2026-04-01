"""
BM25 Sparse Index Service.

Uses bm25s for ultra-fast keyword retrieval.
Index is pre-built offline and loaded at startup.
"""
import logging
import json
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import bm25s — graceful fallback if not installed
try:
    import bm25s
    BM25S_AVAILABLE = True
except ImportError:
    BM25S_AVAILABLE = False
    logger.warning("bm25s not installed. BM25 keyword search will be unavailable.")


class BM25Index:
    """Manages BM25 sparse index for keyword-based legal document retrieval."""
    
    def __init__(self):
        self.index: Optional[object] = None
        self.corpus_tokens = None
        self.is_loaded = False
    
    def build(self, texts: list[str], save_path: str):
        """
        Build BM25 index from a list of document texts.
        Called during offline index building (scripts/build_index.py).
        
        Args:
            texts: List of chunk text strings
            save_path: Directory to save the index
        """
        if not BM25S_AVAILABLE:
            raise RuntimeError("bm25s is required. Install with: pip install bm25s[full]")
        
        logger.info(f"Building BM25 index over {len(texts)} chunks...")
        
        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(texts, stopwords="en")
        
        # Create and index
        self.index = bm25s.BM25()
        self.index.index(corpus_tokens)
        
        # Save to disk
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.index.save(str(save_dir))
        
        self.is_loaded = True
        logger.info(f"BM25 index built and saved to {save_dir}")
    
    def load(self, load_path: str):
        """
        Load a pre-built BM25 index from disk.
        Called at serverless function startup.
        
        Args:
            load_path: Directory containing the saved index
        """
        if not BM25S_AVAILABLE:
            logger.warning("bm25s not available. Skipping BM25 index load.")
            return
        
        load_dir = Path(load_path)
        if not load_dir.exists():
            logger.warning(f"BM25 index not found at {load_dir}")
            return
        
        try:
            self.index = bm25s.BM25.load(str(load_dir), mmap=True)
            self.is_loaded = True
            logger.info(f"BM25 index loaded from {load_dir} (memory-mapped)")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            self.is_loaded = False
    
    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """
        Search the BM25 index for relevant document IDs.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of (doc_index, score) tuples, sorted by score descending
        """
        if not self.is_loaded or self.index is None:
            logger.warning("BM25 index not loaded. Returning empty results.")
            return []
        
        try:
            # Tokenize query
            query_tokens = bm25s.tokenize([query], stopwords="en")
            
            # Retrieve
            results, scores = self.index.retrieve(query_tokens, k=top_k)
            
            # Convert to list of (index, score) tuples
            ranked = []
            for idx, score in zip(results[0], scores[0]):
                if score > 0:
                    ranked.append((int(idx), float(score)))
            
            return ranked
        
        except Exception as e:
            logger.error(f"BM25 search error: {e}")
            return []


# Singleton instance
bm25_index = BM25Index()
