"""
Offline Index Builder for Legal RAG.

Processes legal documents and generates:
1. FAISS vector index (dense embeddings)
2. BM25 sparse index (keyword search)
3. Chunk metadata JSON (text + legal metadata)

Usage:
    python scripts/build_index.py --docs ./legal_docs/ --output ./data/

This script requires sentence-transformers (heavy dependency).
It runs locally ONLY — not deployed to Vercel.
"""
import sys
import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_documents(docs_dir: str) -> list[dict]:
    """
    Load all PDF and TXT files from a directory.
    
    Returns list of dicts: {text, source_file, page}
    """
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {docs_path}")
        sys.exit(1)
    
    documents = []
    
    for file_path in sorted(docs_path.iterdir()):
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            try:
                from langchain_community.document_loaders import PyPDFLoader
                loader = PyPDFLoader(str(file_path))
                pages = loader.load()
                for page in pages:
                    documents.append({
                        "text": page.page_content,
                        "source_file": file_path.name,
                        "page": page.metadata.get("page", 0)
                    })
                logger.info(f"📄 Loaded PDF: {file_path.name} ({len(pages)} pages)")
            except Exception as e:
                logger.error(f"❌ Failed to load PDF {file_path.name}: {e}")
        
        elif ext == '.txt':
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                if text.strip():
                    documents.append({
                        "text": text,
                        "source_file": file_path.name,
                        "page": 0
                    })
                    logger.info(f"📄 Loaded TXT: {file_path.name} ({len(text)} chars)")
            except Exception as e:
                logger.error(f"❌ Failed to load TXT {file_path.name}: {e}")
    
    logger.info(f"\n✅ Total documents loaded: {len(documents)}")
    return documents


def main():
    parser = argparse.ArgumentParser(description="Build indices for Legal RAG")
    parser.add_argument("--docs", type=str, default="./legal_docs/",
                        help="Directory containing legal PDF/TXT documents")
    parser.add_argument("--output", type=str, default="./data/",
                        help="Output directory for indices")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ---- Step 1: Load Documents ----
    logger.info("=" * 60)
    logger.info("STEP 1: Loading documents")
    logger.info("=" * 60)
    documents = load_documents(args.docs)
    
    if not documents:
        logger.error("No documents found. Add PDF/TXT files to the docs directory.")
        sys.exit(1)
    
    # ---- Step 2: Hierarchical Chunking ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Hierarchical chunking")
    logger.info("=" * 60)
    
    from app.services.chunker import chunk_documents
    chunks = chunk_documents(documents)
    logger.info(f"✅ Created {len(chunks)} chunks with legal metadata")
    
    # Show sample metadata
    for chunk in chunks[:3]:
        meta_parts = []
        if chunk.get("article_number"): meta_parts.append(chunk["article_number"])
        if chunk.get("section"): meta_parts.append(chunk["section"])
        if chunk.get("act_name"): meta_parts.append(chunk["act_name"])
        logger.info(f"  Sample chunk: {' | '.join(meta_parts) or 'No legal metadata'} "
                    f"({len(chunk['text'])} chars)")
    
    # ---- Step 3: Generate Embeddings ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Generating embeddings")
    logger.info("=" * 60)
    
    from sentence_transformers import SentenceTransformer
    from app.config import EMBEDDING_MODEL
    
    # Use the model name without the org prefix for sentence-transformers
    model_name = EMBEDDING_MODEL.split("/")[-1]  # "all-MiniLM-L6-v2"
    embed_model = SentenceTransformer(model_name)
    
    texts = [c["text"] for c in chunks]
    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64
    )
    embeddings = np.array(embeddings, dtype=np.float32)
    logger.info(f"✅ Embeddings shape: {embeddings.shape}")
    
    # ---- Step 4: Build FAISS Index ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Building FAISS index")
    logger.info("=" * 60)
    
    from app.services.vector_index import VectorIndex
    
    faiss_path = str(output_dir / "faiss_index")
    vi = VectorIndex()
    vi.build(embeddings, faiss_path)
    logger.info(f"✅ FAISS index saved to {faiss_path}")
    
    # ---- Step 5: Build BM25 Index ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Building BM25 index")
    logger.info("=" * 60)
    
    from app.services.bm25_index import BM25Index
    
    bm25_path = str(output_dir / "bm25_index")
    bi = BM25Index()
    bi.build(texts, bm25_path)
    logger.info(f"✅ BM25 index saved to {bm25_path}")
    
    # ---- Step 6: Save Metadata ----
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Saving chunk metadata")
    logger.info("=" * 60)
    
    meta_path = output_dir / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"✅ Metadata saved to {meta_path}")
    
    # ---- Summary ----
    logger.info("\n" + "=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Documents processed: {len(documents)}")
    logger.info(f"  Chunks created:      {len(chunks)}")
    logger.info(f"  Embedding dims:      {embeddings.shape[1]}")
    logger.info(f"  Output directory:    {output_dir}")
    logger.info(f"\n  Files generated:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            logger.info(f"    {f.relative_to(output_dir)} ({size_kb:.1f} KB)")
    
    logger.info(f"\n🚀 Ready! Deploy with: vercel --prod")


if __name__ == "__main__":
    main()
