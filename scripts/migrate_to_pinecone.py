"""
Pinecone Migration Script

Reads all PDFs in `legal_docs`, chunks them with 400 tokens / 50 overlap,
embeds them locally using fastembed (BAAI/bge-large-en-v1.5), and uploads 
them to the Pinecone index (creating it if necessary). Also rebuilds BM25.
"""
import sys
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

from pinecone import Pinecone, ServerlessSpec
from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from fastembed import TextEmbedding

def main():
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not set.")
        return
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME or "legal-rag"
    
    # 1. Check or Create Pinecone Index
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        logger.info(f"Index '{index_name}' not found. Creating it (dim 1024, cosine)...")
        try:
            pc.create_index(
                name=index_name,
                dimension=1024, # BAAI/bge-large-en-v1.5
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
            logger.info("Index created successfully.")
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
            logger.error("Since you are on the free plan, you can only have 1 index. If you already have an index, delete it in the Pinecone dashboard first, or set PINECONE_INDEX_NAME to match it.")
            return
    else:
        logger.info(f"Index '{index_name}' already exists.")
        
    index = pc.Index(index_name)
    
    # 2. Load Documents
    docs_dir = Path(project_root) / "legal_docs"
    from scripts.build_index import load_documents
    documents = load_documents(str(docs_dir))
    if not documents:
        return
        
    # 3. Chunk Documents (400 / 50)
    logger.info("Chunking documents...")
    from app.services.chunker import chunk_documents
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    
    # 4. Embed using fastembed and Upsert streaming
    logger.info("Embedding chunks locally and upserting in batches...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", threads=4)
    texts = [c["text"] for c in chunks]
    embeddings_generator = embedding_model.embed(texts, batch_size=256)
    
    batch_size_upsert = 100
    vectors = []
    
    for i, emb in enumerate(embeddings_generator):
        chunk = chunks[i]
        chunk_id = str(i)
        chunk["id"] = i
        
        meta = {
            "source_file": chunk.get("source_file", ""),
            "act_name": chunk.get("act_name", ""),
            "section": chunk.get("section", ""),
            "article_number": chunk.get("article_number", ""),
        }
        vectors.append({"id": chunk_id, "values": emb.tolist(), "metadata": meta})
        
        if len(vectors) >= batch_size_upsert:
            index.upsert(vectors=vectors)
            logger.info(f"Upserted up to chunk {i}")
            vectors = []
            
    if vectors:
        index.upsert(vectors=vectors)
        logger.info(f"Upserted final batch up to chunk {len(chunks)-1}")
        
    # 6. Build BM25 and Save chunks_metadata
    output_dir = Path(project_root) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Building BM25 index...")
    from app.services.bm25_index import BM25Index
    bi = BM25Index()
    bi.build(texts, str(output_dir / "bm25_index"))
    
    logger.info("Saving chunk metadata...")
    meta_path = output_dir / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    logger.info("SUCCESS! Migration complete.")

if __name__ == "__main__":
    main()
