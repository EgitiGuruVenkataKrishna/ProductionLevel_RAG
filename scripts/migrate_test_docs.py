"""
Pinecone Migration Script - TEST MODE
Only processes a small subset of documents for quick testing.
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

def load_test_documents(docs_dir: str) -> list[dict]:
    docs_path = Path(docs_dir)
    documents = []
    
    # ONLY load the Contract Act for our test
    target_files = ["Indian contract act.pdf", "Indian contract act -26.pdf"]
    
    for file_path in sorted(docs_path.iterdir()):
        if file_path.name in target_files:
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
                logger.info(f"Loaded {file_path.name} ({len(pages)} pages)")
            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                
    return documents

def main():
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not set.")
        return
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME or "legal-rag"
    
    # 1. Check Pinecone Index
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if index_name not in existing_indexes:
        pc.create_index(name=index_name, dimension=1024, metric="cosine", spec=ServerlessSpec(cloud="aws", region="us-east-1"))
        logger.info("Index created.")
        
    index = pc.Index(index_name)
    
    # Clear index for testing (Optional, but let's just upsert new vectors)
    # 2. Load Documents
    docs_dir = Path(project_root) / "legal_docs"
    documents = load_test_documents(str(docs_dir))
    if not documents:
        logger.error("No test documents found.")
        return
        
    # 3. Chunk Documents
    from app.services.chunker import chunk_documents
    chunks = chunk_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    
    # 4. Embed and Upsert
    logger.info("Embedding chunks locally and upserting in batches...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", threads=4)
    texts = [c["text"] for c in chunks]
    embeddings_generator = embedding_model.embed(texts, batch_size=256)
    
    batch_size_upsert = 100
    vectors = []
    
    for i, emb in enumerate(embeddings_generator):
        chunk = chunks[i]
        chunk_id = f"test_{i}" # use a prefix so we don't conflict with real ingest
        chunk["id"] = chunk_id
        
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
        
    # 5. Build BM25
    output_dir = Path(project_root) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    from app.services.bm25_index import BM25Index
    bi = BM25Index()
    bi.build(texts, str(output_dir / "bm25_index"))
    
    meta_path = output_dir / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
        
    logger.info("SUCCESS! TEST Migration complete.")

if __name__ == "__main__":
    main()
