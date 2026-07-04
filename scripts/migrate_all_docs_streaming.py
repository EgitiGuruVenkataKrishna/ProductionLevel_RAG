"""
Robust Pinecone Migration Script
Processes documents one by one to avoid OOM.
Saves progress incrementally.
"""
import sys
import os
import json
import logging
import traceback
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

def load_single_document(file_path: Path) -> list[dict]:
    documents = []
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
            logger.info(f"Loaded {file_path.name} ({len(pages)} pages)")
        except Exception as e:
            logger.error(f"Failed to load {file_path.name}: {e}")
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
                logger.info(f"Loaded {file_path.name} (TXT)")
        except Exception as e:
            logger.error(f"Failed to load TXT {file_path.name}: {e}")
            
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
    
    docs_dir = Path(project_root) / "legal_docs"
    output_dir = Path(project_root) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = output_dir / "chunks_metadata.json"
    
    # Load existing metadata to resume
    all_chunks = []
    processed_files = set()
    
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                all_chunks = json.load(f)
                processed_files = set(c.get("source_file") for c in all_chunks)
            logger.info(f"Resuming... Already processed {len(processed_files)} files, {len(all_chunks)} chunks.")
        except Exception as e:
            logger.error("Failed to read chunks_metadata.json, starting fresh.")
            all_chunks = []
            
    # Initialize Embedding Model ONCE
    logger.info("Loading FastEmbed Model...")
    embedding_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", threads=4)
    from app.services.chunker import chunk_documents
    
    global_chunk_idx = len(all_chunks)
    
    for file_path in sorted(docs_dir.iterdir()):
        if file_path.name in processed_files:
            logger.info(f"Skipping already processed: {file_path.name}")
            continue
            
        logger.info(f"--- Processing {file_path.name} ---")
        documents = load_single_document(file_path)
        if not documents:
            continue
            
        file_chunks = chunk_documents(documents)
        logger.info(f"Created {len(file_chunks)} chunks for {file_path.name}.")
        
        if not file_chunks:
            processed_files.add(file_path.name)
            continue
            
        # Embed and Upsert
        texts = [c["text"] for c in file_chunks]
        embeddings_generator = embedding_model.embed(texts, batch_size=64)
        
        batch_size_upsert = 50
        vectors = []
        
        for local_i, emb in enumerate(embeddings_generator):
            chunk = file_chunks[local_i]
            
            chunk_id = f"doc_{global_chunk_idx}"
            chunk["id"] = chunk_id
            
            meta = {
                "source_file": chunk.get("source_file") or "",
                "act_name": chunk.get("act_name") or "",
                "section": chunk.get("section") or "",
                "article_number": chunk.get("article_number") or "",
            }
            vectors.append({"id": chunk_id, "values": emb.tolist(), "metadata": meta})
            
            global_chunk_idx += 1
            
            if len(vectors) >= batch_size_upsert:
                try:
                    index.upsert(vectors=vectors)
                except Exception as e:
                    logger.error(f"Pinecone upsert failed: {e}")
                vectors = []
                
        if vectors:
            try:
                index.upsert(vectors=vectors)
            except Exception as e:
                logger.error(f"Pinecone final upsert failed: {e}")
                
        logger.info(f"Upserted all vectors for {file_path.name}.")
        all_chunks.extend(file_chunks)
        
        # Save progress incrementally
        processed_files.add(file_path.name)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
            
    # 5. Build BM25
    logger.info("Building global BM25 Index...")
    from app.services.bm25_index import BM25Index
    bi = BM25Index()
    all_texts = [c["text"] for c in all_chunks]
    bi.build(all_texts, str(output_dir / "bm25_index"))
    
    logger.info("SUCCESS! Full Migration complete.")

if __name__ == "__main__":
    main()
