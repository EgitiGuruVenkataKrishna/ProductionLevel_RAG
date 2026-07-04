import sys
import json
import logging
from pathlib import Path
import numpy as np

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from langchain_community.document_loaders import PyPDFLoader
from app.services.chunker import chunk_documents
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL
from app.services.vector_index import VectorIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
CHECKPOINT_SIZE = 500
DATA_DIR = Path(project_root) / "data"

def check_ocr_required(pages, filename):
    """
    Check if the PDF is likely an image-based scan.
    Returns True if average characters per page is below threshold.
    """
    if not pages:
        return False
    
    total_chars = sum(len(page.page_content.strip()) for page in pages)
    avg_chars = total_chars / len(pages)
    
    logger.info(f"[OCR Check] File: {filename} | Pages: {len(pages)} | Avg chars/page: {avg_chars:.1f}")
    
    if avg_chars < 50:
        logger.warning(f"⚠️  [OCR REQUIRED] '{filename}' appears to be an image-based scan.")
        logger.warning("Standard PyMuPDF/pypdf extraction failed to find text. OCR pipeline must be triggered.")
        return True
    
    logger.info("✅ [OCR Check Passed] Document contains extractable text layers.")
    return False

def ingest_file(file_path_str: str):
    file_path = Path(file_path_str)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return
    
    logger.info("\n========================================")
    logger.info(f"PROCESSING FILE: {file_path.name}")
    logger.info("========================================")
    
    # 1. Load and OCR Check
    documents = []
    if file_path.suffix.lower() == '.pdf':
        try:
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            is_image_based = check_ocr_required(pages, file_path.name)
            if is_image_based:
                logger.error("🛑 HALTING: Tesseract/OCR pipeline is required but not configured locally. Skipping image-based PDF.")
                return
            
            for page in pages:
                documents.append({
                    "text": page.page_content,
                    "source_file": file_path.name,
                    "page": page.metadata.get("page", 0)
                })
        except Exception as e:
            logger.error(f"❌ Failed to load PDF {file_path.name}: {e}")
            return
    elif file_path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        documents.append({"text": text, "source_file": file_path.name, "page": 0})
    
    # 2. Refined Chunking (400 tokens / 50 overlap approx)
    logger.info("\n[Chunking] Splitting text with strict 400 token limit...")
    chunks = chunk_documents(documents)
    total_chunks = len(chunks)
    logger.info(f"✅ Created {total_chunks} legal-aware chunks.")
    
    # 3. Batching & Checkpointing
    logger.info(f"\n[Vectorization] Processing in batches of {CHECKPOINT_SIZE} to prevent OOM.")
    
    model_name = EMBEDDING_MODEL.split("/")[-1]
    embed_model = SentenceTransformer(model_name)
    
    all_embeddings = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for i in range(0, total_chunks, CHECKPOINT_SIZE):
        batch_chunks = chunks[i:i+CHECKPOINT_SIZE]
        batch_texts = [c["text"] for c in batch_chunks]
        
        logger.info(f"  -> Encoding batch {i} to {min(i+CHECKPOINT_SIZE, total_chunks)}...")
        
        # Encode batch
        batch_emb = embed_model.encode(
            batch_texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32
        )
        all_embeddings.append(batch_emb)
        
        # Save checkpoint
        checkpoint_path = DATA_DIR / f"checkpoint_{file_path.stem}_batch_{i}.npz"
        np.savez_compressed(checkpoint_path, embeddings=batch_emb)
        logger.info(f"  💾 Checkpoint saved: {checkpoint_path.name}")
    
    # Combine and save final index (Simplified for this script)
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
        
        # Update FAISS
        faiss_path = str(DATA_DIR / "faiss_index")
        vi = VectorIndex()
        # Note: In a real incremental update, we'd load existing and add.
        # Here we build a new one just for this file for demonstration, 
        # or append if supported.
        vi.build(final_embeddings, faiss_path + f"_{file_path.stem}")
        
        # Save Metadata
        meta_path = DATA_DIR / f"chunks_metadata_{file_path.stem}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
            
        logger.info("\n✅ SUCCESS: File fully processed and checkpoints finalized.")
    else:
        logger.warning("No embeddings generated.")

if __name__ == "__main__":
    # Process the first new file
    target_file = r"e:\RAG_Production level\legal_docs\learning the law-26.pdf"
    ingest_file(target_file)
