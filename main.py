import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

"""
RAG Document Q&A System
-----------------------
A production-ready Retrieval-Augmented Generation (RAG) system that enables 
question-answering over uploaded documents using FastAPI, ChromaDB, and LLMs.

Features:
- PDF and TXT file processing
- Semantic search with confidence scoring
- Edge case handling and validation
- REST API with automatic documentation

Author: Guru Venkata Krishna
Date: February 2026
Version: 1.0.0
"""

# Imports Required
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv(override=True)

# ==================== CONFIGURATION ====================
# File Upload Settings
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MIN_TEXT_LENGTH = 100  # Minimum characters needed

# ChromaDB Settings
CHROMA_PATH = "./chroma_data"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Chunking Settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Confidence Thresholds
HIGH_CONFIDENCE = 0.75
MEDIUM_CONFIDENCE = 0.5
LOW_CONFIDENCE = 0.3

# Paths
UPLOAD_DIR = Path("./uploaded_files")
UPLOAD_DIR.mkdir(exist_ok=True)
# ======================================================


# ==================== UTILITY FUNCTIONS ====================
def sanitize_collection_name(name: str) -> str:
    """
    Convert any string to valid ChromaDB collection name
    
    Rules:
    - Only [a-zA-Z0-9._-]
    - 3-512 characters
    - Must start/end with alphanumeric
    """
    # Replace invalid characters with underscore
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)
    
    # Remove leading/trailing non-alphanumeric
    name = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', name)
    
    # Ensure minimum length
    if len(name) < 3:
        name = f"collection_{name}"
    
    # Ensure maximum length
    if len(name) > 512:
        name = name[:512]
    
    return name.lower()


def get_confidence_level(similarity: float) -> tuple[str, str]:
    """
    Convert similarity score to confidence level and warning
    
    Args:
        similarity: Float between 0 and 1
        
    Returns:
        Tuple of (confidence_level, warning_message)
    """
    if similarity >= HIGH_CONFIDENCE:
        return "high", None
    elif similarity >= MEDIUM_CONFIDENCE:
        return "medium", "Moderate confidence - verify important details"
    elif similarity >= LOW_CONFIDENCE:
        return "low", "Low confidence - answer may not be accurate"
    else:
        return "very_low", "Very low confidence - likely no relevant information"
# ===========================================================


# ==================== PYDANTIC MODELS ====================
class QueryRequest(BaseModel):
    """Request model for asking questions"""
    question: str = Field(..., min_length=3, max_length=500, description="Question to ask")
    collection_name: str = Field(default="documents", description="Collection to search")
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")


class QueryResponse(BaseModel):
    """Response model for answers"""
    answer: str
    confidence: str  # "high", "medium", "low", "very_low"
    avg_similarity: float
    best_similarity: float
    source_documents: list
    warning: str = None
# ===========================================================


# ==================== FASTAPI INSTANCE ====================
app = FastAPI(
    title="Production RAG API",
    description="Upload PDFs/TXT and ask questions using retrieval-augmented generation. "
                "This is a production-level system with edge case handling.",
    version="1.0.0"
)

# Initialize ChromaDB and Embedding Model
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
logger.info(f"Initialized ChromaDB at {CHROMA_PATH}")
logger.info(f"Loaded embedding model: {EMBEDDING_MODEL}")
# ===========================================================


# ==================== ENDPOINTS ====================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(..., description="PDF or TXT file to upload"),
    collection_name: str = "documents"
):
    """
    Upload and process a document (PDF or TXT)
    
    Args:
        file: PDF or TXT file
        collection_name: Name of collection to store in
        
    Returns:
        Success message with processing stats
    """
    try:
        logger.info(f"Received upload request: {file.filename}")
        
        # ===== VALIDATION 1: File Extension =====
        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in ALLOWED_EXTENSIONS:
            logger.warning(f"Invalid file type: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed. Got: {file_extension}"
            )
        
        logger.info(f"✅ File type validation passed: {file_extension}")
        
        # ===== VALIDATION 2: File Size =====
        contents = await file.read()
        file_size = len(contents)
        
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"File too large: {file_size / (1024*1024):.1f}MB")
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.0f}MB. Your file: {file_size / (1024*1024):.1f}MB"
            )
        
        if file_size == 0:
            logger.warning("Empty file uploaded")
            raise HTTPException(
                status_code=400,
                detail="File is empty. Please upload a valid document."
            )
        
        logger.info(f"✅ File size validation passed: {file_size / 1024:.1f}KB")
        
        # Reset file pointer
        await file.seek(0)
        
        # ===== VALIDATION 3: Save Temporarily =====
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            buffer.write(contents)
        
        logger.info(f"Saved file to: {file_path}")
        
        # ===== VALIDATION 4: Text Extraction =====
        try:
            if file_extension == '.pdf':
                loader = PyPDFLoader(str(file_path))
                documents = loader.load()
            elif file_extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                documents = [Document(page_content=text, metadata={"source": file.filename, "page": 0})]
            
            logger.info(f"✅ Extracted from {len(documents)} page(s)")
            
        except Exception as e:
            file_path.unlink(missing_ok=True)
            logger.error(f"Text extraction failed: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract text from file. The file may be corrupted or password-protected. Error: {str(e)}"
            )
        
        # ===== VALIDATION 5: Check Text Content =====
        total_text = " ".join([doc.page_content for doc in documents])
        
        if len(total_text.strip()) < MIN_TEXT_LENGTH:
            file_path.unlink(missing_ok=True)
            logger.warning(f"Too little text extracted: {len(total_text)} chars")
            raise HTTPException(
                status_code=400,
                detail=f"Document contains too little text ({len(total_text)} characters). Minimum required: {MIN_TEXT_LENGTH} characters. The file may be image-based or corrupted."
            )
        
        logger.info(f"✅ Text extraction successful: {len(total_text)} characters")
        
        # ===== Chunking =====
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_documents(documents)
        
        if len(chunks) == 0:
            file_path.unlink(missing_ok=True)
            logger.error("No chunks created")
            raise HTTPException(
                status_code=400,
                detail="No text chunks could be created from this document."
            )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # ===== Generate Embeddings =====
        texts = [chunk.page_content for chunk in chunks]
        embeddings = embedding_model.encode(texts, normalize_embeddings=True)
        
        # Sanitize collection name
        collection_name = sanitize_collection_name(collection_name)
        
        # Get or create collection
        try:
            collection = chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            collection = chroma_client.create_collection(collection_name)
            logger.info(f"Created new collection: {collection_name}")
        
        # ===== Store in ChromaDB =====
        ids = [f"{file.filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": file.filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_id": i,
                "file_size": file_size,
                "total_chunks": len(chunks)
            }
            for i, chunk in enumerate(chunks)
        ]
        
        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"✅ Stored {len(chunks)} chunks in ChromaDB")
        
        # Optional: Clean up uploaded file
        # file_path.unlink()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Document processed successfully",
                "filename": file.filename,
                "file_size_kb": round(file_size / 1024, 2),
                "pages": len(documents),
                "chunks_created": len(chunks),
                "characters_extracted": len(total_text),
                "collection": collection_name,
                "total_documents_in_collection": collection.count()
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing document: {str(e)}"
        )


@app.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question about uploaded documents
    
    Args:
        request: QueryRequest with question and settings
        
    Returns:
        Answer with confidence scoring and sources
    """
    try:
        logger.info(f"Received query: '{request.question}' for collection: '{request.collection_name}'")
        
        # ===== VALIDATION 1: Collection Exists =====
        try:
            collection = chroma_client.get_collection(request.collection_name)
        except:
            logger.warning(f"Collection not found: {request.collection_name}")
            raise HTTPException(
                status_code=404,
                detail=f"Collection '{request.collection_name}' not found. Please upload documents first using /upload-pdf endpoint."
            )
        
        # ===== VALIDATION 2: Collection Has Documents =====
        if collection.count() == 0:
            logger.warning(f"Collection is empty: {request.collection_name}")
            raise HTTPException(
                status_code=400,
                detail=f"Collection '{request.collection_name}' is empty. Please upload documents first."
            )
        
        logger.info(f"Collection has {collection.count()} document chunks")
        
        # ===== Generate Query Embedding =====
        query_embedding = embedding_model.encode(
            request.question,
            normalize_embeddings=True
        )
        
        # ===== Retrieve Relevant Chunks =====
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=5,  # Get top 5 for better context
            include=["documents", "metadatas", "distances"]
        )
        
        # ===== VALIDATION 3: Check If Results Found =====
        if not results['documents'][0]:
            logger.warning("No relevant documents found")
            return QueryResponse(
                answer="I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your question or upload more relevant documents.",
                confidence="none",
                avg_similarity=0.0,
                best_similarity=0.0,
                source_documents=[],
                warning="No relevant documents found"
            )
        
        # ===== Calculate Similarities =====
        similarities = [1 - (dist / 2) for dist in results['distances'][0]]
        avg_similarity = sum(similarities) / len(similarities)
        best_similarity = max(similarities)
        
        logger.info(f"Similarities: {[f'{s:.3f}' for s in similarities]}")
        logger.info(f"Average: {avg_similarity:.3f}, Best: {best_similarity:.3f}")
        
        # ===== Get Confidence Level =====
        confidence_level, warning = get_confidence_level(avg_similarity)
        
        # ===== Check Confidence Threshold =====
        if avg_similarity < request.min_confidence:
            logger.warning(f"Similarity {avg_similarity:.3f} below threshold {request.min_confidence}")
            return QueryResponse(
                answer=f"I found some potentially relevant information, but the confidence is too low (similarity: {avg_similarity:.2f}). The answer might not be accurate. Please try a more specific question or check if the relevant documents are uploaded.",
                confidence="low",
                avg_similarity=round(avg_similarity, 3),
                best_similarity=round(best_similarity, 3),
                source_documents=[
                    {
                        "text": doc[:200] + "...",
                        "source": meta["source"],
                        "page": meta.get("page", "N/A"),
                        "similarity": round(sim, 3)
                    }
                    for doc, meta, sim in zip(
                        results['documents'][0][:2],
                        results['metadatas'][0][:2],
                        similarities[:2]
                    )
                ],
                warning=f"Low confidence answer (similarity: {avg_similarity:.2f})"
            )
        
        # ===== Build Context =====
        context_chunks = results['documents'][0][:3]  # Use top 3
        context = "\n\n---\n\n".join(context_chunks)
        
        # ===== Generate Answer with LLM =====
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.error("GROQ_API_KEY not found")
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not configured. Please add it to your .env file."
            )
        
        llm = ChatGroq(
            model='llama-3.1-8b-instant',
            temperature=0.1,
            api_key=api_key
        )
        
        prompt = f"""You are a helpful AI assistant. Answer the question based ONLY on the context provided below. 

IMPORTANT RULES:
1. If the context contains the answer, provide a clear and concise response
2. If the context does NOT contain enough information, say: "I don't have enough information in the provided documents to answer this question accurately."
3. Do NOT make up information or use knowledge outside the provided context
4. Cite which part of the context you used if possible

Context:
{context}

Question: {request.question}

Answer:"""
        
        try:
            answer = llm.invoke(prompt)
            answer_text = answer.content
            logger.info("✅ Answer generated successfully")
        except Exception as e:
            logger.error(f"LLM error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating answer from LLM: {str(e)}"
            )
        
        # ===== Format Source Documents =====
        source_docs = [
            {
                "text": doc[:300] + ("..." if len(doc) > 300 else ""),
                "source": meta["source"],
                "page": meta.get("page", "N/A"),
                "similarity": round(sim, 3),
                "chunk_id": meta.get("chunk_id", "N/A")
            }
            for doc, meta, sim in zip(
                results['documents'][0][:3],
                results['metadatas'][0][:3],
                similarities[:3]
            )
        ]
        
        return QueryResponse(
            answer=answer_text,
            confidence=confidence_level,
            avg_similarity=round(avg_similarity, 3),
            best_similarity=round(best_similarity, 3),
            source_documents=source_docs,
            warning=warning
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/collections")
async def list_collections():
    """List all available collections with document counts"""
    try:
        collections = chroma_client.list_collections()
        return {
            "total_collections": len(collections),
            "collections": [
                {
                    "name": col.name,
                    "document_count": col.count()
                }
                for col in collections
            ]
        }
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/collection/{collection_name}")
async def delete_collection(collection_name: str):
    """Delete a collection and all its documents"""
    try:
        chroma_client.delete_collection(collection_name)
        logger.info(f"Deleted collection: {collection_name}")
        return {"message": f"Collection '{collection_name}' deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Collection not found or could not be deleted: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """
    Health check endpoint with system information
    
    Returns:
        System status and configuration
    """
    try:
        collections = chroma_client.list_collections()
        
        return {
            "status": "healthy",
            "chroma_path": CHROMA_PATH,
            "total_collections": len(collections),
            "collections": [
                {
                    "name": col.name,
                    "document_count": col.count()
                }
                for col in collections
            ],
            "config": {
                "max_file_size_mb": MAX_FILE_SIZE / (1024 * 1024),
                "allowed_extensions": ALLOWED_EXTENSIONS,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# ==================== RUN ====================
# Should be:
PORT = int(os.getenv("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)  # ✅ Uses PORT variable