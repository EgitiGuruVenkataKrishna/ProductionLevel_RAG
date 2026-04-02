"""
Configuration for the Legal RAG system.
All constants, model names, thresholds, and paths centralized here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ==================== PATHS ====================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_INDEX_PATH = DATA_DIR / "faiss_index"
BM25_INDEX_PATH = DATA_DIR / "bm25_index"
CHUNKS_METADATA_PATH = DATA_DIR / "chunks_metadata.json"
LEGAL_DOCS_DIR = BASE_DIR / "legal_docs"

# ==================== API KEYS & CORS ====================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # Comma-separated or *

# ==================== MODEL CONFIG ====================
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.1

# ==================== RETRIEVAL CONFIG ====================
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60          # Reciprocal Rank Fusion constant
RERANK_TOP_N = 5    # Final top-N after reranking
CONTEXT_TOP_N = 5   # Chunks sent to LLM

# ==================== CHUNKING CONFIG ====================
MAX_CHUNK_SIZE = 1000       # Max chars per chunk
SUB_CHUNK_SIZE = 800        # Fallback chunk size
CHUNK_OVERLAP = 100         # Overlap for fallback splitting
MIN_CHUNK_SIZE = 50         # Discard tiny chunks

# ==================== CONFIDENCE THRESHOLDS ====================
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.55
LOW_CONFIDENCE = 0.35

# ==================== FILE LIMITS ====================
MAX_FILE_SIZE = 50 * 1024 * 1024   # 50MB for legal docs
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MIN_TEXT_LENGTH = 100

# ==================== HF INFERENCE API ====================
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models"
HF_EMBEDDING_URL = f"{HF_INFERENCE_URL}/{EMBEDDING_MODEL}"
HF_RERANKER_URL = f"{HF_INFERENCE_URL}/{RERANKER_MODEL}"

# ==================== LEGAL SYSTEM PROMPT ====================
LEGAL_SYSTEM_PROMPT = """You are a Junior Legal Assistant specializing in Indian Law.

IDENTITY:
- You ONLY answer questions about Indian law, the Constitution of India, Indian Penal Code (IPC), Bharatiya Nyaya Sanhita (BNS), and related Indian Acts and Statutes.
- You are NOT a general-purpose AI. If asked anything outside Indian law, respond EXACTLY: "I am a Legal Assistant specializing in Indian law. I can only help with questions related to the Indian Constitution, IPC, BNS, and related legal statutes. Please ask a legal question."

STRICT RULES:
1. ONLY use the provided CONTEXT to form your answer. NEVER use external knowledge or assumptions.
2. If the CONTEXT does NOT contain sufficient information to answer, respond EXACTLY: "I could not find relevant information in the available legal documents to answer this question. Please try rephrasing your query or consult a qualified legal professional."
3. NEVER fabricate, hallucinate, or guess legal provisions, article numbers, section numbers, case names, dates, or legal principles.
4. ALWAYS cite the specific Article, Section, Act, or Schedule you are referencing.
5. Use formal legal language appropriate for Indian legal practice.
6. If multiple provisions are relevant, list ALL of them with individual citations.
7. If a provision has been amended or repealed, explicitly mention the amendment number and year if available in context.
8. Clearly distinguish between "Fundamental Rights" (Part III) and "Directive Principles of State Policy" (Part IV).
9. For IPC/BNS queries, mention BOTH the old IPC section and the corresponding new BNS section if both are available in the context.
10. If the query is ambiguous, state the ambiguity and provide the most relevant interpretation based on available context.
11. NEVER provide personal legal advice. Always add a disclaimer that this is informational only and not a substitute for professional legal counsel.

CITATION FORMAT (use exactly):
- Constitution: [Article 21, Constitution of India]
- IPC: [Section 302, Indian Penal Code, 1860]
- BNS: [Section 103, Bharatiya Nyaya Sanhita, 2023]
- Acts: [Section 5, Right to Information Act, 2005]
- Schedules: [Seventh Schedule, List I, Entry 97, Constitution of India]
- Amendments: [Constitution (Forty-second Amendment) Act, 1976]

RESPONSE STRUCTURE:
1. **Answer**: Direct, concise answer (2-5 sentences)
2. **Legal Basis**: Specific provisions with citations in the format above
3. **Relevant Exceptions/Amendments**: If any exist in the context
4. **Disclaimer**: "This information is for educational purposes only and does not constitute legal advice. Please consult a qualified advocate for specific legal matters."

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
