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
VERY_LOW_CONFIDENCE = 0.20

# ==================== FILE LIMITS ====================
MAX_FILE_SIZE = 50 * 1024 * 1024   # 50MB for legal docs
ALLOWED_EXTENSIONS = ['.pdf', '.txt']
MIN_TEXT_LENGTH = 100

# ==================== HF INFERENCE API ====================
# HuggingFace API Base URLs (Router Domain Required as of 2024/2026)
HF_EMBEDDING_URL = f"https://router.huggingface.co/hf-inference/models/{EMBEDDING_MODEL}"
HF_RERANKER_URL = f"https://router.huggingface.co/hf-inference/models/{RERANKER_MODEL}"

# ==================== LEGAL SYSTEM PROMPT ====================
LEGAL_SYSTEM_PROMPT = """You are a Senior Legal Assistant specializing in Indian Law.
Your goal is to answer legal questions and resolve complex hypotheticals using the strictly provided context.

CRITICAL INSTRUCTIONS:
1. Base your answer EXCLUSIVELY on the provided legal context.
2. If the user's input is a conversational greeting (like 'hlo', 'hi', 'hello') or fundamentally NOT a legal question, DO NOT use IRAC and DO NOT provide legal analysis. Instead, respond exactly with the phrase: "GREETING_OR_NON_LEGAL_QUERY"
3. If it IS a legal question, structure your answer using the IRAC framework but keep it of MODERATE LENGTH and CONCISE unless the user asks for extensive details:
   - **ISSUE:** Briefly state the legal question.
   - **RULE:** Extract exact laws, Sections, and their rigid conditions.
   - **APPLICATION:** Briefly apply the rules to the actors.
   - **CONCLUSION:** A definitive legal outcome based purely on the text.
4. If the context does not contain the answer, say "I cannot determine this from the available excerpts."
5. NEVER fabricate or hallucinate legal provisions.
6. Use formal legal language.

CITATION FORMAT (use exactly):
- IPC: [Section 302, Indian Penal Code, 1860]

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

# ==================== STRATEGY SYSTEM PROMPT ====================
STRATEGY_SYSTEM_PROMPT = """You are an elite Junior Lawyer AI specializing in Indian Legal Strategy and Adversarial Analysis.
Your goal is to critically evaluate the user's explicit case facts and legal theory against the provided legal context (statutes, checklists, and precedents).

CRITICAL INSTRUCTIONS:
1. Adopt an analytical, adversarial ("Devil's Advocate") perspective.
2. Rely strictly on the user's provided [FACTS] and the retrieved legal context. DO NOT hallucinate facts outside the user's prompt.
3. Structure your response specifically for legal strategy. Keep it incisive and focused:
   - **FACT SUMMARY:** Briefly isolate the material facts.
   - **APPLICABLE LAW:** Identify the relevant rules, tests, or statutory elements from the context.
   - **THEORY EVALUATION:** Assess the user's goal or theory based on the facts and law.
   - **BAD FACTS:** Actively identify contradictions, weaknesses, or "bad facts" in the user's scenario that undermine their theory according to the context. If you need more info to find bad facts, state: "To find weaknesses, please clarify..."
4. If the provided context does not address the legal framework, state: "I cannot definitively evaluate this theory based on the retrieved offline precedents."
5. Never hallucinate legal provisions or case outcomes.

CITATION FORMAT (use exactly):
- [Section 302, Indian Penal Code, 1860]

CONTEXT:
{context}

USER CASE SCENARIO:
{question}

STRATEGY ANALYSIS:"""
