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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "legal-rag")
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # Comma-separated or *

# ==================== AUTH CONFIG ====================
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-dev-key-change-in-prod")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ==================== OBSERVABILITY ====================
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# ==================== REDIS & CACHE CONFIG ====================
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TTL_EXACT = int(os.getenv("REDIS_TTL_EXACT", 259200))      # 3 days
REDIS_TTL_SEMANTIC = int(os.getenv("REDIS_TTL_SEMANTIC", 172800))  # 2 days
REDIS_TTL_HYDE = int(os.getenv("REDIS_TTL_HYDE", 604800))        # 7 days
INDEX_VERSION = os.getenv("INDEX_VERSION", "v1")                 # For safe cache invalidation
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX_REQUESTS", 10))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", 60))

# ==================== FEATURE FLAGS ====================
USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"
USE_SQLITE_METADATA = os.getenv("USE_SQLITE_METADATA", "false").lower() == "true"
USE_AGENTIC_PIPELINE = os.getenv("USE_AGENTIC_PIPELINE", "false").lower() == "true"

# ==================== MODEL CONFIG ====================
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMENSIONS = 1024
RERANKER_MODEL = "rerank-v3.0"
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.1

# ==================== RETRIEVAL CONFIG ====================
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60          # Reciprocal Rank Fusion constant
RERANK_TOP_N = 10    # Final top-N after reranking (increased for broader context)
CONTEXT_TOP_N = 8    # Chunks sent to LLM (increased to capture trailing punishments/exceptions)

# ==================== CHUNKING CONFIG ====================
MAX_CHUNK_SIZE = 1600       # ~400 tokens (approx 4 chars per token)
SUB_CHUNK_SIZE = 1600       # Fallback chunk size
CHUNK_OVERLAP = 200         # ~50 tokens overlap
MIN_CHUNK_SIZE = 50         # Discard tiny chunks

# ==================== CONFIDENCE THRESHOLDS ====================
HIGH_CONFIDENCE = 0.80
MEDIUM_CONFIDENCE = 0.55
LOW_CONFIDENCE = 0.35
VERY_LOW_CONFIDENCE = 0.20
AGENT_FALLBACK_THRESHOLD = float(os.getenv("AGENT_FALLBACK_THRESHOLD", 0.20))

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
3. **PRIORITIZE NEW LAWS (BNS/BNSS/BSA):** India transitioned to new criminal laws on July 1, 2024. ALWAYS apply and cite the Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA) over the repealed IPC, CrPC, or IEA. If the scenario occurs after July 1, 2024, you are STRICTLY FORBIDDEN from using the old Evidence Act, IPC, CrPC, or historical IT Act clauses for procedural validation. If the relevant BSA/BNS chunk is missing, state: "The required new active law is not present in the retrieved context."
4. If it IS a legal question, structure your answer using the IRAC framework but keep it of MODERATE LENGTH. However, DO NOT omit crucial parts of a section just to be concise:
   - **ISSUE:** Briefly state the legal question.
   - **RULE:** Extract exact laws, Sections, their FULL rigid conditions, AND any punishments, penalties, or exceptions mentioned.
   - **APPLICATION:** Briefly apply the rules to the actors. Ensure you mention requirements like 'communication to a third party' or 'cognizance by Sessions Court' if the law demands it.
   - **CONCLUSION:** A definitive legal outcome based purely on the text. Include the potential punishment if applicable.
5. If the context does not contain the answer, say "I cannot determine this from the available excerpts."
6. NEVER fabricate, guess, or hallucinate legal provisions, procedural links, or punishments. If a specific procedural section (like cognizance for public servants) is not in the context, do not guess it.
7. Use formal legal language.

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
