# ⚖️ Legal RAG Assistant — Indian Law Q&A

> Production-grade Retrieval-Augmented Generation system for Indian legal questions — Constitution, IPC, BNS, RTI Act and more.

![Legal RAG](https://img.shields.io/badge/Version-2.1.0-gold) ![Python](https://img.shields.io/badge/Python-3.10+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green) ![Deploy](https://img.shields.io/badge/Deploy-Vercel-black)

---

## 🚀 Why This Project Matters

LLMs alone hallucinate legal provisions. A wrong section number or fabricated article can have **real legal consequences**.

This system implements a **production-grade 8-step RAG pipeline** that:

- Retrieves relevant legal provisions using **hybrid search (BM25 + FAISS)**
- Reranks results using a **cross-encoder model**
- Generates answers grounded in **specific Articles, Sections, and Acts**
- Verifies every claim with an **answer grounding checker**
- Provides **real confidence scoring and citations**

👉 Built as a **serious legal AI backend**, not a demo chatbot.

## 📊 Current Limitations

* Single-user system (no isolation)
* Data indices must be built offline via script
* Rate limited to 10 requests / minute per IP to prevent Abuse

---

## 📈 Planned Improvements

* Multi-user support with isolated collections
* Deployment with Docker + cloud

---

## 🧠 8-Step Pipeline Architecture

```text
User Question
    ↓
Step 1: Query Expansion (multi-query via Groq LLM)
    ↓
Step 2: Multi-Query Hybrid Retrieval (BM25 + FAISS per query)
    ↓
Step 3: Reciprocal Rank Fusion (merge all results)
    ↓
Step 4: Cross-Encoder Reranking (HuggingFace Inference API)
    ↓
Step 5: Context Filtering & Deduplication
    ↓
Step 6: LLM Answer Generation (strict legal prompt)
    ↓
Step 7: Answer Grounding Check (faithfulness verification)
    ↓
Step 8: Real Confidence Scoring → Final Response with Citations
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Vector Store:** FAISS (Dense Embeddings) + BM25S (Keyword Sparse Index)
* **Embeddings:** HuggingFace Serverless (sentence-transformers/all-MiniLM-L6-v2)
* **Reranker:** HuggingFace Serverless (cross-encoder/ms-marco-MiniLM-L-6-v2)
* **LLM:** Groq (Llama-3.1-8b)
* **Processing:** Custom Hierarchical Legal Chunker

---

## 📡 API Endpoints

### `POST /api/ask`
Ask a legal question — runs the full 8-step pipeline.

**Request:**
```json
{
  "question": "What are fundamental rights under the Indian Constitution?",
  "search_mode": "hybrid",
  "min_confidence": 0.35
}
```

**Response includes:** answer, confidence score, citations with Article/Section numbers, grounding metrics, and warnings.

### `GET /api/health`
System health check — index status, model info, chunk count.

---

## 📂 Project Structure

```
├── main.py                  # Local dev server (FastAPI + frontend)
├── api/
│   ├── ask.py               # Vercel serverless: /api/ask
│   └── health.py            # Vercel serverless: /api/health
├── app/
│   ├── config.py            # All configuration constants
│   ├── models.py            # Pydantic request/response models
│   └── services/
│       ├── query_expander.py    # Step 1: Multi-query expansion
│       ├── hybrid_retriever.py  # Steps 2-3: BM25 + FAISS + RRF
│       ├── reranker.py          # Step 4: Cross-encoder reranking
│       ├── context_filter.py    # Step 5: Dedup + sanitize
│       ├── generator.py         # Step 6: LLM generation
│       ├── grounding_checker.py # Step 7: Answer verification
│       ├── bm25_index.py        # BM25 sparse index
│       ├── vector_index.py      # FAISS dense index
│       └── chunker.py           # Legal-aware document chunker
├── frontend/                # Chat UI (HTML/CSS/JS)
├── scripts/
│   └── build_index.py       # Offline index builder
├── data/                    # Pre-built indices + metadata
├── legal_docs/              # Source PDFs (IPC, BNS, RTI Act)
└── vercel.json              # Vercel deployment config
```

---

## 🏃 Quick Start

### Local Development
```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Set your API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# 3. Build indices (first time only)
pip install sentence-transformers langchain-community pypdf
python scripts/build_index.py --docs ./legal_docs/ --output ./data/

# 4. Run
python main.py
# Open http://localhost:8000
```

### Vercel Deployment
```bash
vercel --prod
```

### Docker
```bash
docker build -t legal-rag .
docker run -p 8000:8000 --env-file .env legal-rag
```

---

## 📜 Legal Documents Indexed

| Document | Coverage |
|----------|----------|
| **Bharatiya Nyaya Sanhita (BNS), 2023** | Full text — new criminal code |
| **Indian Penal Code (IPC)** | All sections |
| **RTI Act, 2005** | Amended version |

---

## ⚡ Engineering Highlights

- **Hybrid Search**: BM25 keyword + FAISS semantic, merged via Reciprocal Rank Fusion
- **Legal-Aware Chunking**: Splits at Article/Section boundaries, attaches legal metadata
- **Answer Grounding**: LLM verifies its own answer against source context (faithfulness, relevance, coverage)
- **Strict Legal Persona**: System prompt prevents hallucination and enforces citation format
- **Graceful Degradation**: Falls back to keyword-only if embedding API fails
- **Rate Limiting**: Protects external API quotas from abuse

---

## 👨‍💻 Author

**Guru Venkata Krishna**
Applied AI Engineer

- GitHub: [EgitiGuruVenkataKrishna](https://github.com/EgitiGuruVenkataKrishna)
- LinkedIn: [Guru Venkata Krishna Egiti](https://www.linkedin.com/in/guru-venkata-krishna-egiti-46070a303/)

---

## ⭐ If you find this useful, consider starring the repo.
