# 📚 Production RAG System — Document Q&A API

> Retrieval-Augmented Generation (RAG) system for accurate, source-grounded answers over custom documents

---

## 🚀 Why This Project Matters

LLMs alone hallucinate. Real-world systems require **retrieval + grounding**.

This project implements a **production-style RAG pipeline** that:

* Retrieves relevant document chunks using semantic search
* Generates answers grounded in source data
* Provides **confidence scoring and citations**

👉 Built as an **AI backend system**, not just a demo chatbot.

---

## 🧠 Core Capabilities

### 1. Retrieval Pipeline

* Document ingestion (PDF / TXT)
* Chunking with overlap strategy
* Embedding-based semantic search
* Similarity scoring for relevance

---

### 2. Grounded Answer Generation

* Context-aware LLM responses
* Source attribution (file + page + chunk)
* Confidence scoring based on similarity

---

### 3. API-First Design

* RESTful endpoints for ingestion & querying
* Structured JSON responses
* Validation using typed schemas

---

## 🏗️ System Architecture

```text
User Query
    ↓
Embedding Generation
    ↓
Vector Search (Top-K Retrieval)
    ↓
Context Assembly
    ↓
LLM Generation (Grounded Response)
    ↓
Confidence Scoring + Sources
    ↓
API Response
```

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Vector Store:** ChromaDB
* **Embeddings:** Sentence Transformers
* **LLM:** Groq (Llama 3.1)
* **Processing:** LangChain
* **Parsing:** PyPDF

---

## 📡 API Overview

### `POST /upload-pdf`

* Ingest and process documents
* Returns chunk count and metadata

### `POST /ask`

* Query documents using semantic search
* Returns:

  * Answer
  * Confidence level
  * Source chunks

### `GET /health`

* System status check

---

## ⚡ Engineering Highlights

* Implemented **semantic retrieval pipeline** with similarity scoring
* Designed **confidence evaluation layer** for answer reliability
* Built **modular architecture** for ingestion and querying
* Added **validation and error handling** across endpoints
* Optimized for **low-latency responses (<2s)**

---

## 📊 Current Limitations

* Single-user system (no isolation)
* In-memory / local vector storage
* No hybrid search (keyword + vector)
* No reranking layer

---

## 📈 Planned Improvements

* Multi-user support with isolated collections
* Hybrid retrieval (BM25 + embeddings)
* Reranking for improved relevance
* Deployment with Docker + cloud
* Minimal frontend interface

---

## 👨‍💻 Author

**Guru Venkata Krishna**
Applied AI Engineer (in progress)

* GitHub: https://github.com/EgitiGuruVenkataKrishna
* LinkedIn: (https://www.linkedin.com/in/guru-venkata-krishna-egiti-46070a303/)

---

## ⭐ If you find this useful, consider starring the repo.
