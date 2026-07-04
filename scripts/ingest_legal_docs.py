"""
High-Performance Legal Document Ingestion Pipeline
===================================================

Processes legal documents from `legal_docs/` and ingests them into Pinecone
with optimized chunking for reliable LLM answers.

Key features for SPEED:
  - Parallel PDF parsing via ProcessPoolExecutor
  - FastEmbed ONNX with max threads, batch_size=256
  - Streaming embed → upsert (no full-corpus RAM hold)
  - Large Pinecone upsert batches (100 vectors/call)

Key features for LLM ACCURACY:
  - Contextual chunk enrichment: prepends [Act | Section | Article] header
    to each chunk text BEFORE embedding. This means:
      * The vector captures the legal source in its embedding space
      * When retrieved, the LLM immediately knows which law a passage is from
      * Citation accuracy in generated answers improves dramatically
  - Category tagging for Pinecone metadata filtering
  - Legal-aware hierarchical chunking (splits at Section/Article boundaries)

Usage:
    # Fresh re-index (recommended when adding new docs):
    python scripts/ingest_legal_docs.py --fresh

    # Incremental (skip already-processed files):
    python scripts/ingest_legal_docs.py

    # Custom settings:
    python scripts/ingest_legal_docs.py --fresh --embed-batch 128 --threads 8
"""
import sys
import os
import json
import time
import logging
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Project root setup ──
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# PDF/TXT LOADING — Top-level functions for ProcessPoolExecutor (Windows)
# ═══════════════════════════════════════════════════════════════════════

def _load_single_pdf(file_path_str: str) -> list[dict]:
    """
    Load a single PDF file. Designed to run in a subprocess.
    Must be top-level for Windows multiprocessing (spawn start method).
    """
    file_path = Path(file_path_str)
    documents = []
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        for page in pages:
            text = page.page_content
            if text and len(text.strip()) > 20:
                documents.append({
                    "text": text,
                    "source_file": file_path.name,
                    "page": page.metadata.get("page", 0)
                })
    except Exception as e:
        # Print instead of logger in subprocess
        print(f"[ERROR] Failed to load PDF {file_path.name}: {e}")
    return documents


def _load_single_txt(file_path_str: str) -> list[dict]:
    """Load a single TXT file."""
    file_path = Path(file_path_str)
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        if text.strip() and len(text.strip()) > 20:
            documents.append({
                "text": text,
                "source_file": file_path.name,
                "page": 0
            })
    except Exception as e:
        print(f"[ERROR] Failed to load TXT {file_path.name}: {e}")
    return documents


# ═══════════════════════════════════════════════════════════════════════
# CONTEXTUAL ENRICHMENT — Improves both retrieval AND LLM answer quality
# ═══════════════════════════════════════════════════════════════════════

def create_context_header(chunk: dict) -> str:
    """
    Create a contextual header to prepend to chunk text.

    Example output: "[Indian Penal Code | Part II | Chapter XVI | Section 302]"

    Why this matters:
      1. The embedding model encodes the legal source INTO the vector,
         so queries like "Section 302 IPC" become closer in vector space
         to the actual Section 302 chunk.
      2. When the LLM sees the retrieved chunk, it immediately knows
         which Act/Section/Article it's reading — no guessing.
      3. The IRAC citation format in the system prompt works better
         because the source metadata is right in the text.
    """
    parts = []
    if chunk.get("act_name"):
        parts.append(chunk["act_name"])
    if chunk.get("part"):
        parts.append(chunk["part"])
    if chunk.get("chapter"):
        parts.append(chunk["chapter"])
    if chunk.get("article_number"):
        parts.append(chunk["article_number"])
    if chunk.get("section"):
        parts.append(chunk["section"])

    if parts:
        return f"[{' | '.join(parts)}] "
    return ""


def detect_chunk_category(chunk: dict) -> str:
    """
    Assign a legal category to a chunk for Pinecone metadata filtering.

    This enables the query-time `detect_category()` in hybrid_retriever.py
    to filter Pinecone results by legal domain, dramatically improving
    precision for domain-specific queries.
    """
    text = " ".join([
        chunk.get("text", ""),
        chunk.get("act_name") or "",
        chunk.get("source_file") or ""
    ]).lower()

    # Order matters: more specific patterns first
    if any(kw in text for kw in [
        "tort", "negligence", "nuisance", "trespass", "defamation",
        "malicious prosecution", "strict liability", "vicarious liability",
        "damages", "injunction", "malfeasance"
    ]):
        return "Civil Torts"

    if any(kw in text for kw in [
        "ipc", "indian penal code", "murder", "theft", "criminal",
        "nyaya sanhita", "bns", "culpable homicide", "robbery",
        "cheating", "forgery", "kidnapping", "penal"
    ]):
        return "Criminal Law"

    if any(kw in text for kw in [
        "contract", "agreement", "consideration", "breach",
        "indemnity", "bailment", "pledge", "agency", "indian contract"
    ]):
        return "Contract Law"

    if any(kw in text for kw in [
        "constitution", "fundamental right", "directive principle",
        "amendment", "parliament", "preamble", "article",
        "union", "state legislature"
    ]):
        return "Constitutional Law"

    if any(kw in text for kw in [
        "evidence", "witness", "testimony", "sakshya", "confession",
        "hearsay", "burden of proof", "relevancy"
    ]):
        return "Evidence Law"

    if any(kw in text for kw in [
        "civil procedure", "suit", "decree", "plaint",
        "written statement", "civil court", "order", "code of civil"
    ]):
        return "Civil Procedure"

    if any(kw in text for kw in [
        "criminal procedure", "arrest", "bail", "fir",
        "charge", "nagarik suraksha", "crpc", "bnss",
        "investigation", "cognizable"
    ]):
        return "Criminal Procedure"

    if any(kw in text for kw in [
        "transfer of property", "mortgage", "lease", "sale deed",
        "easement", "immovable property"
    ]):
        return "Property Law"

    if any(kw in text for kw in [
        "rti", "right to information", "public authority",
        "information commission"
    ]):
        return "RTI"

    if any(kw in text for kw in [
        "corporate", "company", "nclt", "ibc", "insolvency",
        "bankruptcy", "shares", "shareholder", "director", "board",
        "oppression", "mismanagement", "companies act"
    ]):
        return "Corporate Law"

    if any(kw in text for kw in [
        "interpretation", "statute", "statutory",
        "maxim", "construction of statute"
    ]):
        return "Statutory Interpretation"

    if any(kw in text for kw in [
        "learning the law", "legal education", "jurisprudence",
        "legal system", "courts", "judiciary"
    ]):
        return "Legal Education"

    return "General Law"


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="High-Performance Legal Document Ingestion → Pinecone"
    )
    parser.add_argument(
        "--docs", type=str, default="./legal_docs/",
        help="Directory containing legal PDF/TXT documents"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete existing Pinecone index and re-ingest from scratch"
    )
    parser.add_argument(
        "--embed-batch", type=int, default=256,
        help="FastEmbed batch size (higher = faster, more RAM)"
    )
    parser.add_argument(
        "--upsert-batch", type=int, default=100,
        help="Pinecone upsert batch size"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel PDF workers (default: CPU count)"
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="FastEmbed ONNX inference threads"
    )
    args = parser.parse_args()

    start_time = time.time()

    from pinecone import Pinecone, ServerlessSpec
    from app.config import PINECONE_API_KEY, PINECONE_INDEX_NAME, EMBEDDING_MODEL
    from fastembed import TextEmbedding
    from app.services.chunker import chunk_documents

    if not PINECONE_API_KEY:
        logger.error("❌ PINECONE_API_KEY is not set in .env")
        return

    # ──────────────────────────────────────────────────────────
    # STEP 0: Pinecone Index Setup
    # ──────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("STEP 0/6: Pinecone Index Setup")
    logger.info("=" * 70)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = PINECONE_INDEX_NAME or "legal-rag"

    if args.fresh:
        logger.info(f"🗑️  FRESH MODE: Deleting and recreating index '{index_name}'...")
        existing = [idx["name"] for idx in pc.list_indexes()]
        if index_name in existing:
            pc.delete_index(index_name)
            logger.info(f"   Deleted index '{index_name}'")
            logger.info("   Waiting for deletion to propagate...")
            time.sleep(8)

        pc.create_index(
            name=index_name,
            dimension=1024,  # BAAI/bge-large-en-v1.5
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"   ✅ Created fresh index '{index_name}' (1024 dims, cosine)")
        logger.info("   Waiting for index to become ready...")
        time.sleep(15)
    else:
        existing = [idx["name"] for idx in pc.list_indexes()]
        if index_name not in existing:
            logger.info(f"   Index '{index_name}' not found. Creating...")
            pc.create_index(
                name=index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"   ✅ Created index '{index_name}'")
            time.sleep(15)
        else:
            logger.info(f"   ✅ Using existing index '{index_name}'")

    index = pc.Index(index_name)

    # ──────────────────────────────────────────────────────────
    # STEP 1: Discover Documents
    # ──────────────────────────────────────────────────────────
    docs_dir = Path(args.docs)
    if not docs_dir.exists():
        logger.error(f"❌ Documents directory not found: {docs_dir}")
        return

    files = sorted([
        f for f in docs_dir.iterdir()
        if f.suffix.lower() in ('.pdf', '.txt') and f.stat().st_size > 0
    ])

    total_size_mb = sum(f.stat().st_size for f in files) / (1024 * 1024)

    logger.info(f"\n📁 Found {len(files)} documents ({total_size_mb:.1f} MB total):")
    for f in files:
        logger.info(f"   {'📄' if f.suffix.lower() == '.pdf' else '📝'} "
                     f"{f.name} ({f.stat().st_size / (1024 * 1024):.1f} MB)")

    # ──────────────────────────────────────────────────────────
    # STEP 2: Load Documents (Parallel)
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1/6: Loading documents (parallel PDF parsing)")
    logger.info("=" * 70)

    all_documents = []
    max_workers = args.workers or min(os.cpu_count() or 4, len(files), 8)

    pdf_files = [f for f in files if f.suffix.lower() == '.pdf']
    txt_files = [f for f in files if f.suffix.lower() == '.txt']

    # TXTs first (fast, no parallelism needed)
    for tf in txt_files:
        docs = _load_single_txt(str(tf))
        all_documents.extend(docs)
        logger.info(f"   ✅ {tf.name}: {len(docs)} document(s)")

    # PDFs in parallel
    if pdf_files:
        logger.info(f"   Parsing {len(pdf_files)} PDFs with {max_workers} parallel workers...")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_load_single_pdf, str(f)): f.name
                for f in pdf_files
            }
            completed = 0
            for future in as_completed(futures):
                fname = futures[future]
                completed += 1
                try:
                    docs = future.result()
                    all_documents.extend(docs)
                    logger.info(
                        f"   ✅ [{completed}/{len(pdf_files)}] {fname}: "
                        f"{len(docs)} pages"
                    )
                except Exception as e:
                    logger.error(f"   ❌ [{completed}/{len(pdf_files)}] {fname}: {e}")

    step1_time = time.time() - start_time
    logger.info(f"\n   📊 Total pages loaded: {len(all_documents)} "
                f"(took {step1_time:.0f}s)")

    if not all_documents:
        logger.error("❌ No documents loaded. Check the docs directory.")
        return

    # ──────────────────────────────────────────────────────────
    # STEP 3: Legal-Aware Chunking
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2/6: Hierarchical legal-aware chunking")
    logger.info("=" * 70)

    chunk_start = time.time()
    chunks = chunk_documents(all_documents)
    chunk_time = time.time() - chunk_start
    logger.info(f"   📊 Created {len(chunks)} chunks ({chunk_time:.1f}s)")

    # ──────────────────────────────────────────────────────────
    # STEP 4: Contextual Enrichment
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3/6: Context enrichment + category tagging")
    logger.info("=" * 70)

    enriched_count = 0
    for i, chunk in enumerate(chunks):
        # Prepend contextual header to text
        header = create_context_header(chunk)
        if header:
            chunk["text"] = header + chunk["text"]
            enriched_count += 1

        # Assign category for Pinecone filtering
        chunk["category"] = detect_chunk_category(chunk)
        chunk["id"] = i

    cat_dist = Counter(c["category"] for c in chunks)
    logger.info(f"   📊 Enriched {enriched_count}/{len(chunks)} chunks with context headers")
    logger.info(f"   📊 Category distribution:")
    for cat, count in cat_dist.most_common():
        logger.info(f"      {cat}: {count} chunks")

    # ──────────────────────────────────────────────────────────
    # STEP 5: Embed + Upsert (Streamed)
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4/6: Embedding + Pinecone upsert (streamed pipeline)")
    logger.info("=" * 70)

    embed_model_name = EMBEDDING_MODEL  # "BAAI/bge-large-en-v1.5"
    logger.info(f"   Model:       {embed_model_name}")
    logger.info(f"   Embed batch: {args.embed_batch}")
    logger.info(f"   Upsert batch: {args.upsert_batch}")
    logger.info(f"   ONNX threads: {args.threads}")
    logger.info(f"   Total chunks: {len(chunks)}")

    embedding_model = TextEmbedding(
        model_name=embed_model_name,
        threads=args.threads
    )
    texts = [c["text"] for c in chunks]

    total_upserted = 0
    vectors_batch = []
    embed_start = time.time()

    logger.info(f"\n   Embedding and upserting {len(texts)} chunks...")

    embeddings_gen = embedding_model.embed(texts, batch_size=args.embed_batch)

    for i, emb in enumerate(embeddings_gen):
        chunk = chunks[i]
        chunk_id_str = str(chunk["id"])

        meta = {
            "source_file": chunk.get("source_file") or "",
            "act_name": chunk.get("act_name") or "",
            "section": chunk.get("section") or "",
            "article_number": chunk.get("article_number") or "",
            "category": chunk.get("category") or "General Law",
            "page": chunk.get("page") if chunk.get("page") is not None else 0,
            # NEW Golden Strategy Metadata for Pinecone Filters:
            "status": chunk.get("status") or "unknown",
            "enactment_year": chunk.get("enactment_year") or 0,
            "doc_type": chunk.get("doc_type") or "statute"
        }

        vectors_batch.append({
            "id": chunk_id_str,
            "values": emb.tolist(),
            "metadata": meta
        })

        if len(vectors_batch) >= args.upsert_batch:
            try:
                index.upsert(vectors=vectors_batch)
                total_upserted += len(vectors_batch)
            except Exception as e:
                logger.error(f"   ❌ Upsert failed at chunk {i}: {e}")
                # Retry once with smaller batch
                try:
                    half = len(vectors_batch) // 2
                    index.upsert(vectors=vectors_batch[:half])
                    index.upsert(vectors=vectors_batch[half:])
                    total_upserted += len(vectors_batch)
                    logger.info(f"   ↻ Retry succeeded with split batch")
                except Exception as e2:
                    logger.error(f"   ❌ Retry also failed: {e2}")
            vectors_batch = []

            # Progress logging
            elapsed = time.time() - embed_start
            rate = total_upserted / elapsed if elapsed > 0 else 0
            pct = (i + 1) / len(chunks) * 100
            eta_sec = (len(chunks) - i - 1) / rate if rate > 0 else 0
            eta_min = eta_sec / 60

            if total_upserted % 500 < args.upsert_batch:
                logger.info(
                    f"   ↑ {total_upserted:,}/{len(chunks):,} "
                    f"({pct:.1f}%) | {rate:.1f} chunks/sec | "
                    f"ETA: {eta_min:.1f} min"
                )

    # Final batch
    if vectors_batch:
        try:
            index.upsert(vectors=vectors_batch)
            total_upserted += len(vectors_batch)
        except Exception as e:
            logger.error(f"   ❌ Final upsert failed: {e}")

    embed_time = time.time() - embed_start
    logger.info(f"\n   ✅ Embedded + upserted {total_upserted:,} vectors "
                f"({embed_time:.0f}s, {total_upserted / embed_time:.1f} chunks/sec)")

    # ──────────────────────────────────────────────────────────
    # STEP 6: Save Metadata JSON
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5/6: Saving chunks_metadata.json")
    logger.info("=" * 70)

    output_dir = Path(project_root) / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_path = output_dir / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    meta_size_mb = meta_path.stat().st_size / (1024 * 1024)
    logger.info(f"   ✅ Saved {len(chunks):,} chunks to {meta_path} ({meta_size_mb:.1f} MB)")

    # ──────────────────────────────────────────────────────────
    # STEP 7: Rebuild BM25 Index
    # ──────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6/6: Building BM25 keyword index")
    logger.info("=" * 70)

    bm25_start = time.time()
    from app.services.bm25_index import BM25Index
    bi = BM25Index()
    bi.build(texts, str(output_dir / "bm25_index"))
    bm25_time = time.time() - bm25_start
    logger.info(f"   ✅ BM25 index rebuilt ({bm25_time:.1f}s)")

    # ──────────────────────────────────────────────────────────
    # SUMMARY
    # ──────────────────────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("🎉 INGESTION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"   Documents processed:  {len(files)}")
    logger.info(f"   Pages loaded:         {len(all_documents):,}")
    logger.info(f"   Chunks created:       {len(chunks):,}")
    logger.info(f"   Chunks enriched:      {enriched_count:,} (with context headers)")
    logger.info(f"   Vectors upserted:     {total_upserted:,}")
    logger.info(f"   Embedding model:      {embed_model_name}")
    logger.info(f"   Pinecone index:       {index_name}")
    logger.info(f"   Total time:           {total_time / 60:.1f} minutes ({total_time:.0f}s)")
    logger.info(f"   Overall rate:         {len(chunks) / total_time:.1f} chunks/sec")
    logger.info("")
    logger.info("   Timing breakdown:")
    logger.info(f"     PDF loading:  {step1_time:.0f}s")
    logger.info(f"     Chunking:     {chunk_time:.0f}s")
    logger.info(f"     Embed+Upsert: {embed_time:.0f}s")
    logger.info(f"     BM25:         {bm25_time:.0f}s")
    logger.info("")
    logger.info(f"🚀 Ready! Push to GitHub → Vercel will use Pinecone cloud vectors.")
    logger.info(f"   Test locally: python main.py")


if __name__ == "__main__":
    main()
