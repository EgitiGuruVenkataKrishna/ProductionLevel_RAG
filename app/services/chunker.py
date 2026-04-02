"""
Hierarchical Legal-Aware Document Chunker.

Splits Indian legal texts at natural boundaries (Articles, Sections, Parts)
instead of blind fixed-size windows. Attaches legal metadata to each chunk.
"""
import re
from typing import Optional
from app.config import MAX_CHUNK_SIZE, SUB_CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE


# ==================== LEGAL PATTERN DEFINITIONS ====================
LEGAL_PATTERNS = {
    "article": re.compile(
        r'(?:^|\n)\s*(?:Article|Art\.?)\s+(\d+[A-Z]?)',
        re.IGNORECASE | re.MULTILINE
    ),
    "section": re.compile(
        r'(?:^|\n)\s*(?:Section|Sec\.?|S\.?)\s+(\d+[A-Z]?)',
        re.IGNORECASE | re.MULTILINE
    ),
    "part": re.compile(
        r'(?:^|\n)\s*Part\s+([IVXLCDM]+|\d+)',
        re.IGNORECASE | re.MULTILINE
    ),
    "chapter": re.compile(
        r'(?:^|\n)\s*Chapter\s+([IVXLCDM]+|\d+)',
        re.IGNORECASE | re.MULTILINE
    ),
    "schedule": re.compile(
        r'(?:^|\n)\s*(?:First|Second|Third|Fourth|Fifth|Sixth|Seventh|Eighth|Ninth|Tenth|Eleventh|Twelfth|\d+(?:st|nd|rd|th)?)\s+Schedule',
        re.IGNORECASE | re.MULTILINE
    ),
    "amendment": re.compile(
        r'(?:Constitution\s*\()?\s*(\w+(?:-\w+)?)\s+Amendment\s*(?:Act)?\s*,?\s*(\d{4})?',
        re.IGNORECASE
    ),
}

# Patterns for splitting at article/section boundaries
SPLIT_PATTERN = re.compile(
    r'(?=(?:^|\n)\s*(?:Article|Art\.?|Section|Sec\.?)\s+\d+)',
    re.IGNORECASE | re.MULTILINE
)

# Act name patterns
ACT_PATTERNS = [
    re.compile(r'(Indian Penal Code|IPC)', re.IGNORECASE),
    re.compile(r'(Bharatiya Nyaya Sanhita|BNS)', re.IGNORECASE),
    re.compile(r'(Constitution of India)', re.IGNORECASE),
    re.compile(r'(Code of Criminal Procedure|CrPC)', re.IGNORECASE),
    re.compile(r'(Bharatiya Nagarik Suraksha Sanhita|BNSS)', re.IGNORECASE),
    re.compile(r'(Indian Evidence Act)', re.IGNORECASE),
    re.compile(r'(Bharatiya Sakshya Adhiniyam|BSA)', re.IGNORECASE),
    re.compile(r'\b(Right to Information Act|RTI)\b', re.IGNORECASE),
    re.compile(r'([\w\s]+Act,?\s*\d{4})', re.IGNORECASE),
]


def extract_legal_metadata(text: str, source_file: str = "", page: int = None) -> dict:
    """
    Extract legal metadata from a chunk of text.
    
    Returns dict with: article_number, section, act_name, part, chapter,
                       schedule, amendment
    """
    metadata = {
        "source_file": source_file,
        "page": page,
        "article_number": None,
        "section": None,
        "act_name": None,
        "part": None,
        "chapter": None,
        "schedule": None,
        "amendment": None,
    }
    
    # Extract article number
    match = LEGAL_PATTERNS["article"].search(text)
    if match:
        metadata["article_number"] = f"Article {match.group(1)}"
    
    # Extract section number
    match = LEGAL_PATTERNS["section"].search(text)
    if match:
        metadata["section"] = f"Section {match.group(1)}"
    
    # Extract part
    match = LEGAL_PATTERNS["part"].search(text)
    if match:
        metadata["part"] = f"Part {match.group(1)}"
    
    # Extract chapter
    match = LEGAL_PATTERNS["chapter"].search(text)
    if match:
        metadata["chapter"] = f"Chapter {match.group(1)}"
    
    # Extract schedule
    match = LEGAL_PATTERNS["schedule"].search(text)
    if match:
        metadata["schedule"] = match.group(0).strip()
    
    # Extract amendment
    match = LEGAL_PATTERNS["amendment"].search(text)
    if match:
        amendment_str = match.group(0).strip()
        metadata["amendment"] = amendment_str
    
    # Extract act name
    for pattern in ACT_PATTERNS:
        match = pattern.search(text)
        if match:
            metadata["act_name"] = match.group(1).strip()
            break
    
    # Infer act from source filename if not found in text
    if not metadata["act_name"] and source_file:
        fname = source_file.lower()
        if "constitution" in fname:
            metadata["act_name"] = "Constitution of India"
        elif "ipc" in fname or "penal" in fname:
            metadata["act_name"] = "Indian Penal Code"
        elif "bns" in fname or "nyaya" in fname:
            metadata["act_name"] = "Bharatiya Nyaya Sanhita"
        elif "crpc" in fname:
            metadata["act_name"] = "Code of Criminal Procedure"
    
    return metadata


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback: split text into overlapping chunks by character count."""
    chunks = []
    separators = ["\n\n", "\n", ". ", " "]
    
    if len(text) <= chunk_size:
        return [text] if len(text.strip()) >= MIN_CHUNK_SIZE else []
    
    # Try each separator
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            current_chunk = ""
            for part in parts:
                candidate = current_chunk + sep + part if current_chunk else part
                if len(candidate) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Overlap: keep tail of previous chunk
                    overlap_text = current_chunk[-overlap:] if overlap > 0 else ""
                    current_chunk = overlap_text + sep + part if overlap_text else part
                else:
                    current_chunk = candidate
            if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
            if chunks:
                return chunks
    
    # Last resort: hard split by character, handling words carefully (BUG-015)
    _words = text.split()
    current_chunk = ""
    for word in _words:
        if len(current_chunk) + len(word) + 1 > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Find overlap if any
            overlap_words = current_chunk.split()[-int(overlap/5):] if overlap > 0 else []
            current_chunk = " ".join(overlap_words + [word])
        else:
            current_chunk = current_chunk + " " + word if current_chunk else word
            
    if len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    return chunks


def hierarchical_chunk(text: str, source_file: str = "", page: int = None) -> list[dict]:
    """
    Split legal text using hierarchical boundaries.
    
    Strategy:
    1. Try splitting at Article/Section boundaries first
    2. If chunks are too large, sub-split at sub-section boundaries
    3. Fallback to RecursiveCharacterTextSplitter-style splitting
    
    Returns:
        List of dicts with keys: text, chunk_id, + all legal metadata
    """
    chunks_with_meta = []
    
    # Pre-process text to remove gazette of india headers and noise (BUG-014)
    lines = text.split("\n")
    clean_lines = []
    for line in lines:
        if "THE GAZETTE OF INDIA" in line or line.strip() == "___" or "EXTRAORDINAR Y" in line:
            continue
        clean_lines.append(line)
    clean_text = "\n".join(clean_lines)

    # Step 1: Split at article/section boundaries
    raw_chunks = SPLIT_PATTERN.split(clean_text)
    
    # Filter empty chunks
    raw_chunks = [c.strip() for c in raw_chunks if c.strip() and len(c.strip()) >= MIN_CHUNK_SIZE]
    
    # If no legal boundaries found, the whole text is one block
    if len(raw_chunks) <= 1:
        raw_chunks = [clean_text.strip()]
    
    # Step 2: Process each chunk
    final_texts = []
    for chunk in raw_chunks:
        if len(chunk) <= MAX_CHUNK_SIZE:
            final_texts.append(chunk)
        else:
            # Sub-split large chunks
            sub_chunks = _recursive_split(chunk, SUB_CHUNK_SIZE, CHUNK_OVERLAP)
            final_texts.extend(sub_chunks)
    
    # Step 3: Attach metadata to each chunk
    for i, chunk_text in enumerate(final_texts):
        if len(chunk_text.strip()) < MIN_CHUNK_SIZE:
            continue
        
        meta = extract_legal_metadata(chunk_text, source_file, page)
        meta["chunk_id"] = i
        meta["text"] = chunk_text.strip()
        chunks_with_meta.append(meta)
    
    return chunks_with_meta


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Process a list of documents (from PDF/TXT loader) into legal chunks.
    
    Args:
        documents: List of dicts with keys: text, source_file, page (optional)
    
    Returns:
        List of chunk dicts with text + legal metadata
    """
    all_chunks = []
    global_id = 0
    
    for doc in documents:
        text = doc.get("text", "")
        source = doc.get("source_file", "unknown")
        page = doc.get("page", None)
        
        doc_chunks = hierarchical_chunk(text, source, page)
        
        # Re-number chunk IDs globally
        for chunk in doc_chunks:
            chunk["chunk_id"] = global_id
            global_id += 1
            all_chunks.append(chunk)
    
    return all_chunks
