"""
Hierarchical Legal-Aware Document Chunker.

Splits Indian legal texts at natural boundaries (Articles, Sections, Parts)
instead of blind fixed-size windows. Attaches legal metadata to each chunk.
"""
import re
import logging
from app.config import MAX_CHUNK_SIZE, SUB_CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE

logger = logging.getLogger(__name__)


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
    r'(?=(?:^|\n)\s*(?:Article|Art\.?|Section|Sec\.?|S\.?)\s+\d+)',
    re.IGNORECASE | re.MULTILINE
)

# Act name patterns with temporal and domain metadata
ACT_PATTERNS = [
    {"pattern": re.compile(r'(Bharatiya Nyaya Sanhita|BNS)', re.IGNORECASE), "name": "Bharatiya Nyaya Sanhita", "status": "active", "enactment_year": 2023, "doc_type": "statute"},
    {"pattern": re.compile(r'(Bharatiya Nagarik Suraksha Sanhita|BNSS)', re.IGNORECASE), "name": "Bharatiya Nagarik Suraksha Sanhita", "status": "active", "enactment_year": 2023, "doc_type": "statute"},
    {"pattern": re.compile(r'(Bharatiya Sakshya Adhiniyam|BSA)', re.IGNORECASE), "name": "Bharatiya Sakshya Adhiniyam", "status": "active", "enactment_year": 2023, "doc_type": "statute"},
    {"pattern": re.compile(r'(Indian Penal Code|IPC)', re.IGNORECASE), "name": "Indian Penal Code", "status": "repealed", "enactment_year": 1860, "doc_type": "statute"},
    {"pattern": re.compile(r'(Code of Criminal Procedure|CrPC)', re.IGNORECASE), "name": "Code of Criminal Procedure", "status": "repealed", "enactment_year": 1973, "doc_type": "statute"},
    {"pattern": re.compile(r'(Indian Evidence Act)', re.IGNORECASE), "name": "Indian Evidence Act", "status": "repealed", "enactment_year": 1872, "doc_type": "statute"},
    {"pattern": re.compile(r'(Constitution of India)', re.IGNORECASE), "name": "Constitution of India", "status": "active", "enactment_year": 1950, "doc_type": "statute"},
    {"pattern": re.compile(r'\b(Right to Information Act|RTI)\b', re.IGNORECASE), "name": "Right to Information Act", "status": "active", "enactment_year": 2005, "doc_type": "statute"},
    {"pattern": re.compile(r'([\w\s]+Act,?\s*\d{4})', re.IGNORECASE), "name": None, "status": "unknown", "enactment_year": None, "doc_type": "statute"},
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
        "status": "unknown",
        "enactment_year": None,
        "doc_type": "statute"
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
    
    # Extract act name and temporal metadata
    for act_dict in ACT_PATTERNS:
        match = act_dict["pattern"].search(text)
        if match:
            if act_dict["name"] is None:
                metadata["act_name"] = match.group(1).strip()
                # Try to extract year from generic "Act, YYYY"
                year_match = re.search(r'\b(\d{4})\b', metadata["act_name"])
                if year_match:
                    metadata["enactment_year"] = int(year_match.group(1))
            else:
                metadata["act_name"] = act_dict["name"]
                metadata["status"] = act_dict["status"]
                metadata["enactment_year"] = act_dict["enactment_year"]
                metadata["doc_type"] = act_dict["doc_type"]
            break
    
    # Infer act from source filename if not found in text
    if not metadata["act_name"] and source_file:
        fname = source_file.lower()
        if "constitution" in fname:
            metadata["act_name"] = "Constitution of India"
            metadata["status"] = "active"
            metadata["enactment_year"] = 1950
        elif "ipc" in fname or "penal code sections" in fname:
            metadata["act_name"] = "Indian Penal Code"
            metadata["status"] = "repealed"
            metadata["enactment_year"] = 1860
        elif "bns" in fname or "nyaya" in fname:
            metadata["act_name"] = "Bharatiya Nyaya Sanhita"
            metadata["status"] = "active"
            metadata["enactment_year"] = 2023
        elif "crpc" in fname or "criminal procedue" in fname:
            metadata["act_name"] = "Code of Criminal Procedure"
            metadata["status"] = "repealed"
            metadata["enactment_year"] = 1973
        elif "evidence-26" in fname or "evidence act" in fname:
            metadata["act_name"] = "Indian Evidence Act"
            metadata["status"] = "repealed"
            metadata["enactment_year"] = 1872
        elif "sakshya" in fname or "bsa" in fname:
            metadata["act_name"] = "Bharatiya Sakshya Adhiniyam"
            metadata["status"] = "active"
            metadata["enactment_year"] = 2023
    
    return metadata


def _recursive_split(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Fallback: split text into overlapping chunks, prioritizing sentence boundaries."""
    if len(text) <= chunk_size:
        return [text] if len(text.strip()) >= MIN_CHUNK_SIZE else []
        
    try:
        import nltk
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
            nltk.download('punkt', quiet=True)
            sentences = nltk.sent_tokenize(text)
            
        logger.warning(f"Using NLTK sentence chunking fallback for text of length {len(text)}")
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            if current_len + sentence_len + 1 > chunk_size and current_chunk:
                # Join the current chunk
                joined_chunk = " ".join(current_chunk)
                chunks.append(joined_chunk)
                
                # Setup next chunk with overlap
                # Estimate words for overlap (roughly 5 chars per word)
                overlap_words = int(overlap / 5)
                if overlap_words > 0:
                    overlap_text = " ".join(joined_chunk.split()[-overlap_words:])
                    current_chunk = [overlap_text, sentence]
                    current_len = len(overlap_text) + 1 + sentence_len
                else:
                    current_chunk = [sentence]
                    current_len = sentence_len
            else:
                current_chunk.append(sentence)
                current_len += sentence_len + 1
                
        if current_chunk:
            final_chunk = " ".join(current_chunk)
            if len(final_chunk.strip()) >= MIN_CHUNK_SIZE:
                chunks.append(final_chunk.strip())
                
        if chunks:
            return chunks

    except Exception as e:
        logger.warning(f"NLTK sent_tokenize failed ({e}). Using raw character-based split fallback.")

    # Last resort: raw character-based split (Original fallback)
    chunks = []
    separators = ["\n\n", "\n", ". ", " "]
    
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
    
    # Hard split by character, handling words carefully (BUG-015)
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
    parent_mapping = {} # Store parent text for each chunk text
    for chunk in raw_chunks:
        # Save the full section as the parent text
        parent_text = chunk.strip()
        
        if len(chunk) <= MAX_CHUNK_SIZE:
            final_texts.append(chunk)
            parent_mapping[chunk] = parent_text
        else:
            # Sub-split large chunks
            sub_chunks = _recursive_split(chunk, SUB_CHUNK_SIZE, CHUNK_OVERLAP)
            for sc in sub_chunks:
                final_texts.append(sc)
                parent_mapping[sc] = parent_text
    
    # Step 3: Attach metadata to each chunk
    for i, chunk_text in enumerate(final_texts):
        if len(chunk_text.strip()) < MIN_CHUNK_SIZE:
            continue
        
        # Use the parent text to extract metadata, so child chunks get the Section/Chapter tags
        parent_text = parent_mapping[chunk_text]
        meta = extract_legal_metadata(parent_text, source_file, page)
        
        meta["chunk_id"] = i
        meta["text"] = chunk_text.strip()
        meta["parent_text"] = parent_text # Crucial: Save the parent text into metadata
        
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
