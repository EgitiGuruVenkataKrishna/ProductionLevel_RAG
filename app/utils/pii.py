import re
from typing import Tuple, Dict

# Regex patterns for common Indian PII
PII_PATTERNS = {
    "EMAIL": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "PHONE": r'(?:\+91[-\s]?)?[6-9]\d{2}[-\s]?\d{3}[-\s]?\d{4}',
    "AADHAAR": r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    "PAN": r'\b[A-Z]{5}\d{4}[A-Z]\b'
}

def redact_pii(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Redact PII from text and return the redacted text along with a mapping 
    to restore the PII later.
    """
    pii_map = {}
    redacted_text = text
    counter = 1
    
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.finditer(pattern, redacted_text)
        # Process matches in reverse to not mess up indices if we were replacing by index,
        # but with re.sub it's easier, though we need to capture exactly what was matched.
        
        # We will iterate and replace one by one to build the map
        for match in set(re.findall(pattern, redacted_text)):
            placeholder = f"[{pii_type}_{counter}]"
            pii_map[placeholder] = match
            redacted_text = redacted_text.replace(match, placeholder)
            counter += 1
            
    return redacted_text, pii_map

def restore_pii(text: str, pii_map: Dict[str, str]) -> str:
    """
    Restore PII in the text using the provided mapping.
    """
    restored_text = text
    for placeholder, original_value in pii_map.items():
        restored_text = restored_text.replace(placeholder, original_value)
    return restored_text
