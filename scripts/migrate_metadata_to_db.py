"""
Script to migrate chunks_metadata.json to an SQLite database.
"""
import sys
import os
import json
import sqlite3
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import CHUNKS_METADATA_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

def migrate():
    json_path = Path(CHUNKS_METADATA_PATH)
    db_path = json_path.with_suffix(".db")
    
    if not json_path.exists():
        logger.error(f"Source JSON not found: {json_path}")
        return
        
    logger.info(f"Reading JSON from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        
    logger.info(f"Loaded {len(chunks)} chunks. Creating SQLite DB at {db_path}...")
    
    # We use standard sqlite3 here because this is a one-off synchronous script
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                article_number TEXT,
                section TEXT,
                act_name TEXT,
                part TEXT,
                page TEXT,
                source_file TEXT
            )
        """)
        
        # Clear existing
        conn.execute("DELETE FROM chunks")
        
        # Insert all
        records = []
        for idx, chunk in enumerate(chunks):
            records.append((
                idx,  # Force ID to match list index exactly
                chunk.get("text", ""),
                chunk.get("article_number"),
                chunk.get("section"),
                chunk.get("act_name"),
                chunk.get("part"),
                chunk.get("page"),
                chunk.get("source_file")
            ))
            
        conn.executemany("""
            INSERT INTO chunks (id, text, article_number, section, act_name, part, page, source_file)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        
        conn.commit()
        
    logger.info("Migration complete! You can now enable USE_SQLITE_METADATA=true")

if __name__ == "__main__":
    migrate()
